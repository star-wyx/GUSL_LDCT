import os
import xgboost as xgb
import numpy as np
from skimage.util import view_as_windows
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from LNT import LNT
from RFT_feature import FeatureTest
from PixelHop import Pixelhop
from single_xgboost import SingleXGBoost
from utils import shrink_patches, add_neighbor_saab, plot_train_val_rank, upscale_pred, \
    add_neighbor_pixels, load_model, save_model
import logging
from args import shrinkArgs, SaabArgs, concatArg, srhinkArgs_prev_pred, data_range

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, args, lq_patches=None, gt_patches=None) -> None:
        self.args = args

        # {pixelHop0, pixelHop1, pixelHop2, pixelHop3}
        self.pixelhop_lst = dict()
        # {level1, level2, level3, level4}
        self.rft_lst = dict()
        # {level1, level2, level3, level4}
        self.lnt_lst = dict()
        self.pred_pca = {}

        # {level1, level2, level3, level4}
        self.xgb_lst = dict()
        self.depth = len(args['patch_sizes'])
        self.kmeans = None

        # intermediate results
        # N, H, W, C
        self.saab_features = {}
        # N, H, W, C
        self.level_rft_lnt_feat = {}
        self.level_pred = {}
        self.lq_patches = {f'level{self.depth}': lq_patches}
        self.gt_patches = {f'level{self.depth}': gt_patches}

    def load_pixelHops(self):
        logger.info(f'loading PixelHops...')
        self.pixelhop_lst = load_model(self.args['pixelHop_paths'])

    def Train_PixelHop(self, cur_level):
        lq_patches, _ = self.Get_patches_per_level(cur_level)
        logger.info(f'training PixelHops...')
        sample_size = int(self.args['pixelhop_sample_ratio'] * lq_patches.shape[0])
        sample_idx = np.random.choice(lq_patches.shape[0], size=sample_size, replace=False)
        sample_train_data = lq_patches[sample_idx]
        logger.info(f'PixelHop train sample data: {sample_train_data.shape}')

        shrinkArg = shrinkArgs[cur_level]
        pixelHop = Pixelhop(depth=1, TH1=0, TH2=0,
                            SaabArgs=SaabArgs, shrinkArgs=shrinkArg,
                            concatArg=concatArg)
        pixelHop.fit(sample_train_data)
        self.pixelhop_lst[f'pixelHop{cur_level}'] = pixelHop

    # Feature from all pixelHops, all hops
    def Get_Saab_Feature(self, cur_level):
        if f'pixelHop{cur_level}' in self.saab_features:
            return self.saab_features[f'pixelHop{cur_level}']

        lq_patches, _ = self.Get_patches_per_level(cur_level)
        pixelhop = self.pixelhop_lst[f'pixelHop{cur_level}']
        res = pixelhop.transform_singleHop(lq_patches, layer=0)
        res = add_neighbor_pixels(res, lq_patches, self.args['neighbor_size'])
        logger.info(f'pixelHop{cur_level}: shape {res.shape}')
        self.saab_features[f'pixelHop{cur_level}'] = res
        return res

    def Get_patches_per_level(self, level):
        if f'level{level}' in self.lq_patches:
            return self.lq_patches[f'level{level}'], self.gt_patches[f'level{level}']
        else:
            lq_patches = self.lq_patches[f'level{self.depth}']
            gt_patches = self.gt_patches[f'level{self.depth}']
            target_size = (self.args['patch_sizes'][level - 1], self.args['patch_sizes'][level - 1])
            self.lq_patches[f'level{level}'] = shrink_patches(lq_patches, target_size)
            self.gt_patches[f'level{level}'] = shrink_patches(gt_patches, target_size)
            return self.lq_patches[f'level{level}'], self.gt_patches[f'level{level}']

    def load_level(self):
        model_paths = self.args['model_paths']
        for i, path in enumerate(model_paths):
            logger.info(f'loading previous level{i + 1}...')
            logger.info(f'path: {path}')
            if not os.path.exists(path):
                logger.error(f'Path {path} not exists')
                break
            level = load_model(path)
            self.pixelhop_lst[f'pixelHop{i + 1}'] = level['pixelHop']
            self.rft_lst[f'level{i + 1}'] = level['rft']
            self.xgb_lst[f'level{i + 1}'] = level['xgb']
            self.lnt_lst[f'level{i + 1}'] = level['lnt']
            self.pred_pca[f'level{i + 1}'] = level['pred_pca']
            self.kmeans = level['Kmeans']

    def save_level(self, cur_level, save_path):
        res = {
            'pixelHop': self.pixelhop_lst[f'pixelHop{cur_level}'],
            'rft': self.rft_lst[f'level{cur_level}'],
            'xgb': self.xgb_lst[f'level{cur_level}'],
            'lnt': self.lnt_lst[f'level{cur_level}'],
            'pred_pca': self.pred_pca[f'level{cur_level}'],
            'Kmeans': self.kmeans
        }
        save_model(res, os.path.join(save_path, f"level_{cur_level}.pkl"))
        self.args['model_paths'].append(os.path.join(save_path, f"level_{cur_level}.pkl"))

    def Train_RFT_LNT(self, y, cur_level, save_path):
        saab_features = self.Get_Saab_Feature(cur_level)
        logger.info(f'Train RFT and LNT for level {cur_level}')
        y = y.flatten()

        sample_size = min(int(self.args['rft_sample_size']), y.shape[0])
        sample_idx = np.random.choice(y.shape[0], size=sample_size, replace=False)
        sample_y = y[sample_idx]

        N, H, W, C = saab_features.shape
        x = saab_features.reshape(-1, C)
        sample_x = x[sample_idx]

        prev_pred = self.Upscale_Prev_Pred_PCA(cur_level)
        sample_x = np.concatenate((sample_x, prev_pred[sample_idx]), axis=-1)

        X_train, X_val, y_train, y_val = train_test_split(sample_x, sample_y, test_size=self.args['rft_val_ratio'],
                                                          random_state=42)

        logger.info(f'Train RFT, shape: {X_train.shape}')
        rft = FeatureTest('rmse')
        rft.fit(X_train, y_train, n_bins=16, outliers=True)
        rft.plot(path=os.path.join(save_path, f"train_rft_{cur_level}.png"))

        logger.info(f'Val RFT, shape: {X_val.shape}')
        rft_val = FeatureTest('rmse')
        rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
        rft_val.plot(path=os.path.join(save_path, f"val_rft_{cur_level}.png"))

        plot_train_val_rank(rft, rft_val, path=os.path.join(save_path, f"joint_{cur_level}.png"))

        # Train and validation overlap
        n_selected = self.args['num_rft_features'][cur_level - 1]
        overlap_feat = np.intersect1d(rft.sorted_features[:n_selected], rft_val.sorted_features[:n_selected])
        rft.sorted_features = overlap_feat
        rft.n_selected = len(overlap_feat)

        X_train = rft.transform(X_train, n_selected=rft.n_selected)
        X_val = rft.transform(X_val, n_selected=rft.n_selected)
        self.rft_lst[f'level{cur_level}'] = rft

        lnt = LNT(self.args)
        lnt.fit(X_train, y_train, X_val, y_val)
        logger.info(f"Get LNT {lnt.dim} features")
        self.lnt_lst[f'level{cur_level}'] = lnt

        if self.args['show_lnt_rank']:
            label_dims = {}
            X_train = np.concatenate((X_train, lnt.transform(X_train)), axis=-1)
            N, C = X_train.shape
            plt_rft = FeatureTest('rmse')
            plt_rft.fit(X_train, y_train, n_bins=16, outliers=True)
            label_dims['LNT Features'] = list(range(C - lnt.dim, C))
            plt_rft.plot(path=os.path.join(save_path, f"train_rft_lnt_{cur_level}.png"), label_dims=label_dims)

    def Get_RFT_LNT_Feature(self, cur_level, flatten=True):
        if cur_level in self.level_rft_lnt_feat:
            logger.info(f'Reload RFT/LNT Features from Level {cur_level}')
            res = self.level_rft_lnt_feat[cur_level]
            N, H, W, C = res.shape
            if flatten:
                return res.reshape(N * H * W, C)
            else:
                return res

        res = self.Get_Saab_Feature(cur_level)
        N, H, W, C = res.shape
        res = res.reshape(N * H * W, C)

        # add last level's denoised patches & PCA
        prev_pred = self.Upscale_Prev_Pred_PCA(cur_level)
        res = np.concatenate((res, prev_pred), axis=-1)

        rft = self.rft_lst[f'level{cur_level}']
        res = rft.transform(res, n_selected=rft.n_selected)
        logger.info(f"Get {rft.n_selected} RFT features from {rft.dim}")

        lnt = self.lnt_lst[f'level{cur_level}']
        lnt_feat = lnt.transform(res)
        res = np.concatenate((res, lnt_feat), axis=-1)

        self.level_rft_lnt_feat[cur_level] = res.reshape(N, H, W, -1)
        if flatten:
            return res
        else:
            return self.level_rft_lnt_feat[cur_level]

    def Train_xgb(self, resd, cur_level, save_path):
        logger.info(f'Train XGBoost for level {cur_level}')
        resd = resd.flatten()

        sample_size = int(self.args['xgb_sample_ratio'][cur_level - 1] * resd.shape[0])
        sample_idx = np.random.choice(resd.shape[0], size=sample_size, replace=False)
        sample_y = resd[sample_idx]

        X = self.Get_RFT_LNT_Feature(cur_level, flatten=True)
        sample_x = X[sample_idx]

        logger.info(f'Train XGBoost, input data {X.shape}')
        X_train, X_val, y_train, y_val = train_test_split(sample_x, sample_y, test_size=self.args['xgb_val_ratio'],
                                                          random_state=42)
        sxgb = SingleXGBoost(self.args)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        sxgb.fit(dtrain, dval)

        # plot learning curve
        sxgb.plot_learning_curve(eval_metric="rmse", path=os.path.join(save_path, "rmse.png"))

        # Train score report
        y_pred_train = sxgb.predict(dtrain)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        logger.info(f'mse_train: {mse_train} mae_train: {mae_train}')

        # Validation score report
        y_pred_val = sxgb.predict(dval)
        mse_val = mean_squared_error(y_val, y_pred_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        logger.info(f"mse_val: {mse_val} mae_val: {mae_val}")

        self.xgb_lst[f'level{cur_level}'] = sxgb

    def Upscale_Prev_Pred_PCA(self, cur_level):
        # construct neighborhood
        lq_patch, _ = self.Get_patches_per_level(cur_level)
        y_prev_pred = self.Get_Prediction(cur_level - 1)
        target_size = (self.args['patch_sizes'][cur_level - 1], self.args['patch_sizes'][cur_level - 1])
        y_prev_pred = upscale_pred(y_prev_pred, target_size)
        resd = lq_patch - y_prev_pred

        if f'level{cur_level}' not in self.pred_pca:
            pixelHop = Pixelhop(depth=1, TH1=0, TH2=0,
                                SaabArgs=SaabArgs, shrinkArgs=srhinkArgs_prev_pred,
                                concatArg=concatArg)
            pixelHop.fit(resd)
            self.pred_pca[f'level{cur_level}'] = pixelHop
        features = self.pred_pca[f'level{cur_level}'].transform_singleHop(resd, layer=0)
        features = add_neighbor_saab(features, self.args['neighbor_pred_size'])
        res = add_neighbor_pixels(features, resd, neighbor=1)
        N, H, W, C = res.shape
        res = res.reshape(N * H * W, C)
        logger.info(f'Upscale previous prediction & PCA, shape: {res.shape}')
        return res

    def Get_Prediction(self, cur_level):
        if cur_level in self.level_pred:
            logger.info(f'Reload Prediction from Level {cur_level}')
            return self.level_pred[cur_level]

        logger.info(f"Compute Prediction of Level {cur_level}")
        features = self.Get_RFT_LNT_Feature(cur_level, flatten=True)

        dtest = xgb.DMatrix(features)
        y_pred = self.xgb_lst[f'level{cur_level}'].predict(dtest)
        y_pred = y_pred.reshape(-1, self.args['patch_sizes'][cur_level - 1], self.args['patch_sizes'][cur_level - 1])
        logger.info(f"Current level {cur_level} xgb predict {y_pred.shape}")

        prev_res = self.Get_Prediction(cur_level - 1)
        target_size = (self.args['patch_sizes'][cur_level - 1], self.args['patch_sizes'][cur_level - 1])
        prev_res = upscale_pred(prev_res, target_size)
        y_pred = np.clip(prev_res + y_pred, data_range[0], data_range[1])
        self.level_pred[cur_level] = y_pred
        return y_pred

    def Train_Kmeans(self, window_size=4, n_clusters=16):
        lq_patches, gt_patches = self.Get_patches_per_level(level=1)
        N, H, W = lq_patches.shape

        lq_patches = view_as_windows(
            lq_patches,
            window_shape=(1, window_size, window_size),
            step=(1, window_size, window_size)
        )
        lq_patches = lq_patches.reshape(-1, window_size * window_size)

        logger.info('Fitting KMeans model.')
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(lq_patches)

    def Get_Kmean_Prediction(self, window_size=4):
        lq_patches, _ = self.Get_patches_per_level(level=1)
        N, H, W = lq_patches.shape

        x = view_as_windows(
            lq_patches,
            window_shape=(1, window_size, window_size),
            step=(1, window_size, window_size)
        )
        x = x.reshape(-1, window_size * window_size)

        logger.info('Using existing KMeans model to predict cluster labels.')
        cluster_labels = self.kmeans.predict(x)

        res = self.kmeans.cluster_centers_[cluster_labels]
        num_patches_h = H // window_size
        num_patches_w = W // window_size
        res = res.reshape(N, num_patches_h, num_patches_w, window_size, window_size)
        res = res.transpose(0, 1, 3, 2, 4).reshape(N, H, W)
        self.level_pred[0] = res

    def Calculate_residual(self, cur_level):
        lq_patch, gt_patch = self.Get_patches_per_level(cur_level)
        y_prev_pred = self.Get_Prediction(cur_level - 1)
        target_size = (self.args['patch_sizes'][cur_level - 1], self.args['patch_sizes'][cur_level - 1])
        y_prev_pred_upsize = upscale_pred(y_prev_pred, target_size)
        residual = gt_patch - y_prev_pred_upsize
        return residual

    def clear(self):
        self.level_pred = {}
        self.level_rft_lnt_feat = {}
        self.saab_features = {}
        self.lq_patches = {}
        self.gt_patches = {}
