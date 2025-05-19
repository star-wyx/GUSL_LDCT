import logging

from sklearn.linear_model import LinearRegression
import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


class LNT:
    def __init__(self, args):
        self.A = None
        self.no_tree = args['lnt_tree']
        self.max_depth = args['lnt_depth']
        self.lr = args['lnt_lr']
        self.feature_in_comb = args['lnt_feat_in_comb']

    def fit(self, X_train, y_train, X_val, y_val):
        logger.info(f"Training LNT")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': self.max_depth,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
            'colsample_bylevel': 1,
            'colsample_bynode': 1,
            'learning_rate': self.lr,
            # 'tree_method': "gpu_hist",
            # 'gpu_id': args['gpu_id']
        }

        watchlist = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(params, dtrain,
                          num_boost_round=self.no_tree,
                          evals=watchlist, early_stopping_rounds=10,
                          verbose_eval=False, evals_result={})

        best_iteration = model.best_iteration
        logger.info(f"Best iteration: {best_iteration}")

        tree_paths = self.get_path(model)

        num_features = X_train.shape[1]
        A = np.zeros((num_features, len(tree_paths)))

        logger.info(f'# of combination {len(tree_paths)}')
        for i in range(len(tree_paths)):
            selected = tree_paths[i]
            selected = [int(idx) for idx in selected]
            X_sel = X_train[:, selected]

            lin_reg = LinearRegression()
            lin_reg.fit(X_sel, y_train)

            theta = np.zeros(num_features)
            theta[selected] = lin_reg.coef_

            A[:, i] = theta

        self.A = A
        self.dim = A.shape[1]

    def transform(self, X):
        return X @ self.A

    def get_path(self, model):
        trees_df = model.trees_to_dataframe()
        tree_ids = trees_df['Tree'].unique()
        tree_feat = []

        for tree_id in tree_ids:
            tree_df = trees_df[trees_df['Tree'] == tree_id]
            tmp = set()
            for _, row in tree_df.iterrows():
                feature = row['Feature']
                if feature != 'Leaf':
                    tmp.add(int(feature[1:]))
            tree_feat.append(sorted(tmp))

        res = []
        tmp = set()
        for i in range(len(tree_feat)):
            if len(tmp) < self.feature_in_comb:
                tmp = tmp.union(tree_feat[i])
            else:
                res.append(list(tmp))
                tmp = set()
        if len(tmp) > 0:
            res.append(list(tmp))

        return res