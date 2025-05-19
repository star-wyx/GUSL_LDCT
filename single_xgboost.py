import os
import time
import torch
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import logging

from xgboost.callback import TrainingCallback

logger = logging.getLogger(__name__)


class LogCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        log_message = f"[{epoch}]"
        for data_name, metrics in evals_log.items():
            for metric_name, metric_value in metrics.items():
                log_message += f"   {data_name}-{metric_name}:{metric_value[-1]:.5f}"
        logger.info(log_message)
        return False


class SingleXGBoost:
    def __init__(self, args, clf=False):
        params = {
            "tree_method": "hist",
            'max_depth': args['xgb_depth'],
            'learning_rate': args['xgb_lr'],
        }
        
        if clf:
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'base_score': 0.5
            })
        else:
            params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            })
        
        self.params = params
        self.num_boost_round = args['xgb_num_round']
        self.early_stopping_rounds = args['xgb_early_stopping_rounds']

    def fit(self, DMatrix_train, DMatrix_val, *args, **kwargs):
        self.evals_result = {}
        watchlist = [(DMatrix_train, 'train'), (DMatrix_val, 'val')]
        self.bst = xgb.train(self.params, DMatrix_train, self.num_boost_round,
                             evals=watchlist, early_stopping_rounds=self.early_stopping_rounds,
                             verbose_eval=False, evals_result=self.evals_result,
                             callbacks=[LogCallback()], *args, **kwargs)
        return self

    def predict(self, X, iteration_range=None):
        start = time.time()
        if isinstance(X, np.ndarray) or isinstance(X, torch.Tensor):
            X = xgb.DMatrix(X)
        if iteration_range is None:
            iteration_range = (0, self.bst.best_iteration + 1)
        logger.info(f'iter range{iteration_range}')
        output = self.bst.predict(X, iteration_range=iteration_range)
        logger.info(f"XGBoost Predict Finished in {time.time() - start} seconds.")
        return output

    def inplace_predict(self, X, iteration_range=None):
        start = time.time()
        if iteration_range is None:
            iteration_range = (0, self.bst.best_iteration + 1)
        logger.info(iteration_range)
        output = self.bst.inplace_predict(X, iteration_range=iteration_range)
        logger.info(f"XGBoost Inplace Predict Finished in {time.time() - start} seconds.")
        return output

    def cuda(self, gpu_id=0):
        self.bst.set_param({"predictor": "gpu_predictor"})
        self.bst.set_param({"gpu_id": gpu_id})

    def cpu(self):
        self.bst.set_param({"predictor": "cpu_predictor"})

    def to(self, device):
        if device == 'cuda':
            self.cuda()
        else:
            self.cpu()

    def plot_learning_curve(self, eval_metric='logloss', path=None):
        plt.figure()
        plt.plot(self.evals_result['train'][eval_metric], label='train')
        plt.plot(self.evals_result['val'][eval_metric], label='val')
        plt.xlabel('Iteration')
        plt.ylabel(eval_metric)
        plt.legend()
        if path is not None:
            plt.savefig(path)
            plt.show()
            plt.close()
        else:
            plt.show()

    def get_feature_importance(self, importance_type='gain'):
        feature_importance = self.bst.get_score(importance_type=importance_type)
        feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        feature_importance = np.array(feature_importance)
        feature_importance = [int(i[1:]) for i in feature_importance[:, 0]]
        return feature_importance

    def get_feature_importance_score(self, importance_type='gain'):
        feature_importance = self.bst.get_score(importance_type=importance_type)
        feature_importance = feature_importance.items()
        feature_importance = [y for x, y in feature_importance]
        return feature_importance

    def get_num_boost_round(self):
        return self.bst.num_boost_round

    def save_model(self, model_path):
        start = time.time()
        self.bst.save_model(model_path)
        logger.info(f"Save XGBoost Finished in {time.time() - start} seconds.")


def plot_distribution(y, prob):
    plt.hist(prob[y == 0], bins=100, alpha=0.5, label='0')
    plt.hist(prob[y == 1], bins=100, alpha=0.5, label='1')
    plt.legend()
    plt.show()
