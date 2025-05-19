from utils import Concat, Shrink

data_range = (-160, 240)
data_path = './data/aapm_npy_img/'

args = {
    'random_seed': 0,
    'patch_sizes': [64, 128, 256, 512],
    'patch_sample_ratio': [1, 0.25, 0.06, 0.02],
    'neighbor_pred_size': 3,  # Get neighbor residual's saab
    'neighbor_size': 5,  # Get neighbor pixel

    # PixelHop
    'pixelhop_sample_ratio': 0.5,
    'pixelHop_paths': [],
    'model_paths': [],

    # RFT
    'rft_sample_size': 2000000,
    'rft_val_ratio': 0.2,
    'num_rft_features': [500, 800, 800, 800, 800],

    # LNT
    'lnt_tree': 500,
    'lnt_depth': 3,
    'show_lnt_rank': 1,
    'lnt_lr': 0.1,
    'lnt_feat_in_comb': 50,

    # XGB
    'xgb_sample_ratio': [1, 1, 1, 1, 1],
    'xgb_val_ratio': 0.2,
    'xgb_depth': 3,
    'xgb_lr': 0.2,
    'xgb_num_round': 8000,
    'xgb_early_stopping_rounds': 10,

    # other
    'dump_fig': True,
}

SaabArgs = [
    {'num_AC_kernels': -1, 'needBias': False, 'cw': False}
]
shrinkArgs = {
    4: [{'func': Shrink, 'win': 5, 'stride': 1, 'pad': 2, 'pool': 1}],
    3: [{'func': Shrink, 'win': 5, 'stride': 1, 'pad': 2, 'pool': 1}],
    2: [{'func': Shrink, 'win': 5, 'stride': 1, 'pad': 2, 'pool': 1}],
    1: [{'func': Shrink, 'win': 5, 'stride': 1, 'pad': 2, 'pool': 1}]
}

srhinkArgs_prev_pred = [
    {'func': Shrink, 'win': 7, 'stride': 1, 'pad': 3, 'pool': 1}
]

concatArg = {'func': Concat}
