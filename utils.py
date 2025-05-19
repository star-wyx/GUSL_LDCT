import json
import logging
import pickle
import os.path
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from skimage.util import view_as_windows
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def Shrink(X, shrinkArg):
    # ---- max pooling----
    pool = shrinkArg['pool']
    if pool != 1:
        X = shrink_patches(X, (X.shape[1] // pool, X.shape[2] // pool), order="lanczos")
    # ---- neighborhood construction
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    # numpy padding
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    X = view_as_windows(X, (1, win, win, X.shape[-1]), (1, stride, stride, X.shape[-1]))

    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)


# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_model(model, path):
    with open(path, 'wb') as file:
        # Dump the dictionary into the file
        pickle.dump(model, file)


def load_model(path):
    with open(path, 'rb') as file:
        # Load the dictionary back from the pickle file.
        model = pickle.load(file)
    return model


def add_neighbor_saab(saab_feature, neighbor):
    res = saab_feature
    if neighbor != 1:
        N, H, W, C = saab_feature.shape
        padding = neighbor // 2
        res = np.pad(saab_feature, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='reflect')
        res = view_as_windows(res, (1, neighbor, neighbor, 1), step=(1, 1, 1, 1))
        res = res.reshape(N, H, W, -1)
    return res


def add_neighbor_pixels(feat, lq_patches, neighbor=1):
    N, H, W = lq_patches.shape
    if neighbor == 1:
        res = lq_patches[:, :, :, np.newaxis]
    else:
        padding = neighbor // 2
        res = np.pad(lq_patches, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')
        res = view_as_windows(res, (1, neighbor, neighbor), step=(1, 1, 1))
        res = res.reshape(N, H, W, -1)

    feat = np.concatenate([feat, res], axis=-1)
    return feat


def plot_train_val_rank(dft, val_dft, path=None):
    training_rank = [dft.dim_rank[dim] for dim in range(dft.dim)]
    validation_rank = [val_dft.dim_rank[dim] for dim in range(val_dft.dim)]
    plt.figure(figsize=(8, 6))
    plt.scatter(training_rank, validation_rank, color='blue', alpha=1)

    plt.title('Feature Rank in Training vs. Validation')
    plt.xlabel('Training Rank')
    plt.ylabel('Validation Rank')
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()


def duplicate_feat(data, target_size, order="mean"):
    logger.info(f'Start duplicate feature to {target_size} using order {order}')
    data_tensor = torch.from_numpy(data).to('cpu')
    N, H, W, C = data.shape
    ratio = target_size[0] / H

    if order == "mean":
        repeat = int(ratio)
        resized = data_tensor.repeat_interleave(repeat, dim=1).repeat_interleave(repeat, dim=2)
    elif order == "lanczos":
        data_tensor = data_tensor.permute(0, 3, 1, 2)
        resized = F.interpolate(
            data_tensor,
            scale_factor=ratio,
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        resized = resized.permute(0, 2, 3, 1)
    else:
        raise ValueError("Unsupported order: choose 'mean' or 'lanczos'")

    resized_cpu = resized.cpu().numpy()
    logger.info('End Resize')
    return resized_cpu


def upscale_pred(pred, target_size, order="mean"):
    N, H, W = pred.shape
    ratio = target_size[0] / H
    logger.info(f'Start upscale pred to {target_size} using order {order}')

    if order == "mean":
        repeat = int(ratio)
        resized = np.repeat(np.repeat(pred, repeat, axis=1), repeat, axis=2)

    elif order == "lanczos":
        scale = (1, ratio, ratio)
        resized = zoom(pred, scale, order=3)

    logger.info('End Resize')
    return resized


def shrink_patches(patch, target_size, padding=0, order="mean"):
    logger.info(f'Start shrink patches to {target_size} using order {order}')
    N = patch.shape[0]
    H = patch.shape[1]
    W = patch.shape[2]
    is_4d = len(patch.shape) == 4

    target_size = (target_size[0] + padding * 2, target_size[1] + padding * 2)
    ratio = target_size[0] / H

    if order == "mean":
        num_pools = int(np.log2(1 / ratio))
        pooled_patch = patch

        for _ in range(num_pools):
            if is_4d:
                # (N, H, W, C) -> (N, H//2, 2, W//2, 2, C)
                shape = (N, pooled_patch.shape[1] // 2, 2,
                         pooled_patch.shape[2] // 2, 2, pooled_patch.shape[3])
                pooled_patch = pooled_patch.reshape(shape).mean(axis=(2, 4))
            else:
                # (N, H, W) -> (N, H//2, 2, W//2, 2)
                shape = (N, pooled_patch.shape[1] // 2, 2,
                         pooled_patch.shape[2] // 2, 2)
                pooled_patch = pooled_patch.reshape(shape).mean(axis=(2, 4))
        return pooled_patch

    elif order == 'lanczos':
        # Handle both 3D and 4D cases
        scale = (1, ratio, ratio, 1) if is_4d else (1, ratio, ratio)
        return zoom(patch, scale, order=3)


def write_args_to_json(args, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(args, json_file, indent=4)


def read_args_from_json(file_path):
    with open(file_path, 'r') as json_file:
        args = json.load(json_file)
    return args
