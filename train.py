import random
import argparse
from prep.loader import ct_dataset
from prep.test_LDCT_psnr import trunc, denormalize_
import json
import logging
from skimage.metrics import mean_squared_error
from args import args, data_path
from Model import Model
import os
import numpy as np
from utils import read_args_from_json, write_args_to_json, check_mkdir
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()


def update_logger(cur_level):
    save_path = os.path.join(ckpt_path, exp_name, f'level_{cur_level}')
    check_mkdir(save_path)

    log_filename = os.path.join(save_path, f'logfile.log')

    # Remove existing file handlers from the root logger
    root_logger = logging.getLogger()
    handlers_to_remove = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)
        handler.close()

    # Create and add new file handler to root logger
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(module)s %(levelname)s %(message)s'))
    root_logger.addHandler(file_handler)

    # Ensure all loggers can propagate to root
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).propagate = True

    return save_path


def load_aapm_train(args, training_level):
    logger.info("loading AAPM training data...")
    sample_ratio = args['patch_sample_ratio'][training_level - 1]

    dataset = ct_dataset(mode='train', load_mode=1, saved_path=data_path, test_patient='L506')
    x = np.stack(dataset.input_, axis=0).astype(np.float32)
    y = np.stack(dataset.target_, axis=0).astype(np.float32)

    sample_count = int(y.shape[0] * sample_ratio)
    sample_idx = random.sample(range(y.shape[0]), sample_count)
    x = x[sample_idx]
    y = y[sample_idx]
    logger.info(f'train_gt: {y.shape}, train_lq: {x.shape}')

    x = trunc(denormalize_(x))
    y = trunc(denormalize_(y))

    mse = [mean_squared_error(x[i], y[i]) for i in range(len(y))]
    logger.info(f"Training MSE: {np.mean(mse)}")
    return y, x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="./ckpt", help="Path to save checkpoints")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    input_args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(module)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Exp name: {input_args.exp_name}")

    ckpt_path = input_args.ckpt_path
    exp_name = input_args.exp_name

    json_path = os.path.join(ckpt_path, exp_name, f'args.json')
    depth = len(args['patch_sizes'])
    np.random.seed(args["random_seed"])

    for training_level in range(1, depth + 1):
        if os.path.exists(json_path):
            args = read_args_from_json(json_path)

        save_path = update_logger(training_level)
        logger.info(f"Training level {training_level}")

        gt_patches, lq_patches = load_aapm_train(args, training_level)

        model = Model(args, lq_patches, gt_patches)
        model.load_level()

        if not model.kmeans:
            model.Train_Kmeans(window_size=4, n_clusters=1024)
        model.Get_Kmean_Prediction(window_size=4)

        model.Train_PixelHop(training_level)

        residual = model.Calculate_residual(training_level)
        logger.info(f'residual: {residual.shape}, type: {residual.dtype}')
        logger.info(f'residual MSE: {np.mean(np.square(residual))}')

        model.Train_RFT_LNT(residual, training_level, save_path)
        model.Train_xgb(residual, training_level, save_path)
        model.save_level(training_level, save_path)

        # Train performance
        pred = model.Get_Prediction(training_level)

        down_gt_patches = model.gt_patches[f"level{training_level}"]
        down_lq_patches = model.lq_patches[f"level{training_level}"]
        logger.info(f"Config: {json.dumps(args, indent=4)}")
        logger.info(f"For level {training_level}")
        logger.info(f'Original lq MSE, {mean_squared_error(down_lq_patches, down_gt_patches)}')
        logger.info(f"Training MSE, {mean_squared_error(pred, down_gt_patches)}")

        write_args_to_json(model.args, json_path)
