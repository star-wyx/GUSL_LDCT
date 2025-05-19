import argparse
import logging
import cv2
from prep.loader import ct_dataset
from prep.measure import compute_measure
from prep.test_LDCT_psnr import trunc, denormalize_
from args import data_path
from Model import Model
import os
import numpy as np
from utils import read_args_from_json, check_mkdir


def test_all_levels(model, current_lq, current_gt):
    model.clear()
    model.lq_patches = {f'level{model.depth}': current_lq}
    model.gt_patches = {f'level{model.depth}': current_gt}

    depth = len(model.args['patch_sizes'])
    model.Get_Kmean_Prediction(window_size=4)
    restored = model.Get_Prediction(depth)
    return restored


def load_aapm_test():
    dataset = ct_dataset(mode='test', load_mode=1, saved_path=data_path, test_patient='L506')
    x = np.stack(dataset.input_, axis=0).astype(np.float32)
    y = np.stack(dataset.target_, axis=0).astype(np.float32)
    x = trunc(denormalize_(x))
    y = trunc(denormalize_(y))
    return y, x


def save_as_png(img, filename, hu_min=-160, hu_max=240):
    img = img.squeeze()
    img = (img - hu_min) / (hu_max - hu_min)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(filename, img)


def main():
    gt_patches, lq_patches = load_aapm_test()
    ori_psnr_lst, final_psnr_lst = [], []
    ori_ssims_lst, final_ssims_lst = [], []
    ori_rmse_lst, final_rmse_lst = [], []

    json_path = os.path.join(ckpt_path, exp_name, f'args.json')
    args = read_args_from_json(json_path)
    model = Model(args)
    model.load_level()

    for img_idx in range(gt_patches.shape[0]):
        gt_img = gt_patches[img_idx:img_idx + 1]
        lq_img = lq_patches[img_idx:img_idx + 1]

        # Initialize restored image
        restored = test_all_levels(model, lq_img, gt_img)
        restored = trunc(restored)
        gt_img = trunc(gt_img)
        lq_img = trunc(lq_img)

        # Calculate and log MSE
        original_result, pred_result = compute_measure(lq_img, gt_img, restored, 240 - (-160))

        ori_psnr_lst.append(original_result[0])
        ori_ssims_lst.append(original_result[1])
        ori_rmse_lst.append(original_result[2])

        final_psnr_lst.append(pred_result[0])
        final_ssims_lst.append(pred_result[1])
        final_rmse_lst.append(pred_result[2])

        # print(f'Image {img_idx} Original PSNR: {original_result[0]:.2f}, '
        #       f'Original SSIM: {original_result[1]:.2f}, '
        #       f'Original RMSE: {original_result[2]:.2f}')
        print(f'Image {img_idx} Final PSNR: {pred_result[0]:.2f}, '
              f'Final SSIM: {pred_result[1]:.2f}, '
              f'Final RMSE: {pred_result[2]:.2f}')

        # Save results
        if args['dump_fig']:
            save_as_png(restored, os.path.join(save_path, f'{img_idx}_restored.png'))
            save_as_png(gt_img, os.path.join(save_path, f'{img_idx}_gt.png'))
            save_as_png(lq_img, os.path.join(save_path, f'{img_idx}_lq.png'))

    print(f'Original PSNR: {np.mean(ori_psnr_lst):.4f}, {np.std(ori_psnr_lst):.4f}')
    print(f'Original SSIM: {np.mean(ori_ssims_lst):.4f}, {np.std(ori_ssims_lst):.4f}')
    print(f'Original RMSE: {np.mean(ori_rmse_lst):.4f}, {np.std(ori_rmse_lst):.4f}')

    print(f'Final PSNR: {np.mean(final_psnr_lst):.4f}, {np.std(final_psnr_lst):.4f}')
    print(f'Final SSIM: {np.mean(final_ssims_lst):.4f}, {np.std(final_ssims_lst):.4f}')
    print(f'Final RMSE: {np.mean(final_rmse_lst):.4f}, {np.std(final_rmse_lst):.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="./ckpt", help="Path to save checkpoints")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    input_args = parser.parse_args()

    ckpt_path = input_args.ckpt_path
    exp_name = input_args.exp_name
    save_path = os.path.join(ckpt_path, exp_name, 'test_result')
    check_mkdir(save_path)

    logging.basicConfig(level=logging.CRITICAL)
    main()
