import numpy as np
from prep.loader import get_loader
from prep.measure import compute_measure

norm_range_max = 3072
norm_range_min = -1024
trunc_max = 240
trunc_min = -160


def mayo_npy_mse(test_patient):
    data_loader = get_loader(mode="test",
                             load_mode=0,
                             saved_path='./data/mayo_npy_img_1mm',
                             test_patient=test_patient,
                             patch_n=None,
                             patch_size=None,
                             transform=False,
                             batch_size=1,
                             num_workers=1)

    ori_rmse = []
    ori_psnr = []

    for i, (x, y) in enumerate(data_loader):
        shape_ = x.shape[-1]
        x = x.unsqueeze(0).float().to('cpu')
        y = y.unsqueeze(0).float().to('cpu')

        x = trunc(denormalize_(x.view(shape_, shape_).cpu().detach()))
        y = trunc(denormalize_(y.view(shape_, shape_).cpu().detach()))

        original_result, pred_result = compute_measure(x, y, x, trunc_max - trunc_min)
        ori_psnr.append(original_result[0])
        ori_rmse.append(original_result[2])

    print(f"Mean RMSE: {np.mean(ori_rmse)}")
    print(f"Mean PSNR: {np.mean(ori_psnr)}")
    return np.mean(ori_rmse), np.mean(ori_psnr)


def denormalize_(image):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image


def trunc(mat):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    return mat

# LDCT PSNR for Testing data (1mm L506)
if __name__ == "__main__":
    rmse_lst = []
    psnr_lst = []

    rmse, psnr = mayo_npy_mse('L506')
    rmse_lst.append(rmse)
    psnr_lst.append(psnr)
    print(f"Mean RMSE: {np.mean(rmse_lst)}")
    print(f"Mean PSNR: {np.mean(psnr_lst)}")
    print(f"Std PSNR: {np.std(psnr_lst)}")
