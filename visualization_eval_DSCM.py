from utils import *
from options.options import *
from net.unet import Unet
import tifffile
from tqdm import tqdm

from torchvision import transforms
from skimage.metrics import structural_similarity as SSIM

size_list = [256, 512, 1024]
zoom_list = ["z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z18", "z22", "z25", "z28", "z38", "z48", "z60", "z90", "z150", "z225"]           

convert_table = [
    [664, 332, 166.0, 'nm/pixel'],
    [320, 160, 80.0, 'nm/pixel'],
    [210, 105, 52.0, 'nm/pixel'],
    [164, 82, 41.0, 'nm/pixel'],
    [125, 62, 31.0, 'nm/pixel'],
    [105, 53, 26.0, 'nm/pixel'],
    [94, 47, 23.0, 'nm/pixel'],
    [78, 39, 20.0, 'nm/pixel'],
    [70, 35, 18.0, 'nm/pixel'],
    [63, 31, 16.0, 'nm/pixel'],
    [59, 29, 15.0, 'nm/pixel'],
    [55, 27, 14.0, 'nm/pixel'],
    [51, 25, 13.0, 'nm/pixel'],
    [47, 23, 12.0, 'nm/pixel'],
    [43, 21, 11.0, 'nm/pixel'],
    [39, 20, 10.0, 'nm/pixel'],
    [35, 18, 9.0, 'nm/pixel'],
    [31, 16, 8.0, 'nm/pixel'],
    [27, 14, 7.0, 'nm/pixel'],
    [23, 12, 6.0, 'nm/pixel'],
    [20, 10, 5.0, 'nm/pixel'],
    [16, 8, 4.0, 'nm/pixel'],
    [12, 6, 3.0, 'nm/pixel'],
    [8, 4, 2.0, 'nm/pixel'],
    [4, 3, 1.5, 'nm/pixel'],
    [1, 2, 1.0, 'nm/pixel']
]

def norm_statistic(Input, device, std=None):
    mean = torch.mean(Input).to(device)
    mean_zero = torch.zeros_like(mean).to(device)
    std = torch.std(Input).to(device) #if std == None else std
    output = transforms.Normalize(mean_zero, std)(Input)
    return output, mean_zero, std


def calculate_nrmae(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    计算两张 float 类型图像的 NRMAE（归一化平均绝对误差）。

    参数:
        image1: numpy.ndarray，float 类型图像
        image2: numpy.ndarray，float 类型图像

    返回:
        NRMAE 值（float）
    """
    if image1.shape != image2.shape:
        raise ValueError(f"图像尺寸不一致: {image1.shape} vs {image2.shape}")

    # 计算 MAE
    mae = np.mean(np.abs(image1 - image2))

    # 使用 image1 的动态范围归一化
    data_range = image1.max() - image1.min()
    if data_range == 0:
        raise ValueError("图像1的动态范围为0，无法归一化")

    nrmae = mae / data_range
    return round(nrmae, 4)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def compute_pcc_per_channel(img1, img2):
    assert img1.shape == img2.shape, "图像尺寸不一致"
    h, w, c = img1.shape
    pccs = []
    for ch in range(c):
        a = img1[:, :, ch].flatten()
        b = img2[:, :, ch].flatten()
        pcc = np.corrcoef(a, b)[0, 1]
        pccs.append(pcc)
    return pccs



def DSCM_enhancement(read_dir, weights_dir, weights_dir_dn=None, 
                     device='cuda', save_dir=False, name=None, show_image=False):
    file_list = natsort.natsorted(os.listdir(read_dir))    
    check_existence(save_dir)
    if weights_dir_dn: 
        model_dn = torch.load(weights_dir, weights_only=False) 
    model = torch.load(weights_dir, weights_only=False)
    #model.eval()
    Input_list = []
    output_list = []
    save_list = []
    bar = tqdm(total=len(file_list))
    with torch.no_grad():
        for index in range(len(file_list)):
            if file_list[index].find('.tif') != -1 or file_list[index].find('.png') != -1:
                read_dir_file = os.path.join(read_dir, file_list[index])
                zoom = 0
                for i in range(len(zoom_list)):
                        if read_dir_file.find(zoom_list[i]) != -1:
                            zoom = zoom_list[i]
                #img = np.array(Image.open(read_dir_file))
                img = tifffile.imread(read_dir_file)
                if len(img.shape) == 3:
                    GT_2 = img[0,:,:]
                    GT_1 = img[2,:,:]
                    img = img[1,:,:]
                raw_h, raw_w = img.shape
                '''for zoom_index in range(len(zoom_list)):
                        if zoom == zoom_list[zoom_index]:
                            for size_index in range(len(size_list)):
                                if raw_h == size_list[size_index]:
                                    #print(read_dir, convert_table[zoom_index][size_index])
                                    convert_ratio = convert_table[zoom_index][size_index]/20
                                    convert_ratio = 22.7 / 20
                                    img = resize(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                                    GT_1 = resize(GT_1, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                                    GT_2 = resize(GT_2, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))'''
                convert_ratio = 22.7 / 20
                img = resize(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                GT_1 = resize(GT_1, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                GT_2 = resize(GT_2, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                #convert_ratio = 22.7 / 20
                #img = resize(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                #img = resize_image_bicubic(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                #GT_1 = resize_image_bicubic(GT_1, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                #GT_2 = resize_image_bicubic(GT_2, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                #img = upscale_lanczos(img, convert_ratio)
                #GT_1 = upscale_lanczos(GT_1, convert_ratio)
                #GT_2 = upscale_lanczos(GT_2, convert_ratio)
                h, w = img.shape
                size = min(h, w)
                upper_limit = 1440
                if size > upper_limit:
                    num_row = (h // upper_limit) + 1
                    num_col = (w // upper_limit) + 1
                    size = upper_limit
                else:
                    upper_limit = (size // 16) * 16
                    num_row = (h // upper_limit) + 1
                    num_col = (w // upper_limit) + 1
                    size = upper_limit
                
                image_list = []
                pred_list = []
                GT_list = []
                row_cood_list = []
                col_cood_list = []
                for row in range(num_row):
                    if (row+1)*size < h:
                        row_cood_list.append([row*size, (row+1)*size])
                    else:
                        row_cood_list.append([h-size, h])
                for col in range(num_col):
                    if (col+1)*size < w:
                        col_cood_list.append([col*size, (col+1)*size])
                    else:
                        col_cood_list.append([w-size, w])
                for row in range(num_row):
                    image_list.append([])
                    pred_list.append([])
                    GT_list.append([])
                    for col in range(num_col):
                        image = img[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        GT_1_image = GT_1[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        GT_2_image = GT_2[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        image = torch.tensor(np.float64(image), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                        image, mean, std = norm_statistic(image, device)
                        if weights_dir_dn:
                            dn = model_dn(image)
                            pred = model(dn)
                        else:
                            pred = model(image)
                        image = image*std+mean
                        pred = pred*std+mean
                        image[image<0] = 0
                        pred[pred<0] = 0
                        image = to_cpu(image.squeeze(0).squeeze(0))
                        pred = to_cpu((pred.squeeze(0).permute(1,2,0))) 
                        GT = np.transpose(np.stack((GT_1_image, GT_2_image), axis=0), (1,2,0))

                        #MAE_value = calculate_nrmae(GT, pred)
                        #PCC_value = compute_pcc_per_channel(GT, pred)
                        #print(MAE_value, PCC_value)
                        #SSIM_value = SSIM(GT, pred)
                        #print(MAE_value, SSIM_value)
                        image_list[row].append(image)
                        pred_list[row].append(pred)
                        GT_list[row].append(GT)
                full_pred = np.zeros((h, w ,pred_list[0][0].shape[-1]))
                for row in range(num_row):
                    for col in range(num_col):
                        full_pred[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]] = pred_list[row][col]
                full_GT = np.zeros((h, w ,GT_list[0][0].shape[-1]))
                for row in range(num_row):
                    for col in range(num_col):
                        full_GT[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]] = GT_list[row][col]
                h, w = img.shape
                #h = int(h * 0.05)
                #w = int(w * 0.05)
                #img = img[h:, w:]
                #full_pred = full_pred[h:, w:, :]
                img = np.uint8(img/np.max(img)*255)
                if file_list[index].find('tif') != -1:
                    tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", "_Input.tif")), img)
                    save_list.append(img)
                    full_pred = np.uint8(full_pred/np.max(full_pred)*255)
                    for i in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", f"_{i}_SR.tif")), (full_pred[:,:,i]))
                        save_list.append(full_pred[:,:,i])
                    full_GT = np.uint8(full_GT/np.max(full_GT)*255)
                    for i in range(full_GT.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", f"_{i}_GT.tif")), (full_GT[:,:,i]))
                        save_list.append(full_GT[:,:,i])
                bar.set_description("{}".format(zoom))
                if show_image:
                    plt.figure(figsize=(10, 6))

                    plt.subplot(231)
                    plt.imshow(img)
                    plt.title("Input Image")

                    plt.subplot(232)
                    plt.imshow(full_GT[:,:,0])
                    plt.title("Ground Truth - Channel 0")

                    plt.subplot(233)
                    plt.imshow(full_GT[:,:,1])
                    plt.title("Ground Truth - Channel 1")

                    plt.subplot(234)
                    plt.imshow(full_pred[:,:,0])
                    plt.title("Prediction - Channel 0")

                    plt.subplot(235)
                    plt.imshow(full_pred[:,:,1])
                    plt.title("Prediction - Channel 1")

                    plt.tight_layout()
                    plt.show()
            bar.update(1)
        if name:
            stack = np.stack(save_list, axis=0)
            tifffile.imwrite(r"D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\resized results\{}.tif".format(name), stack, imagej=True)


if __name__ == "__main__":
    if 0:
        name_list = [
            '228_0.010_4', '228_0.030_4', '228_0.050_4', '228_0.070_4', 
            '240_0.010_4', '240_0.030_4', '240_0.050_4', '240_0.070_4', 
            '260_0.010_4', '260_0.030_4', '260_0.050_4', '260_0.070_4', 
            '228_0.008_4', 
            '240_0.008_4', 
                     ]
        for i in range(len(name_list)):
            read_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\data'
            name = name_list[i]
            weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_{}_DSCM_384_Unet_fea_loss_0.1_SSIM_loss_1_grad_loss_1_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth".format(name)
            #weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_228_0.010_4_DSCM_384_Unet_fea_loss_0_SSIM_loss_0_grad_loss_0_GAN_loss_0_real-time_1000_epoches\weights\1\main_G.pth"
            save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\results'
            DSCM_enhancement(read_dir=read_dir, weights_dir=weights_dir, save_dir=save_dir, name=name)

        name_list = [
            '260_0.008_4', 
        ]
        for i in range(len(name_list)):
            read_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\data'
            name = name_list[i]
            weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_{}_DSCM_384_Unet_fea_loss_0_SSIM_loss_0_grad_loss_0_GAN_loss_0_real-time_1000_epoches\weights\1\main_G.pth".format(name)
            #weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_228_0.010_4_DSCM_384_Unet_fea_loss_0_SSIM_loss_0_grad_loss_0_GAN_loss_0_real-time_1000_epoches\weights\1\main_G.pth"
            save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\results'
            DSCM_enhancement(read_dir=read_dir, weights_dir=weights_dir, save_dir=save_dir, name=name)

        name_list = [
            '215_0.010_4', 
        ]
        for i in range(len(name_list)):
            read_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\data'
            name = name_list[i]
            weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_{}_DSCM_384_Unet_fea_loss_0.1_SSIM_loss_1_grad_loss_1_GAN_loss_1_real-time_1000_epoches_temp\weights\1\main_G.pth".format(name)
            #weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_228_0.010_4_DSCM_384_Unet_fea_loss_0_SSIM_loss_0_grad_loss_0_GAN_loss_0_real-time_1000_epoches\weights\1\main_G.pth"
            save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\results'
            DSCM_enhancement(read_dir=read_dir, weights_dir=weights_dir, save_dir=save_dir, name=name)
    
    if 0:
        name_list = [
            '228_0.005_4', '228_0.010_4', '228_0.015_4', '228_0.025_4', '228_0.035_4', '228_0.045_4', '228_0.055_4', 
            '240_0.005_4', '240_0.010_4', '240_0.015_4', '240_0.025_4', '240_0.035_4', '240_0.045_4', '240_0.055_4', 
            '260_0.005_4', '260_0.010_4', '260_0.015_4', '260_0.025_4', '260_0.035_4', '260_0.045_4', '260_0.055_4', 
                     ]
        for i in range(len(name_list)):
            read_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\data'
            name = name_list[i]
            weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_{}_DSCM_384_Unet_grad_loss_real-time\weights\1\main_G.pth".format(name)
            save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\results'
            DSCM_enhancement(read_dir=read_dir, weights_dir=weights_dir, save_dir=save_dir, name=name)

    if 1:
        read_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\data'
        weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_228_0.010_4_DSCM_384_Unet_fea_loss_0.01_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches_20_Inf_samples\weights\1\main_G.pth"
        save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Mitochondria_eval\results'
        DSCM_enhancement(read_dir=read_dir, weights_dir=weights_dir, save_dir=save_dir)
   