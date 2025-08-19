import os
import torch
import natsort
import math
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np


from PIL import Image
from skimage.metrics import mean_squared_error
from skimage import transform, measure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from openpyxl import Workbook
from random import random,  randint
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset


def check_existence(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        remove_list = os.listdir(dir)
        for i in range(len(remove_list)):
            remove_dir = os.path.join(dir, remove_list[i])
            os.remove(remove_dir)

def pseudo_norm(input, thresh=700):
    input[input < 0] = 0
    input = input / thresh * 255
    #input = input / np.max(input) * 255
    input[input > 255] = 255

def image_displacement(input, x=50, y=-50):
    plain = np.zeros_like(input)
    if y > 0:
        if x > 0:
            x = abs(x)
            y = abs(y)
            plain[:-y,x:] = input[y:,:-x]
        elif x == 0:
            plain[:-y,:] = input[y:,:]
        elif x < 0:
            x = abs(x)
            y = abs(y)
            plain[:-y,:-x] = input[y:,x:]
    elif y < 0:
        if x > 0:
            x = abs(x)
            y = abs(y)
            plain[y:,x:] = input[:-y,:-x]
        elif x == 0:
            plain[-y:,:] = input[:y,:]
        elif x < 0:
            x = abs(x)
            y = abs(y)
            plain[y:,:-x] = input[:-y,x:]
    elif y == 0:
        if x > 0:
            x = abs(x)
            y = abs(y)
            plain[:,x:] = input[:,:-x]
        elif x == 0:
            plain[:,:] = input[:,:]
        elif x < 0:
            x = abs(x)
            y = abs(y)
            plain[:,:-x] = input[:,x:]
        
    return plain

def resize(input, size):
    m1 = np.max(input)
    resized = transform.resize(input, size)
    m2 = np.max(resized)
    output = resized * m1 / m2
    return np.uint16(output)

def resize_image_bicubic(image_np, new_size):
    """
    Resize a numpy image using bicubic interpolation without changing the value range.
    
    Parameters:
    - image_np: np.ndarray, input image (H, W) or (H, W, C)
    - new_size: tuple, (new_width, new_height)

    Returns:
    - resized_np: np.ndarray, resized image with same value range
    """
    value_min = image_np.min()
    value_max = image_np.max()

    # Normalize to 0~255 for PIL compatibility
    img_norm = (image_np - value_min) / (value_max - value_min + 1e-8) * 255
    img_uint8 = img_norm.astype(np.uint8)

    # Convert to PIL Image
    if image_np.ndim == 2:
        img_pil = Image.fromarray(img_uint8, mode='L')
    elif image_np.ndim == 3 and image_np.shape[2] == 3:
        img_pil = Image.fromarray(img_uint8, mode='RGB')
    else:
        raise ValueError("Unsupported image shape")

    # Resize using BICUBIC
    img_resized = img_pil.resize(new_size, resample=Image.BICUBIC)

    # Convert back to numpy and rescale to original range
    resized_np = np.asarray(img_resized).astype(np.float32)
    resized_np = resized_np / 255.0 * (value_max - value_min) + value_min

    return resized_np

def upscale_lanczos(img, scale):
    """
    使用 Lanczos4 内核上采样，兼顾锐利度和抑制振铃。
    
    参数
    ----
    img   : H×W×C 或 H×W ndarray，uint8 / float32 都可
    scale : 放大倍数，可为 2、4 或任意正数
    
    返回
    ----
    up_img: (H*scale)×(W*scale) same dtype as input
    """
    if scale <= 1:
        raise ValueError("scale 必须 > 1 才是上采样")
    h, w = img.shape[:2]
    # INTER_LANCZOS4 内核支撑为 8，能更好抑制振铃
    up_img = cv2.resize(img, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_LANCZOS4)
    return up_img


def to_cpu(input):
    input = Tensor.cpu(input)
    input = input.detach().numpy()
    return input

def show_image(input):
    input = to_cpu(input)
    plt.figure(1)
    plt.imshow(input)
    plt.show()


def get_crop_params(img_size, output_size):
    h, w= img_size
    th = output_size
    tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w
    i = randint(0, h - th)
    j = randint(0, w - tw)
    return i, j, th, tw


def getStat(cal_dataset, device):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(cal_dataset))
    cal_loader = DataLoader(
        cal_dataset, batch_size=1, shuffle=False, num_workers=0,
        )
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for X, _ in cal_loader:
        for d in range(1):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(cal_dataset))
    std.div_(len(cal_dataset))
    #return list(mean.numpy()), list(std.numpy())
    return list(mean), list(std)



def calculate_psnr(img1, img2):
    # 计算均方差（MSE）
    mse = mean_squared_error(img1, img2)

    # 计算信号图像的均值平方
    signal_power = np.max(img1)

    # 计算SNR
    snr = 20 * np.log10(signal_power / np.sqrt(mse))
    #print(snr)
    return snr


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    

def cdist(x, y):
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

def Hausdorffdistance(SR_pos , HR_pos):
    distances = cdist(SR_pos, HR_pos)
    SR_dis = torch.mean(torch.min(distances,1)[0])
    HR_dis = torch.mean(torch.min(distances,0)[0])
    avg_dis =(SR_dis + HR_dis) * 0.5
    return avg_dis

def cal_Hausdorff(SRimg,HRimg):
    HRpos = peak_local_max(HRimg, min_distance=1)
    HR_pos = torch.from_numpy(np.flip(HRpos,axis=0).copy()).float()
    SRpos = peak_local_max(SRimg, min_distance=1)
    SR_pos = torch.from_numpy(np.flip(SRpos,axis=0).copy()).float()
    dis = Hausdorffdistance(SR_pos , HR_pos)
    return dis



def cal_colcalization(img_1, img_2):
    thresh_1 = threshold_otsu(img_1)
    binary_1 = img_1 > thresh_1

    thresh_2 = threshold_otsu(img_2)
    binary_2 = img_2 > thresh_2

    label_image_1 = measure.label(binary_1)
    label_image_2 = measure.label(binary_2)
    colocalized_regions = np.logical_and(label_image_1 > 0, label_image_2 > 0)
    #print(np.mean(colocalized_regions))
    #plt.figure(1)
    #plt.imshow(colocalized_regions)
    #plt.show()

    return colocalized_regions, np.mean(colocalized_regions)


def write2Yaml(data, save_path="test.yaml"):
    with open(save_path, "w") as f:
        yaml.dump(data, f)