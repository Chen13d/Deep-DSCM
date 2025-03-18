import os
import sys
import copy 
import time
import glob
import torch
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from random import random,  randint
from torchvision import transforms
from torchvision.transforms import Resize 
from skimage.filters import threshold_otsu

import torch.nn.functional as F
from torch import nn
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset
sys.path.append('./utils')
from utils import *

import scipy.ndimage as ndimage

def calculate_fwhm(psf):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a 2D Point Spread Function (PSF).
    
    Parameters:
    psf (2D numpy array): The 2D PSF array.

    Returns:
    float: The FWHM value.
    """
    # Normalize the PSF
    psf = psf / np.max(psf)
    h, w = psf.shape
    psf = resize(psf*255, (1024, 1024))
    
    psf = psf / np.max(psf)
    # Find the half maximum value
    half_max = 0.5
    
    # Find the coordinates where the PSF is greater than half max
    indices = np.where(psf >= half_max)
    
    # Get the bounding box of these coordinates
    x_min, x_max = np.min(indices[1]), np.max(indices[1])
    y_min, y_max = np.min(indices[0]), np.max(indices[0])
    
    # Calculate the width in both directions
    width_x = (x_max - x_min) * w / 1024 * 20
    width_y = (y_max - y_min) * h / 1024 * 20
    # Calculate the FWHM as the mean of the widths in x and y directions
    return width_x, width_y


# get the parameters of cropping
def get_crop_params(img_size, output_size):
    h, w = img_size
    th = output_size
    tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w
    i = randint(0, h - th)
    j = randint(0, w - tw)
    #print(h, w, h - th, w - tw, i, j)
    #print(i, j, th, tw)
    return i, j, th, tw


def rand_crop_single(img, size):
    i,j,height,width = get_crop_params(img_size=img.shape[0:2], output_size=size) 
    img = img[i:i+height, j:j+width]
    return img
def rand_crop_single_with_mask(img, mask, size, min_size):
    i,j,height,width = get_crop_params(img_size=(min_size, min_size), output_size=size) 
    img = img[i:i+height, j:j+width]
    mask = mask[i:i+height, j:j+width]    
    return img, mask
def rand_crop_with_mask(img_1, img_2, mask, size, min_size):
    i,j,height,width = get_crop_params(img_size=(min_size, min_size), output_size=size) 
    img_1 = img_1[i:i+height, j:j+width]
    img_2 = img_2[i:i+height, j:j+width]
    mask = mask[i:i+height, j:j+width]    
    return img_1, img_2, mask
# crop image, both images are cropped in same size
def rand_crop(img1, img2, size):
    i,j,height,width = get_crop_params(img_size=img1.shape[0:2], output_size=size) 
    img1 = img1[i:i+height, j:j+width]
    img2 = img2[i:i+height, j:j+width]
    return img1, img2
# crop image, both images are cropped in same ratio but different size
def rand_crop_up(img1, img2, size):
    i,j,height,width = get_crop_params(img_size=img2.shape[0:2], output_size=size) 
    img1 = img1[i:i+height, j:j+width]
    img2 = img2[i:i+height, j:j+width]
    return img1, img2

# Dataset
class Dataset_decouple_SR(Dataset):
    def __init__(self, GT_dir_list_DS, GT_dir_list_D, device, num_file, up_factor, factor_list, 
                 size=512, noise_level=0.5, output_list = None, denoise=False, train_flag=True, 
                 random_selection=False, crop_flag=True, flip_flag=True, eval_flag=False, w0=2.05, read_LR=False):
        super(Dataset_decouple_SR, self).__init__()
        self.num_file = num_file
        self.up_factor = up_factor
        self.noise_level = noise_level
        self.random_selection = random_selection
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        self.output_list = output_list
        self.denoise = denoise
        self.train_flag = train_flag
        self.eval_flag = eval_flag
        self.dir_list_DS = []
        self.dir_list_D = []
        self.dir_list_S = []
        self.w0 = w0
        self.read_LR = read_LR
        for i in range(len(GT_dir_list_DS)):
            self.dir_list_DS.append(glob.glob(GT_dir_list_DS[i]+'/*'))
        for i in range(len(GT_dir_list_D)):
            self.dir_list_D.append(glob.glob(GT_dir_list_D[i]+'/*'))
        #if not crop_flag:            
        '''for i in range(len(GT_dir_list_DS)):
            if i == 0:
                self.num_min = len(self.dir_list_DS[i])
            else:
                if len(self.dir_list_DS[i]) < self.num_min: self.num_min = len(self.dir_list_DS[i])   
        if self.num_file <= self.num_min: self.num_min = self.num_file'''                 
        self.size = size
        self.device = device  
        self.factor_list = factor_list
        self.plain = np.zeros((self.size, self.size))
        if self.w0 > 0:
            N = 129
            w0 = 0.65
            span = 12
            psf_STED = self.generate_psf(m=0, N=N, w0=w0, span=span)
            fwhm = calculate_fwhm(psf_STED)
            #print("FHWM of STED psf:", fwhm[0])
            cv2.imencode('.tif', psf_STED)[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\STED_psf.tif')

            N = 129
            w0 = self.w0
            span = 12
            psf_confocal = self.generate_psf(m=0, N=N, w0=w0, span=span)
            fwhm = calculate_fwhm(psf_confocal)
            print("FHWM of confocal psf:", fwhm[0])
            cv2.imencode('.tif', psf_confocal)[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\confocal_psf.tif')

            # 计算OTF
            otf_STED = np.fft.fftshift(np.fft.fft2(psf_STED))
            #otf_magnitude_STED = np.abs(otf_STED)
            otf_confocal = np.fft.fftshift(np.fft.fft2(psf_confocal))
            #otf_magnitude_confocal = np.abs(otf_confocal)
            otf_cal = otf_confocal / otf_STED
            psf_cal = np.fft.fftshift(np.fft.fft2(otf_cal))
            psf_cal = np.abs(psf_cal)
            #psf_cal = tifffile.imread(r"D:\CQL\codes\microscopy_decouple_on_submission\models\cal_psf.tif")
            fwhm = calculate_fwhm(psf_cal)
            print("FHWM of cal psf:", fwhm[0])
            psf_cal = psf_cal / np.sum(psf_cal)
            self.psf_cal = torch.FloatTensor(psf_cal).to(self.device).unsqueeze(0)
            self.psf_cal = nn.Parameter(data=self.psf_cal, requires_grad=False).to(self.device)

                
    def generate_psf(self, m, N=1024, span=6, lamb=635e-9, w0=2):
        k = 2 * np.pi / lamb
        beta = 50 * np.pi / 180
        x = np.linspace(-span, span, N)
        y = np.linspace(-span, span, N)
        [X, Y] = np.meshgrid(x, y)
        [r, theta] = cv2.cartToPolar(X, Y)
        E = np.power((r / w0), m) * np.exp(-np.power(r, 2) / np.power(w0, 2)) * np.exp(1j*beta) * np.exp(-1j * m * theta)
        I = np.real(E * np.conj(E))
        I /= np.sum(I) 
        #I = torch.FloatTensor(I).to(self.device).unsqueeze(0)
        #I = nn.Parameter(data=I, requires_grad=False).to(self.device)
        return I
    # set the number of enumerations according to options
    def __len__(self):
        return self.num_file        
    def add_poisson_pytorch(self, Input, intensity=1.0):
        Input_max = torch.max(Input)
        temp = Input / Input_max
        temp = torch.poisson(temp * intensity) / intensity * Input_max
        #return temp
        ratio = Input / Input_max
        Input = ratio * Input + (1 - ratio) * temp
        return Input
        
    def add_poisson_numpy(self, Input, intensity):        
        Input_max = torch.max(Input)
        Input = to_cpu(Input / Input_max)
        Input = np.random.poisson(Input * intensity) / intensity
        Input = torch.tensor(Input, dtype=torch.float, device=self.device) * Input_max
        return Input
    def add_noise(self, Input):   
        Input = self.add_poisson_pytorch(Input, intensity=self.noise_level)
        #Input = self.add_poisson_numpy(Input, intensity=self.noise_level)
        return Input
    def norm_statistic(self, Input, std=None):
        mean = torch.mean(Input).to(self.device)
        mean_zero = torch.zeros_like(mean).to(self.device)
        std = torch.std(Input).to(self.device) if std == None else std
        output = transforms.Normalize(mean_zero, std)(Input)
        return output, mean_zero, std
        #tensor_max = torch.tensor(np.float64(50), dtype=torch.float, device=self.device)
        #output = Input / tensor_max
        #return output, mean_zero, tensor_max
    def map_values(self, image, new_min=0, new_max=1, min_val=None, max_val=None, percentile=100, index=0):
        if index == 0:
            # 计算指定百分位数的最小值和最大值
            min_val = torch.quantile(image, (100 - percentile) / 100)
            max_val = torch.quantile(image, percentile / 100)
            # 避免除以零的情况
            if max_val == min_val:
                raise ValueError("最大值和最小值相等，无法进行归一化。")
            
        # 将图像值缩放到新范围
        scaled = (image - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
        
        # 可选：将值限制在新范围内
        #scaled = torch.clamp(scaled, min=new_min, max=new_max)
        return scaled, min_val, max_val
    def gen_mask(self, Input, kernel_size=(7,7), iteration=7):
        thresh = threshold_otsu(Input)
        Input[Input<thresh] = 0
        Input[Input>=thresh] = 1
        return Input
    def rand_crop_single_with_mask(self, img, mask, size, min_size, index):
        self.i,self.j,self.height,self.width = get_crop_params(img_size=(min_size, min_size), output_size=size) 
        #if index == 0: self.i,self.j,self.height,self.width = get_crop_params(img_size=(min_size, min_size), output_size=size) 
        img = img[self.i:self.i+self.height, self.j:self.j+self.width]
        #mask = mask[self.i:self.i+self.height, self.j:self.j+self.width]    
        return img, mask
    def rand_crop_with_mask(self, img_1, img_2, mask, size, min_size, index):
        if index == 0: self.i,self.j,self.height,self.width = get_crop_params(img_size=(min_size, min_size), output_size=size) 
        img_1 = img_1[self.i:self.i+self.height, self.j:self.j+self.width]
        img_2 = img_2[self.i:self.i+self.height, self.j:self.j+self.width]
        mask = mask[self.i:self.i+self.height, self.j:self.j+self.width]    
        return img_1, img_2, mask
    def __getitem__(self, index):
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=int(random()>0.5))
        self.vertical_flip = transforms.RandomVerticalFlip(p=int(random()>0.5))       
        # list for different LR and HR components
        GT_list_D = []    
        GT_list_DS = []
        HR_list = []
        LR_list = []
        #max_list = []
        for it in range(len(self.dir_list_DS)):
            D_index = index % len(self.dir_list_DS[it])
            #print(D_index, len(self.dir_list_DS[it]))
            HR = Image.open(self.dir_list_DS[it][D_index])
            HR_list.append(HR)
            if self.read_LR:
                LR = Image.open(self.dir_list_D[it][D_index])
                LR_list.append(LR)
        min_list = []
        mask_list = []
        for it in range(len(self.dir_list_D)):
            if self.random_selection:
                D_index = randint(0, len(self.dir_list_DS[it])-1)
            else:
                D_index = index % len(self.dir_list_DS[it])
            
            if self.flip_flag:   
                HR_list[it] = np.array(self.horizontal_flip(self.vertical_flip(HR_list[it])))
                if self.read_LR: LR_list[it] = np.array(self.horizontal_flip(self.vertical_flip(LR_list[it])))
            else:
                HR_list[it] = np.array(HR_list[it])
                if self.read_LR: LR_list[it] = np.array(LR_list[it])
            #mask_list.append(self.gen_mask(HR_list[it].copy(), kernel_size=(3,3), iteration=2))
            mask_list.append(self.plain)
            min_list.append(min(HR_list[it].shape))
        min_size = (min(min_list) // 16) * 16
        #size = min_size if self.eval_flag else self.size
        size = self.size
        # zeros tensor for addition of different LR components
        Input = torch.zeros((1, size//2, size//2), device=self.device) if self.up_factor != 1\
            else torch.zeros((1, size, size), device=self.device)
        Input = torch.zeros((1, size//2, size//2), device=self.device) if self.up_factor != 1\
            else torch.zeros((1, size, size), device=self.device)
        # zeros tensor for addition of different HR components
        GT_S = torch.zeros((1, size, size), device=self.device)
        self.resize = Resize([self.size // self.up_factor, self.size // self.up_factor])
        for it in range(len(self.dir_list_D)):
            # crop the data randomly or not
            if self.crop_flag:
                if not self.read_LR:
                    HR_list[it], mask_list[it] = self.rand_crop_single_with_mask(img=HR_list[it], mask=mask_list[it], size=size, min_size=min_size, index=it)
                else:
                    HR_list[it], LR_list[it], mask_list[it] = self.rand_crop_with_mask(img_1=HR_list[it], img_2=LR_list[it], mask=mask_list[it], size=size, min_size=min_size, index=it)
            else:
                h, w = HR_list[it].shape    
                HR_list[it] = HR_list[it][((h//2)-(size//2)):((h//2)+(size//2)), ((w//2)-(size//2)):((w//2)+(size//2))]
                if self.read_LR: LR_list[it] =LR_list[it][((h//2)-(size//2)):((h//2)+(size//2)), ((w//2)-(size//2)):((w//2)+(size//2))]    
                mask_list[it] = mask_list[it][((h//2)-(size//2)):((h//2)+(size//2)), ((w//2)-(size//2)):((w//2)+(size//2))]
            
            HR_list[it] = torch.tensor(np.float64(HR_list[it]), dtype=torch.float, device=self.device).unsqueeze(0)
            if self.read_LR:
                LR_list[it] = torch.tensor(np.float64(LR_list[it]), dtype=torch.float, device=self.device).unsqueeze(0)
            else:
                if self.w0 > 0: 
                    LR_list.append(F.conv2d(input=HR_list[it].unsqueeze(0), weight=self.psf_cal.unsqueeze(0), padding=64, stride=1).squeeze(0))
                else:
                    LR_list.append(HR_list[it])

            
            GT_list_DS.append(HR_list[it]) 
            GT_list_D.append(LR_list[it])
            GT_S += self.factor_list[it]*HR_list[it]
            if self.read_LR: Input += self.factor_list[it]*LR_list[it]
        GT_DS = torch.concat(GT_list_DS, dim=0).to(self.device)
        GT_D = torch.concat(GT_list_D, dim=0).to(self.device)
        if self.noise_level > 0:
            GT_D = self.add_noise(GT_D)
        if not self.read_LR: 
            if self.w0 > 0:
                #temp = to_cpu(GT_S[0,:,:])
                #tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\SR\Micro\degeneration\HR.tif', temp)
                Input = F.conv2d(input=GT_S.unsqueeze(0), weight=self.psf_cal.unsqueeze(0), padding=64, stride=1).squeeze(0)
                #temp = to_cpu(Input[0,:,:])
                #tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\SR\Micro\degeneration\Blurred.tif', temp)
            else:
                Input = GT_S
        if self.noise_level > 0:
            Denoised = Input
            Input = self.add_noise(Input)
            #temp = to_cpu(Input[0,:,:])
            #tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\SR\Micro\degeneration\noised.tif', temp)
        # mask        
        mask = np.zeros((size, size))
        '''
        for i in range(len(mask_list)):
            mask += mask_list[i]
        #mask = np.power(2, mask)
        mask[mask==1] = 0
        #mask += 1
        #mask[mask != 0] = 5
        mask = cv2.dilate(mask, kernel=(7,7), iterations=1)
        mask = cv2.GaussianBlur(mask, (15,15), 0)'''

        mask = torch.tensor(np.float64(mask), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        # normalization
        '''Input, Input_mean, Input_std = self.norm_statistic(Input, std=None)
        GT_DS, GT_DS_mean, GT_DS_std = self.norm_statistic(GT_DS, Input_std)
        GT_D, GT_D_mean, GT_D_std = self.norm_statistic(GT_D, Input_std)
        GT_S, GT_S_mean, GT_S_std = self.norm_statistic(GT_S, Input_std)'''
        Input, _, Input_std = self.norm_statistic(Input=Input, std=None)
        Input_min = torch.tensor(0, dtype=torch.float, device=self.device)
        GT_DS, _, _ = self.norm_statistic(GT_DS, Input_std)
        GT_D, _, _ = self.norm_statistic(GT_D, Input_std)
        GT_S, _, _ = self.norm_statistic(GT_S, Input_std)
        if self.noise_level > 0: Denoised, _, _ = self.norm_statistic(Denoised, Input_std)
        '''Input, Input_val_min, Input_val_max = self.map_values(Input)
        GT_DS, _, _ = self.map_values(GT_DS, min_val=Input_val_min, max_val=Input_val_max, index=1)
        GT_D, _, _ = self.map_values(GT_D, min_val=Input_val_min, max_val=Input_val_max, index=1)
        GT_S, _, _ = self.map_values(GT_S, min_val=Input_val_min, max_val=Input_val_max, index=1)
        if self.noise_level > 0: Denoised, _, _ = self.map_values(Denoised, min_val=Input_val_min, max_val=Input_val_max, index=1)'''
        # generate statistic dict when validation
        statistic_dict = {
            "Input_mean":Input_min, "Input_std":Input_std
            }
        
        # to device
        Input = Input.to(self.device)
        GT_DS = GT_DS.to(self.device)
        GT_D = GT_D.to(self.device)
        GT_S = GT_S.to(self.device)
        '''print(self.dir_list_DS[it][D_index])
        plt.figure(1)
        plt.imshow(to_cpu(Input[0,:,:]), cmap='gray')
        plt.show()'''
        if self.noise_level > 0: Denoised = Denoised.to(self.device)
        
        if self.noise_level > 0:
            return Input, GT_DS, GT_D, GT_S, statistic_dict
        else:
            return Input, GT_DS, GT_D, GT_S, statistic_dict
        

if __name__ == "__main__":
    cwd = os.getcwd()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    if torch.cuda.is_available():        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        print('-----------------------------Using GPU-----------------------------')

    def gen_temp_dataloader(GT_tag_list, noise_level, num_test, size, up_factor=1):
        output_list = GT_tag_list
        denoise = "None"
        factor_list = [1, 1, 1]
        train_dir_LR =  os.path.join(cwd, "data\\train_LR")
        test_dir_LR = os.path.join(cwd, "data\\test_LR")
        train_dir_HR = os.path.join(cwd, "data\\train_HR")
        test_dir_HR = os.path.join(cwd, "data\\test_HR")
        train_dir_GT_HR_list = []
        test_dir_GT_HR_list = []
        train_dir_GT_LR_list = []
        test_dir_GT_LR_list = []
        for i in range(len(GT_tag_list)):
            train_dir_GT_HR_list.append(os.path.join(train_dir_HR, GT_tag_list[i]))
            test_dir_GT_HR_list.append(os.path.join(test_dir_HR, GT_tag_list[i]))
            train_dir_GT_LR_list.append(os.path.join(train_dir_LR, GT_tag_list[i]))
            test_dir_GT_LR_list.append(os.path.join(test_dir_LR, GT_tag_list[i]))
        eval_dataset = Dataset_decouple_SR(
                GT_dir_list_DS=test_dir_GT_HR_list, GT_dir_list_D=test_dir_GT_LR_list,
                size=size, device=device, noise_level=noise_level, output_list=output_list, denoise=denoise, 
                train_flag=False, num_file=num_test, up_factor=up_factor, factor_list=factor_list, 
                random_selection=False, crop_flag=False, flip_flag=False, read_LR=False
            )    
        eval_dataloader = DataLoader(dataset=eval_dataset, shuffle=False, batch_size=1)
        return eval_dataloader
    
    category = ['Microtubes', 'Mitochondria', 'Lysosome']
    
    eval_dataloader = gen_temp_dataloader(GT_tag_list=category, noise_level=100, num_test=100, size=512)
    for batch_index, data in enumerate(eval_dataloader):
        Input, GT, _, _, denoised, sta = data
        Input = to_cpu(Input.seqeeze(0).permute(1,2,0))
        denoised = to_cpu(denoised.sequeeze(0).permute(1,2,0))
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(denoised)
        plt.subplot(122)
        plt.show()

        
