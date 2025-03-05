import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
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

if __name__ == "__main__":
    from degradation_model import *
else:
    from dataset.degradation_model import *

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

def map_values_numpy(image, new_min=-15, new_max=15, percentile=99):
    # Flatten the image and calculate the percentiles
    sorted_vals = np.sort(image.flatten())
    max_val = sorted_vals[int(len(sorted_vals) * percentile / 100)]
    min_val = sorted_vals[int(len(sorted_vals) * (100 - percentile) / 100)]
    # Rescale values to the new range
    image = np.uint16(image)
    return (image - min_val) * (new_max - new_min) / (max_val - min_val) + new_min

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
def rand_crop_dual(img1, img2, size):
    print(img1.shape[1:3],size)
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
                 random_selection=False, crop_flag=True, flip_flag=True, eval_flag=False, w0=2.05, 
                 read_LR=False, normalization=True):
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
        self.normalization = normalization
        for i in range(len(GT_dir_list_DS)):
            self.dir_list_DS.append(natsort.natsorted(glob.glob(GT_dir_list_DS[i]+'/*')))
        for i in range(len(GT_dir_list_D)):
            self.dir_list_D.append(natsort.natsorted(glob.glob(GT_dir_list_D[i]+'/*')))
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

        self.deg = Degradation_model(w0=self.w0, noise_scale=self.noise_level, size=self.size)
       
    # set the number of enumerations according to options
    def __len__(self):
        return self.num_file        
    
    def norm_statistic(self, Input, std=None):
        mean = torch.mean(Input).to(self.device)
        mean_zero = torch.zeros_like(mean).to(self.device)
        std = torch.std(Input).to(self.device) if std == None else std
        output = transforms.Normalize(mean_zero, std)(Input)
        return output, mean_zero, std
    def gen_mask(self, Input, kernel_size=(7,7), iteration=7):
        thresh = threshold_otsu(Input)
        Input[Input<thresh] = 0
        Input[Input>=thresh] = 1
        return Input
    def rand_crop_single_with_mask(self, img, mask, size, min_size, index):
        self.i,self.j,self.height,self.width = get_crop_params(img_size=size, output_size=min_size) 
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
        self.deg.create_stack()
        binary_list = []
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
            #print(self.dir_list_DS[it][D_index])
            HR = Image.open(self.dir_list_DS[it][D_index])
            HR_list.append(HR)
            if self.read_LR:
                LR = Image.open(self.dir_list_D[it][D_index])
                LR_list.append(LR)
        min_list = []
        mask_list = []
        min_crop = 10000
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

            # remap values - abandoned
            #HR_list[it] = map_values_numpy(HR_list[it], percentile=99, new_max=100, new_min=0)
            # calculate minsize
            temp_min = min(HR_list[it].shape)
            if temp_min < min_crop:
                min_crop = temp_min
            
            #mask_list.append(self.gen_mask(HR_list[it].copy(), kernel_size=(3,3), iteration=2))
            mask_list.append(self.plain)
            min_list.append(min(HR_list[it].shape))
        min_size = (min(min_list) // 16) * 16
        min_crop = min(min_crop, self.size)
        #size = min_size if self.eval_flag else self.size
        # zeros for addition of different LR components
        Input = np.zeros((min_crop//2, min_crop//2, 1)) if self.up_factor != 1\
            else np.zeros((self.size, self.size, 1))
        Input = np.zeros((min_crop//2, min_crop//2, 1)) if self.up_factor != 1\
            else np.zeros((min_crop, min_crop, 1)) 
        #self.deg.generate_plain(size=(HR_list[0].shape))
        self.deg.generate_plain(size=min_crop)
        # zeros for addition of different HR components
        self.resize = Resize([min_crop // self.up_factor, min_crop // self.up_factor])
        for it in range(len(self.dir_list_D)):
            # crop the data randomly or not
            if self.crop_flag:
                if not self.read_LR:
                    HR_list[it], mask_list[it] = self.rand_crop_single_with_mask(img=HR_list[it], mask=mask_list[it], size=HR_list[it].shape, min_size=min_crop, index=it)
                else:
                    HR_list[it], LR_list[it], mask_list[it] = self.rand_crop_with_mask(img_1=HR_list[it], img_2=LR_list[it], mask=mask_list[it], size=HR_list[it].shape, min_size=min_crop, index=it)
            else:
                pass
            HR_list[it] = HR_list[it] / HR_list[it].max() * 500
            # ge#nerate binary map for noising h,w-h,w,c
            if self.noise_level > 0:
                binary_list.append(self.deg.get_binary(HR_list[it]))
            # generate individual LR images h,w,c
            if self.read_LR:
                pass
            else:
                if self.w0 > 0: 
                    #print("HR", HR_list[it].mean())
                    LR_list.append(self.deg.degrade_resolution_numpy(np.expand_dims(HR_list[it], -1)))
                    #print("LR", LR_list[it].mean())
                else:
                    LR_list.append(np.expand_dims(HR_list[it], -1))
            # add images to stack HR_list-h,w Stack-h,w,c
            self.deg.add_image(Input_HR=HR_list[it], Input_LR=LR_list[it])

            # if degradation is not performed
            if self.read_LR: Input += self.factor_list[it]*LR_list[it]
        # add noise for individual LR images stack_LR-h,w,c
        if self.noise_level > 0:
            for j in range(len(LR_list)):                
                # Modified binary_map to None
                #self.deg.stack_LR[j] = self.deg.degrade_noise(self.deg.stack_LR[j], binary_map=binary_list[j], version="numpy")
                self.deg.stack_LR[j] = self.deg.degrade_noise(self.deg.stack_LR[j], binary_map=None, version="numpy")
                
        # concatenation h,w,c
        GT_DS, GT_D = self.deg.images_concatenation()
        # channel degradation - combination h,w,c
        GT_S = self.deg.combination(factor_list=self.factor_list)
        
        # get binary mask for single-LR noising map h,w,c
        #binary_mask = self.deg.merge_binary(binary_list=binary_list)
        # modified to None
        binary_mask = None
        
        #tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\estimate_noise\map.tif', binary_map)
        # resolution degradation Input h,w,c
        if not self.read_LR: 
            if self.w0 > 0:
                Input = np.expand_dims(self.deg.degrade_resolution_numpy(GT_S), -1)
            else:
                Input = GT_S
        # noise degradation h,w,c
        if self.noise_level > 0:
            Denoised = copy.deepcopy(Input)
            Input = self.deg.degrade_noise(Input, binary_mask, version="numpy")
        
        # convert to tensor
        #Input = torch.tensor(np.float64(Input), dtype=torch.float, device=self.device).permute(2,0,1)
        #GT_S = torch.tensor(np.float64(GT_S), dtype=torch.float, device=self.device).permute(2,0,1)
        #GT_D = torch.tensor(np.float64(GT_D), dtype=torch.float, device=self.device).permute(2,0,1)
        #GT_DS = torch.tensor(np.float64(GT_DS), dtype=torch.float, device=self.device).permute(2,0,1)
       
       # do normalization if needed
        statistic_dict = {} 
        if self.normalization:
            # normalization
            Input, Input_mean, Input_std = self.norm_statistic(Input, std=None)
            GT_DS, GT_DS_mean, GT_DS_std = self.norm_statistic(GT_DS, Input_std)
            GT_D, GT_D_mean, GT_D_std = self.norm_statistic(GT_D, Input_std)
            GT_S, GT_S_mean, GT_S_std = self.norm_statistic(GT_S, Input_std)
            if self.noise_level > 0: 
                Denoised = torch.tensor(np.float64(Denoised), dtype=torch.float, device=self.device).permute(2,0,1)
                Denoised, Denoised_mean, Denoised_std = self.norm_statistic(Denoised, Input_std)
            # generate statistic dict for validation
            statistic_dict = {
                "Input_mean":Input_mean, "Input_std":Input_std, "GT_main_mean":GT_DS_mean, "GT_main_std":GT_DS_std, 
                "GT_D_mean":GT_D_mean, "GT_D_std":GT_D_std, "GT_S_mean":GT_S_mean, "GT_S_std":GT_S_std,                 
                }
            if self.noise_level > 0:
                statistic_dict.update({"Denoised_mean":Denoised_mean, "Denoised_std":Denoised_std})
                      
        return Input, GT_DS, GT_D, GT_S, 0, statistic_dict
        

if __name__ == "__main__":
    cwd = os.getcwd()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    if torch.cuda.is_available():        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        print('-----------------------------Using GPU-----------------------------')

    def gen_temp_dataloader(GT_tag_list, noise_level, w0, num_file_train, num_file_val, size, read_LR, factor_list, up_factor=1):
        output_list = GT_tag_list
        denoise = "None"
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
        train_dataset = Dataset_decouple_SR(
            GT_dir_list_DS=train_dir_GT_HR_list, GT_dir_list_D=train_dir_GT_LR_list,
            size=size, device=device, noise_level=noise_level, output_list=output_list, denoise=denoise, 
            train_flag=False, num_file=num_file_train, up_factor=up_factor, factor_list=factor_list, read_LR=read_LR, 
            random_selection=False, crop_flag=True, flip_flag=False, normalization=False, w0=w0
        )    
        eval_dataset = Dataset_decouple_SR(
            GT_dir_list_DS=test_dir_GT_HR_list, GT_dir_list_D=test_dir_GT_LR_list,
            size=size, device=device, noise_level=noise_level, output_list=output_list, denoise=denoise, 
            train_flag=False, num_file=num_file_val, up_factor=up_factor, factor_list=factor_list, read_LR=read_LR, 
            random_selection=False, crop_flag=True, flip_flag=False, normalization=False, w0=w0
            )    
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1)
        eval_dataloader = DataLoader(dataset=eval_dataset, shuffle=False, batch_size=1)
        return train_dataloader, eval_dataloader
    
    # generate folders
    noise_level = 0
    w0 = 2.05
    org_list = ['NPCs', 'Mito_inner', 'Membrane']
    factor_list = [1, 1, 1]
    combination_name = "_".join(org_list) + "_" + str(noise_level)
    folder_list = []
    
    cwd = os.getcwd()
    save_dir_train = os.path.join(cwd, "data\\prepared_data\\train")
    save_dir_val = os.path.join(cwd, "data\\prepared_data\\val")

    save_dir_folder = os.path.join(save_dir_train, combination_name)
    Input_dir = os.path.join(save_dir_folder, "Input")
    GT_S_dir = os.path.join(save_dir_folder, "GT_S")
    GT_DS_dir = os.path.join(save_dir_folder, "GT_DS")
    GT_D_dir = os.path.join(save_dir_folder, "GT_D")
    #GT_denoised_dir = os.path.join(save_dir_folder, "denoised")
    check_existence(Input_dir)                                                                                                                                                                                                                     
    check_existence(GT_S_dir)
    check_existence(GT_D_dir)
    check_existence(GT_DS_dir)
    
    train_dataloader, eval_dataloader = gen_temp_dataloader(GT_tag_list=org_list, noise_level=noise_level, w0=w0, factor_list=factor_list, read_LR=False, num_file_train=4, num_file_val=4, size=1000)
    for batch_index, data in enumerate(train_dataloader):
        Input, GT_DS, GT_D, GT_S, _, sta = data
        Input = to_cpu(Input)
        GT_DS = to_cpu(GT_DS)
        GT_D = to_cpu(GT_D)
        GT_S = to_cpu(GT_S)

        # save to folder
        tifffile.imwrite(os.path.join(Input_dir, f"{batch_index+1}.tif"), np.uint16(Input[0,:,:,0]))
        tifffile.imwrite(os.path.join(GT_S_dir, f"{batch_index+1}.tif"), np.uint16(GT_S[0,:,:,0]))
        
        for i in range(len(org_list)):
            tifffile.imwrite(os.path.join(GT_D_dir, f'{batch_index+1}_{org_list[i]}.tif'), np.uint16(GT_D[0,:,:,i]))
            tifffile.imwrite(os.path.join(GT_DS_dir, f'{batch_index+1}_{org_list[i]}.tif'), np.uint16(GT_DS[0,:,:,i]))
        
    save_dir_folder = os.path.join(save_dir_val, combination_name)
    Input_dir = os.path.join(save_dir_folder, "Input")
    GT_S_dir = os.path.join(save_dir_folder, "GT_S")
    GT_DS_dir = os.path.join(save_dir_folder, "GT_DS")
    GT_D_dir = os.path.join(save_dir_folder, "GT_D")
    #GT_denoised_dir = os.path.join(save_dir_folder, "denoised")
    check_existence(Input_dir)
    check_existence(GT_S_dir)
    check_existence(GT_D_dir)
    check_existence(GT_DS_dir)
    for batch_index, data in enumerate(eval_dataloader):
        Input, GT_DS, GT_D, GT_S, _, sta = data
        Input = to_cpu(Input)
        GT_DS = to_cpu(GT_DS)
        GT_D = to_cpu(GT_D)
        GT_S = to_cpu(GT_S)
        # save to folder
        tifffile.imwrite(os.path.join(Input_dir, f"{batch_index+1}.tif"), np.uint16(Input[0,:,:,0]))
        tifffile.imwrite(os.path.join(GT_S_dir, f"{batch_index+1}.tif"), np.uint16(GT_S[0,:,:,0]))
        
        for i in range(len(org_list)):
            tifffile.imwrite(os.path.join(GT_D_dir, f'{batch_index+1}_{org_list[i]}.tif'), np.uint16(GT_D[0,:,:,i]))
            tifffile.imwrite(os.path.join(GT_DS_dir, f'{batch_index+1}_{org_list[i]}.tif'), np.uint16(GT_DS[0,:,:,i]))