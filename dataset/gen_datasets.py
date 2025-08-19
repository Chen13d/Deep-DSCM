import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import copy 
import time
import glob
from tqdm import tqdm
import torch
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from random import random,  randint
from torchvision import transforms
from torchvision.transforms import Resize 
from skimage.filters import threshold_otsu

from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
sys.path.append('./utils')
from utils import *

try:
    if __name__ == "__main__":
        from degradation_model import *
    else:
        from dataset.degradation_model import *
except:
    from degradation_model import *


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
class Dataset_degradation(Dataset):
    def __init__(self, GT_dir_list_DS, GT_dir_list_D, device, num_file, up_factor, factor_list, 
                 target_resolution, STED_resolution_dict, degradation_method, average, generate_FLIM, 
                 size=512, noise_level=0.5, output_list = None, denoise=False, train_flag=True, 
                 random_selection=False, crop_flag=True, flip_flag=True, eval_flag=False, w0_T=2.05, 
                 read_LR=False, real_time=True, find_psf=True):
        super(Dataset_degradation, self).__init__()
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
        self.w0_T = w0_T
        self.read_LR = read_LR
        self.real_time = real_time
        self.find_psf = find_psf
        self.STED_resolution_dict = STED_resolution_dict
        self.degradation_method = degradation_method
        self.average = average
        self.generate_FLIM = generate_FLIM
        for i in range(len(GT_dir_list_DS)):
            self.dir_list_DS.append(natsort.natsorted(glob.glob(GT_dir_list_DS[i]+'/*')))
        for i in range(len(GT_dir_list_D)):
            self.dir_list_D.append(natsort.natsorted(glob.glob(GT_dir_list_D[i]+'/*')))

        self.size = size
        self.device = device 
        self.factor_list = factor_list
        self.plain = np.zeros((self.size, self.size))

        self.deg = Degradation_base_model(target_resolution=target_resolution, noise_level=self.noise_level, average=self.average, size=self.size, STED_resolution_dict=self.STED_resolution_dict, factor_list=self.factor_list)
        self.run_time_list = []

        self.psf_list = []
        self.w0_S_list = []
        fixed_w0_S_list = [2.44, 2.30, 2.26]
        if not read_LR:
            if STED_resolution_dict:
                self.w0_T = self.deg.find_psf_for_resolution(resolution=target_resolution)
                #self.w0_T = 8.17 # 280
                #self.w0_T = 6.64 # 228
                # self.w0_T = 6.97 # 240
                for resolution_index in range(len(self.STED_resolution_dict)):
                    self.w0_S_org = self.deg.find_psf_for_resolution(resolution=self.STED_resolution_dict[resolution_index])
                    #self.w0_S_org = 2.44
                    #self.w0_S_org = fixed_w0_S_list[resolution_index]
                    #self.w0_S_list.append(self.w0_S_org)
                    self.psf_list.append(self.deg.generate_cal_psf(w0_S=self.w0_S_org, w0_T=self.w0_T))
            else:
                for i in range(len(self.factor_list)):
                    self.psf_list.append(self.deg.cal_psf)

        #self.general_psf = self.deg.generate_cal_psf(w0_S=np.mean(self.w0_S_list), w0_T=self.w0_T)

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
    def rand_crop_single(self, img, image_size, crop_size):
        self.i,self.j,self.height,self.width = get_crop_params(img_size=image_size, output_size=crop_size) 
        #if index == 0: self.i,self.j,self.height,self.width = get_crop_params(img_size=(min_size, min_size), output_size=size) 
        img = img[self.i:self.i+self.height, self.j:self.j+self.width]
        return img
    def rand_crop(self, img_1, img_2, image_size, crop_size):
        self.i,self.j,self.height,self.width = get_crop_params(img_size=image_size, output_size=crop_size) 
        img_1 = img_1[self.i:self.i+self.height, self.j:self.j+self.width]
        img_2 = img_2[self.i:self.i+self.height, self.j:self.j+self.width]
        return img_1, img_2
    def numpy_flip(self, Input):
        if len(Input.shape) == 2:
            if self.v_flip_flag:
                Input = np.flipud(Input)
                if self.h_flip_flag:
                    Input = np.fliplr(Input)
                else:
                    pass
            else:
                pass
        elif len(Input.shape) == 3:
            for i in range(Input.shape[0]):
                if self.v_flip_flag:
                    Input[i,:,:] = np.flipud(Input[i,:,:])
                    if self.h_flip_flag:
                        Input[i,:,:] = np.fliplr(Input[i,:,:])
                    else:
                        pass
                else:
                    pass
        return Input
    def __getitem__(self, index):
        # measure run time in a simple way
        """temp = time.time()
        self.run_time_list.append(temp)
        if index > 1:
            print(self.run_time_list[-1] - self.run_time_list[-2])"""
        # creat a stack for GT_DS and GT_D
        self.deg.create_stack()
        binary_list = []
        # list for different LR and HR components
        HR_list = []
        confocal_list = []
        LR_list = []
        # give a random noise level
        #rand_noise_scale = uniform(0.15, 1)
        # read HR images random_selection - select images randomly, or select images base on dataloader
        for it in range(len(self.dir_list_DS)):
            if self.random_selection:
                D_index = randint(0, len(self.dir_list_DS[it])-1)
                #D_index = randint(0, 20)
            else:
                D_index = index % len(self.dir_list_DS[it])
            # STED images
            HR = tifffile.imread(self.dir_list_DS[it][D_index])
            HR_list.append(HR)
            # Confocal images, not necessary
            confocal = tifffile.imread(self.dir_list_D[it][D_index])
            confocal_list.append(confocal)
        # list to acquire smallest size of component images
        min_list = []
        # abandoned
        mask_list = []
        # apply image flip among components
        for it in range(len(self.dir_list_D)):
            if self.flip_flag:
                self.h_flip_flag = int(random()>0.5)
                self.v_flip_flag = int(random()>0.5)
                HR_list[it] = np.float32(self.numpy_flip(HR_list[it]))
                confocal_list[it] = np.float32(self.numpy_flip(confocal_list[it]))
            else:
                HR_list[it] = np.array(HR_list[it])
                confocal_list[it] = np.array(confocal_list[it])
            if not self.read_LR:
                # remap images to a certain range
                HR_list[it] = np.float32(HR_list[it])
                HR_list[it] = self.deg.map_values_numpy(HR_list[it], new_max=255, new_min=0, percentile=99.9)
                HR_list[it][HR_list[it] < 0] = 0

            mask_list.append(self.plain)
            # send min size to the list
            min_list.append(min(HR_list[it].shape))
        # set the crop_size for final output image lateral size
        # in real_time, the crop_size = size in yml file
        if not self.real_time: 
            crop_size = min(min_list) if self.size < min(min_list) else self.size
        else: 
            crop_size = self.size
        # zeros for addition of different LR components
        Input = np.zeros((crop_size//2, crop_size//2, 1)) if self.up_factor != 1\
            else np.zeros((crop_size, crop_size, 1)) 
        self.deg.generate_plain(size=crop_size)
        # enumerate in components
        for it in range(len(self.dir_list_D)):
            # crop the data randomly or not
            if self.crop_flag:
                HR_list[it], confocal_list[it] = self.rand_crop(img_1=HR_list[it], img_2=confocal_list[it], image_size=HR_list[it].shape, crop_size=crop_size)
            else:
                HR_list[it] = HR_list[it][:crop_size,:crop_size]
                confocal_list[it] = confocal_list[it][:crop_size,:crop_size]
            #HR_list[it] = HR_list[it] / HR_list[it].max() * 500
            # generate binary map for noising h,w-h,w,c
            if self.noise_level > 0:
                binary_list.append(self.deg.get_binary(HR_list[it]))
            # generate individual LR images h,w,c
            if self.read_LR:
                pass
            else:
                if self.w0_T != 0: 
                    LR_list.append(self.deg.degrade_resolution_numpy(np.expand_dims(HR_list[it], -1), self.psf_list[it]))
                else:
                    LR_list.append(np.expand_dims(HR_list[it], -1))

            # add images to stack HR_list-h,w Stack-h,w,c
            if not self.read_LR:
                self.deg.add_image(Input_HR=HR_list[it], Input_LR=LR_list[it])
            else:
                self.deg.add_image(Input_HR=HR_list[it], Input_LR=confocal_list[it])
            
            # if degradation is not performed
            if self.read_LR: 
                Input += self.factor_list[it]*np.expand_dims(confocal_list[it], axis=-1)
        
        # concatenation h,w,c
        GT_DS, GT_D = self.deg.images_concatenation()
        # channel degradation - composition h,w,c
        blurred, GT_S = self.deg.composition(factor_list=self.factor_list)
        # get binary mask for single-LR noising map h,w,c, abondoned
        #binary_mask = self.deg.merge_binary(binary_list=binary_list)
            # do pseudo FLIM images
        if self.generate_FLIM:
            lifetime_list = []
            # generate mask for lifetime
            for i in range(len(HR_list)):
                lifetime_list.append(self.deg.make_threshold(HR_list[i]))
                '''plt.figure()
                plt.subplot(121)
                plt.imshow(LR_list[i])
                plt.subplot(122)
                plt.imshow(lifetime_list[i])
                plt.show()
                #lifetime_list.append(LR_list[i])'''
            sorted_image = self.deg.make_lifetime_distribution(lifetime_list, size=crop_size)
            sorted_image = np.expand_dims(sorted_image, -1)
            sorted_image = np.transpose(sorted_image, (2,0,1))
        else:
            sorted_image = 0
        # add noise for individual LR images stack_LR-h,w,c
        if self.noise_level > 0 and not self.read_LR:
            for j in range(len(HR_list)):                
                self.deg.stack_LR[j] = self.deg.degrade_noise(self.deg.stack_LR[j], version="numpy", noise_scale=self.noise_level, average=self.average)
        # concatenation h,w,c
        _, GT_D = self.deg.images_concatenation()
        
        # resolution degradation Input h,w,c
        if not self.read_LR: 
            if self.w0_T > 0:
                if self.degradation_method == "composite-blur-noise":
                    Input = np.expand_dims(self.deg.degrade_resolution_numpy(GT_S, self.general_psf), -1)
                elif self.degradation_method == "blur-composite-noise":
                    Input = blurred
            else:
                Input = GT_S
        # noise degradation h,w,c
        Denoised = copy.deepcopy(Input)
        if self.noise_level > 0 and not self.read_LR: # and self.degradation_method == "composite-blur-noise":
            Input = self.deg.degrade_noise(Input, version="numpy", noise_scale=self.noise_level, average=self.average)
        
        #Check the distance between confocal and lr images
        #print(np.mean(confocal_list[it] - Input))
        
        # make to (b,c,h,w)
        Input = np.transpose(Input, (2,0,1))
        GT_S = np.transpose(GT_S, (2,0,1))
        GT_D = np.transpose(GT_D, (2,0,1))
        GT_DS = np.transpose(GT_DS, (2,0,1))
        Denoised = np.transpose(Denoised, (2,0,1))
        
        # do normalization if needed
        statistic_dict = {} 
        if self.real_time:
            # send to tensor
            Input = torch.tensor(np.float32(Input), dtype=torch.float32, device=self.device)
            GT_DS = torch.tensor(np.float32(GT_DS), dtype=torch.float32, device=self.device)
            GT_D = torch.tensor(np.float32(GT_D), dtype=torch.float32, device=self.device)
            GT_S = torch.tensor(np.float32(GT_S), dtype=torch.float32, device=self.device)
            Denoised = torch.tensor(np.float32(Denoised), dtype=torch.float32, device=self.device)
            # normalization
            Input, Input_mean, Input_std = self.norm_statistic(Input, std=None)
            GT_DS, GT_DS_mean, GT_DS_std = self.norm_statistic(GT_DS, Input_std)
            GT_D, GT_D_mean, GT_D_std = self.norm_statistic(GT_D, Input_std)
            GT_S, GT_S_mean, GT_S_std = self.norm_statistic(GT_S, Input_std)
            Denoised, _, _ = self.norm_statistic(Denoised)
            # generate statistic dict for recovering
            statistic_dict = {
                "Input_mean":Input_mean, "Input_std":Input_std                
                }
            if self.generate_FLIM:
                #sorted_image = np.float32(sorted_image) / np.max(sorted_image)
                sorted_image = torch.tensor(np.float32(sorted_image), dtype=torch.float, device=self.device)
                sorted_image, _, _ = self.norm_statistic(sorted_image, std=None)
            return Input, GT_DS, GT_D, GT_S, Denoised, sorted_image, statistic_dict
        return Input, GT_DS, GT_D, GT_S, Denoised, sorted_image, statistic_dict


def gen_degradation_dataloader(GT_tag_list, noise_level, w0_T, STED_resolution_dict, target_resolution, degradation_method, average, generate_FLIM, 
                real_time, num_file_train, num_file_val, size, read_LR, factor_list, num_workers, cwd, device, up_factor=1, batch_size=1, 
                eval_flag=False):
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
    STED_resolution_list = [STED_resolution_dict[org] for org in GT_tag_list]
    for i in range(len(GT_tag_list)):
        train_dir_GT_HR_list.append(os.path.join(train_dir_HR, GT_tag_list[i]))
        test_dir_GT_HR_list.append(os.path.join(test_dir_HR, GT_tag_list[i]))
        train_dir_GT_LR_list.append(os.path.join(train_dir_LR, GT_tag_list[i]))
        test_dir_GT_LR_list.append(os.path.join(test_dir_LR, GT_tag_list[i]))

    if not eval_flag:
        train_dataset = Dataset_degradation(
            GT_dir_list_DS=train_dir_GT_HR_list, GT_dir_list_D=train_dir_GT_LR_list,
            size=size, device=device, noise_level=noise_level, output_list=output_list, 
            STED_resolution_dict=STED_resolution_list, target_resolution=target_resolution, degradation_method=degradation_method, 
            average=average, generate_FLIM=generate_FLIM, real_time=real_time, denoise=denoise, 
            train_flag=False, num_file=num_file_train, up_factor=up_factor, factor_list=factor_list, read_LR=read_LR, 
            random_selection=True, crop_flag=True, flip_flag=True, w0_T=w0_T
        )
        if num_workers:
            train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
        else:
            train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
    eval_dataset = Dataset_degradation(
        GT_dir_list_DS=test_dir_GT_HR_list, GT_dir_list_D=test_dir_GT_LR_list,
        size=size, device=device, noise_level=noise_level, output_list=output_list, 
        STED_resolution_dict=STED_resolution_list, target_resolution=target_resolution, degradation_method=degradation_method, 
        average=average, generate_FLIM=generate_FLIM, real_time=real_time, denoise=denoise, 
        train_flag=False, num_file=num_file_val, up_factor=up_factor, factor_list=factor_list, read_LR=read_LR, 
        random_selection=False, crop_flag=False, flip_flag=False, w0_T=w0_T
    )
    eval_dataloader = DataLoader(dataset=eval_dataset, shuffle=False, batch_size=1)
    
    if not eval_flag:
        return train_dataloader, eval_dataloader
    else:
        return eval_dataloader


if __name__ == "__main__":
    cwd = os.getcwd()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    if torch.cuda.is_available():        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        print('-----------------------------Using GPU-----------------------------')

    # generate folders
    w0_T = 2.05
    target_resolution = 280
    noise_level = 0.030
    average = 1
    org_list = ['Micro']
    factor_list = [1]
    #degradation_method = "composite-blur-noise"
    degradation_method = "blur-composite-noise"
    generate_FLIM = False
    
    num_workers = 0

    num_train_image = 10
    num_val_image = 10

    STED_resolution_dict = {
        "Micro": 85.93, "Mito": 87.09, "Lyso": 91.97, "Membrane": 81.24, "NPCs": 85.92, "Mito_inner": 82.88, "Mito_inner_deconv": 81.19
    }
    
    if generate_FLIM:
        combination_name = "_".join(org_list) + "_" + str(target_resolution) + "_" + str(noise_level) + "_" + str(average) + "_FLIM"
    else:
        combination_name = "_".join(org_list) + "_" + str(target_resolution) + "_" + str(noise_level) + "_" + str(average)
    
    cwd = os.getcwd()
    save_dir_train = os.path.join(cwd, "data\\prepared_data\\train")
    save_dir_val = os.path.join(cwd, "data\\prepared_data\\val")

    save_dir_folder = os.path.join(save_dir_train, combination_name)
    Input_dir = os.path.join(save_dir_folder, "Input")
    GT_S_dir = os.path.join(save_dir_folder, "GT_S")
    GT_DS_dir = os.path.join(save_dir_folder, "GT_DS")
    GT_D_dir = os.path.join(save_dir_folder, "GT_D")
    denoised_dir = os.path.join(save_dir_folder, "denoised")
    check_existence(Input_dir)                                                                                                                                                                                                                     
    check_existence(GT_S_dir)
    check_existence(GT_D_dir)
    check_existence(GT_DS_dir)
    check_existence(denoised_dir)
    if generate_FLIM:
        lifetime_dir = os.path.join(save_dir_folder, "lifetime")
        check_existence(lifetime_dir)
    
    train_dataloader, eval_dataloader = gen_degradation_dataloader(GT_tag_list=org_list, noise_level=noise_level, w0_T=w0_T, 
            factor_list=factor_list, STED_resolution_dict=STED_resolution_dict, target_resolution=target_resolution, generate_FLIM=generate_FLIM, 
            degradation_method=degradation_method, average=average, read_LR=False, num_file_train=num_train_image, num_file_val=num_val_image, size=512, num_workers=num_workers, 
            device=device, cwd=cwd, real_time=False)
    bar = tqdm(total=num_train_image)
    for batch_index, data in enumerate(train_dataloader):
        #print(f"{batch_index+1} / {10000}")
        Input, GT_DS, GT_D, GT_S, denoised, sorted_image, sta = data
        Input = to_cpu(Input)
        GT_DS = to_cpu(GT_DS)
        GT_D = to_cpu(GT_D)
        GT_S = to_cpu(GT_S)
        denoised = to_cpu(denoised)
        
        # save to folder
        tifffile.imwrite(os.path.join(Input_dir, f"{batch_index+1}.tif"), np.uint16(Input[0,0,:,:]))
        tifffile.imwrite(os.path.join(GT_S_dir, f"{batch_index+1}.tif"), np.uint16(GT_S[0,0,:,:]))
        tifffile.imwrite(os.path.join(GT_D_dir, f'{batch_index+1}.tif'), np.uint16(GT_D[0,:,:,:]), imagej=True)
        tifffile.imwrite(os.path.join(GT_DS_dir, f'{batch_index+1}.tif'), np.uint16(GT_DS[0,:,:,:]), imagej=True)
        tifffile.imwrite(os.path.join(denoised_dir, f'{batch_index+1}.tif'), np.uint16(denoised[0,0,:,:]))
        if generate_FLIM:
            sorted_image = to_cpu(sorted_image)
            tifffile.imwrite(os.path.join(lifetime_dir, f'{batch_index+1}.tif'), np.uint16(sorted_image[0,0,:,:]))
        
        #for i in range(len(org_list)):
        #    tifffile.imwrite(os.path.join(GT_D_dir, f'{batch_index+1}_{org_list[i]}.tif'), np.uint16(GT_D[0,:,:,i]))
        #    tifffile.imwrite(os.path.join(GT_DS_dir, f'{batch_index+1}_{org_list[i]}.tif'), np.uint16(GT_DS[0,:,:,i]))
        bar.update(1)
    bar.close()
    save_dir_folder = os.path.join(save_dir_val, combination_name)
    Input_dir = os.path.join(save_dir_folder, "Input")
    GT_S_dir = os.path.join(save_dir_folder, "GT_S")
    GT_DS_dir = os.path.join(save_dir_folder, "GT_DS")
    GT_D_dir = os.path.join(save_dir_folder, "GT_D")
    denoised_dir = os.path.join(save_dir_folder, "denoised")
    check_existence(Input_dir)                                                                                                                                                                                                                     
    check_existence(GT_S_dir)
    check_existence(GT_D_dir)
    check_existence(GT_DS_dir)
    check_existence(denoised_dir)
    if generate_FLIM:
        lifetime_dir = os.path.join(save_dir_folder, "lifetime")
        check_existence(lifetime_dir)
    for batch_index, data in enumerate(eval_dataloader):
        Input, GT_DS, GT_D, GT_S, denoised, sorted_image, sta = data
        Input = to_cpu(Input)
        GT_DS = to_cpu(GT_DS)
        GT_D = to_cpu(GT_D)
        GT_S = to_cpu(GT_S)
        denoised = to_cpu(denoised)
        
        # save to folder
        tifffile.imwrite(os.path.join(Input_dir, f"{batch_index+1}.tif"), np.uint16(Input[0,0,:,:]))
        tifffile.imwrite(os.path.join(GT_S_dir, f"{batch_index+1}.tif"), np.uint16(GT_S[0,0,:,:]))
        tifffile.imwrite(os.path.join(GT_D_dir, f'{batch_index+1}.tif'), np.uint16(GT_D[0,:,:,:]), imagej=True)
        tifffile.imwrite(os.path.join(GT_DS_dir, f'{batch_index+1}.tif'), np.uint16(GT_DS[0,:,:,:]), imagej=True)
        tifffile.imwrite(os.path.join(denoised_dir, f'{batch_index+1}.tif'), np.uint16(denoised[0,0,:,:]))
        if generate_FLIM:
            sorted_image = to_cpu(sorted_image)
            tifffile.imwrite(os.path.join(lifetime_dir, f'{batch_index+1}.tif'), np.uint16(sorted_image[0,0,:,:]))
        #for i in range(len(org_list)):
        #    tifffile.imwrite(os.path.join(GT_D_dir, f'{batch_index+1}_{org_list[i]}.tif'), np.uint16(GT_D[0,:,:,i]))
        #    tifffile.imwrite(os.path.join(GT_DS_dir, f'{batch_index+1}_{org_list[i]}.tif'), np.uint16(GT_DS[0,:,:,i]))