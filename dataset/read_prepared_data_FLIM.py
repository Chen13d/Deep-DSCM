import os
import torch
import tifffile
import numpy as np
from random import randint, random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

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


def rand_crop_pair_with_lifetime(Input, GT, denoised, lifetime, size):
    #print(Input.size(), lifetime.size(), GT.size())
    i,j,height,width = get_crop_params(img_size=Input.size(), output_size=size) 
    Input = Input[i:i+height, j:j+width]
    lifetime = lifetime[i:i+height, j:j+width]
    denoised = denoised[i:i+height, j:j+width]
    GT = GT[:,i:i+height, j:j+width]
    return Input, GT, denoised, lifetime


class prepared_dataset_FLIM(Dataset):
    def __init__(self, read_dir, num_file, num_org, org_list, size, device, random_selection=True, crop_flag=True, flip_flag=True):
        super(prepared_dataset_FLIM, self).__init__()
        self.read_dir = read_dir
        self.num_file = num_file
        self.num_org = num_org
        self.org_list = org_list
        self.device = device
        self.size = size 
        self.random_selection = random_selection
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        self.generate_read_dir()

        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.ConvertImageDtype(torch.float32)
        ])


    def generate_read_dir(self):
        self.Input_dir = os.path.join(self.read_dir, "Input")
        self.GT_DS_dir = os.path.join(self.read_dir, "GT_DS")
        self.GT_D_dir = os.path.join(self.read_dir, "GT_D")
        self.GT_S_dir = os.path.join(self.read_dir, "GT_S")
        self.lifetime_dir = os.path.join(self.read_dir, "Lifetime")
        self.denoised_dir = os.path.join(self.read_dir, "denoised")

    def __len__(self):
        self.dataset_length = len(os.listdir(self.Input_dir))
        return self.num_file
    
    def __get_file_num__(self):
        return len(os.listdir(self.Input_dir))
    
    
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
    
    def norm_statistic(self, Input, std=None):        
        mean = torch.mean(Input).to(self.device)
        mean_zero = torch.zeros_like(mean).to(self.device)
        std = torch.std(Input).to(self.device) if std == None else std
        output = transforms.Normalize(mean_zero, std)(Input)
        return output, mean_zero, std
    
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
        # random selection / constrained index
        if self.random_selection:
            D_index = randint(0, self.num_file-1)
        else:
            D_index = index
        if self.flip_flag:
            self.h_flip_flag = int(random()>0.5)
            self.v_flip_flag = int(random()>0.5)
            Input = torch.tensor(np.float64(self.numpy_flip(tifffile.imread(os.path.join(self.Input_dir, f"{index+1}.tif")))), dtype=torch.float, device=self.device)
            sorted_image = torch.tensor(np.float64(self.numpy_flip(tifffile.imread(os.path.join(self.lifetime_dir, f"{index+1}.tif")))), dtype=torch.float, device=self.device)
            GT_DS = torch.tensor(np.float64(self.numpy_flip(tifffile.imread(os.path.join(self.GT_DS_dir, f"{index+1}.tif")))), dtype=torch.float, device=self.device)
            denoised = torch.tensor(np.float64(self.numpy_flip(tifffile.imread(os.path.join(self.denoised_dir, f"{D_index+1}.tif")))), dtype=torch.float, device=self.device)
        else:
            Input = torch.tensor(np.float64(tifffile.imread(os.path.join(self.Input_dir, f"{index+1}.tif"))), dtype=torch.float, device=self.device)
            sorted_image = torch.tensor(np.float64(tifffile.imread(os.path.join(self.lifetime_dir, f"{index+1}.tif"))), dtype=torch.float, device=self.device)
            GT_DS = torch.tensor(np.float64(tifffile.imread(os.path.join(self.GT_DS_dir, f"{index+1}.tif"))), dtype=torch.float, device=self.device)
            denoised = torch.tensor(np.float64(tifffile.imread(os.path.join(self.denoised_dir, f"{D_index+1}.tif"))), dtype=torch.float, device=self.device)
        if self.crop_flag:
            Input, GT_DS, denoised, sorted_image = rand_crop_pair_with_lifetime(Input=Input, lifetime=sorted_image, GT=GT_DS, denoised=denoised, size=self.size)
        else:
            Input = Input[:self.size, :self.size]
            GT_DS = GT_DS[:, :self.size, :self.size]
            denoised = denoised[:self.size, :self.size]
            sorted_image = sorted_image[:self.size, :self.size]
        # normalizations
        Input = Input.unsqueeze(0)
        sorted_image = sorted_image.unsqueeze(0)
        denoised = sorted_image.unsqueeze(0)
        Input, Input_mean, Input_std = self.norm_statistic(Input)
        #lifetime, _, _ = self.norm_statistic(lifetime, Input_std)
        #sorted_image = sorted_image / torch.max(sorted_image)
        sorted_image, _, _ = self.norm_statistic(sorted_image, Input_std)
        GT_DS, _, _ = self.norm_statistic(GT_DS, Input_std)
        denoised, _, _ = self.norm_statistic(denoised, Input_std)
        '''GT_DS, GT_DS_mean, GT_DS_std = self.norm_statistic(GT_DS, Input_std)
        GT_D, GT_D_mean, GT_D_std = self.norm_statistic(GT_D, Input_std)
        GT_S, GT_S_mean, GT_S_std = self.norm_statistic(GT_S, Input_std)  '''      
        # generate statistic dict for validation
        '''statistic_dict = {
            "Input_mean":Input_mean, "Input_std":Input_std, "GT_main_mean":GT_DS_mean, "GT_main_std":GT_DS_std, 
            "GT_D_mean":GT_D_mean, "GT_D_std":GT_D_std, "GT_S_mean":GT_S_mean, "GT_S_std":GT_S_std,                 
            }'''
        statistic_dict = {
            "Input_mean":Input_mean, "Input_std":Input_std
            }
        #return Input, GT_DS, GT_D, GT_S, statistic_dict
        return Input, GT_DS, 0, 0, 0, sorted_image, statistic_dict
    

def gen_prepared_dataloader_FLIM(read_dir_train, read_dir_val, num_file_train, num_file_val, num_org, org_list, batch_size, device):
    train_dataset = prepared_dataset_FLIM(read_dir=read_dir_train, num_file=num_file_train, num_org=num_org, org_list=org_list, device=device)
    val_dataset = prepared_dataset_FLIM(read_dir=read_dir_val, num_file=num_file_val, num_org=num_org, org_list=org_list, device=device)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1)
    return train_dataloader, val_dataloader, num_file_train
