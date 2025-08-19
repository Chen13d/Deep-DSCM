import os
import sys
import glob 
import tifffile

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np

from random import random,  randint
from torchvision import transforms
from torchvision.transforms import Resize 
from skimage.filters import threshold_otsu

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



class DSCM_dataset(Dataset):
    def __init__(self, 
                 GT_dir_list_DS, GT_dir_list_D, 
                 device, 
                 num_file, 
                 size, 
                 random_selection=False, crop_flag=True, flip_flag=True):
        # Designate the file_dir of raw data
        self.dir_list_DS = []
        self.dir_list_D = []
        for i in range(len(GT_dir_list_DS)):
            self.dir_list_DS.append(natsort.natsorted(glob.glob(GT_dir_list_DS[i]+'/*')))
        for i in range(len(GT_dir_list_D)):
            self.dir_list_D.append(natsort.natsorted(glob.glob(GT_dir_list_D[i]+'/*')))

        # Data augmentation, random_selection indicates to select the component images, regradless of their file index. 
        self.random_selection = random_selection
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        # Number of images included in each epoch
        self.num_file = num_file
        # Output image size for training/validation
        self.size = size
        # Designate the device 
        self.device = device
    
    # Data augmentation - cropping
    def rand_crop(self, img_1, img_2, image_size, crop_size):
        self.i,self.j,self.height,self.width = get_crop_params(img_size=image_size, output_size=crop_size) 
        img_1 = img_1[self.i:self.i+self.height, self.j:self.j+self.width]
        img_2 = img_2[self.i:self.i+self.height, self.j:self.j+self.width]
        return img_1, img_2
    # Replace the length of file list by a designated value
    def __len__(self):
        return self.num_file 
    # Data augmentation - flipping using numpy
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
        HR_list = []
        confocal_list = []
        #minsize_list = []
        # Enumerate in categories - c_index
        for c_index in range(len(self.dir_list_DS)):
            # D_index to index the file directory list
            # Data augmentation - random selection
            if self.random_selection:
                D_index = randint(0, len(self.dir_list_DS[c_index])-1)
            else:
                D_index = index % len(self.dir_list_DS[c_index])
            # Read raw images
            HR = tifffile.imread(self.dir_list_DS[c_index][D_index])
            HR_list.append(HR)
            confocal = tifffile.imread(self.dir_list_D[c_index][D_index])
            confocal_list.append(confocal)
            
            # Data augmentation - flipping
            if self.flip_flag:
                self.h_flip_flag = int(random()>0.5)
                self.v_flip_flag = int(random()>0.5)
                HR_list[c_index] = np.float32(self.numpy_flip(HR_list[c_index]))
                confocal_list[c_index] = np.float32(self.numpy_flip(confocal_list[c_index]))
            else:
                HR_list[c_index] = np.float32(HR_list[c_index])
                confocal_list[c_index] = np.float32(confocal_list[c_index])
            # get the minsize to have consistant output image size
            #minsize_list.append(min(HR_list[c_index].shape))
            if self.crop_flag:
                HR_list[c_index], confocal_list[c_index] = self.rand_crop(img_1=HR_list[c_index], img_2=confocal_list[c_index], image_size=HR_list[c_index].shape, crop_size=self.size)
            else:
                HR_list[c_index] = HR_list[c_index][:self.size,:self.size]
                confocal_list[c_index] = confocal_list[c_index][:self.size,:self.size]
        return HR_list, confocal_list

# avoid to return tensor
def my_collate_fn(batch):
    return batch[0][0], batch[0][1]


"""
-- Data loader
cwd: cwd
category: involved components - ["Micro", "Mito", "Lyso"]
device: cpu / cuda(gpu)
size: image size
batch_size: batch_size
num_workers: number of workers includes in dataloader, may accelerate the training while increasing the memory usage
"""
def DSCM_dataloader(
        cwd, 
        categories, 
        device, 
        num_file_train, 
        num_file_val,
        size, 
        batch_size, 
        num_workers
    ):
    train_dir_LR =  os.path.join(cwd, "data\\train_LR")
    val_dir_LR = os.path.join(cwd, "data\\test_LR")
    train_dir_HR = os.path.join(cwd, "data\\train_HR")
    val_dir_HR = os.path.join(cwd, "data\\test_HR")
    train_dir_GT_HR_list = []
    val_dir_GT_HR_list = []
    train_dir_GT_LR_list = []
    val_dir_GT_LR_list = []
    for category in categories:
        train_dir_GT_HR_list.append(os.path.join(train_dir_HR, category))
        val_dir_GT_HR_list.append(os.path.join(val_dir_HR, category))
        train_dir_GT_LR_list.append(os.path.join(train_dir_LR, category))
        val_dir_GT_LR_list.append(os.path.join(val_dir_LR, category))
    train_dataset = DSCM_dataset(
        GT_dir_list_DS=train_dir_GT_HR_list, 
        GT_dir_list_D=train_dir_GT_LR_list, 
        device=device, 
        num_file=num_file_train,
        size=size, 
        random_selection=True, crop_flag=True, flip_flag=True
    )
    val_dataset = DSCM_dataset(
        GT_dir_list_DS=val_dir_GT_HR_list, 
        GT_dir_list_D=val_dir_GT_LR_list, 
        device=device, 
        num_file=num_file_val,
        size=size, 
        random_selection=False, crop_flag=False, flip_flag=False
    )
    if num_workers:
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, collate_fn=my_collate_fn)
    else:
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size, collate_fn=my_collate_fn)
    eval_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1, collate_fn=my_collate_fn)
    return train_dataloader, eval_dataloader


        

            
            
