import os, sys
import cv2
import natsort
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *


def distribution(read_dir, train_dir_HR, val_dir_HR, train_dir_LR, val_dir_LR, train_dir_HR_deconv, val_dir_HR_deconv, tags, save_list, val_interval, shrink, size=512):
    total_filelist = []
    for i in range(len(tags)):
        total_filelist.append([[], [], []])
    for tag_index in range(len(tags)):
        read_folder_dir = os.path.join(read_dir, tags[tag_index])
        file_list = os.listdir(read_folder_dir)
        file_list = natsort.natsorted(file_list)
        for i in range(len(file_list)):            
            img = np.uint16(Image.open(os.path.join(read_folder_dir, file_list[i])))#[20:-20, 20:-20]            
            #print(img.shape)
            if file_list[i].find('STED') != -1:
                #img = np.uint16(img // 50)
                if file_list[i].find('deconv') == -1:
                    total_filelist[tag_index][1].append(img)
                else:
                    total_filelist[tag_index][2].append(img)
            elif file_list[i].find('Confocal') != -1:
                total_filelist[tag_index][0].append(img)
        
    for tag_index in range(len(save_list)):
        check_existence(os.path.join(train_dir_HR, save_list[tag_index]))
        check_existence(os.path.join(val_dir_HR, save_list[tag_index]))
        check_existence(os.path.join(train_dir_LR, save_list[tag_index]))
        check_existence(os.path.join(val_dir_LR, save_list[tag_index]))      
        check_existence(os.path.join(train_dir_HR_deconv, save_list[tag_index]))  
        check_existence(os.path.join(val_dir_HR_deconv, save_list[tag_index]))
        for index in range(len(total_filelist[tag_index][1])):
            HR = total_filelist[tag_index][1][index]
            #HR = HR / np.max(HR) * 255
            HR = np.uint16(HR)
            #HR = HR - 32768
            if HR.shape[0] < size or HR.shape[1] < size:
                print(HR.shape, save_list[tag_index], index)
            else:
                if index % val_interval != 0:
                    save_dir_file = os.path.join(os.path.join(train_dir_HR, save_list[tag_index]), '{}.tif'.format((index+1)))       
                else:
                    save_dir_file = os.path.join(os.path.join(val_dir_HR, save_list[tag_index]), '{}.tif'.format((index+1)))
                cv2.imencode('.tif', np.uint8(HR))[1].tofile(save_dir_file)

        for index in range(len(total_filelist[tag_index][0])):
            LR = total_filelist[tag_index][0][index]
            LR = LR / np.max(LR) * 255
            LR = np.uint16(LR)
            H,W = LR.shape
            if H < size or W < size:
                print(H, W, save_list[tag_index], index)
            else:
                if shrink: LR = resize(LR, (H//2, W//2))
                if index % val_interval != 0:
                    save_dir_file = os.path.join(os.path.join(train_dir_LR, save_list[tag_index]), '{}.tif'.format((index+1)))
                else:
                    save_dir_file = os.path.join(os.path.join(val_dir_LR, save_list[tag_index]), '{}.tif'.format((index+1)))
                cv2.imencode('.tif', np.uint8(LR))[1].tofile(save_dir_file)
        for index in range(len(total_filelist[tag_index][2])):
            HR_deconv = total_filelist[tag_index][2][index]
            HR_deconv = HR_deconv / np.max(HR_deconv) * 255
            HR_deconv = np.uint16(HR_deconv)
            H,W = HR_deconv.shape
            if H < size or W < size:
                print(H, W, save_list[tag_index], index)
            else:
                if shrink: HR_deconv = resize(HR_deconv, (H//2, W//2))
                if index % val_interval != 0:
                    save_dir_file = os.path.join(os.path.join(train_dir_HR_deconv, save_list[tag_index]), '{}.tiff'.format((index+1)))
                else:
                    save_dir_file = os.path.join(os.path.join(val_dir_HR_deconv, save_list[tag_index]), '{}.tiff'.format((index+1)))
                cv2.imencode('.tif', np.uint8(HR_deconv))[1].tofile(save_dir_file)


def distribution_demo(read_dir, train_dir_HR, val_dir_HR, tags, HR_name, save_list, val_interval, size=512):
    """
    tags: 
    HR_name: Filenames which includes HR_name will loaded. 
    size: Those images' size smaller than this value will be excluded. 
    """
    total_filelist = []
    for i in range(len(tags)):
        total_filelist.append([[], [], []])
    for tag_index in range(len(tags)):
        read_folder_dir = os.path.join(read_dir, tags[tag_index])
        file_list = os.listdir(read_folder_dir)
        file_list = natsort.natsorted(file_list)
        for i in range(len(file_list)):            
            if file_list[i].find(HR_name+".") != -1:
                img = np.uint16(Image.open(os.path.join(read_folder_dir, file_list[i])))#[20:-20, 20:-20]            
                total_filelist[tag_index][1].append(img)
        
    for tag_index in range(len(save_list)):
        check_existence(os.path.join(train_dir_HR, save_list[tag_index]))
        check_existence(os.path.join(val_dir_HR, save_list[tag_index]))
        for index in range(len(total_filelist[tag_index][1])):
            HR = total_filelist[tag_index][1][index]
            #HR = HR / np.max(HR) * 255
            HR = np.uint16(HR)
            #HR = HR - 32768
            if HR.shape[0] < size or HR.shape[1] < size:
                #print(HR.shape, save_list[tag_index], index)
                pass
            else:
                if index % val_interval != 0:
                    save_dir_file = os.path.join(os.path.join(train_dir_HR, save_list[tag_index]), '{}.tif'.format((index+1)))       
                else:
                    save_dir_file = os.path.join(os.path.join(val_dir_HR, save_list[tag_index]), '{}.tif'.format((index+1)))
                cv2.imencode('.tif', np.uint8(HR))[1].tofile(save_dir_file)


if __name__ == '__main__':
    if 1:
        read_dir = r"D:\CQL\codes\microscopy_decouple\data\STED_data\deconv_8_bit_smooth"
        train_dir_HR = r'D:\CQL\codes\microscopy_decouple\data\train_HR'
        test_dir_HR = r'D:\CQL\codes\microscopy_decouple\data\test_HR'
        train_dir_LR = r'D:\CQL\codes\microscopy_decouple\data\train_LR'
        test_dir_LR = r'D:\CQL\codes\microscopy_decouple\data\test_LR'
        train_dir_HR_deconv = r'D:\CQL\codes\microscopy_decouple\data\train_HR_deconv'
        test_dir_HR_deconv = r'D:\CQL\codes\microscopy_decouple\data\test_HR_deconv'
        tags = ['Micro', 'Mito', 'Membrane', 'NPCs', 'Lyso', 'Mito_inner']
        #tags = ['Microtubes', 'Mitochondria', 'Lysosome']
        save_list = ['Micro', 'Mito', 'Membrane', 'NPCs', 'Lyso', 'Mito_inner']
        #save_list = ['Micro', 'Mito', 'Lyso']

        val_interval = 9
        distribution(read_dir=read_dir, train_dir_HR=train_dir_HR, val_dir_HR=test_dir_HR, train_dir_HR_deconv=train_dir_HR_deconv, val_dir_HR_deconv=test_dir_HR_deconv, 
                    train_dir_LR=train_dir_LR, val_dir_LR=test_dir_LR, tags=tags, save_list=save_list, val_interval=val_interval, shrink=False)
