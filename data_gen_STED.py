import os
import cv2
import tifffile
import random
import time
import copy
import natsort
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import randint
from torchvision import transforms

from utils import *


def get_crop_params(img_size, output_size):
    h, w= img_size
    th = output_size
    tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w
    i = randint(0, h - th)
    j = randint(0, w - tw)
    return i, j, th, tw


def read_stack(read_dir, save_dir):
    file_list = os.listdir(read_dir)
    file_list = natsort.natsorted(file_list)
    name_list = ["Microtubes", "Mitochrondria", "Composed"]
    total_filelist = [[], [], []]
    for _ in range(len(file_list)):
        for name_index in range(len(name_list)): 
            if file_list[_].find(name_list[name_index]) != -1 and file_list[_].find("tif") != -1:
                #print(os.path.join(read_dir, file_list[_]))
                stack = tifffile.imread(os.path.join(read_dir, file_list[_]))                
                r = 0
                c = 0
                for i in range(stack.shape[0]):
                    img = stack[i,:,:]
                    for row in range(img.shape[0]):
                        if img[row,:].sum() == 0:                
                            break
                    for col in range(img.shape[1]):
                        if img[:,col].sum() == 0:                
                            break
                    if (r == row and c == col) and (row > 512 and col > 512):
                        total_filelist[name_index].append(stack[i-1,:row,:col])
                        total_filelist[name_index].append(stack[i,:row,:col])
                        r = 0
                        c = 0
                    else:
                        r = row
                        c = col
    print(len(total_filelist[0]), len(total_filelist[1]), len(total_filelist[2]))
    for name_index in range(len(name_list)):
        save_folder_dir = os.path.join(save_dir, name_list[name_index])
        check_existence(save_folder_dir)
        for i in range(len(total_filelist[name_index])):
            RL = 'Confocal' if i % 2 == 0 else 'STED'            
            save_file_dir = os.path.join(save_folder_dir, '{}_{}.tif'.format(((i // 2)+1), RL))
            #print(save_file_dir)           
            cv2.imencode('.tif', total_filelist[name_index][i] - 32768)[1].tofile(save_file_dir)

                    
def read_raw(read_dir, read_dir_list, save_dir, tag_list):    
    count_list = []
    for i in range(len(tag_list)):
        check_existence(os.path.join(save_dir, tag_list[i]))
        count_list.append(-1)
    for i in range(len(read_dir_list)):
        folder_dir = os.path.join(read_dir, read_dir_list[i])
        file_list = os.listdir(folder_dir)
        file_list = natsort.natsorted(file_list)
        for j in range(len(file_list)):            
            for k in range(len(tag_list)):                
                if file_list[j].find(tag_list[k]) != -1 and file_list[j].find('hgsb') == -1 and file_list[j].find('txt') == -1 and file_list[j].find("zip") == -1:
                    img = np.array(Image.open(os.path.join(folder_dir, file_list[j])))
                    if tag_list[k] == 'Microtubes' or tag_list[k] == 'Mitochondria':
                        img = resize(img, (1000, 1000))
                    elif tag_list[k] == 'Mitochondria_inner':
                        pass
                    elif tag_list[k] == "Lysosome":
                        h, w = img.shape
                        img = resize(img, (int(h*1.5), int(w*1.5)))
                    elif tag_list[k] == "Membrane":
                        h, w = img.shape
                        img = resize(img, (int(h*2.5), int(w*2.5)))
                    save_dir_folder = os.path.join(save_dir, tag_list[k])
                    if file_list[j].find('Confocal.raw') != -1:
                        count_list[k] += 1
                        cv2.imencode('.tif', img)[1].tofile(os.path.join(save_dir_folder, '{}_Confocal.tif').format(count_list[k]))
                    elif file_list[j].find('STED') != -1 and file_list[j].find('cmle') == -1:
                        cv2.imencode('.tif', img)[1].tofile(os.path.join(save_dir_folder, '{}_STED.tif').format(count_list[k]))
                    elif file_list[j].find('cmle') != -1:
                        cv2.imencode('.tif', img)[1].tofile(os.path.join(save_dir_folder, '{}_STED_deconv.tif').format(count_list[k]))
                    


def distribution(read_dir, train_dir_HR, test_dir_HR, train_dir_LR, test_dir_LR, train_dir_HR_deconv, test_dir_HR_deconv, tags, save_list, test_interval, shrink, size=512):
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
        check_existence(os.path.join(test_dir_HR, save_list[tag_index]))
        check_existence(os.path.join(train_dir_LR, save_list[tag_index]))
        check_existence(os.path.join(test_dir_LR, save_list[tag_index]))      
        check_existence(os.path.join(train_dir_HR_deconv, save_list[tag_index]))  
        check_existence(os.path.join(test_dir_HR_deconv, save_list[tag_index]))
        for index in range(len(total_filelist[tag_index][1])):
            HR = total_filelist[tag_index][1][index]
            #HR = HR / np.max(HR) * 255
            HR = np.uint16(HR)
            #HR = HR - 32768
            if HR.shape[0] < size or HR.shape[1] < size:
                print(HR.shape, save_list[tag_index], index)
            else:
                if index % test_interval != 0:
                    save_dir_file = os.path.join(os.path.join(train_dir_HR, save_list[tag_index]), '{}.tif'.format((index+1)))       
                else:
                    save_dir_file = os.path.join(os.path.join(test_dir_HR, save_list[tag_index]), '{}.tif'.format((index+1)))
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
                if index % test_interval != 0:
                    save_dir_file = os.path.join(os.path.join(train_dir_LR, save_list[tag_index]), '{}.tif'.format((index+1)))
                else:
                    save_dir_file = os.path.join(os.path.join(test_dir_LR, save_list[tag_index]), '{}.tif'.format((index+1)))
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
                if index % test_interval != 0:
                    save_dir_file = os.path.join(os.path.join(train_dir_HR_deconv, save_list[tag_index]), '{}.tiff'.format((index+1)))
                else:
                    save_dir_file = os.path.join(os.path.join(test_dir_HR_deconv, save_list[tag_index]), '{}.tiff'.format((index+1)))
                cv2.imencode('.tif', np.uint8(HR_deconv))[1].tofile(save_dir_file)


if 0:
    read_dir = r'D:\CQL\codes\microscopy_decouple\data\STED_data_raw'
    read_dir_list = [
        '20240305_2_Microtubes', '20240306_Microtubes', '20240307_Mitochondria', '20240308_1_Mitochondria', '20240308_2_Mitochondria', 
        '20240311_1_Membrane_3', '20240311_2_Membrane_3', '20240312_Membrane_1_3', '20240312_Membrane_2_3', '20240315_Microtubes', '20240315_Mitochondria', 
        '20240316_Microtubes_Mitochondria', '20240324_Membrane_Microtubes_1_3', '20240325_Mitochondria', '20240328_Mitochondria_inner_1_20', 
        '20240328_Mitochondria_inner_2_20', '20240329_Mitochondria_inner_lysosome', '20240330_Mitochondria_inner_Hela', '20240331_Lysosome_MDA'
    ]
    save_dir = r"D:\CQL\codes\microscopy_decouple\data\STED_data\original"
    tag_list = ['Microtubes', 'Mitochondria', 'Mitochondria_inner', 'Lysosome', 'Membrane']

    read_raw(read_dir=read_dir, read_dir_list=read_dir_list, save_dir=save_dir, tag_list=tag_list)


if __name__ == '__main__':
    if 1:
        read_dir = r"D:\CQL\codes\microscopy_decouple\data\STED_data\deconv_8_bit"
        train_dir_HR = r'D:\CQL\codes\microscopy_decouple\data\train_HR'
        test_dir_HR = r'D:\CQL\codes\microscopy_decouple\data\test_HR'
        train_dir_LR = r'D:\CQL\codes\microscopy_decouple\data\train_LR'
        test_dir_LR = r'D:\CQL\codes\microscopy_decouple\data\test_LR'
        train_dir_HR_deconv = r'D:\CQL\codes\microscopy_decouple\data\train_HR_deconv'
        test_dir_HR_deconv = r'D:\CQL\codes\microscopy_decouple\data\test_HR_deconv'
        tags = ['Micro', 'Mito', 'Mito_inner', 'Lyso', 'Membrane', 'NPCs']
        #tags = ['Microtubes', 'Mitochondria', 'Lysosome']
        save_list = ['Micro', 'Mito', 'Mito_inner', 'Lyso', 'Membrane', 'NPCs']
        #save_list = ['Micro', 'Mito', 'Lyso']

        test_interval = 9
        distribution(read_dir=read_dir, train_dir_HR=train_dir_HR, test_dir_HR=test_dir_HR, train_dir_HR_deconv=train_dir_HR_deconv, test_dir_HR_deconv=test_dir_HR_deconv, 
                    train_dir_LR=train_dir_LR, test_dir_LR=test_dir_LR, tags=tags, save_list=save_list, test_interval=test_interval, shrink=False)
