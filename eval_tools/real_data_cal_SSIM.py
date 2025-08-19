import os, sys, natsort, tifffile, torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
from tqdm import tqdm
from loss.SSIM_loss import SSIM



def real_data_cal_SSIM(read_dir, save_dir):
    file_list = os.listdir(read_dir)
    file_list = natsort.natsorted(file_list)
    pred_list_1 = []
    GT_list_1 = []
    pred_list_2 = []
    GT_list_2 = []
    SSIM_criterion = SSIM().to('cuda')
    ssim_list_1 = []
    ssim_list_2 = []
    for i in range(len(file_list)):
        file_dir = os.path.join(read_dir, file_list[i])
        img = tifffile.imread(file_dir)
        img = torch.tensor(np.float32(img), dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
        if file_list[i].find("Convolved_1") != -1:
            pred_list_1.append(img)
        elif file_list[i].find("GT_1") != -1:
            GT_list_1.append(img)
        elif file_list[i].find("Convolved_2") != -1:
            pred_list_2.append(img)
        elif file_list[i].find("GT_2") != -1:
            GT_list_2.append(img)
    for i in range(len(pred_list_1)):
        ssim_index_1 = SSIM_criterion(GT_list_1[i], pred_list_1[i])
        print(ssim_index_1.item())



        

if __name__ == "__main__":
    read_dir = r"D:\CQL\DSCM\real_data_eval\Micro_Mito_eval\results\228_0.015_4"
    save_dir = r""
    real_data_cal_SSIM(read_dir=read_dir, save_dir=save_dir)
