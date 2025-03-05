import os, sys, shutil
import pandas as pd

# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *


def find_max_SSIM(read_dir, save_dir, name_list):
    data_frame = pd.read_csv(read_dir)
    data_col_list = [[] for i in range(len(name_list))]
    sta_list = [[] for i in range(len(name_list))]
    sta_name_list = ['argmin', 'min', 'argmax', 'max', 'mean', 'std']
    df_dict = {}
    # enumerate in name_list to obtain data in column
    for i in range(len(name_list)):
        data_col_list[i].append(pd.to_numeric(data_frame.iloc[1:,i]).tolist())
        sta_list[i] = [np.argmin(data_col_list[i][0])+1, np.min(data_col_list[i][0]), np.argmax(data_col_list[i][0])+1, np.max(data_col_list[i][0]), np.mean(data_col_list[i][0]), np.std(data_col_list[i][0])]
        df_dict[f'{name_list[i]}'] = sta_list[i]
    df = pd.DataFrame(df_dict, index=sta_name_list)
    df.to_csv(save_dir)

    return df_dict

        
if __name__ == "__main__":
    if 0:
        read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data\MAE_SSIM.csv'
        save_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data\MAE_SSIM_sta.csv'
        name_list = [
        'mae_micro_micro_lyso', 'mae_lyso_micro_lyso', 'SSIM_micro_micro_lyso', 'SSIM_lyso_micro_lyso', 'mae_micro_lyso', 'SSIM_micro_lyso', 
        'mae_mito_mito_lyso', 'mae_lyso_mito_lyso', 'SSIM_mito_mito_lyso', 'SSIM_lyso_mito_lyso', 'mae_mito_lyso', 'SSIM_mito_lyso', 
        'mae_micro_micro_mito', 'mae_mito_micro_mito', 'SSIM_micro_micro_mito', 'SSIM_mito_micro_mito', 'mae_micro_mito', 'SSIM_micro_mito', 
        'mae_micro_micro_mito_lyso', 'mae_mito_micro_mito_lyso', 'mae_lyso_micro_mito_lyso', 'SSIM_micro_micro_mito_lyso', 'SSIM_mito_micro_mito_lyso', 'SSIM_lyso_micro_mito_lyso', 'mae_micro_mito_lyso', 'SSIM_micro_mito_lyso'
        ]
        df = find_max_SSIM(read_dir=read_dir, save_dir=save_dir, name_list=name_list)
    
    if 0:
        read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\SIM_synthetic_data\MAE_SSIM.csv'
        save_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\SIM_synthetic_data\MAE_SSIM_sta.csv'
        name_list = [
        'mae_CCPS_CCPS_ER', 'mae_ER_CCPS_ER', 'SSIM_CCPS_CCPS_ER', 'SSIM_ER_CCPS_ER', 'mae_CCPS_ER', 'SSIM_CCPS_ER', 
        'mae_CCPS_CCPS_Micro', 'mae_Micro_CCPS_Micro', 'SSIM_CCPS_CCPS_Micro', 'SSIM_Micro_CCPS_Micro', 'mae_CCPS_Micro', 'SSIM_CCPS_Micro', 
        'mae_ER_ER_Micro', 'mae_Micro_ER_Micro', 'SSIM_ER_ER_Micro', 'SSIM_Micro_ER_Micro', 'mae_ER_Micro', 'SSIM_ER_Micro', 
        'mae_CCPS_CCPS_ER_Micro', 'mae_ER_CCPS_ER_Micro', 'mae_Micro_CCPS_ER_Micro', 'SSIM_CCPS_CCPS_ER_Micro', 'SSIM_ER_CCPS_ER_Micro', 'SSIM_Micro_CCPS_ER_Micro', 'mae_CCPS_ER_Micro', 'SSIM_CCPS_ER_Micro'
        ]
        find_max_SSIM(read_dir=read_dir, save_dir=save_dir, name_list=name_list)