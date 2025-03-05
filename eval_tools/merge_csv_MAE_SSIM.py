import os, sys
import pandas as pd

# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *



def merge_csv_MAE_SSIM(read_dir_list, save_dir, name_list):
    data_list = [[] for i in range(len(name_list))]
    name_count = 0
    # enumerate combination]
    for i in range(len(read_dir_list)):
        read_csv_dir = os.path.join(read_dir_list[i], 'MAE_SSIM.csv')
        # enumerate noise_list
        df = pd.read_csv(read_csv_dir)
        # cal num_org, discard index column
        num_col = len(df.iloc[0]) - 1
        num_org = num_col // 2
        for j in range(num_col):
            #data_list[name_count] = [name_list[name_count]]
            data_list[name_count] = pd.to_numeric(df.iloc[:,j+1]).tolist()
            name_count += 1
    data_dict = {}
    for i in range(len(data_list)):
        data_dict[f'{name_list[i]}'] = data_list[i]
    data_frame = pd.DataFrame(data_dict)

    data_frame.to_csv(save_dir, index=False)
        
if __name__ == "__main__":
    if 1:
        read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data'
        save_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data\MAE_SSIM.csv'
        noise_list = [0]
        name_list = [
            'mae_micro_micro_lyso', 'mae_lyso_micro_lyso', 'SSIM_micro_micro_lyso', 'SSIM_lyso_micro_lyso', 'mae_micro_lyso', 'SSIM_micro_lyso', 
            'mae_mito_mito_lyso', 'mae_lyso_mito_lyso', 'SSIM_mito_mito_lyso', 'SSIM_lyso_mito_lyso', 'mae_mito_lyso', 'SSIM_mito_lyso', 
            'mae_micro_micro_mito', 'mae_mito_micro_mito', 'SSIM_micro_micro_mito', 'SSIM_mito_micro_mito', 'mae_micro_mito', 'SSIM_micro_mito', 
            'mae_micro_micro_mito_lyso', 'mae_mito_micro_mito_lyso', 'mae_lyso_micro_mito_lyso', 'SSIM_micro_micro_mito_lyso', 'SSIM_mito_micro_mito_lyso', 'SSIM_lyso_micro_mito_lyso', 'mae_micro_mito_lyso', 'SSIM_micro_mito_lyso', 
            'mae_NPCs_NPCs_mito_inner', 'mae_mito_inner_NPCs_mito_inner', 'SSIM_NPCS_NPCs_mito_inner', 'SSIM_mito_inner_NPCs_mito_inner', 'mae_NPCs_mito_inner', 'SSIM_NPCs_mito_inner', 
            'mae_NPCs_NPCs_membrane', 'mae_Membrane_NPCs_membrane', 'SSIM_NPCs_NPCs_membrane', 'SSIM_Membrane_NPCs_membrane', 'mae_NPCs_membrane', 'SSIM_NPCs_membrane', 
            'mae_mito_inner_mito_inner_membrane', 'mae_membrane_mito_inner_membrane', 'SSIM_mito_inner_mito_inner_membrane', 'SSIM_membrane_mito_inner_membrane', 'mae_mito_inner_membrane', 'SSIM_mito_inner_membrane', 
            'mae_NPCs_NPCs_mito_inner_membrane', 'mae_mito_inner_NPCs_mito_inner_membrane', 'mae_membrane_NPCs_mito_inner_membrane', 'SSIM_NPCs_NPCs_mito_inner_membrane', 'SSIM_mito_inner_NPCs_mito_inner_membrane', 'SSIM_membrane_NPCs_mito_inner_membrane', 'mae_NPCs_mito_inner_membrane', 'SSIM_NPCs_mito_inner_membrane', 
        ]
        combination_list = ['Micro_Lyso', 'Mito_Lyso', 'Micro_Mito', 'Micro_Mito_Lyso', 
                            'NPCs_Mito_inner', 'NPCs_Membrane', 'Mito_inner_Membrane', 'NPCs_Mito_inner_Membrane']
        read_dir_list = [os.path.join(os.path.join(read_dir, combination_list[i]), "noise_"+str(noise_list[0])) for i in range(len(combination_list))]
        
        merge_csv_MAE_SSIM(read_dir_list=read_dir_list, save_dir=save_dir, name_list=name_list)
    if 0:
        read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\SIM_synthetic_data'
        save_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\SIM_synthetic_data\MAE_SSIM.csv'
        noise_list = [0]
        name_list = [
            'mae_CCPS_CCPS_ER', 'mae_ER_CCPS_ER', 'SSIM_CCPS_CCPS_ER', 'SSIM_ER_CCPS_ER', 'mae_CCPS_ER', 'SSIM_CCPS_ER', 
            'mae_CCPS_CCPS_Micro', 'mae_Micro_CCPS_Micro', 'SSIM_CCPS_CCPS_Micro', 'SSIM_Micro_CCPS_Micro', 'mae_CCPS_Micro', 'SSIM_CCPS_Micro', 
            'mae_CCPS_CCPS_F_actin', 'mae_F-actin_CCPS_F-actin', 'SSIM_CCPS_CCPS_F-actin', 'SSIM_F-actin_CCPS_F-actin', 'mae_CCPS_F-actin', 'SSIM_CCPS_F-actin', 
            'mae_ER_ER_F-actin', 'mae_F-actin_ER_F-actin', 'SSIM_ER_ER_F-actin', 'SSIM_F-actin_ER_F-actin', 'mae_ER_F-actin', 'SSIM_ER_F-actin', 
            'mae_Micro_Micro_F-actin', 'mae_F-actin_Micro_F-actin', 'SSIM_Micro_Micro_F-actin', 'SSIM_F-actin_Micro_F-actin', 'mae_Micro_F-actin', 'SSIM_Micro_F-actin', 
            'mae_ER_ER_Micro', 'mae_Micro_ER_Micro', 'SSIM_ER_ER_Micro', 'SSIM_Micro_ER_Micro', 'mae_ER_Micro', 'SSIM_ER_Micro', 
            'mae_CCPS_CCPS_ER_Micro', 'mae_ER_CCPS_ER_Micro', 'mae_Micro_CCPS_ER_Micro', 'SSIM_CCPS_CCPS_ER_Micro', 'SSIM_ER_CCPS_ER_Micro', 'SSIM_Micro_CCPS_ER_Micro', 'mae_CCPS_ER_Micro', 'SSIM_CCPS_ER_Micro', 
            'mae_CCPS_CCPS_ER_F-actin', 'mae_ER_CCPS_ER_F-actin', 'mae_F-actin_CCPS_ER_Micro', 'SSIM_CCPS_CCPS_ER_F-actin', 'SSIM_ER_CCPS_ER_F-actin', 'SSIM_F-actin_CCPS_ER_F-actin', 'mae_CCPS_ER_F-actin', 'SSIM_CCPS_ER_F-actin', 
            #'mae_CCPS_CCPS_Micro_F-actin', 'mae_Micro_CCPS_Micro_F-actin', 'mae_F-actin_CCPS_Micro_F-actin', 'SSIM_CCPS_CCPS_Micro_F-actin', 'SSIM_Micro_CCPS_Micro_F-actin', 'SSIM_F-actin_CCPS_Micro_F-actin', 
            #'mae_ER_ER_Micro_F-actin', 'mae_Micro_ER_Micro_F-actin', 'mae_F-actin_ER_Micro_F-actin', 'SSIM_ER_ER_Micro_F-actin', 'SSIM_Micro_ER_Micro_F-actin', 'SSIM_F-actin_ER_Micro_F-actin', 
        ]
        combination_list = ['CCPS_ER', 'CCPS_Micro', 'CCPS_F-actin', 'ER_F-actin', 'Micro_F-actin', 'ER_Micro', 'CCPS_ER_Micro', 'CCPS_ER_F-actin'] #'CCPS_Micro_F-actin', 'ER_Micro_F-actin'
        read_dir_list = [os.path.join(os.path.join(read_dir, combination_list[i]), "noise_"+str(noise_list[0])) for i in range(len(combination_list))]

        merge_csv_MAE_SSIM(read_dir_list=read_dir_list, save_dir=save_dir, name_list=name_list)