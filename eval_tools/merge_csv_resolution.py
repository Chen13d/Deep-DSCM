import os, sys
import pandas as pd

# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *

def merge_csv(read_dir, save_dir, Org_list, name_list, combination_list, noise_list=[0]):
    name_count = 0
    data_list = [[] for i in range(len(name_list))]
    half_len_name_list = len(name_list) // 2
    # 遍历组合
    for combination_index in range(len(combination_list)):
        folder_path = os.path.join(read_dir, "{}\\noise_{}\\Resolution_csv".format(combination_list[combination_index], noise_list[0]))
        org_list = Org_list[combination_index]
        num_org = len(org_list)
        Input_list = []
        GT_list = [[] for i in range(num_org)]
        Output_list = [[] for i in range(num_org)]

        # 获取文件夹中所有的CSV文件名
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        csv_files = natsort.natsorted(csv_files)
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            # 读取CSV文件
            df = pd.read_csv(file_path)
            # 将文件名作为一列添加到数据中（可选）
            name = file[:file.find(".csv")]
            #df['source_file'] = file[:file.find(".csv")]
            if name.find("Input") != -1:
                Input_list.append(df)
            for i in range(num_org):
                if name.find("Output") != -1 and name.find(org_list[i]) != -1:
                    Output_list[i].append(df)
                    #print(len(Output_list))
                elif name.find("GT") != -1 and name.find(org_list[i]) != -1:
                    GT_list[i].append(df)

        # 将数据追加到all_data中
        Input_df = pd.concat([*Input_list], ignore_index=True)
        Output_df = [[] for i in range(num_org)]
        GT_df = [[] for i in range(num_org)]
        for i in range(num_org):
            Output_df[i] = pd.concat([*Output_list[i]], ignore_index=True)
            GT_df[i] = pd.concat([*GT_list[i]], ignore_index=True)
        for i in range(num_org):
            data_list[name_count] = (pd.to_numeric(GT_df[i].iloc[:,0]).tolist())
            data_list[name_count+half_len_name_list] = (pd.to_numeric(GT_df[i].iloc[:,2]).tolist())
            #if name_count == 16 or name_count+half_len_name_list-1 == 16: print(name_count)
            
            name_count += 1
            data_list[name_count] = (pd.to_numeric(Output_df[i].iloc[:,0]).tolist())
            data_list[name_count+half_len_name_list] = (pd.to_numeric(Output_df[i].iloc[:,2]).tolist())
            #if name_count == 16 or name_count+half_len_name_list-1 == 16: print(name_count)
            name_count += 1
    #print(len(name_list), len(data_list))
    #for i in range(len(name_list)):
    #    print(name_list[i], len(data_list[i]))
    #print(name_list[16], len(data_list[16]))
    data_dict = {}
    for i in range(len(name_list)):
        data_dict[f'{name_list[i]}'] = data_list[i]
    data_frame = pd.DataFrame(data_dict)
    data_frame.to_csv(save_dir, index=False)
        
            


if __name__ == "__main__":
    if 0:
        noise_list = [0]
        combination_list = ['Microtubules_Lysosome', 'Mitochondria_outer_Lysosome', 'Microtubules_Mitochondria_outer', 'Microtubules_Mitochondria_outer_Lysosome']
        org_list = [['micro', 'lyso'], ['mito_outer', 'lyso'], ['micro', 'mito_outer'], ['micro', 'mito_outer', 'lyso']]
        name_list = [
            'mean_micro_micro_lyso_GT', 'mean_micro_micro_lyso_Output', 'mean_lyso_micro_lyso_GT', 'mean_lyso_micro_lyso_Output', 
            'mean_mito_mito_lyso_GT', 'mean_mito_mito_lyso_Output', 'mean_lyso_mito_lyso_GT', 'mean_lyso_mito_lyso_Output',
            'mean_micro_micro_mito_GT', 'mean_micro_micro_mito_Output', 'mean_mito_micro_mito_GT', 'mean_mito_micro_mito_Output', 
            'mean_micro_micro_mito_lyso_GT', 'mean_micro_micro_mito_lyso_Output', 'mean_mito_micro_mito_lyso_GT', 'mean_mito_micro_mito_lyso_Output', 'mean_lyso_micro_mito_lyso_GT', 'mean_lyso_micro_mito_lyso_Output', 
            'min_micro_micro_lyso_GT', 'min_micro_micro_lyso_Output', 'min_lyso_micro_lyso_GT', 'min_lyso_micro_lyso_Output', 
            'min_mito_mito_lyso_GT', 'min_mito_mito_lyso_Output', 'min_lyso_mito_lyso_GT', 'min_lyso_mito_lyso_Output',
            'min_micro_micro_mito_GT', 'min_micro_micro_mito_Output', 'min_mito_micro_mito_GT', 'min_mito_micro_mito_Output', 
            'min_micro_micro_mito_lyso_GT', 'min_micro_micro_mito_lyso_Output', 'min_mito_micro_mito_lyso_GT', 'min_mito_micro_mito_lyso_Output', 'min_lyso_micro_mito_lyso_GT', 'min_lyso_micro_mito_lyso_Output', 
        ]    
        read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data'
        save_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data\Resolution.csv'
        merge_csv(read_dir=read_dir, save_dir=save_dir, Org_list=org_list, name_list=name_list, combination_list=combination_list)
    if 1:
        noise_list = [0]
        combination_list = ['CCPS_ER', 'CCPS_Micro', 'CCPS_F-actin', 'ER_Micro', 'CCPS_ER_Micro']
        org_list = [['CCPS', 'ER'], ['CCPS', 'Micro'], ['CCPS', 'F-actin'], ['ER', 'Micro'], ['CCPS', 'ER', 'Micro']]
        name_list = [
            'mean_CCPS_CCPS_ER_GT', 'mean_CCPS_CCPS_ER_Output', 'mean_ER_CCPS_ER_GT', 'mean_ER_CCPS_ER_Output', 
            'mean_CCPS_CCPS_Micro_GT', 'mean_CCPS_CCPS_Micro_Output', 'mean_Micro_CCPS_Micro_GT', 'mean_Micro_CCPS_Micro_Output', 
            'mean_CCPS_CCPS_F-actin_GT', 'mean_CCPS_CCPS_F-actin_Output', 'mean_F-actin_CCPS_F-actin_GT', 'mean_F-actin_CCPS_F-actin_Output', 
            'mean_ER_ER_Micro_GT', 'mean_ER_ER_Micro_Output', 'mean_Micro_ER_Micro_GT', 'mean_Micro_ER_Micro_Output', 
            'mean_CCPS_CCPS_ER_Micro_GT', 'mean_CCPS_CCPS_ER_Micro_Output', 'mean_ER_CCPS_ER_Micro_GT', 'mean_ER_CCPS_ER_Micro_Output', 'mean_Micro_CCPS_ER_Micro_GT', 'mean_Micro_CCPS_ER_Micro_Output', 
            #'mean_CCPS_CCPS_ER_F-actin_GT', 'mean_CCPS_CCPS_ER_F-actin_Output', 'mean_ER_CCPS_ER_F-actin_GT', 'mean_ER_CCPS_ER_F-actin_Output', 'mean_F-actin_CCPS_ER_F-actin_GT', 'mean_F-actin_CCPS_ER_F-actin_Output', 
            #'mean_CCPS_CCPS_Micro_F-actin_GT', 'mean_CCPS_CCPS_Micro_F-actin_Output', 'mean_Micro_CCPS_Micro_F-actin_GT', 'mean_Micro_CCPS_Micro_F-actin_Output', 'mean_F-actin_CCPS_Micro_F-actin_GT', 'mean_F-actin_CCPS_Micro_F-actin_Output', 
            #'mean_ER_ER_Micro_F-actin_GT', 'mean_ER_ER_Micro_F-actin_Output', 'mean_Micro_ER_Micro_F-actin_GT', 'mean_Micro_ER_Micro_F-actin_Output', 'mean_F-actin_ER_Micro_F-actin_GT', 'mean_F-actom_ER_Micro_F-actin_Output', 
            'min_CCPS_CCPS_ER_GT', 'min_CCPS_CCPS_ER_Output', 'min_ER_CCPS_ER_GT', 'min_ER_CCPS_ER_Output', 
            'min_CCPS_CCPS_Micro_GT', 'min_CCPS_CCPS_Micro_Output', 'min_Micro_CCPS_Micro_GT', 'min_Micro_CCPS_Micro_Output', 
            'min_CCPS_CCPS_F-actin_GT', 'min_CCPS_CCPS_F-actin_Output', 'min_F-actin_CCPS_F-actin_GT', 'min_F-actin_CCPS_F-actin_Output', 
            'min_ER_ER_Micro_GT', 'min_ER_ER_Micro_Output', 'min_Micro_ER_Micro_GT', 'min_Micro_ER_Micro_Output', 
            'min_CCPS_CCPS_ER_Micro_GT', 'min_CCPS_CCPS_ER_Micro_Output', 'min_ER_CCPS_ER_Micro_GT', 'min_ER_CCPS_ER_Micro_Output', 'min_Micro_CCPS_ER_Micro_GT', 'min_Micro_CCPS_ER_Micro_Output', 
            #'min_CCPS_CCPS_ER_F-actin_GT', 'min_CCPS_CCPS_ER_F-actin_Output', 'min_ER_CCPS_ER_F-actin_GT', 'min_ER_CCPS_ER_F-actin_Output', 'min_F-actin_CCPS_ER_F-actin_GT', 'min_F-actin_CCPS_ER_F-actin_Output', 
            #'min_CCPS_CCPS_Micro_F-actin_GT', 'min_CCPS_CCPS_Micro_F-actin_Output', 'min_Micro_CCPS_Micro_F-actin_GT', 'min_Micro_CCPS_Micro_F-actin_Output', 'min_F-actin_CCPS_Micro_F-actin_GT', 'min_F-actin_CCPS_Micro_F-actin_Output', 
            #'min_ER_ER_Micro_F-actin_GT', 'min_ER_ER_Micro_F-actin_Output', 'min_Micro_ER_Micro_F-actin_GT', 'min_Micro_ER_Micro_F-actin_Output', 'min_F-actin_ER_Micro_F-actin_GT', 'min_F-actom_ER_Micro_F-actin_Output', 
        ]
        read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\SIM_synthetic_data'
        save_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\SIM_synthetic_data\Resolution.csv'
        merge_csv(read_dir=read_dir, save_dir=save_dir, Org_list=org_list, name_list=name_list, combination_list=combination_list)
                



