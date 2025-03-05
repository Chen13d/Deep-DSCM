import os, sys, shutil, tifffile
import pandas as pd
from sklearn.metrics import mean_absolute_error
# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *


def mae(index, y_true, y_pred, save_dir=False):
    error_map = torch.abs((y_true - y_pred)) / (y_true.max() - y_true.min())
    error_map_list = []
    for i in range(error_map.size(1)):
        error_map_cpu = to_cpu(error_map[0,i,:,:])
        error_map_list.append(error_map_cpu)
    mae = torch.mean(error_map)
    #mae = mae
    return mae

def estimate_MAE(read_dir, save_dir, org_list, name):
    read_dir = os.path.join(read_dir, "ROI")
    file_list = natsort.natsorted(os.listdir(read_dir))
    interval = len(org_list)*2+1
    MAE_list = [[] for i in range(len(org_list))]
    for i in range(len(file_list)):
        res = i % interval
        if res != 0:
            if res % 2 != 0:
                print("GT", file_list[i])
                GT = np.array(Image.open(os.path.join(read_dir, file_list[i])))
            else:
                print("Output", file_list[i])
                Output = np.array(Image.open(os.path.join(read_dir, file_list[i])))
                GT = GT / (np.max(GT)-np.min(GT))
                Output = Output / (np.max(Output)-np.min(Output))
                #GT = GT / np.max(GT)
                #Output = Output / np.max(Output)
                '''plt.figure(1)
                plt.subplot(121)
                plt.imshow(GT)
                plt.subplot(122)
                plt.imshow(Output)
                plt.show()'''
                MAE_list[(res//2)-1].append(mean_absolute_error(GT, Output))
                #GT = torch.tensor(np.float64(GT), dtype=torch.float).unsqueeze(0).unsqueeze(0)
                #Output = torch.tensor(np.float64(Output), dtype=torch.float).unsqueeze(0).unsqueeze(0)
                #MAE_list[(res//2)-1].append(mae(index=0, y_true=GT, y_pred=Output))
        else:
            pass

    for i in range(len(org_list)):
        data_dict[f'{name}_{org_list[i]}'] = MAE_list[i]
    

if __name__ == "__main__":
    read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data'
    org_list = [['NPCs', 'Mito_inner_deconv'], ['NPCs', 'Membrane'], ['Mito_inner_deconv', 'Membrane'], ['NPCs', 'Mito_inner_deconv', 'Membrane']]
    name_list = ['NPCs_Mitochondria_inner', 'NPCs_Membrane', 'Mitochondria_inner_Membrane', 'NPCs_Mitochondria_inner_Membrane']
    noise_list = [0]
    for j in range(len(noise_list)):
        data_dict = {}
        for i in range(len(org_list)):
            read_dir_folder = os.path.join(os.path.join(read_dir, name_list[i]), f'noise_{noise_list[j]}')
            #save_dir_folder = os.path.join(read_dir_folder, "MAE.csv")
            estimate_MAE(read_dir=read_dir_folder, save_dir=None, org_list=org_list[i], name=name_list[i])

    data_frame = pd.DataFrame(data_dict)
    save_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data\MAE.csv'
    data_frame.to_csv(save_dir, index=False)