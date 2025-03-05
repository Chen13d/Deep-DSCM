import os, sys
 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)


import pandas as pd
from utils import *



def sort_Decorrelation(read_dir, org_list, pixel_size=20, save_dir=None):
    df = pd.read_csv(read_dir)
    #filtered_df = df[df['Images'].str.contains('Output')]
    #image_count = len(df['Images'].dropna())
    #print((len(filtered_df)))
    num_total = len(df['Label'])
    interval = 1 + len(org_list)*2
    dec = df['Res.']
    dec_list = [[[] for i in range(len(org_list))] for i in range(2)]
    #print(dec[0]*20)
    for i in range(num_total):
        res = i % interval
        redu = res % 2
        quot = res // 2
        if res != 0:
            if redu != 0:
                #print(res, quot, "GT:", dec[i])
                dec_list[0][quot].append(dec[i]*pixel_size)
            else:
                #print(res, quot, "Output:", dec[i])
                dec_list[1][quot-1].append(dec[i]*pixel_size)
    temp_list = ['GT', 'Output']
    for i in range(len(org_list)):
        for j in range(2):
            data_dict[f'{org_list[i]}_{temp_list[j]}'] = dec_list[j][i]



if __name__ == "__main__":
    pixel_size = 20
    org_list = ['NPCs', 'Mito_inner', 'Membrane']
    
    read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data\NPCs_Mito_inner_Membrane\noise_0\Decorrelation.csv'
    data_dict = {}
    sort_Decorrelation(read_dir=read_dir, org_list=org_list, pixel_size=pixel_size)

    
    data_frame = pd.DataFrame(data_dict)
    data_frame.to_csv(r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_synthetic_data\Decorrelation.csv', index=False)