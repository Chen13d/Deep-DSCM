import os, sys
import pandas as pd
from skimage.metrics import structural_similarity as ssim_
# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *
from loss.SSIM_loss import *

device = 'cuda'
ssim_criterion = SSIM(device=device)

read_dir = r'D:\CQL\codes\microscopy_decouple\evaluation\DSRM\DSRM_SIM_data\SIM'
size = 512
imlist = os.listdir(read_dir)
SSIM_list = [[] for i in range(3)]
MAE_list = [[] for i in range(3)]

for i in range(len(imlist)):
    image = np.array(Image.open(os.path.join(read_dir, imlist[i])))
    for j in range(3):
        GT = image[:size, (j+1)*size:(j+2)*size]
        Output = image[size:size*2, (j+1)*size:(j+2)*size]

        #GT = torch.tensor(GT).to(device).unsqueeze(0).unsqueeze(0)
        #Output = torch.tensor(Output).to(device).unsqueeze(0).unsqueeze(0)
        #val_std = torch.std(GT)
        temp_max = max(np.max(GT), np.max(Output))
        GT = GT / temp_max * 255
        Output = Output / temp_max * 255

        #print(np.max(GT), np.min(GT), np.max(Output), np.min(Output))

        GT[GT<0] = 0
        Output[Output<0] = 0
        #print(np.min(GT), np.max(GT), np.min(Output), np.max(Output))
        GT = np.uint8(GT)
        Output = np.uint8(Output)
        
        #score = ssim_criterion(Output, GT)
        score = ssim_(GT, Output)
        print(score)

        
        #print(ssim_score)
