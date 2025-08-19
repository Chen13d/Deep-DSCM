import os, sys, natsort, tifffile, re, copy
import torch
import numpy as np
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from utils import *
from loss.NRMAE import *
from loss.SSIM_loss import SSIM
from loss.pearson_loss import *

from skimage.metrics import structural_similarity as ssim

def extract_number(item):
	# 从文件名中提取数字并转换为整数
	number = re.findall(r'\d+', item)
	return int(number[0]) if number else 0


def check_existence(dir_path):
	"""确保输出目录存在并清空旧文件"""
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	else:
		for fname in os.listdir(dir_path):
			os.remove(os.path.join(dir_path, fname))



def make_threshold(Input):
        thresh = threshold_otsu(Input)
        Output = copy.deepcopy(Input)
        Output[Output<thresh] = 0
        Output[Output>=thresh] = 255
        kernel_size = (3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        Output = cv2.dilate(Output, kernel, 
                 anchor=None, 
                 iterations=1, 
                 borderType=cv2.BORDER_CONSTANT, 
                 borderValue=0)
        return Output


def generate_overlapped_map(image_list):
	for i in range(len(image_list)):
		img = image_list[i]
		img_thresh = make_threshold(img)
		if i == 0:
			img_overlapped = np.zeros_like(img)
			img_sum = np.zeros_like(img)
		img_sum += img
		img_overlapped += img_thresh
	img_overlapped[img_overlapped == 255] = 0
	img_overlapped[img_overlapped != 0] = 1
	return img_overlapped, img_sum


def get_lists_2(read_dir):
	# 按数字顺序读取文件
	file_list = sorted(os.listdir(read_dir), key=extract_number)

	# 根据命名规则分类
	input_list, gt1_list, gt2_list, pred1_list, pred2_list = [], [], [], [], []
	for name in file_list:
		p = os.path.join(read_dir, name)
		if "Input" in name:
			input_list.append(p)
		elif "GT_0" in name:
			gt1_list.append(p)
		elif "GT_1" in name:
			gt2_list.append(p)
		elif "pred_0" in name:  # 注意: 改为 pred_0
			pred1_list.append(p)
		elif "pred_1" in name:  # 注意: 改为 pred_1
			pred2_list.append(p)
	return input_list, gt1_list, pred1_list, gt2_list, pred2_list

def get_lists_3(read_dir):
	# 按数字顺序读取文件
	file_list = sorted(os.listdir(read_dir), key=extract_number)

	# 根据命名规则分类
	input_list, gt1_list, gt2_list, gt3_list, pred1_list, pred2_list, pred3_list = [], [], [], [], [], [], []
	for name in file_list:
		p = os.path.join(read_dir, name)
		if "Input" in name:
			input_list.append(p)
		elif "GT_0" in name:
			gt1_list.append(p)
		elif "GT_1" in name:
			gt2_list.append(p)
		elif "GT_2" in name:
			gt3_list.append(p)
		elif "pred_0" in name:  # 注意: 改为 pred_0
			pred1_list.append(p)
		elif "pred_1" in name:  # 注意: 改为 pred_1
			pred2_list.append(p)
		elif "pred_2" in name:
			pred3_list.append(p)
	return input_list, gt1_list, pred1_list, gt2_list, pred2_list, gt3_list, pred3_list


def cal_NRMAE_PSNR_SSIM(read_dir, save_dir, num_org):
	file_list = os.listdir(read_dir)
	file_list = natsort.natsorted(file_list)
	data_list = get_lists_3(read_dir=read_dir) if num_org == 3 else get_lists_2(read_dir=read_dir)
	SSIM_criterion = SSIM().to('cuda')
	MAE_list = []
	SSIM_list = []
	PSNR_list = []
	for index in range(len(data_list[0])):
		for cat_index in range(num_org):
			GT = tifffile.imread(data_list[1+cat_index*2][index])[:,:,0]
			Pred = tifffile.imread(data_list[1+cat_index*2+1][index])[:,:,0]
			GT = GT / (np.max(GT) - np.min(GT))
			Pred = Pred / (np.max(Pred) - np.min(Pred))
			# cal PSNR
			PSNR_value = calculate_psnr(GT, Pred)
			GT = torch.tensor(np.float32(GT), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
			Pred = torch.tensor(np.float32(Pred), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
			# cal MAE
			MAE_value = nrmae(GT, Pred).item()
			# cal SSIM
			SSIM_value = SSIM_criterion(GT, Pred).item()
			MAE_list.append(MAE_value)
			SSIM_list.append(SSIM_value)
			PSNR_list.append(PSNR_value)

	result_df = pd.DataFrame({
        'MAE': MAE_list,
        'SSIM': SSIM_list,
        'PSNR': PSNR_list,
    })

	result_df.to_csv(save_dir, index=False, float_format='%.4f')

# ChatGPT
import torch
import torch.nn.functional as F
from functools import partial

def _gaussian_kernel(channels, kernel_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel

def masked_ssim_torch(img1, img2, mask=None, data_range=1.0, kernel_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    if mask is None:                       # 默认假设 0 表示被 mask
        mask = ((img1 != 0) & (img2 != 0)).float()
    else:
        mask = mask.float()

    kernel = _gaussian_kernel(img1.shape[1], kernel_size, sigma, img1.device)

    # 1）对图像与掩膜做加权平均
    def filter2d(x):
        return F.conv2d(x, kernel, padding=kernel_size//2, groups=x.shape[1])

    # 有效像素数（不同位置可能不一样）
    mask_mean = filter2d(mask)
    mask_mean = torch.clamp(mask_mean, min=1e-12)      # 避免除零

    mu1 = filter2d(img1 * mask) / mask_mean
    mu2 = filter2d(img2 * mask) / mask_mean

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = filter2d(img1**2 * mask) / mask_mean - mu1_sq
    sigma2_sq = filter2d(img2**2 * mask) / mask_mean - mu2_sq
    sigma12   = filter2d(img1 * img2 * mask) / mask_mean - mu1_mu2

    # SSIM map
    C1 *= data_range**2
    C2 *= data_range**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 2）只在有效区域求均值
    ssim_mean = (ssim_map * mask).sum() / mask.sum()
    return ssim_mean.item(), ssim_map         # 返回全局 SSIM 和局部 SSIM map


def masked_pcc_torch(img1, img2, mask=None, eps=1e-8):
	"""
	img*: Tensor，形状 (N, C, H, W) 或 (C, H, W) 或 (H, W)
	mask: 同形状布尔张量；若为 None，则默认两图非零处为有效像素
	返回: (N, C) 或 (C,) 或 float，具体取决于输入维度
	"""
	if mask is None:
		mask = (img1 != 0) & (img2 != 0)
	mask = mask.float()

	# 展平成 (batch, channel, num_valid)
	def flatten_valid(x):
		return x.reshape(x.shape[0], x.shape[1], -1)  # N C L
	
	x = flatten_valid(img1*mask)
	y = flatten_valid(img2*mask)
	m = flatten_valid(mask)          # 有效像素计数

	# 有效像素数 (N, C, 1)
	n_valid = m.sum(dim=-1, keepdim=True).clamp_(min=2)

	# 均值
	mean_x = x.sum(dim=-1, keepdim=True) / n_valid
	mean_y = y.sum(dim=-1, keepdim=True) / n_valid

	# 去均值
	x_cent = x - mean_x
	y_cent = y - mean_y

	# 协方差与方差
	cov_xy = (x_cent * y_cent).sum(dim=-1) / (n_valid.squeeze(-1) - 1 + eps)
	var_x  = (x_cent**2).sum(dim=-1) / (n_valid.squeeze(-1) - 1 + eps)
	var_y  = (y_cent**2).sum(dim=-1) / (n_valid.squeeze(-1) - 1 + eps)

	pcc = cov_xy / (var_x.sqrt() * var_y.sqrt() + eps)

	# 去掉多余维度，尽量返回标量
	while pcc.dim() > 0 and pcc.shape[0] == 1:
		pcc = pcc.squeeze(0)
	return pcc.item()

def masked_NRMAE_torch(img1, img2, mask=None):
	x = img1 * mask
	y = img2 * mask
	m = torch.sum(mask)
	mae = torch.abs(x - y)
	mask_mae = torch.sum(mae) / m
	return mask_mae.item()

def cal_NRMAE_PSNR_SSIM_overlapped(read_dir, save_dir_overlapped, save_dir_non_overlapped, num_org):
	file_list = os.listdir(read_dir)
	file_list = natsort.natsorted(file_list)
	data_list = get_lists_3(read_dir=read_dir) if num_org == 3 else get_lists_2(read_dir=read_dir)
	SSIM_criterion = SSIM().to('cuda')
	MAE_list_overlapped = []
	SSIM_list_overlapped = []
	PSNR_list_overlapped = []
	PCC_list_overlapped = []
	MAE_list_non_overlapped = []
	SSIM_list_non_overlapped = []
	PSNR_list_non_overlapped = []
	PCC_list_non_overlapped = []
	for index in range(len(data_list[0])):
		for cat_index in range(num_org):
			if cat_index == 0: GT_list, Pred_list = [], []
			GT = tifffile.imread(data_list[1+cat_index*2][index])[:,:,0]
			GT_list.append(GT)

		overlapped_map, image_sum = generate_overlapped_map(GT_list)
		for cat_index in range(num_org):
			GT = tifffile.imread(data_list[1+cat_index*2][index])[:,:,0] #* overlapped_map
			Pred = tifffile.imread(data_list[1+cat_index*2+1][index])[:,:,0] #* overlapped_map
			GT = GT / (np.max(GT) - np.min(GT))
			Pred = Pred / (np.max(Pred) - np.min(Pred))
			
			# overlapped part
			# cal PSNR
			#PSNR_value = calculate_psnr(GT, Pred)
			GT = torch.tensor(np.float32(GT), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
			Pred = torch.tensor(np.float32(Pred), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
			overlapped_map_torch = torch.tensor(np.float32(overlapped_map), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
			# cal PCC
			PCC_value = masked_pcc_torch(GT, Pred, overlapped_map_torch)
			# cal MAE
			MAE_value = masked_NRMAE_torch(GT, Pred, overlapped_map_torch)
			# cal SSIM
			SSIM_value, _ = masked_ssim_torch(GT, Pred, mask=overlapped_map_torch)
			MAE_list_overlapped.append(MAE_value)
			SSIM_list_overlapped.append(SSIM_value)
			#PSNR_list_overlapped.append(PSNR_value)
			PCC_list_overlapped.append(PCC_value)

			# non-overlapped part
			non_overlapped_map = 1 - overlapped_map
			GT = tifffile.imread(data_list[1+cat_index*2][index])[:,:,0] #* non_overlapped_map
			Pred = tifffile.imread(data_list[1+cat_index*2+1][index])[:,:,0] #* non_overlapped_map
			GT = GT / (np.max(GT) - np.min(GT))
			Pred = Pred / (np.max(Pred) - np.min(Pred))
			# cal PSNR
			#PSNR_value = calculate_psnr(GT, Pred)
			GT = torch.tensor(np.float32(GT), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
			Pred = torch.tensor(np.float32(Pred), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
			non_overlapped_map_torch = torch.tensor(np.float32(non_overlapped_map), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
			# cal PCC
			PCC_value = masked_pcc_torch(GT, Pred, non_overlapped_map_torch)
			# cal MAE
			MAE_value = masked_NRMAE_torch(GT, Pred, non_overlapped_map_torch)
			# cal SSIM
			SSIM_value, _ = masked_ssim_torch(GT, Pred, mask=non_overlapped_map_torch)
			MAE_list_non_overlapped.append(MAE_value)
			SSIM_list_non_overlapped.append(SSIM_value)
			#PSNR_list_non_overlapped.append(PSNR_value)
			PCC_list_non_overlapped.append(PCC_value)

			print("SSIM: ", SSIM_list_overlapped[-1], SSIM_list_non_overlapped[-1], end='\n')
			print("PCC: ", PCC_list_overlapped[-1], PCC_list_non_overlapped[-1], end='\n')
			print("MAE: ", MAE_list_overlapped[-1], MAE_list_non_overlapped[-1], end='\n')


	result_df_overlapped = pd.DataFrame({
		'MAE': MAE_list_overlapped,
        'PCC': PCC_list_overlapped,
        'SSIM': SSIM_list_overlapped
    })

	result_df_non_overlapped = pd.DataFrame({
		'MAE': MAE_list_non_overlapped, 
        'PCC': PCC_list_non_overlapped,
        'SSIM': SSIM_list_non_overlapped,
    })

	result_df_overlapped.to_csv(save_dir_overlapped, index=False, float_format='%.4f')	
	result_df_non_overlapped.to_csv(save_dir_non_overlapped, index=False, float_format='%.4f')	
			

if __name__ == "__main__":
	if 1:
		read_dir = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\Real_comparison\data_20250811_real_Mito"
		save_dir = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\Real_comparison\MAE_SSIM_PSNR.csv"
		num_org = 1
		cal_NRMAE_PSNR_SSIM(read_dir=read_dir, save_dir=save_dir, num_org=num_org)
	if 0:
		# IMM Micro
		if 0:
			read_dir = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\Mito_inner_deconv_Micro\data_20250720_1"
			save_dir_overlapped = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\Mito_inner_deconv_Micro\MAE_SSIM_PSNR_overlapped.csv"
			save_dir_non_overlapped = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\Mito_inner_deconv_Micro\MAE_SSIM_PSNR_non_overlapped.csv"
		# NPCs Micro
		if 0:
			read_dir = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Micro\data_20250721"
			save_dir_overlapped = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Micro\MAE_SSIM_PSNR_overlapped.csv"
			save_dir_non_overlapped = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Micro\MAE_SSIM_PSNR_non_overlapped.csv"
		# NPCs IMM
		if 0:
			read_dir = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv\data_20250722"
			save_dir_overlapped = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv\MAE_SSIM_PSNR_overlapped.csv"
			save_dir_non_overlapped = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv\MAE_SSIM_PSNR_non_overlapped.csv"
		# Tubulin IMM NPCs
		if 0:
			read_dir = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv_Micro\data_20250720_1"
			save_dir_overlapped = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv_Micro\MAE_SSIM_PSNR_overlapped.csv"
			save_dir_non_overlapped = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv_Micro\MAE_SSIM_PSNR_non_overlapped.csv"
		num_org = 2
		cal_NRMAE_PSNR_SSIM_overlapped(read_dir=read_dir, 
								 save_dir_overlapped=save_dir_overlapped, 
								 save_dir_non_overlapped=save_dir_non_overlapped, 
								 num_org=num_org)

	
			
            
			
		
	

