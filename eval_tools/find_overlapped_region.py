import os, sys, tifffile, copy
import pandas as pd
from skimage.metrics import structural_similarity as ssim_
# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *


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
        if i == 0: img_overlapped = np.zeros_like(img)
        img_overlapped += img_thresh
    img_overlapped[img_overlapped == 255] = 0
    img_overlapped[img_overlapped != 0] = 1
    return img_overlapped

             
        

img1 = tifffile.imread(r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv_Micro\data_20250720_1\0_GT_0.tif")
img2 = tifffile.imread(r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv_Micro\data_20250720_1\0_GT_1.tif")
img3 = tifffile.imread(r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\NPCs_Mito_inner_deconv_Micro\data_20250720_1\0_GT_2.tif")

img1_thresh = make_threshold(img1)
img2_thresh = make_threshold(img2)
img3_thresh = make_threshold(img3)

plt.figure()
plt.subplot(131)
plt.imshow(img1_thresh)
plt.subplot(132)
plt.imshow(img2_thresh)
plt.subplot(133)
plt.imshow(img3_thresh)

img_overlapped = img1_thresh + img2_thresh + img3_thresh
img_overlapped[img_overlapped == 255] = 0
img_overlapped[img_overlapped != 0] = 1

plt.figure()
plt.imshow(img_overlapped)

plt.show()


