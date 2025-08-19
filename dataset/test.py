import os, sys
import cv2
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from skimage import io, img_as_float
from skimage.restoration import rolling_ball

from skimage import img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity


def enhance_contrast(image, method='clahe', clip_limit=1.0, tile_grid_size=(8,8)):
    """
    对灰度图像进行对比度增强，并在 'stretch' 模式下保留原始区间与类型。

    参数
    ----
    image : ndarray
        输入的 2D 灰度图像，float 或 uint8。
    method : {'global', 'clahe', 'stretch'}
        'global'：全局直方图均衡化；
        'clahe'：自适应直方图均衡化（CLAHE）；
        'stretch'：对比度拉伸并保留原始区间。
    clip_limit : float
        仅在 method='clahe' 时使用，CLAHE 的 clipLimit。
    tile_grid_size : tuple
        仅在 method='clahe' 时使用，CLAHE 的 tileGridSize。
    
    返回
    ----
    enhanced : ndarray
        增强后的图像，数据类型与输入一致，数值区间与原图一致。
    """
    # 记录原始类型和区间
    orig_dtype = image.dtype
    orig_min, orig_max = image.min(), image.max()

    # 先转换为 float [0,1] 处理
    img_float = img_as_float(image)
    img_uint8 = img_as_ubyte(img_float)

    if method == 'global':
        out_uint8 = cv2.equalizeHist(img_uint8)
        enhanced = img_as_float(out_uint8)

    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        out_uint8 = clahe.apply(img_uint8)
        enhanced = img_as_float(out_uint8)

    elif method == 'stretch':
        # 计算 2%–98% 百分位
        p2, p98 = np.percentile(img_float, (2, 98))
        # 首先拉伸到 [0,1]
        stretched = rescale_intensity(img_float, in_range=(p2, p98), out_range=(0, 1))
        # 再映射回原始区间
        enhanced = stretched * (orig_max - orig_min) + orig_min

    else:
        raise ValueError("Unsupported method: choose 'global', 'clahe', or 'stretch'.")

    enhanced = np.clip(enhanced, orig_min, orig_max)
    return enhanced


def fast_rolling_ball(image, radius):
    """
    调用 skimage.optimized 版本的 rolling_ball，
    比传统 opening + ball 结构元素要快很多。
    """
    img = img_as_float(image)
    background = rolling_ball(img, radius=radius)
    corrected = img - background
    corrected[corrected < 0] = 0
    return background, corrected

read_dir = r"D:\CQL\codes\microscopy_decouple\data2\train_HR\Micro\2.tif"
img = tifffile.imread(read_dir)

#bg, processed = fast_rolling_ball(img, radius=100)
processed = enhance_contrast(image=img, clip_limit=1)

tifffile.imwrite(r"C:\Users\18923\Desktop\DSRM_paper_on_submission_material\DSRM paper\Real data\Micro_Mito_Lyso\3\1.tif", processed)

plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(processed)
plt.show()


