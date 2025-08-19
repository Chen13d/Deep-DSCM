import numpy as np
import cv2
import matplotlib.pyplot as plt

def estimate_resolution_via_fft(img, show_plot=True):
    """
    使用傅里叶功率谱方法估算图像的空间分辨率（以像素为单位）
    
    参数:
        image_path (str): 图像路径
        show_plot (bool): 是否可视化功率谱和截断位置

    返回:
        resolution_px (float): 估算分辨率（单位：像素）
    """
    # 读取灰度图
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    log_spectrum = np.log1p(magnitude_spectrum)

    # 计算频率半径图
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)

    # 径向平均频谱
    max_r = np.max(r)
    radial_profile = np.bincount(r.ravel(), weights=log_spectrum.ravel()) / np.bincount(r.ravel())

    # 找能量下跌到某阈值处（如 1/e），作为估计的频率边界
    norm_profile = radial_profile / np.max(radial_profile)
    threshold = 1 / np.e
    resolution_px = np.argmax(norm_profile < threshold)

    if show_plot:
        plt.plot(norm_profile)
        plt.axvline(resolution_px, color='red', linestyle='--', label=f'Est. Res = {resolution_px} px')
        plt.xlabel('Spatial Frequency (radius in px⁻¹)')
        plt.ylabel('Normalized Power')
        plt.title('Radial Power Spectrum')
        plt.legend()
        plt.show()

    return resolution_px