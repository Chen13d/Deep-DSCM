import os, sys
import cv2
import tifffile

import copy
import joblib
from random import randint
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from loss.SSIM_loss import SSIM

def rand_lines(img_shape, num_lines, min_len=50, max_len=150, max_tries=1000):
    H, W = img_shape
    lines = []
    tries = 0
    while len(lines) < num_lines and tries < max_tries:
        x1, y1 = np.random.randint(W), np.random.randint(H)
        angle = np.random.rand() * 2 * np.pi
        length = np.random.randint(min_len, max_len + 1)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        if 0 <= x2 < W and 0 <= y2 < H:
            lines.append(((x1, y1), (x2, y2)))
        tries += 1
    return lines

def draw_lines(img_shape, lines, int_range, width=1):
    intensity = randint(*int_range)
    img = np.zeros(img_shape, dtype=np.uint8)
    for p0, p1 in lines:
        cv2.line(img, p0, p1, color=intensity, thickness=width)
    return img

def draw_ellipse_like_lines_fixed(img_shape, lines, alpha, int_range, max_thickness=8, min_thickness=2):
    """
    alpha = 0 时是椭圆，alpha = 1 时是线段，中间为过渡状态
    使用传入的 lines 位置绘图
    """
    H, W = img_shape
    img = np.zeros(img_shape, dtype=np.uint8)
    intensity = randint(*int_range)

    for p0, p1 in lines:
        x1, y1 = p0
        x2, y2 = p1

        # 计算中心点与角度
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))

        # 主轴长度
        major = int(np.hypot(dx, dy))
        minor = int(max_thickness * (1 - alpha)) + min_thickness

        if alpha < 1.0:
            # 椭圆
            cv2.ellipse(img, (cx, cy), (major // 2, minor), angle, 0, 360, intensity, -1)
        else:
            # 线段
            pt1 = (np.clip(x1, 0, W - 1), np.clip(y1, 0, H - 1))
            pt2 = (np.clip(x2, 0, W - 1), np.clip(y2, 0, H - 1))
            cv2.line(img, pt1, pt2, color=intensity, thickness=2)

    return img


def draw_ellipse_like_lines(img_shape, num_shapes, alpha, int_range, min_len=12, max_len=15, max_thickness=8, min_thickness=2):
    """
    alpha = 0 时是椭圆，alpha = 1 时是线段，中间为过渡状态
    """
    H, W = img_shape
    img = np.zeros(img_shape, dtype=np.uint8)
    intensity = randint(*int_range)

    for _ in range(num_shapes):
        cx, cy = np.random.randint(W), np.random.randint(H)
        angle = np.random.rand() * 360

        # 主轴长度
        major = np.random.randint(min_len, max_len + 1)
        # 短轴逐渐减小
        minor = int(max_thickness * (1 - alpha)) + min_thickness

        if alpha < 1.0:
            # 椭圆阶段
            cv2.ellipse(img, (cx, cy), (major // 2, minor), angle, 0, 360, intensity, -1)
        else:
            # 退化为线段
            angle_rad = np.deg2rad(angle)
            dx = int((major / 2) * np.cos(angle_rad))
            dy = int((major / 2) * np.sin(angle_rad))
            pt1 = (np.clip(cx - dx, 0, W - 1), np.clip(cy - dy, 0, H - 1))
            pt2 = (np.clip(cx + dx, 0, W - 1), np.clip(cy + dy, 0, H - 1))
            cv2.line(img, pt1, pt2, color=255, thickness=2)
    return img


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1
    dx = gaussian_filter(dx, sigma, mode="reflect") * alpha
    dy = gaussian_filter(dy, sigma, mode="reflect") * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    distorted = map_coordinates(image.astype(np.float32), indices, order=1, mode='reflect').reshape(shape)
    return distorted

def add_poisson_gaussian_noise(img, poisson_scale=30, gauss_sigma=0.04):
    vals = np.random.poisson(img * poisson_scale) / float(poisson_scale)
    gauss = np.random.normal(0, gauss_sigma, img.shape)
    noisy = vals + gauss
    return np.clip(noisy, 0, 1)


def check_existence(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        remove_list = os.listdir(dir)
        for i in range(len(remove_list)):
            remove_dir = os.path.join(dir, remove_list[i])
            os.remove(remove_dir)
import torch
def kmeans_torch(X, n_clusters, n_iters=100, n_init=10, device='cuda'):
    X = torch.tensor(X, device=device)
    X = X.to(device)
    N, D = X.shape
    best_inertia = float('inf')
    best_labels = None
    best_centers = None

    for _ in range(n_init):
        # 随机初始化质心
        indices = torch.randperm(N)[:n_clusters]
        centers = X[indices]

        for _ in range(n_iters):
            dists = torch.cdist(X, centers, p=2)
            labels = dists.argmin(dim=1)
            new_centers = torch.stack([
                X[labels == k].mean(dim=0) if (labels == k).sum() > 0 else centers[k]
                for k in range(n_clusters)
            ])
            if torch.allclose(centers, new_centers, rtol=1e-4, atol=1e-4):
                break
            centers = new_centers

        # 计算总误差（inertia）
        inertia = torch.sum(torch.min(torch.cdist(X, centers, p=2) ** 2, dim=1).values)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
            best_centers = centers.clone()

    return best_labels.cpu(), best_centers.cpu()

def make_lifetime_distribution(lifetime_list, size):
    # size - crop size
    lifetime_image = np.zeros((size, size))
    '''temp_rand = np.random.randint(2000, 3300, size=size)
    rand_mask[mask==2] = temp_rand[mask==2]'''
    ori_min = 2000
    ori_max = 3000
    for i in range(len(lifetime_list)):
        temp_rand = np.random.randint(ori_min, ori_max, size=(size, size))
        if i == 0:
            ori_min += 2000
            ori_max += 2000
        temp_rand[lifetime_list[i] == 0] = 0
        lifetime_image += temp_rand
    vector = lifetime_image.reshape(-1, 1)
    #kmeans = KMeans(n_clusters=3, random_state=42)
    #kmeans.fit(vector)
    
    with joblib.parallel_backend('loky', n_jobs=-1):  # 使用所有可用CPU核心
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        kmeans.fit(vector)
    labels = kmeans.labels_
    # kmeans by gpu
    #labels, _ = kmeans_torch(X=vector, n_clusters=4)
    # 5. 将标签重塑回图像的形状
    segmented_image = labels.reshape(lifetime_image.shape)
    threshold_image = copy.deepcopy(lifetime_image)
    threshold_image[threshold_image>0] = 1
    background_image = 1 - threshold_image
    min_max_list = []
    for j in range(4):
        temp_image = np.zeros_like(segmented_image)
        temp_image[segmented_image == j] = 1
        temp_image = temp_image * background_image
        min_max_list.append(np.mean(temp_image))
    max_sq = np.argmax(min_max_list)
    if max_sq != 0:
        segmented_image[segmented_image == max_sq] = -1
        segmented_image[segmented_image == 0] = max_sq
        segmented_image[segmented_image == -1] = 0

    return segmented_image


def make_dataset(num_levels, 
                 save_dir, 
                 num_file, 
                 sigma, 
                 image_size = 512, 
                 intensity_range_1=[128, 255], 
                 intensity_range_2=[128, 255], 
                 num_structure_range_1=[60, 80], 
                 num_structure_range_2=[60, 80], 
                 length_range_1 = [100, 200], 
                 length_range_2 = [20, 80], 
                 min_thickness=2, 
                 max_thickness=8):
    SSIM_criterion = SSIM().to(device='cuda')
    for i in tqdm(range(1, num_levels+2), desc="Simulating transition from ellipse to line"):
        sub_dir = os.path.join(save_dir, f"{sigma}_level_{i}")
        os.makedirs(sub_dir, exist_ok=True)
        
        GT_DS_dir = os.path.join(sub_dir, "GT_DS")
        Input_dir = os.path.join(sub_dir, "Input")
        GT_S_dir = os.path.join(sub_dir, "GT_S")
        denoised_dir = os.path.join(sub_dir, "denoised")
        lifetime_dir = os.path.join(sub_dir, "lifetime")
        check_existence(GT_DS_dir)
        check_existence(Input_dir)
        check_existence(GT_S_dir)
        check_existence(denoised_dir)
        check_existence(lifetime_dir)

        # 控制椭圆向线过渡
        alpha = (i - 1) / (num_levels-1)  # 从 0 到 1
        SSIM_list = []
        for index in tqdm(range(1, num_file+1)):
            #if i == 3:
            if 1:
                n_lines = np.random.randint(*num_structure_range_1)
                n_ellipses = np.random.randint(*num_structure_range_2)

                lines = rand_lines(image_size, n_lines, min_len=length_range_1[0], max_len=length_range_1[1])
                line_img = draw_lines(image_size, lines, int_range=intensity_range_1, width=min_thickness)
                
                #if i == 4:
                #    ellipse_img = rand_lines(image_size, n_lines, min_len=length_range_1[0], max_len=length_range_1[1])
                #else:
                if i < num_levels+1: 
                    ellipse_img = draw_ellipse_like_lines(image_size, n_ellipses, alpha, int_range=intensity_range_2, min_len=length_range_2[0], max_len=length_range_2[1], max_thickness=max_thickness, min_thickness=min_thickness)
                    #ellipse_img = draw_ellipse_like_lines_fixed(image_size, lines, alpha, intensity_range_2, max_thickness=max_thickness, min_thickness=min_thickness)
                    SSIM_index = SSIM_criterion(
                        torch.tensor(np.float32(line_img), dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0), 
                        torch.tensor(np.float32(ellipse_img), dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0), 
                        )
                    SSIM_list.append(SSIM_index.item())
                else:
                    lines = rand_lines(image_size, n_lines, min_len=length_range_2[0], max_len=length_range_2[1])
                    ellipse_img = draw_lines(image_size, lines, int_range=intensity_range_2, width=min_thickness)

                

                line_img = np.clip(line_img, 0, 255)
                ellipse_img = np.clip(ellipse_img, 0, 255)

                # 合并图像（线 + 椭圆/线）
                combined = line_img.astype(np.uint16) + ellipse_img.astype(np.uint16)

                # 高斯模糊模拟 PSF
                blurred = gaussian_filter(combined, sigma=sigma)

                # 模糊GT
                line_img = gaussian_filter(line_img, sigma=1)
                ellipse_img = gaussian_filter(ellipse_img, sigma=1)

                line_img_thresholded = copy.deepcopy(line_img)
                ellipse_img_thresholded = copy.deepcopy(ellipse_img)
                thresh_line = threshold_otsu(line_img_thresholded)
                thresh_ellipse = threshold_otsu(ellipse_img_thresholded)
                line_img_thresholded[line_img_thresholded < thresh_line] = 0
                line_img_thresholded[line_img_thresholded >= thresh_line] = 1
                ellipse_img_thresholded[ellipse_img_thresholded < thresh_ellipse] = 0
                ellipse_img_thresholded[ellipse_img_thresholded >= thresh_ellipse] = 1
                lifetime_list = [line_img_thresholded, ellipse_img_thresholded]
                lifetime_image = make_lifetime_distribution(lifetime_list=lifetime_list, size=image_size[0])
                # 保存图像
                line_img = np.expand_dims(line_img, axis=0)
                ellipse_img = np.expand_dims(ellipse_img, axis=0)
                stack = np.stack((line_img, ellipse_img), axis=0)
                
                tifffile.imwrite(os.path.join(GT_DS_dir, f"{index}.tif"), np.uint16(stack), imagej=True)
                tifffile.imwrite(os.path.join(GT_S_dir, f"{index}.tif"), np.uint16(combined))
                tifffile.imwrite(os.path.join(Input_dir, f"{index}.tif"), np.uint16(blurred))
                tifffile.imwrite(os.path.join(denoised_dir, f"{index}.tif"), np.uint16(blurred))
                tifffile.imwrite(os.path.join(lifetime_dir, f"{index}.tif"), np.uint16(lifetime_image))

        #if i == 3:
        #    print(np.mean(SSIM_list))

# ----------------------------
# —— 主流程
# ----------------------------

num_levels = 3
sigma = 5.0
image_size = (512, 512)
intensity_range_1 = [145, 200]
intensity_range_2 = [200, 255]
num_structure_range_1 = [60, 80]
num_structure_range_2 = [60, 80]
length_range_1 = [100, 200]
length_range_2 = [100, 200]
min_thickness=2
max_thickness=8

make_dataset(num_levels=num_levels, 
            save_dir=r'D:\CQL\codes\microscopy_decouple\data\simulated_data\train', 
            num_file=1000, 
            sigma=sigma, 
            image_size=image_size,
            intensity_range_1=intensity_range_1, 
            intensity_range_2=intensity_range_2, 
            num_structure_range_1=num_structure_range_1, 
            num_structure_range_2=num_structure_range_2, 
            length_range_1 = length_range_1, 
            length_range_2 = length_range_2, 
            min_thickness=min_thickness, 
            max_thickness=max_thickness)

make_dataset(num_levels=num_levels, 
            save_dir=r'D:\CQL\codes\microscopy_decouple\data\simulated_data\val', 
            num_file=100, 
            sigma=sigma, 
            image_size=image_size, 
            intensity_range_1=intensity_range_1, 
            intensity_range_2=intensity_range_2, 
            num_structure_range_1=num_structure_range_1, 
            num_structure_range_2=num_structure_range_2, 
            length_range_1 = length_range_1, 
            length_range_2 = length_range_2, 
            min_thickness=min_thickness, 
            max_thickness=max_thickness)


'''
for i in tqdm(range(1, 11), desc="Simulating transition from ellipse to line"):
    sub_dir = maindir + f"_level_{i}"
    os.makedirs(sub_dir, exist_ok=True)
    
    GT_DS_dir = os.path.join(sub_dir, "GT_DS")
    Input_dir = os.path.join(sub_dir, "Input")
    GT_S_dir = os.path.join(sub_dir, "GT_S")
    denoised_dir = os.path.join(sub_dir, "denoised")
    check_existence(GT_DS_dir)
    check_existence(Input_dir)
    check_existence(GT_S_dir)
    check_existence(denoised_dir)

    # 控制椭圆向线过渡
    alpha = (i - 1) / 9  # 从 0 到 1

    for index in tqdm(range(1, 1001)):
        n_lines = np.random.randint(60, 80)
        n_ellipses = np.random.randint(60, 80)

        lines = rand_lines(image_size, n_lines, min_len=100, max_len=200)
        line_img = draw_lines(image_size, lines, int_range=intensity_range, width=2)

        ellipse_img = draw_ellipse_like_lines(image_size, n_ellipses, alpha, int_range=intensity_range, min_len=20, max_len=80, max_thickness=8, min_thickness=2)

        line_img = np.clip(line_img, 0, 255)
        ellipse_img = np.clip(ellipse_img, 0, 255)

        # 合并图像（线 + 椭圆/线）
        combined = line_img.astype(np.uint16) + ellipse_img.astype(np.uint16)

        # 高斯模糊模拟 PSF
        blurred = gaussian_filter(combined, sigma=sigma)

        # 模糊GT
        line_img = gaussian_filter(line_img, sigma=1.0)
        ellipse_img = gaussian_filter(ellipse_img, sigma=1.0)

        # 保存图像
        line_img = np.expand_dims(line_img, axis=0)
        ellipse_img = np.expand_dims(ellipse_img, axis=0)
        stack = np.stack((line_img, ellipse_img), axis=0)
        tifffile.imwrite(os.path.join(GT_DS_dir, f"{index}.tif"), np.uint16(stack), imagej=True)
        tifffile.imwrite(os.path.join(GT_S_dir, f"{index}.tif"), np.uint16(combined))
        tifffile.imwrite(os.path.join(Input_dir, f"{index}.tif"), np.uint16(blurred))
        tifffile.imwrite(os.path.join(denoised_dir, f"{index}.tif"), np.uint16(blurred))
        #tifffile.imwrite(os.path.join(Input_dir, f"{index}.tif"), np.uint8(noisy * 255))


maindir = r'D:\CQL\codes\microscopy_decouple\data\simulated_data\val'
maindir = os.path.join(maindir, str(sigma))
#os.makedirs(maindir, exist_ok=True)

for i in tqdm(range(1, 11), desc="Simulating transition from ellipse to line"):
    sub_dir = maindir + f"_level_{i}"
    os.makedirs(sub_dir, exist_ok=True)
    
    GT_DS_dir = os.path.join(sub_dir, "GT_DS")
    Input_dir = os.path.join(sub_dir, "Input")
    GT_S_dir = os.path.join(sub_dir, "GT_S")
    denoised_dir = os.path.join(sub_dir, "denoised")
    check_existence(GT_DS_dir)
    check_existence(Input_dir)
    check_existence(GT_S_dir)
    check_existence(denoised_dir)

    # 控制椭圆向线过渡
    alpha = (i - 1) / 9  # 从 0 到 1

    for index in tqdm(range(1, 101)):
        n_lines = np.random.randint(60, 80)
        n_ellipses = np.random.randint(60, 80)

        lines = rand_lines(image_size, n_lines, min_len=100, max_len=200)
        line_img = draw_lines(image_size, lines, width=2, int_range=intensity_range)

        ellipse_img = draw_ellipse_like_lines(image_size, n_ellipses, alpha, int_range=intensity_range, min_len=20, max_len=80, max_thickness=8, min_thickness=2)

        line_img = np.clip(line_img, 0, 255)
        ellipse_img = np.clip(ellipse_img, 0, 255)

        # 合并图像（线 + 椭圆/线）
        combined = line_img.astype(np.uint16) + ellipse_img.astype(np.uint16)

        # 高斯模糊模拟 PSF
        blurred = gaussian_filter(combined, sigma=sigma)

        # 模糊GT
        line_img = gaussian_filter(line_img, sigma=1.0)
        ellipse_img = gaussian_filter(ellipse_img, sigma=1.0)

        # 保存图像
        line_img = np.expand_dims(line_img, axis=0)
        ellipse_img = np.expand_dims(ellipse_img, axis=0)
        stack = np.stack((line_img, ellipse_img), axis=0)
        tifffile.imwrite(os.path.join(GT_DS_dir, f"{index}.tif"), np.uint16(stack), imagej=True)
        tifffile.imwrite(os.path.join(GT_S_dir, f"{index}.tif"), np.uint16(combined))
        tifffile.imwrite(os.path.join(Input_dir, f"{index}.tif"), np.uint16(blurred))
        tifffile.imwrite(os.path.join(denoised_dir, f"{index}.tif"), np.uint16(blurred))
        #tifffile.imwrite(os.path.join(Input_dir, f"{index}.tif"), np.uint8(noisy * 255))

print("Simulation done!")'''
