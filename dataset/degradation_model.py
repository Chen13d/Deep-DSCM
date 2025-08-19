import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import copy
import torch
import tifffile
import joblib
from random import uniform
#import cupy as cp
from utils import *
from sklearn.cluster import KMeans
from skimage.filters import threshold_local


# ChatGPT

from skimage import io, img_as_float, img_as_ubyte
from skimage.filters import threshold_local

def local_threshold_segmentation(image, block_size=51, offset=0, method='mean'):
    """
    对灰度图做局部阈值分割。

    参数
    ----
    image : ndarray
        输入的 2D 灰度图（float 或 uint）。
    block_size : int
        局部窗大小，必须为奇数。影响局部统计范围。
    offset : float
        从局部统计值中减去的偏移量，调节分割敏感度。
    method : {'mean', 'gaussian'}
        选择局部统计方式：'mean' 为简单平均，'gaussian' 为加权高斯。

    返回
    ----
    binary : ndarray
        二值分割结果（bool）。
    thresh : ndarray
        每个像素的自适应阈值图。
    """
    img = img_as_float(image)
    # 计算每个像素的局部阈值
    thresh = threshold_local(img,
                             block_size=block_size,
                             method=method,
                             offset=offset)
    # 二值化
    binary = img > thresh
    return binary, thresh


class Degradation_base_model():
    def __init__(self, target_resolution, noise_level, average, STED_resolution_dict, size, factor_list):
        super(Degradation_base_model, self).__init__()
        self.target_resolution = target_resolution
        self.noise_level = noise_level
        self.average = average
        self.STED_resolution_dict = STED_resolution_dict
        self.size = size
        self.factor_list = factor_list

    def calculate_fwhm(self, psf):
        """
        Calculate the Full Width at Half Maximum (FWHM) of a 2D Point Spread Function (PSF).
        
        Parameters:
        psf (2D numpy array): The 2D PSF array.

        Returns:
        float: The FWHM value.
        """
        # Normalize the PSF
        psf = psf / np.max(psf)
        h, w = psf.shape
        psf = resize(psf*255, (1024, 1024))
        
        psf = psf / np.max(psf)
        # Find the half maximum value
        half_max = 0.5
        # Find the coordinates where the PSF is greater than half max
        indices = np.where(psf >= half_max)        
        # Get the bounding box of these coordinates
        x_min, x_max = np.min(indices[1]), np.max(indices[1])
        y_min, y_max = np.min(indices[0]), np.max(indices[0])
        
        # Calculate the width in both directions
        width_x = (x_max - x_min) * w / 1024 * 20
        width_y = (y_max - y_min) * h / 1024 * 20
        # Calculate the FWHM as the mean of the widths in x and y directions
        return width_x, width_y
    
    def generate_psf(self, m, N=36, span=12, lamb=635e-9, w0=2):
        k = 2 * np.pi / lamb
        beta = 50 * np.pi / 180
        x = np.linspace(-span, span, N)
        y = np.linspace(-span, span, N)
        [X, Y] = np.meshgrid(x, y)
        [r, theta] = cv2.cartToPolar(X, Y)
        E = np.power((r / w0), m) * np.exp(-np.power(r, 2) / np.power(w0, 2)) * np.exp(1j*beta) * np.exp(-1j * m * theta)
        I = np.real(E * np.conj(E))
        I /= np.sum(I) 
        return I
    
    def generate_cal_psf(self, w0_S, w0_T):
        N = 36
        span = 12
        self.STED_psf = self.generate_psf(m=0, N=N, w0=w0_S, span=span)
        #fwhm = calculate_fwhm(psf_STED)
        
        self.confocal_psf = self.generate_psf(m=0, N=N, w0=w0_T, span=span)
        #fwhm = self.calculate_fwhm(self.confocal_psf)
        #print("Target fwhm: ", fwhm)
    
        # 计算OTF
        STED_otf = (np.fft.fftshift(np.fft.fft2(self.STED_psf)))
        confocal_otf = (np.fft.fftshift(np.fft.fft2(self.confocal_psf)))
        
        confocal_otf[np.abs(confocal_otf) < 1e-3] = 0
        STED_otf[np.abs(STED_otf) < 1e-3] = 0
        cal_otf = (confocal_otf) / ((STED_otf) + 1e-9)
        
        cal_psf = np.fft.fftshift(np.fft.fft2(cal_otf))
        cal_psf = np.abs(cal_psf)
        #fwhm = calculate_fwhm(psf_cal)
        #tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradadtion\2.4.tif', psf_cal)
        self.cal_psf = cal_psf / np.sum(cal_psf)
        self.cal_psf = np.expand_dims(self.cal_psf, axis=-1)
        self.cal_psf /= np.sum(self.cal_psf)

        if 0:
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\confocal_psf.tif', self.confocal_psf)
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\STED_psf.tif', self.STED_psf)
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\cal_psf.tif', self.cal_psf)
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\confocal_otf.tif', np.abs(confocal_otf))
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\STED_otf.tif', np.abs(STED_otf))
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\cal_OTF.tif', (cal_otf))
            raise KeyboardInterrupt
        return self.cal_psf
    
    def find_psf_for_resolution(self, resolution, interval=0.01):
        count = 40
        prev_fwhm = None
        best_fwhm = None
        best_diff = float('inf')
        best_param = None

        while count * interval <= 15:
            w0_current = count * interval
            psf = self.generate_psf(m=0, w0=count*interval)
            
            fwhm = self.calculate_fwhm(psf)[0]
            diff = abs(fwhm - resolution)
            #print(f"FWHM = {fwhm:.4f}, Target = {resolution:.4f}, w0_S = {w0_current:.4f}")

            # 保存当前结果图像（可选）
            '''psf = psf / np.max(psf) * 255
            tifffile.imwrite(
                os.path.join(r'D:\CQL\codes\microscopy_decouple\data\prepared_data\train\temp', 
                            f'{w0_current:.4f}.tif'), 
                np.uint8(psf)
            )'''

            # 记录当前最接近 resolution 的参数
            if diff < best_diff:
                best_diff = diff
                best_fwhm = fwhm
                best_param = w0_current

            # 如果当前 fwhm 超过 resolution，并且是在增长，提前停止
            if prev_fwhm is not None:
                if fwhm > resolution and fwhm > prev_fwhm:
                    print("\nFWHM 超过目标且正在递增，提前停止搜索。")
                    break

            prev_fwhm = fwhm
            count += 1

        print(f"\n最接近目标分辨率 {resolution:.4f} 的参数为 w0_S = {best_param:.4f}，对应 FWHM = {best_fwhm:.4f}")
        return best_param

    
    def map_values_numpy(self, image, new_min=20, new_max=255, percentile=99):
        # 计算对称百分位数
        low_p = (100 - percentile) / 2
        high_p = 100 - low_p

        min_val = np.percentile(image, low_p)
        max_val = np.percentile(image, high_p)

        if max_val == min_val:
            return np.full_like(image, new_min, dtype=np.float32)

        # 转 float32
        image = image.astype(np.float32)

        # 一次性线性缩放
        scale = (new_max - new_min) / (max_val - min_val)
        mapped = (image - min_val) * scale + new_min

        # clip 限幅（比手动 mask +赋值更快）
        np.clip(mapped, new_min, new_max, out=mapped)

        return mapped
    
    def map_values_numpy1(self, image, new_min=20, new_max=255, percentile=99):
        """
        内部用 GPU 加速，输入输出都保留 numpy CPU 格式，功能不变
        """
        # 输入 numpy，转 GPU tensor
        image_gpu = torch.from_numpy(image.astype(np.float32)).to('cuda')

        # 计算对称百分位
        low_p = (100 - percentile) / 2 / 100
        high_p = 1 - low_p

        min_val = torch.quantile(image_gpu.flatten(), low_p)
        max_val = torch.quantile(image_gpu.flatten(), high_p)

        if max_val == min_val:
            return np.full_like(image, new_min, dtype=np.float32)

        scale = (new_max - new_min) / (max_val - min_val)
        mapped_gpu = (image_gpu - min_val) * scale + new_min
        mapped_gpu = torch.clamp(mapped_gpu, new_min, new_max)

        # 转回 CPU numpy
        return mapped_gpu.cpu().numpy()


    def add_poisson_pytorch(self, Input, percentile=99, intensity=1.0):
        noise = torch.ones_like(Input)
        noise = (torch.poisson(noise*intensity) / intensity - 1) 
        noise = self.map_values_pytorch(noise, new_min=self.new_min, new_max=self.new_max, percentile=percentile)
        #print(torch.max(noise), torch.min(noise))
        Input = Input + noise
        return Input
    
    # 20250429
    def add_poisson_numpy(self, Input, binary_map=None, percentile=90, poisson_gain=200):
        noise = np.ones_like(Input)
        noise = np.random.poisson(noise * poisson_gain) / poisson_gain  #- 1
        noise = self.map_values_numpy(noise, new_min=self.new_min, new_max=self.new_max, percentile=percentile)
        Input = Input + noise * binary_map if binary_map else Input + noise
        return Input
    # 20250518
    def add_poisson_gaussian_numpy(self,
                               Input,
                               poisson_gain=0.05,
                               gauss_sigma=1,
                               average=3):
        """
        Add mixed Poisson-Gaussian noise to an image.
        
        Parameters
        ----------
        Input : ndarray
            Input image (float, assumed in [0, 1] or scaled accordingly).
        poisson_gain : float
            Factor to scale image intensities before Poisson sampling.
            Higher gain → lower relative Poisson noise :contentReference[oaicite:3]{index=3}.
        gauss_sigma : float
            Standard deviation of added Gaussian noise (in same units as Input).
        percentile : float
            Percentile for remapping noise values.
        intensity : float
            Legacy parameter (unused here; kept for compatibility).
        """
        Output = np.zeros_like(Input)
        #temp_Input = self.map_values_numpy(Input, new_min=0, new_max=255, percentile=99.9)
        for i in range(average):
            # 1. Poisson noise: scale, sample, then rescale
            scaled = np.clip(Input * poisson_gain, 0, None)
            #print(np.min(scaled), np.max(scaled), np.min(Input*poisson_gain), np.max(Input*poisson_gain))
            poisson_noise = np.random.poisson(scaled) / float(poisson_gain)
            
            # 2. Gaussian noise: zero-mean, user-defined sigma
            gauss_noise = np.random.normal(loc=0.0,
                                        scale=gauss_sigma,
                                        size=Input.shape)
            
            # 3. Combine noises
            total_noise = poisson_noise + gauss_noise
            
            # 4. Remap noise distribution if desired
            #total_noise = self.map_values_numpy(total_noise,
            #                                    new_min=self.new_min,
            #                                    new_max=self.new_max,
            #                                    percentile=percentile)  # :contentReference[oaicite:6]{index=6}
            
            # 5. Apply to image
            Output += (Input+total_noise)
        Output /= average
        Output = self.map_values_numpy(Output, new_min=0, new_max=np.max(Input), percentile=99.9)
        return Output
    
    def get_binary(self, Input):
        thresh = threshold_otsu(Input)
        binary = copy.deepcopy(Input)
        binary[binary<thresh]=0
        binary[binary>=thresh]=1
        #plt.figure()
        #plt.imshow(binary)
        #plt.show()
        return np.expand_dims(binary, -1)
    def merge_binary(self, binary_list):
        binary_mask = np.zeros_like(binary_list[0])
        for i in range(len(binary_list)):
            binary_mask += binary_list[i]
        binary_mask[binary_mask>0] = 1
        return binary_mask
    
    def degrade_resolution_numpy(self, Input, psf):
        blurred = cv2.filter2D(Input, -1, psf)      
        return blurred    
    
    def degrade_noise(self, Input, version, average=1, noise_scale=0, intensity=200, percentile=90):
        if version == "numpy":
            #noised = self.add_poisson_numpy(Input=Input, binary_map=binary_map, percentile=percentile)
            noised = self.add_poisson_gaussian_numpy(Input=Input, poisson_gain=noise_scale, average=average)
        elif version == "pytorch":
            noised = self.add_poisson_pytorch(Input=Input, intensity=intensity, percentile=percentile)
        noised[noised<0] = 0
        return noised
    def create_stack(self):
        self.stack_LR = []
        self.stack_HR = []
    def add_image(self, Input_LR, Input_HR):
        self.stack_LR.append(np.expand_dims(Input_LR, axis=-1))
        self.stack_HR.append(np.expand_dims(Input_HR, axis=-1))
    def remove_small_objects(self, binary_img, min_size=150):
        # 连接组件标记
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        # 创建输出图像
        output = np.zeros(binary_img.shape, dtype=np.uint8)
        
        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                output[labels == i] = 255
                
        return output

    def make_threshold(self, Input):
        thresh = threshold_otsu(Input)
        Output = copy.deepcopy(Input)
        Output[Output<thresh] = 0
        Output[Output>=thresh] = 255
        kernel_size = (5, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        Output = cv2.dilate(Output, kernel, 
                 anchor=None, 
                 iterations=3, 
                 borderType=cv2.BORDER_CONSTANT, 
                 borderValue=0)
        return Output

        
    
    def kmeans_torch(self, X, n_clusters, n_iters=100, device='cuda'):
        # X: shape (N, D)
        X = torch.tensor(X, device=device)
        X = X.to(device)
        N, D = X.shape

        # 随机初始化质心
        indices = torch.randperm(N)[:n_clusters]
        centers = X[indices]

        for _ in range(n_iters):
            # 计算所有点到质心的距离 (N, K)
            dists = torch.cdist(X, centers, p=2)

            # 分配每个点的最近质心
            labels = dists.argmin(dim=1)

            # 更新质心
            new_centers = torch.stack([X[labels == k].mean(dim=0) if (labels == k).sum() > 0 else centers[k] for k in range(n_clusters)])

            # 如果质心没有变化，则提前停止
            if torch.allclose(centers, new_centers, rtol=1e-4, atol=1e-4):
                break

            centers = new_centers

        return labels.cpu(), centers.cpu()

    def make_lifetime_distribution(self, lifetime_list, size):
        n_cluster = 4 # if len(lifetime_list) == 2 else 7
        # size - crop size
        thresh_image = np.zeros((size, size))
        '''temp_rand = np.random.randint(2000, 3300, size=size)
        rand_mask[mask==2] = temp_rand[mask==2]'''
        ori_min = 2000
        ori_max = 3000
        for i in range(len(lifetime_list)):
            temp_rand = np.random.randint(ori_min, ori_max, size=(size, size))
            ori_min += 2000
            ori_max += 2000
            temp_rand[lifetime_list[i] == 0] = 0
            if i < 2: 
                thresh_image += temp_rand
            else:
                temp_rand[lifetime_list[0] != 0] = 0
                thresh_image += temp_rand

        #plt.figure()
        #plt.imshow(thresh_image)
        #plt.show()
        vector = thresh_image.reshape(-1, 1)
        '''kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(vector)'''
        #labels, _ = self.kmeans_torch(X=vector, n_clusters=n_cluster)
        with joblib.parallel_backend('loky', n_jobs=-1):  # 使用所有可用CPU核心
            kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init='auto')
            kmeans.fit(vector)
        # 4. 获取每个像素的聚类标签
        labels = kmeans.labels_
        # 5. 将标签重塑回图像的形状
        sorted_image = labels.reshape(thresh_image.shape)
        for index in range(n_cluster):
            temp_thresh = np.where(thresh_image == 0, 1, 0)
            temp_seg = np.where(sorted_image == index, 1, 0)
            temp_coloc = np.sum(temp_thresh * temp_seg)
            if temp_coloc > 1e-3 and index != 0:
                temp_index = index
                sorted_image[sorted_image == 0] = -1
                sorted_image[sorted_image == temp_index] = 0
                sorted_image[sorted_image == -1] = temp_index
                break
        return sorted_image

    def images_concatenation(self):
        self.GT_DS = np.concatenate([*self.stack_HR], axis=-1)
        self.GT_D = np.concatenate([*self.stack_LR], axis=-1)
        return self.GT_DS, self.GT_D
    
    def generate_plain(self, size):
        self.plain_blurred = np.zeros((size, size, 1))
        self.plain_GT_S = np.zeros((size, size, 1))
    def composition(self, factor_list):
        for i in range(len(self.stack_HR)):
            self.plain_blurred += factor_list[i] * self.stack_LR[i]
            self.plain_GT_S += factor_list[i] * self.stack_HR[i]
        return self.plain_blurred, self.plain_GT_S
    def composition_LR(self, x):
        self.plain_blurred = np.zeros((self.size, self.size))
        for index in range(len(x)):
            self.plain_blurred += x[index]
        return self.plain_blurred
    def composition_HR(self, x):
        self.plain_GT_S = np.zeros((self.size, self.size))
        for index in range(len(x)):
            self.plain_GT_S += x[index]
        return self.plain_GT_S

        
if __name__ == "__main__":
    read_dir_LR = r'D:\CQL\codes\microscopy_decouple\visualization\estimate_noise\LR_low_noise.tif'
    read_dir_HR = r'D:\CQL\codes\microscopy_decouple\visualization\estimate_noise\HR_low_noise.tif'
    save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\estimate_noise'
    LR = np.array(Image.open(read_dir_LR))
    HR = np.array(Image.open(read_dir_HR))

    deg = Degradation_base_model(w0_T=1.95, noise_level=6)
    binary = deg.get_binary(HR)
    blurred_HR = deg.degrade_resolution_numpy(Input=HR)
    
    noised_HR = deg.degrade_noise(Input=blurred_HR, binary_map=binary, version="numpy")

    tifffile.imwrite(os.path.join(save_dir, 'LR.tif'), (np.uint8(LR)))
    tifffile.imwrite(os.path.join(save_dir, "blurred_HR.tif"), (np.uint8(blurred_HR)))
    tifffile.imwrite(os.path.join(save_dir, "noised_HR.tif"), (np.uint8(noised_HR)))

    
    