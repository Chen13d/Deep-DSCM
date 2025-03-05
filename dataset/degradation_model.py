import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import copy
import torch
import tifffile
from torch import nn
from utils import *
from sklearn.cluster import KMeans

class Degradation_model():
    def __init__(self, w0, noise_scale, size):
        super(Degradation_model, self).__init__()
        self.cal_psf = self.generate_cal_psf(w0_T=w0)
        self.noise_scale = noise_scale
        self.new_min = -noise_scale
        self.new_max = noise_scale
        self.size = size

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
    
    def generate_psf(self, m, N=1024, span=6, lamb=635e-9, w0=2):
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
    
    def generate_cal_psf(self, w0_T):
        N = 129
        w0 = 0.65
        span = 12
        self.psf_STED = self.generate_psf(m=0, N=N, w0=w0, span=span)
        #fwhm = calculate_fwhm(psf_STED)
        
        N = 129
        #w0 = self.w0
        span = 12
        self.psf_confocal = self.generate_psf(m=0, N=N, w0=w0_T, span=span)
        fwhm = self.calculate_fwhm(self.psf_confocal)
        print("Target fwhm: ", fwhm)
    
        # 计算OTF
        #self.psf_confocal = tifffile.imread(r"D:\CQL\codes\microscopy_decouple\visualization\degradation\temp\confocal_psf.tif")
        #self.psf_STED = tifffile.imread(r"D:\CQL\codes\microscopy_decouple\visualization\degradation\temp\STED_psf.tif")

        otf_STED = (np.fft.fftshift(np.fft.fft2(self.psf_STED)))
        otf_confocal = (np.fft.fftshift(np.fft.fft2(self.psf_confocal)))

        
        otf_cal = otf_confocal / otf_STED
        psf_cal = np.fft.fftshift(np.fft.fft2(otf_cal))
        psf_cal = np.abs(psf_cal)
        #fwhm = calculate_fwhm(psf_cal)
        self.psf_cal = psf_cal / np.sum(psf_cal)
        self.psf_cal = np.expand_dims(self.psf_cal, axis=-1)
        self.psf_cal /= np.sum(self.psf_cal)

        if 0:
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\confocal_psf.tif', self.psf_confocal)
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\STED_psf.tif', self.psf_STED)
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\cal_psf.tif', self.psf_cal)
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\confocal_otf.tif', np.abs(otf_confocal))
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\STED_otf.tif', np.abs(otf_STED))
            tifffile.imwrite(r'D:\CQL\codes\microscopy_decouple\visualization\degradation\cal_OTF.tif', (otf_cal))
            raise KeyboardInterrupt
        return self.psf_cal
    
    def map_values_pytorch(self, image, new_min=-15, new_max=15, percentile=99):
        # Flatten the image and get the 99th percentile value
        sorted_vals = image.flatten().sort().values
        max_val = sorted_vals[int(len(sorted_vals) * percentile / 100)]
        min_val = sorted_vals[int(len(sorted_vals) * (100-percentile) / 100)]
        # Rescale values to the new range
        return (image - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
    
    def map_values_numpy(self, image, new_min=-15, new_max=15, percentile=99):
        # Flatten the image and calculate the percentiles
        sorted_vals = np.sort(image.flatten())
        max_val = sorted_vals[int(len(sorted_vals) * percentile / 100)]
        min_val = sorted_vals[int(len(sorted_vals) * (100 - percentile) / 100)]
        # Rescale values to the new range
        return (image - min_val) * (new_max - new_min) / (max_val - min_val) + new_min

    def add_poisson_pytorch(self, Input, binary_map, percentile=99, intensity=1.0):
        noise = torch.ones_like(Input)
        noise = (torch.poisson(noise*intensity) / intensity - 1) 
        noise = self.map_values_pytorch(noise, new_min=self.new_min, new_max=self.new_max, percentile=percentile)
        #print(torch.max(noise), torch.min(noise))
        Input = Input + noise * binary_map
        return Input
    
    def add_poisson_numpy(self, Input, binary_map, percentile=90, intensity=200):
        noise = np.ones_like(Input)
        noise = np.random.poisson(noise * intensity) / intensity  #- 1
        noise = self.map_values_numpy(noise, new_min=self.new_min, new_max=self.new_max, percentile=percentile)
        Input = Input + noise * binary_map if binary_map else Input + noise
        return Input
    
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
    
    def degrade_resolution_numpy(self, Input):
        blurred = cv2.filter2D(Input, -1, self.cal_psf)      
        return blurred    
    
    def degrade_noise(self, Input, binary_map, version, kernel_size=6, intensity=200, percentile=90):
        #dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  
        if binary_map:
            binary_map = cv2.dilate(binary_map, kernel, iterations=1)        
            binary_map = np.expand_dims(binary_map, -1)
            binary_map = cv2.filter2D(np.float64(binary_map), -1, self.cal_psf)
            binary_map /= np.max(binary_map)
            binary_map = np.expand_dims(binary_map, -1)
        if version == "numpy":
            noised = self.add_poisson_numpy(Input=Input, intensity=intensity, binary_map=binary_map, percentile=percentile)
        elif version == "pytorch":
            noised = self.add_poisson_pytorch(Input=Input, intensity=intensity, binary_map=binary_map, percentile=percentile)
        noised[noised<0] = 0
        return noised
    def create_stack(self):
        self.stack_LR = []
        self.stack_HR = []
    def add_image(self, Input_LR, Input_HR):
        self.stack_LR.append(np.expand_dims(Input_LR, axis=-1))
        self.stack_HR.append(np.expand_dims(Input_HR, axis=-1))
    def make_lifetime_distribution(self, lifetime_list, size):
        # size - crop size
        lifetime_image = np.zeros((size, size))
        '''temp_rand = np.random.randint(2000, 3300, size=size)
        rand_mask[mask==2] = temp_rand[mask==2]'''
        ori_min = 2000
        ori_max = 3000
        for i in range(len(lifetime_list)):
            temp_rand = np.random.randint(ori_min, ori_max, size=(size, size))
            ori_min += 2000
            ori_max += 2000
            temp_rand[lifetime_list[i] == 0] = 0
            lifetime_image += temp_rand
        vector = lifetime_image.reshape(-1, 1)
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(vector)
        # 4. 获取每个像素的聚类标签
        labels = kmeans.labels_
        # 5. 将标签重塑回图像的形状
        segmented_image = labels.reshape(lifetime_image.shape)
        '''plt.figure()
        plt.subplot(121)
        plt.imshow(lifetime_image)
        plt.subplot(122)
        plt.imshow(segmented_image)
        plt.show()'''
        return segmented_image



    def images_concatenation(self):
        self.GT_DS = np.concatenate([*self.stack_HR], axis=-1)
        self.GT_D = np.concatenate([*self.stack_LR], axis=-1)
        return self.GT_DS, self.GT_D
    def generate_plain(self, size):
        self.plain_Input = np.zeros((size, size, 1))
        #print(size)
        self.plain_GT_S = np.zeros((size, size, 1))
    def combination(self, factor_list):
        for i in range(len(self.stack_HR)):
            self.plain_GT_S += factor_list[i] * self.stack_HR[i]
        return self.plain_GT_S

        
if __name__ == "__main__":
    read_dir_LR = r'D:\CQL\codes\microscopy_decouple\visualization\estimate_noise\LR_low_noise.tif'
    read_dir_HR = r'D:\CQL\codes\microscopy_decouple\visualization\estimate_noise\HR_low_noise.tif'
    save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\estimate_noise'
    LR = np.array(Image.open(read_dir_LR))
    HR = np.array(Image.open(read_dir_HR))

    deg = Degradation_model(w0=1.95, noise_scale=6)
    binary = deg.get_binary(HR)
    blurred_HR = deg.degrade_resolution_numpy(Input=HR)
    
    noised_HR = deg.degrade_noise(Input=blurred_HR, binary_map=binary, version="numpy")

    tifffile.imwrite(os.path.join(save_dir, 'LR.tif'), (np.uint8(LR)))
    tifffile.imwrite(os.path.join(save_dir, "blurred_HR.tif"), (np.uint8(blurred_HR)))
    tifffile.imwrite(os.path.join(save_dir, "noised_HR.tif"), (np.uint8(noised_HR)))

    
    