from utils import *
from options.options import *
import tifffile
from tqdm import tqdm
from torchvision import transforms
from skimage.util import view_as_windows
from sklearn.linear_model import LinearRegression
from copy import deepcopy

size_list = [256, 512, 1024]
zoom_list = ["z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z18", "z22", "z25", "z28", "z38", "z48", "z60", "z90", "z150", "z225"]           

convert_table = [
    [664, 332, 166.0, 'nm/pixel'],
    [320, 160, 80.0, 'nm/pixel'],
    [210, 105, 52.0, 'nm/pixel'],
    [164, 82, 41.0, 'nm/pixel'],
    [125, 62, 31.0, 'nm/pixel'],
    [105, 53, 26.0, 'nm/pixel'],
    [94, 47, 23.0, 'nm/pixel'],
    [78, 39, 20.0, 'nm/pixel'],
    [70, 35, 18.0, 'nm/pixel'],
    [63, 31, 16.0, 'nm/pixel'],
    [59, 29, 15.0, 'nm/pixel'],
    [55, 27, 14.0, 'nm/pixel'],
    [51, 25, 13.0, 'nm/pixel'],
    [47, 23, 12.0, 'nm/pixel'],
    [43, 21, 11.0, 'nm/pixel'],
    [39, 20, 10.0, 'nm/pixel'],
    [35, 18, 9.0, 'nm/pixel'],
    [31, 16, 8.0, 'nm/pixel'],
    [27, 14, 7.0, 'nm/pixel'],
    [23, 12, 6.0, 'nm/pixel'],
    [20, 10, 5.0, 'nm/pixel'],
    [16, 8, 4.0, 'nm/pixel'],
    [12, 6, 3.0, 'nm/pixel'],
    [8, 4, 2.0, 'nm/pixel'],
    [4, 3, 1.5, 'nm/pixel'],
    [1, 2, 1.0, 'nm/pixel']
]

def norm_statistic(Input, device, std=None):
    mean = torch.mean(Input).to(device)
    mean_zero = torch.zeros_like(mean).to(device)
    std = torch.std(Input).to(device) if std == None else std
    output = transforms.Normalize(mean_zero, std)(Input)
    return output, mean_zero, std

from sklearn.cluster import KMeans
from matplotlib.colors import hsv_to_rgb

device = 'cuda'

def gen_FLIM(tm_data, intensity_image, t_range=[0, 6000]):
    #print(intensity_dir, asc_dir)
    #tm_data = np.loadtxt(asc_dir)[:512, :512]
    #tm_data = resize(np.loadtxt(asc_dir)[160:864, 160:864], (1024, 1024))
    tm_data[tm_data < t_range[0]] = t_range[0]
    tm_data[tm_data > t_range[1]] = t_range[1]
    #in_data = cv2.imdecode(np.fromfile(intensity_dir, dtype=np.uint8), flags=cv2.IMREAD_COLOR).astype(np.float64)
    #in_data = np.array(Image.open(intensity_dir))
    intensity_image = np.expand_dims(intensity_image, axis=-1)
    intensity_image = np.repeat(intensity_image, axis=-1, repeats=3)
    tm_DATA = (tm_data - np.min(tm_data)) / (np.max(tm_data) - np.min(tm_data))
    in_DATA = (intensity_image - np.min(intensity_image)) / (np.max(intensity_image) - np.min(intensity_image))
    hue_channel = 2 * tm_DATA / 3  # Map normalized lifetime to [0, 2/3]
    value_channel = in_DATA  # Use normalized intensity as the value channel

    h, w = intensity_image.shape[0:2]
    # Create HSV image
    hsv_img = np.zeros((h, w, 3))
    hsv_img[:, :, 0] = hue_channel  # Hue channel
    hsv_img[:, :, 1] = 1  # Fixed saturation at 1 (max saturation)
    hsv_img[:, :, 2] = value_channel[:,:,0]  # Value channel

    # Convert HSV to RGB
    rgb_color1 = hsv_to_rgb(hsv_img)
    rgb_color2 = np.zeros_like(rgb_color1)
    rgb_color2[:,:,0] = rgb_color1[:,:,0]
    rgb_color2[:,:,1] = rgb_color1[:,:,1]
    rgb_color2[:,:,2] = rgb_color1[:,:,2]

    return np.uint16(rgb_color2*255)

def smooth_mask_edges(image, blur_ksize=(5, 5), morph_ksize=(5, 5), iterations=1, threshold=True):
    """
    平滑图像边缘毛刺，适用于二值图（如掩膜）。

    参数：
        image (ndarray): 输入图像，灰度图或二值图。
        blur_ksize (tuple): 高斯模糊核大小（奇数），如 (5, 5)。
        morph_ksize (tuple): 形态学操作核大小，如 (5, 5)。
        iterations (int): 膨胀/腐蚀次数，默认 1。
        threshold (bool): 是否进行自动二值化处理（用于灰度图）。

    返回：
        final (ndarray): 平滑后的二值图。
    """
    # 复制原图避免修改输入
    img = image.copy()

    # 若为灰度图，则先模糊再二值化
    #if threshold:
    #    img = cv2.GaussianBlur(img, blur_ksize, 0)
    #    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 创建结构元素（推荐椭圆结构更自然）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_ksize)

    # 闭运算 = 膨胀 -> 腐蚀，可平滑小缺口和毛刺
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # 可选再次模糊边缘
    final = cv2.GaussianBlur(closed, blur_ksize, 0)

    return final

def DSCM_FLIM_enhancement(read_dir, read_dir_asc, weights_dir, weights_dir_dn=None,  device='cuda', save_dir=False, resize_to_const=False, tm_min=0, tm_max=5000):
    file_list = natsort.natsorted(os.listdir(read_dir))    
    asc_list = natsort.natsorted(os.listdir(read_dir_asc))
    check_existence(save_dir)
    if weights_dir_dn: 
        model_dn = torch.load(weights_dir, weights_only=False) 
    model = torch.load(weights_dir, weights_only=False)
    Input_list = []
    output_list = []
    bar = tqdm(total=len(file_list))
    with torch.no_grad():
        for index in range(len(file_list)):
            if (file_list[index].find('.tif') != -1 or file_list[index].find('.png') != -1) and file_list[index].find('17') != -1:
                read_dir_file = os.path.join(read_dir, file_list[index])
                read_dir_ASC = os.path.join(read_dir_asc, asc_list[index])
                zoom = 0
                for i in range(len(zoom_list)):
                        if read_dir_file.find(zoom_list[i]) != -1:
                            zoom = zoom_list[i]
                #img = np.array(Image.open(read_dir_file))
                img = tifffile.imread(read_dir_file)
                tm_data = np.loadtxt(read_dir_ASC)#[:1008, :1008]
                tifffile.imwrite(os.path.join(save_dir, asc_list[index].replace(".asc", "_tm_data_raw.tif")), tm_data)
                #FLIM_image = gen_FLIM(tm_data=tm_data, intensity_image=img)
                FLIM_image = gen_FLIM(tm_data=tm_data, intensity_image=img, t_range=[tm_min, tm_max])
                vector = FLIM_image.reshape(-1, 3)
                #vector = tm_data.reshape(-1, 1)
                n_cluster = 4
                kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
                kmeans.fit(vector)
                labels = kmeans.labels_
                #segmented_image = labels.reshape(lifetime_image.shape)
                sorted_image = labels.reshape(tm_data.shape)

                thresh_image = deepcopy(img)
                thresh = threshold_otsu(thresh_image)
                thresh_image[thresh_image < thresh] = 0
                thresh_image[thresh_image >= thresh] = 1

                coloc_list = []
                for cluster_index in range(n_cluster):
                    temp_thresh = np.where(thresh_image == 0, 1, 0)
                    temp_sorted = np.where(sorted_image == cluster_index, 1, 0)
                    temp_coloc = np.sum(temp_thresh * temp_sorted) / np.sum(temp_thresh)
                    coloc_list.append(temp_coloc)
                #min_coloc_index = coloc_list.index(min(coloc_list))
                max_coloc_index = coloc_list.index(max(coloc_list))
                if  max_coloc_index != 0:
                    temp_index = max_coloc_index
                    sorted_image[sorted_image == 0] = -1
                    sorted_image[sorted_image == temp_index] = 0
                    sorted_image[sorted_image == -1] = temp_index
                    #break

                '''sorted_image[sorted_image == 3] = -1
                sorted_image[sorted_image == 1] = 3
                sorted_image[sorted_image == -1] = 1

                sorted_image[sorted_image == 3] = -1
                sorted_image[sorted_image == 2] = 3
                sorted_image[sorted_image == -1] = 2'''
                
                tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", "_thresh.tif")), thresh_image)
                #sorted_image *= np.int32(thresh_image)
                sorted_image *= 50
                sorted_image = np.uint8(sorted_image)
                #tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", "_temp.tif")), sorted_image)

                sorted_image = tifffile.imread(r"C:\Users\18923\Desktop\17_z8_intensity_image_temp_Smooth.tif")
                #sorted_image = (np.float64(sorted_image) / 50)
                '''kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                sorted_image = cv2.dilate(sorted_image, kernel, 
                        anchor=None, 
                        iterations=1, 
                        borderType=cv2.BORDER_CONSTANT, 
                        borderValue=0)'''
                
                #sorted_image = cv2.GaussianBlur(sorted_image, (5, 5), 1)
                #sorted_image = np.uint8(np.float64(sorted_image) / 50)

                '''kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                sorted_image = cv2.dilate(sorted_image, kernel, 
                        anchor=None, 
                        iterations=3, 
                        borderType=cv2.BORDER_CONSTANT, 
                        borderValue=0)

                #sorted_image = cv2.GaussianBlur(sorted_image, (5, 5), 2)
                sorted_image = smooth_mask_edges(sorted_image, morph_ksize=(3, 3), iterations=2, blur_ksize=(5, 5))'''
                
                raw_h, raw_w = img.shape
                for zoom_index in range(len(zoom_list)):
                        if zoom == zoom_list[zoom_index]:
                            for size_index in range(len(size_list)):
                                if raw_h == size_list[size_index]:
                                    #print(read_dir, convert_table[zoom_index][size_index])
                                    convert_ratio = convert_table[zoom_index][size_index]/20
                                    img = resize(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                                    img = img[int(0.05*(raw_h*convert_ratio)):int(0.95*(raw_h*convert_ratio)), int(0.05*(raw_w*convert_ratio)):int(0.95*(raw_w*convert_ratio))]
                                    sorted_image = resize(sorted_image, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                                    sorted_image = sorted_image[int(0.05*(raw_h*convert_ratio)):int(0.95*(raw_h*convert_ratio)), int(0.05*(raw_w*convert_ratio)):int(0.95*(raw_w*convert_ratio))]

                h, w = img.shape
                size = min(h, w)
                upper_limit = 1440
                if size > upper_limit:
                    num_row = (h // upper_limit) + 1
                    num_col = (w // upper_limit) + 1
                    size = upper_limit
                else:
                    upper_limit = (size // 16) * 16
                    num_row = (h // upper_limit) + 1
                    num_col = (w // upper_limit) + 1
                    size = upper_limit
                
                image_list = []
                pred_list = []
                FLIM_list = []
                row_cood_list = []
                col_cood_list = []
                for row in range(num_row):
                    if (row+1)*size < h:
                        row_cood_list.append([row*size, (row+1)*size])
                    else:
                        row_cood_list.append([h-size, h])
                for col in range(num_col):
                    if (col+1)*size < w:
                        col_cood_list.append([col*size, (col+1)*size])
                    else:
                        col_cood_list.append([w-size, w])
                for row in range(num_row):
                    image_list.append([])
                    pred_list.append([])
                    FLIM_list.append([])
                    for col in range(num_col):
                        image = img[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        sorted_Input = sorted_image[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        #temp = image / np.max(image) * 255
                        #estimate_poisson_gaussian_noise(img=temp)
                        image = torch.tensor(np.float64(image), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                        image, mean, std = norm_statistic(image, device)
                        plt.figure()
                        plt.imshow(sorted_Input)
                        plt.show()
                        sorted_Input = torch.tensor(np.float64(sorted_Input), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                        sorted_Input /= 50
                        sorted_Input, _, _  = norm_statistic(sorted_Input, device=device, std=std)
                        Input = torch.concat([image, sorted_Input], dim=1)
                        '''plt.figure()
                        plt.subplot(121)
                        plt.imshow(to_cpu(image[0,0,:,:]))
                        plt.subplot(122)
                        plt.imshow(to_cpu(sorted_Input[0,0,:,:]))
                        plt.show()'''
                        pred = model(Input)
                        image = image*std+mean
                        pred = pred*std+mean
                        image[image<0] = 0
                        pred[pred<0] = 0
                        image_list[row].append(to_cpu(image.squeeze(0).squeeze(0)))
                        pred_list[row].append(to_cpu(pred.squeeze(0).permute(1,2,0)))
                        FLIM_list[row].append(to_cpu(sorted_Input.squeeze(0).squeeze(0)))
                full_pred = np.zeros((h, w ,pred_list[0][0].shape[-1]))
                for row in range(num_row):
                    for col in range(num_col):
                        full_pred[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]] = pred_list[row][col]
                h, w = img.shape
                #h = int(h * 0.05)
                #w = int(w * 0.05)
                #img = img[h:, w:]
                #full_pred = full_pred[h:, w:, :]
                img = np.uint8(img/np.max(img)*255)
                sorted_image = np.uint8(sorted_image/np.max(sorted_image)*255)
                if resize_to_const:
                    img = resize(img, (1024, 1024))
                    full_pred = resize(full_pred, (1024,1024,full_pred.shape[-1]))
                    sorted_image = resize(sorted_image, (1024, 1024))
                    FLIM_image = resize(FLIM_image, (1024, 1024))
                    tm_data = resize(tm_data, (1024, 1024))
                if file_list[index].find('tif') != -1:
                    tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", "_Input.tif")), img)
                    tifffile.imwrite(os.path.join(save_dir, asc_list[index].replace(".asc", "_sorted_image.tif")), sorted_image)
                    tifffile.imwrite(os.path.join(save_dir, asc_list[index].replace(".asc", "_FLIM_image.tif")), FLIM_image)
                    tifffile.imwrite(os.path.join(save_dir, asc_list[index].replace(".asc", "_tm_data.tif")), tm_data)
                    full_pred = np.uint8(full_pred/np.max(full_pred)*255)
                    for i in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", f"_{i}_SR.tif")), np.uint8(full_pred[:,:,i]))
                elif file_list[index].find('png') != -1:
                    tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".png", "_Input.tif")), img)
                    tifffile.imwrite(os.path.join(save_dir, asc_list[index].replace(".asc", "_FLIM.tif")))
                    full_pred = np.uint8(full_pred/np.max(full_pred)*255)
                    for i in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".png", f"_{i}_SR.tif")), np.uint8(full_pred[:,:,i]))
                bar.set_description("{}".format(zoom))

                SR_image = np.zeros_like(img, dtype=np.uint16)
                for i in range(full_pred.shape[-1]):
                    SR_image += full_pred[:,:,i]
                tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", "_SR.tif")), SR_image)
                #SR_FLIM = gen_FLIM(tm_data=tm_data[int(0.05*(raw_h*convert_ratio)):int(0.95*(raw_h*convert_ratio)), int(0.05*(raw_w*convert_ratio)):int(0.95*(raw_w*convert_ratio))], intensity_image=SR_image, t_range=[tm_min, tm_max])
                #tifffile.imwrite(os.path.join(save_dir, file_list[index].replace(".tif", "_SR_FLIM.tif")), SR_FLIM)
            bar.update(1)

if __name__ == "__main__":
    # Mito Lyso
    if 0:
        read_dir = r'D:\CQL\codes\microscopy_decouple\visualization\FLIM_decouple\Mito_Lyso\FLIM_data\Intensity'
        read_dir_asc = r'D:\CQL\codes\microscopy_decouple\visualization\FLIM_decouple\Mito_Lyso\FLIM_data\ASC'
        weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Mito_Lyso_280_0.070_1_DSCM_FLIM_384_Unet_fea_loss_0.01_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth"
        save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\FLIM_decouple\Mito_Lyso\FLIM_results'
        DSCM_FLIM_enhancement(read_dir=read_dir, read_dir_asc=read_dir_asc, weights_dir=weights_dir, 
                              save_dir=save_dir, resize_to_const=False, tm_min=0, tm_max=4000) 
        
    if 1:
        read_dir = r'D:\CQL\codes\microscopy_decouple\visualization\FLIM_decouple\Micro_Mito_Lyso_DSCM_FLIM\data\Intensity'
        read_dir_asc = r'D:\CQL\codes\microscopy_decouple\visualization\FLIM_decouple\Micro_Mito_Lyso_DSCM_FLIM\data\ASC'
        weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Mito_Lyso_280_0.030_1_DSCM_FLIM_384_Unet_fea_loss_0.01_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches_4_genre\weights\1\main_G.pth"
        save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\FLIM_decouple\Micro_Mito_Lyso_DSCM_FLIM\FLIM_results'
        DSCM_FLIM_enhancement(read_dir=read_dir, read_dir_asc=read_dir_asc, weights_dir=weights_dir, 
                              save_dir=save_dir, resize_to_const=False, tm_min=0, tm_max=6000) 