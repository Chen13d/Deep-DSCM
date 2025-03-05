import os, sys
import argparse
import pandas as pd
from tqdm import tqdm
cwd = os.getcwd()
from options.options import parse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
if torch.cuda.is_available():        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
    print('-----------------------------Using GPU-----------------------------')
from net.make_model import *
from dataset.dataset_decouple_SR import *
from skimage.metrics import structural_similarity as ssim_
from loss.SSIM_loss import SSIM as SSIM_cri
cwd = os.getcwd()


class synthetic_dataset(Dataset):
    def __init__(self, read_dir, num_file, num_org, org_list, device):
        super(synthetic_dataset, self).__init__()
        self.read_dir = read_dir
        self.num_file = num_file
        self.num_org = num_org
        self.org_list = org_list
        self.device = device
        self.generate_read_dir()

        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.to(torch.float32))
        ])

    def generate_read_dir(self):
        self.Input_dir = os.path.join(self.read_dir, "Input")
        self.GT_DS_dir = os.path.join(self.read_dir, "GT_DS")
        self.GT_D_dir = os.path.join(self.read_dir, "GT_D")
        self.GT_S_dir = os.path.join(self.read_dir, "GT_S")

    def __len__(self):
        return self.num_file
    
    
    def map_values(self, image, new_min=0, new_max=1, min_val=None, max_val=None, percentile=100, index=0):
        if index == 0:
            # 计算指定百分位数的最小值和最大值
            min_val = torch.quantile(image, (100 - percentile) / 100)
            max_val = torch.quantile(image, percentile / 100)
            # 避免除以零的情况
            if max_val == min_val:
                raise ValueError("最大值和最小值相等，无法进行归一化。")
            
        # 将图像值缩放到新范围
        scaled = (image - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
        
        # 可选：将值限制在新范围内
        #scaled = torch.clamp(scaled, min=new_min, max=new_max)
        
        return scaled, min_val, max_val
    
    def norm_statistic(self, Input, std=None):        
        mean = torch.mean(Input).to(self.device)
        mean_zero = torch.zeros_like(mean).to(self.device)
        std = torch.std(Input).to(self.device) if std == None else std
        output = transforms.Normalize(mean_zero, std)(Input)
        return output, mean_zero, std
    
    def __getitem__(self, index):
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=int(random()>0.5))
        self.vertical_flip = transforms.RandomVerticalFlip(p=int(random()>0.5))   
        Input = self.transform(self.vertical_flip(self.horizontal_flip(Image.open(os.path.join(self.Input_dir, f"{index+1}.tif"))))).to(self.device)
        GT_DS_list = []
        GT_D_list = []
        for i in range(self.num_org):
            GT_DS_list.append(self.transform(self.vertical_flip(self.horizontal_flip(Image.open(os.path.join(self.GT_DS_dir, f"{index+1}_{self.org_list[i]}.tif"))))).to(self.device))
            #GT_D_list.append(self.transform(Image.open(os.path.join(self.GT_D_dir, f"{index+1}_{self.org_list[i]}.tif"))).to(self.device))
        #GT_S = self.transform(Image.open(os.path.join(self.GT_S_dir, f"{index+1}.tif"))).to(self.device)

        GT_DS = torch.concatenate([*GT_DS_list], dim=0)
        #Input, GT_DS = rand_crop_dual(img1=Input, img2=GT_DS, size=512)
        # normalizations
        Input, Input_mean, Input_std = self.norm_statistic(Input)
        GT_DS, GT_DS_mean, GT_DS_std = self.norm_statistic(GT_DS, Input_std)
        # generate statistic dict for validation
        statistic_dict = {
            "Input_mean":Input_mean, "Input_std":Input_std
            }
        #return Input, GT_DS, GT_D, GT_S, statistic_dict
        return Input, GT_DS, 0, 0, statistic_dict

def gen_eval_dataloader(test_dir_HR, test_dir_LR, GT_tag_list, noise_level, factor_list, num_test, size, w0):
    output_list = GT_tag_list
    denoise = "None"
    train_dir_GT_HR_list = []
    test_dir_GT_HR_list = []
    train_dir_GT_LR_list = []
    test_dir_GT_LR_list = []
    for i in range(len(GT_tag_list)):
        test_dir_GT_HR_list.append(os.path.join(test_dir_HR, GT_tag_list[i]))
        test_dir_GT_LR_list.append(os.path.join(test_dir_LR, GT_tag_list[i]))

    eval_dataset = Dataset_decouple_SR(GT_dir_list_DS=test_dir_GT_HR_list, GT_dir_list_D=test_dir_GT_LR_list, up_factor=1, train_flag=False, num_file=num_test, noise_level=noise_level, w0=w0, size=size, factor_list=factor_list, 
                        eval_flag=True, random_selection=False, crop_flag=False, flip_flag=False, device=device)
    eval_dataloader = DataLoader(dataset=eval_dataset, shuffle=False, batch_size=1)
    return eval_dataloader

def nrmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)  # 计算均方误差 (MSE)
    rmse = torch.sqrt(mse)  # 计算根均方误差 (RMSE)
    nrmse = rmse / (y_true.max() - y_true.min())  # 归一化 RMSE 得到 NRMSE
    return nrmse

def mae(file_index, org_index, y_true, y_pred, save_dir=False):
    error_map = torch.abs((y_true - y_pred)) / (y_true.max() - y_true.min())
    error_map_list = []
    #for i in range(num_org):
    error_map_cpu = to_cpu(error_map)
    error_map_list.append(error_map_cpu)
    if save_dir:
        save_dir_file = os.path.join(save_dir, '{}_{}.tif'.format((file_index+1), (org_index+1)))
        tifffile.imwrite(save_dir_file, error_map_cpu)
    mae = torch.mean(error_map)
    #mae = mae
    return mae

def evaluation(model, eval_dataloader, num_test, org_list, device='cuda', noise_level=0, save_dir=None, save_flag=True, combination_name=None): 
    data_frame = pd.DataFrame()
    l2_criterion = nn.MSELoss().to(device)
    SSIM_criterion = SSIM_cri(device=device)
    PCC_criterion = Pearson_loss().to(device)
    model.train()
    l1_loss_list = []
    Pearson_list = [[] for i in range(len(org_list))]
    PCC_list = [[] for i in range(len(org_list))]
    eval_list = []
    whole_Pearson = []
    whole_mae = []
    save_dir_error_map = save_dir.replace("raw_data", "error_map")
    save_dir_ROI = save_dir.replace("raw_data", "ROI")
    if save_flag:
        check_existence(save_dir)
        check_existence(save_dir_error_map)
        check_existence(save_dir_ROI)
    bar = tqdm(total=num_test)
    mae_list = [[] for i in range(len(org_list))]
    with torch.no_grad():
        for batch_index, data in enumerate(eval_dataloader):
            if noise_level > 0:
                Input, GT_DS, GT_D, _, sta = data
            else:
                Input, GT_DS, GT_D, _, sta = data
            Output = model(Input)
            temp_Pearson = []
            temp_mae = []
            for org_index in range(len(org_list)):
                # MAE
                #l1_loss = mae(y_true=GT_DS[0,org_index,:,:], y_pred=Output[0,org_index,:,:], file_index=batch_index, org_index=org_index, save_dir=save_dir_error_map)
                l1_loss = mae(y_true=GT_DS[0,org_index,:,:], y_pred=Output[0,org_index,:,:], file_index=batch_index, org_index=org_index)
                #l1_loss = l2_criterion(Output[:,org_index:org_index+1,:,:], GT_DS[:,org_index:org_index+1,:,:])
                mae_list[org_index].append(l1_loss.item())
                # SSIM
                '''SSIM_output = Output[:,org_index:org_index+1,:,:].detach()
                SSIM_GT = GT_DS[:,org_index:org_index+1,:,:].detach()
                temp_max = max(torch.max(SSIM_output).item(), torch.max(SSIM_GT).item())
                SSIM_output = SSIM_output / temp_max * 1
                SSIM_GT = SSIM_GT / temp_max * 1
                SSIM_loss = SSIM_criterion(SSIM_output, SSIM_GT)
                SSIM_list[org_index].append(SSIM_loss.item())'''
                # Pearson
                PCC_output = Output[:,org_index:org_index+1,:,:].detach()
                PCC_GT = GT_DS[:,org_index:org_index+1,:,:].detach()
                
                PCC_loss = PCC_criterion(PCC_output, PCC_GT)
                Pearson_list[org_index].append(PCC_loss.item())
                temp_mae.append(l1_loss.item())
                temp_Pearson.append(PCC_loss.item())

            whole_mae.append(np.mean(temp_mae))
            whole_Pearson.append(np.mean(temp_Pearson))
            
            if save_flag:
                Input = to_cpu((Input * sta['Input_std'] + sta['Input_mean']).squeeze(0).permute(1,2,0))
                Output = to_cpu((Output * sta['Input_std'] + sta['Input_mean']).squeeze(0).permute(1,2,0))
                GT_DS = to_cpu((GT_DS * sta['Input_std'] + sta['Input_mean']).squeeze(0).permute(1,2,0))
                #GT_D = to_cpu((GT_D * sta['Input_std'] + sta['Input_mean']).squeeze(0).permute(1,2,0))
                plain = np.zeros_like(Input)
                col_input = np.array((np.vstack((Input, plain)) / np.max(Input)) * np.max(GT_DS))
                Input[Input<0] = 0
                tifffile.imwrite(os.path.join(save_dir_ROI, f'{batch_index+1}_Input.tif'), np.array(Input))
                for i in range(Output.shape[-1]):
                    fake_temp = np.hstack((fake_temp, Output[:,:,i:i+1])) if i > 0 else Output[:,:,0:1]
                    GT_temp = np.hstack((GT_temp, GT_DS[:,:,i:i+1])) if i > 0 else GT_DS[:,:,0:1]
                    fake_temp[fake_temp<0] = 0
                    GT_temp[GT_temp<0] = 0
                    tifffile.imwrite(os.path.join(save_dir_ROI, f'{batch_index+1}_{org_list[i]}_GT.tif'), np.array(GT_DS[:,:,i:i+1]))
                    tifffile.imwrite(os.path.join(save_dir_ROI, f'{batch_index+1}_{org_list[i]}_Output.tif'), np.array(Output[:,:,i:i+1]))
                    #tifffile.imwrite(os.path.join(save_dir_ROI, f'{batch_index+1}_{org_list[i]}_GT_low.tif'), np.uint16(GT_D[:,:,i:i+1]))
                col_output = np.vstack((GT_temp, fake_temp))
                plot = np.hstack((col_input, col_output))
                plot[plot < 0] = 0
                tifffile.imwrite(os.path.join(save_dir, '{}.tif'.format(batch_index+1)), np.array(plot))
            bar.set_description_str(
                f'{batch_index+1} / {num_test}'
            )
            bar.update(1)
            if __name__ != "__main__":
                plt.figure(1)
                plt.imshow(plot)
                plt.show()
        data_dict = {}
        for i in range(len(org_list)):
            data_dict[f'mae_{org_list[i]}'] = mae_list[i]
        for i in range(len(org_list)):
            data_dict[f'PCC_{org_list[i]}'] = Pearson_list[i]
        data_dict['mae_{}'.format(combination_name)] = whole_mae
        data_dict['PCC_{}'.format(combination_name)] = whole_Pearson
        data_frame = pd.DataFrame(data_dict)
        #for i in range(len(org_list)):
        #    data_frame = pd.concat([data_frame, pd.DataFrame(data_dict)], ignore_index=True)
        #for i in range(len(org_list)):
        #    data_frame = pd.concat([data_frame, pd.DataFrame({'SSIM': SSIM_list[i]})], ignore_index=True)
        #print(os.path.join(save_dir.replace("\\raw_data", ""), '{}.csv'.format(noise_level)))
        data_frame.to_csv(os.path.join(save_dir.replace("\\raw_data", ""), 'MAE_PCC.csv'))
        bar.close()
    for org_index in range(len(org_list)):
        mae_list[org_index] = np.mean(mae_list[org_index])
    return mae_list

if __name__ == "__main__":
    cwd = os.getcwd()
    train_dir_LR =  os.path.join(cwd, "data\\train_LR")
    test_dir_LR = os.path.join(cwd, "data\\test_LR")
    train_dir_HR = os.path.join(cwd, "data\\train_HR")
    test_dir_HR = os.path.join(cwd, "data\\test_HR")

    org_list = ['NPCs', 'Mito_inner_deconv', 'Membrane']
    factor_list = [1, 1, 1]
    noise_level = 0
    eval_dir = os.path.join(cwd, 'data\\demo_synthetic_data\\NPCs_Mito_inner_Membrane')
    model_dir = os.path.join(cwd, "models\\NPCs_Mito_inner_Membrane.pth")
    #model_dir = r'D:\CQL\codes\microscopy_decouple\validation\DSRM_NPCs_Mitochondria_inner_Membrane_noise_level_0_Unet_re-distributed\weights\1\main_G.pth'
    model = torch.load(model_dir)
    save_dir = os.path.join(cwd, 'evaluation\\NPCs_Mito_inner_Membrane.pth')
    num_eval = 1
    eval_dataloader = gen_eval_dataloader(test_dir_HR=test_dir_HR, test_dir_LR=test_dir_LR, GT_tag_list=org_list, noise_level=noise_level, factor_list=factor_list, num_test=num_eval, size=512, w0=2.0)
    
    score_list = evaluation(model=model, eval_dataloader=eval_dataloader, num_test=num_eval, device=device, noise_level=noise_level, save_dir=save_dir, org_list=org_list)
    print("NPCs_Mito_inner_Membrane", noise_level, score_list)
