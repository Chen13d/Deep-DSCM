import os
import pandas as pd
from tqdm import tqdm
cwd = os.getcwd()
from options.options import parse
from dataset.read_prepared_data import *
from net.make_model import *
from loss.SSIM_loss import *
from loss.NRMAE import *


#options of .yml format in "options" folder
opt_path = 'options/Simulation_eval.yml'

# read options
opt = parse(opt_path=os.path.join(cwd, opt_path))
# set rank of GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_rank']

import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('-----------------------------Using GPU-----------------------------')
else:
    print('-----------------------------Using CPU-----------------------------')


def main():
    toolbox = ToolBox(opt=opt)    
    net_main = DSCM_with_dataset(opt, in_channels=1, num_classes=len(opt['category']), model_name_G=opt['net_G']['model_decouple_name'], 
                        model_name_D=opt['net_D']['model_name'], initialize=opt['net_G']['initialize'], mode=opt['net_G']['mode_decouple'], 
                        scheduler_name=opt['train']['scheduler'], device=device, weight_list=opt['net_G']['weight_decouple'], lr_G=opt['train']['lr_G'], lr_D=opt['train']['lr_D'])
    net_main.net_G = torch.load(opt['net_G']['pretrain_dir'], weights_only=False)
    net_main.net_G.train()
    # "old" = read data pairs, "new" = generate pseudo data pairs
    if opt['read_version'] == "real-time":
        # generate Dataloader random selection + cropping + flipping
        _, val_loader, num_train_image = gen_data_loader(opt=opt, random_selection=False, crop_flag=False, flip_flag=False)
    elif opt['read_version'] == "prepared":
        combination_name = f"{opt['degeneration_w0']}" + f"_level_{opt['noise_level']}"
        read_dir_train = os.path.join(r'data\simulated_data\train', combination_name)
        read_dir_val = os.path.join(r'data\simulated_data\val', combination_name)
        #_, val_loader, num_train_image = gen_prepared_dataloader(read_dir_train=read_dir_train, read_dir_val=read_dir_val, num_file_train=opt['num_train'], 
        #                                                        num_file_val=opt['num_test'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], 
        #                                                            batch_size=opt['train']['batch_size'], device=opt['device'])
        train_dataset = prepared_dataset(read_dir=read_dir_train, num_file=opt['num_train'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], device=opt['device'], crop_flag=False, flip_flag=False)
        val_dataset = prepared_dataset(read_dir=read_dir_val, num_file=opt['num_test'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], device=opt['device'], random_selection=False, crop_flag=False, flip_flag=False)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt['train']['batch_size'], num_workers=1, persistent_workers=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)
        num_train_image = opt['num_train']
        num_val_image = opt['num_test']
    # generate folders for validation
    toolbox.make_folders()
    excel_path = opt['excel_path']
    Input_list = []
    GT_list = []
    sta_list = []
    # list for validation loss
    mae_list = []
    ssim_list = []
    SSIM_criterion = SSIM().to(device)
    PCC_list = []
    print('======================== evaluating ========================')    
    bar = tqdm(total=num_val_image)
    # enumerate in test Dataloader 
    with torch.no_grad():
        for batch_index, data in enumerate(val_loader):
            Input, GT_DS, denoised, _, sta = data
            fake_main = net_main.feed_data(Input=Input, GT=GT_DS)
            loss_main, pearson_coef = net_main.validation(mask=None)
            # append to list for epoches to save
            Input_list.append(Input)
            GT_list.append([fake_main, GT_DS])
            sta_list.append(sta)
            
            #GT_DS = to_cpu(GT_DS[0,0,:,:])
            # MAE loss
            mae_loss = nrmae(fake_main, GT_DS)
            mae_list.append(mae_loss.item())
            # SSIM loss
            temp_SSIM_list = []
            for i in range(fake_main.size()[1]):
                ssim_loss = SSIM_criterion(fake_main[:,i:i+1,:,:], GT_DS[:,i:i+1,:,:])
                temp_SSIM_list.append(ssim_loss.item())
            ssim_list.append(np.mean(temp_SSIM_list))
            # PCC loss
            PCC_list.append(pearson_coef.item())
            bar.update(1)
        

    bar.close()

    # 将这些列表打包在一起，以 mae_list 为排序依据，从小到大
    sorted_items = sorted(zip(mae_list, Input_list, GT_list, sta_list), key=lambda x: x[0])

    # 解包排序后的前50项
    mae_list_sorted, Input_list_sorted, GT_list_sorted, sta_list_sorted = zip(*sorted_items)

    # 如果你需要的是列表形式（而不是 tuple）
    mae_list = list(mae_list_sorted)
    #Input_list = list(Input_list_sorted)
    #GT_list = list(GT_list_sorted)
    #sta_list = list(sta_list_sorted)

    # 保存验证图像
    toolbox.gen_validation_images_with_dataset(data_list=[Input_list, GT_list, sta_list])
    toolbox.save_val_list(name="main")

    ssim_list = sorted(ssim_list, reverse=True)#[:50]
    PCC_list = sorted(PCC_list, reverse=True)#[:50]
    print(np.mean(mae_list[:50]))
    print(np.mean(ssim_list[:50]))
    print(np.mean(PCC_list[:50]))

    df = pd.DataFrame({
        "MAE": mae_list, 
        "SSIM": ssim_list, 
        "PCC": PCC_list
    })    

    df.to_excel(os.path.join(excel_path, f"{opt['degeneration_w0']}_{opt['noise_level']}_intensity_varied.xlsx"))

main()