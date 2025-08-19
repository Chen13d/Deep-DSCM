import os, sys
from tqdm import tqdm

from options.options import parse
from dataset.read_prepared_data_FLIM import *

cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from loss.SSIM_loss import SSIM
from loss.NRMAE import nrmae
from loss.FRC_cal import estimate_resolution_via_fft

from net.make_model import *
from dataset.gen_datasets import *


#options of .yml format in "options" folder
opt_path = 'options/Synthetic_eval_FLIM.yml'

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
    #net_main.net_G.train()
    # "old" = read data pairs, "new" = generate pseudo data pairs
    if opt['read_version'] == "real-time":
        val_loader = gen_degradation_dataloader(
            GT_tag_list=opt['category'], 
            noise_level=opt['noise_level'], 
            w0_T=6.9, 
            factor_list=opt['factor_list'], 
            STED_resolution_dict=opt['resolution'], 
            target_resolution=opt['degradation_resolution'], 
            generate_FLIM=opt['FLIM'], 
            degradation_method=opt['degradation_method'], 
            average=opt['average'], 
            read_LR=opt['read_LR'], 
            num_file_train=opt['num_file_train'], 
            num_file_val=opt['num_file_val'], 
            size=opt['size'], 
            num_workers=opt['num_workers'], 
            cwd=cwd, 
            device=device, 
            real_time=True, 
            eval_flag=True
        )
        num_val_image = opt['num_file_val']
    elif opt['read_version'] == "prepared":
        combination_name = "_".join(opt['category']) + f"_{opt['degradation_resolution']}" + f"_{opt['noise_level']}" + f"_{opt['average']}"
        if opt['FLIM']: combination_name += "_FLIM"
        read_dir_train = os.path.join(r'data\prepared_data\train', combination_name)
        read_dir_val = os.path.join(r'data\prepared_data\val', combination_name)
        #train_loader, val_loader, num_tr  ain_image = gen_prepared_dataloader(read_dir_train=read_dir_train, read_dir_val=read_dir_val, num_file_train=opt['num_train'], 
        #                                                        num_file_val=opt['num_test'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], 
        #                                                        batch_size=opt['train']['batch_size'], device=opt['device'])
        train_dataset = prepared_dataset_FLIM(read_dir=read_dir_train, num_file=opt['num_file_train'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], device=opt['device'])
        val_dataset = prepared_dataset_FLIM(read_dir=read_dir_val, num_file=opt['num_file_val'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], device=opt['device'])
        if opt['num_workers']:
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt['train']['batch_size'], num_workers=opt['num_workers'], persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt['train']['batch_size'])
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)
        num_train_image = opt['num_file_train']
        num_val_image = opt['num_file_val']
    # generate folders for validation
    toolbox.make_folders()
    Input_list = []
    GT_list = []
    GT_D_list = []
    lifetime_list = []
    sta_list = []
    # list for validation loss
    mae_list = []
    ssim_list = []
    SSIM_criterion = SSIM().to(device)
    pearson_coef_list = []
    print('======================== evaluating ========================')    
    bar = tqdm(total=num_val_image)
    # enumerate in test Dataloader 
    with torch.no_grad():
        for batch_index, data in enumerate(val_loader):
            Input, GT_DS, GT_D, _, _, lifetime, sta = data
            fake_main = net_main.feed_data(Input=torch.concatenate([Input, lifetime], dim=1), GT=GT_DS)
            loss_main, pearson_coef = net_main.validation(mask=None)
            Input = Input / (torch.max(Input) - torch.min(Input))
            fake_main = fake_main / (torch.max(fake_main) - torch.min(fake_main))
            GT_DS = GT_DS / (torch.max(GT_DS) - torch.min(GT_DS))
            # append to list for epoches to save
            Input_list.append(Input)
            GT_list.append([fake_main, GT_DS])
            lifetime_list.append(lifetime)
            #GT_D_list.append(to_cpu((GT_D*sta["Input_std"]).squeeze(0).permute(1,2,0)))
            sta_list.append(sta)
            mae_loss = nrmae(fake_main, GT_DS)
            # cal SSIM
            for i in range(fake_main.size()[1]):
                temp_fake = fake_main[:,i:i+1,:,:].detach()
                temp_GT = GT_DS[:,i:i+1,:,:].detach()
                temp_fake = temp_fake / torch.max(temp_fake)
                temp_GT = temp_GT / torch.max(temp_GT)
                SSIM_value = SSIM_criterion(temp_fake, temp_GT)
            mae_list.append(mae_loss.item())
            ssim_list.append(SSIM_value.item())
            # PCC loss
            pearson_coef_list.append(pearson_coef.item())
            bar.update(1)
        pearson_aver = np.mean(pearson_coef_list)
        pearson_coef_list.append(pearson_aver)
        # save val stack and model
        toolbox.gen_validation_images_FLIM(data_list=[Input_list, GT_list, lifetime_list, sta_list])
        toolbox.save_val_list(name="main")

        save_dir = r"C:\Users\18923\Desktop\DSRM_paper_on_submission_material\DSRM paper\synthetic_data_eval\Micro_Mito_Lyso_280_0.030_1_FLIM\data"
        check_existence(save_dir)
        save_list = toolbox.val_list
        size = opt['size']
        for index in range(len(save_list)):
            temp_img = save_list[index]
            Input = temp_img[:size, :size]
            sorted_image = temp_img[size:2*size, :size]
            for org_index in range(len(opt['category'])):
                GT = temp_img[:size, (org_index+1)*size:(org_index+2)*size]
                pred = temp_img[size:2*size, (org_index+1)*size:(org_index+2)*size]
                tifffile.imwrite(os.path.join(save_dir, f"{index}_GT_{org_index}.tif"), np.uint8(GT))
                tifffile.imwrite(os.path.join(save_dir, f"{index}_pred_{org_index}.tif"), np.uint8(pred))
                # save GT_D for RSP RSE
                #temp_GT_D = GT_D_list[index][:,:,org_index]
                #tifffile.imwrite(os.path.join(save_dir, f"{index}_GT_D_{org_index}.tif"), np.uint16(temp_GT_D))

            tifffile.imwrite(os.path.join(save_dir, f"{index}_Input.tif"), Input)
            tifffile.imwrite(os.path.join(save_dir, f"{index}_sorted_image.tif"), sorted_image)

    bar.close()

    print(np.mean(mae_list))
    print(np.mean(ssim_list))
    print(np.mean(pearson_coef_list))


main()