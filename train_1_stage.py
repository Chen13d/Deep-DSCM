import os, sys
import argparse
from tqdm import tqdm

cwd = os.getcwd()
from options.options import parse
from dataset.gen_data_loader import *
from dataset.read_prepared_data import *


opt_path = 'options/5_train_1_stage_STED.yml'


# read options
opt = parse(opt_path=os.path.join(cwd, opt_path))
# set rank of GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_rank']

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('-----------------------------Using {}-----------------------------'.format(device))

from net.make_model import *
from dataset.prepare_data import *


def train():
    # define dir for prepared data
    cwd = os.getcwd()
    print("current dir:  {}".format(cwd))
    read_dir_train = os.path.join(cwd, "data\\prepared_data\\train")
    read_dir_val = os.path.join(cwd, "data\\prepared_data\\val")

    # define toolbox
    toolbox = ToolBox(opt=opt)    
    # generate model for training 
    net_main = make_model(opt, in_channels=1, num_classes=len(opt['category']), model_name_G=opt['net_G']['model_decouple_name'], model_name_D=opt['net_D']['model_name'], initialize=opt['net_G']['initialize'], mode=opt['net_G']['mode_decouple'], scheduler_name=opt['train']['scheduler'], device=device, weight_list=opt['net_G']['weight_decouple'], lr_G=opt['train']['lr_G'], lr_D=opt['train']['lr_D'])
    # generate Dataloader random selection + cropping + flipping
    # old: generate data during training; new: read prepared data
    if opt['read_version'] == "old":
        train_loader, val_loader, num_train_image = gen_data_loader(opt=opt, random_selection=True, crop_flag=True, flip_flag=True)
    elif opt['read_version'] == "new":
        combination_name = "_".join(opt['category']) + f"_{opt['noise_level']}"
        read_dir_train = os.path.join(read_dir_train, combination_name)
        read_dir_val = os.path.join(read_dir_val, combination_name)
        train_loader, val_loader, num_train_image = gen_prepared_dataloader(read_dir_train=read_dir_train, read_dir_val=read_dir_val, num_file_train=opt['num_train'], 
                                                                num_file_val=opt['num_test'], num_org=len(opt['category']), org_list=opt['category'], batch_size=opt['train']['batch_size'], 
                                                                device=opt['device'])
    # generate folders for validation
    toolbox.make_folders()
    # list for training and validation loss
    epoch_list_train = []
    epoch_list_val = []
    cols = 7
    loss_list_train = [[] for _ in range(cols)]
    loss_list_val = [[] for _ in range(cols)]
    lr_list = []
    pearson_list = []
    if opt['net_G']['weight_decouple'][5] > 0:
        net_main.net_D_1.train()
    # iter by epoches
    num_org = len(opt['category'])
    net_main.net_G.train()
    # flag to choose leaving the bar or not
    leave_flag = True if __name__ == "__main__" else False
    for epoch in range(opt['train']['epoches']):
        #print('======================== training epoch %d ========================'%(epoch+1))    
        #print(f'Epoch {epoch+1}, Learning Rate: {net_main.optim_G.param_groups[0]["lr"]}')
        bar = tqdm(total=num_train_image//opt['train']['batch_size']*opt['train']['num_iter'], leave=leave_flag)
        # list for training loss
        curr_list = [[] for _ in range(cols)]
        # enumerate in train Dataloader
        for iter in range(opt['train']['num_iter']):
            for batch_index, data in enumerate(train_loader):  
                Input, GT_DS, GT_D, GT_S, sta = data
                fake_main = net_main.feed_data(Input=Input, GT=GT_DS)
                loss_main = net_main.calculate_loss(batch_index=batch_index, mask=None, stage="train")
                net_main.update_net(loss_list=loss_main)
                for col in range(cols):
                    curr_list[col].append(loss_main[col].item()) if loss_main[col] != 0 else curr_list[col].append(0)
                bar.set_description_str(
                    f'========Epoch {epoch+1}, Learning Rate: {net_main.optim_G.param_groups[0]["lr"]}==== pixel: {np.mean(curr_list[0]):.4f}, fea: {np.mean(curr_list[1]):.4f}, SSIM: {1 - (np.mean(curr_list[2])/num_org):.4f}, grad: {np.mean(curr_list[3]):.4f}, corr: {np.mean(curr_list[4]):.4f}, G: {np.mean(curr_list[5]):.4f}, D: {np.mean(curr_list[6]):.4f}'
                )
                bar.update(1)
        # update scheduler
        lr_list.append(round(net_main.optim_G.param_groups[0]["lr"], 10))
        if opt['train']['scheduler'] != "None":
            net_main.update_scheduler()
        # append loss to the train list
        epoch_list_train.append(epoch+1)
        for col in range(cols):
            loss_list_train[col].append(np.mean(curr_list[col]))
        
        # validate per "epoches_per_test"
        if (epoch+1) % opt['train']['epoches_per_val'] == 0:
            with torch.no_grad():
                #net_main.net_G.eval()
                Input_list = []
                denoise_list = []
                sr_list = []
                GT_list = []
                mask_list = []
                sta_list = []
                curr_list = [[] for _ in range(cols)]
                pearson_coef_list = []
                # enumerate in test Dataloader 
                for batch_index, data in enumerate(val_loader):
                    Input, GT_DS, GT_D, GT_S, sta = data
                    fake_main = net_main.feed_data(Input=Input, GT=GT_DS)
                    loss_main, pearson_coef = net_main.validation(mask=None)
                    if (epoch+1) % opt['train']['epoches_per_save'] == 0:
                        Input_list.append(Input)
                        GT_list.append([fake_main, GT_DS])
                        mask_list.append(None)
                        sta_list.append(sta)
                    for col in range(7):
                        curr_list[col].append(loss_main[col].item()) if loss_main[col] != 0 else curr_list[col].append(0)
                    pearson_coef_list.append(pearson_coef.item())
                # append loss to the test list
                epoch_list_val.append(epoch+1)
                for col in range(cols):
                    loss_list_val[col].append(np.mean(curr_list[col]))
                pearson_aver = np.mean(pearson_coef_list)
                pearson_list.append(pearson_aver)
                # save val stack and model
                if (epoch+1) % opt['train']['epoches_per_save'] == 0:
                    toolbox.gen_validation_images(data_list=[Input_list, GT_list, mask_list, sta_list])
                    toolbox.save_val_list(name="main")
                    #toolbox.save_val_list(name="{}".format(epoch+1))
                    toolbox.save_model(model=[net_main.net_G], name=["main_G"])
                    #toolbox.save_model(model=[net_main.net_G], name=["{}".format(epoch+1)])
                    if opt['net_G']['weight_decouple'][5] > 0:
                        toolbox.save_model(model=[net_main.net_D_1], name=["main_D"])
                        #if opt['net_G']['weight_decouple'][4] > 0:
                        #    toolbox.save_model(model=[net_main.net_D_2], name=["main_D"])
                    pixel_aver = np.mean(curr_list[0])
                    if epoch+1 == opt['train']['epoches_per_save']: 
                        best_score_pixel = pixel_aver
                        best_score_pear = pearson_aver
                    else:
                        if pixel_aver < best_score_pixel:
                            best_score_pixel = pixel_aver
                            toolbox.save_model(model=[net_main.net_G], name=["best_G_pixel"])
                            toolbox.save_val_list(name="best_pixel")
                        if pearson_aver > best_score_pear:
                            best_score_pear = pearson_aver
                            toolbox.save_model(model=[net_main.net_G], name=["best_G_pear"])
                            toolbox.save_val_list(name="best_pear")
                    
        bar.close()
        train_list = [loss_list_train]
        val_list = [loss_list_val]
        model_name_list = ['main']
        loss_name_list = ['pixel', 'fea', 'freq', 'grad', 'corr']
        toolbox.make_loss_plots(epoch_list_train=epoch_list_train, epoch_list_val=epoch_list_val, 
                                train_list=train_list, val_list=val_list, model_name_list=model_name_list, 
                                loss_name_list=loss_name_list, lr_list=lr_list, pearson_list=pearson_list)

    return 0


if __name__ == "__main__":
    train()