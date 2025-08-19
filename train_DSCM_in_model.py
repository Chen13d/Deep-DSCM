import os
cwd = os.getcwd()
from tqdm import tqdm
from options.options import parse

from dataset.DSCM_dataset import *
from dataset.read_prepared_data import *
from net.make_model import *

#options of .yml format in "options" folder
opt_path = 'options/train_in_model.yml'

# read options
opt = parse(opt_path=os.path.join(cwd, opt_path))

import torch
if torch.cuda.is_available():
    # set rank of GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_rank']
    device = torch.device("cuda:0")
    print('-----------------------------Using GPU-----------------------------')
else:
    device = "cpu"
    print('-----------------------------Using CPU-----------------------------')


def main():
    toolbox = ToolBox(opt=opt)    
    DSCM = DSCM_in_model(
        target_resolution=opt['degradation_resolution'], 
        STED_resolution_dict=opt['resolution'], 
        noise_level=opt['noise_level'], 
        average=opt['average'], 
        size = opt['size'], 
        factor_list=opt['factor_list'], 
        org_list=opt['category'],
        device=device
        )
    # initialize the training
    print("========initiating the training process========")
    DSCM.init_training(
        opt=opt,
        in_channels=1, 
        num_classes=len(opt['category']),
        model_name_G=opt['net_G']['model_decouple_name'], 
        model_name_D=opt['net_D']['model_name'],
        initialize=opt['net_G']['initialize'],
        weight_list=opt['net_G']['weight_decouple'],
        lr_G=opt['train']['lr_G'], 
        lr_D=opt['train']['lr_D'], 
        optimizer_name=opt['train']['optimizer'],
        scheduler_name=opt['train']['scheduler'],
        index_per_D=opt['train']['index_per_D'],
        device=device
    )
    # GAN discriminator
    if opt['net_G']['weight_decouple'][5] > 0:
        DSCM.set_train(DSCM.net_D_1)
    # generate the dataloader
    train_loader, val_loader = DSCM_dataloader(
        cwd=cwd, 
        categories=opt['category'], 
        device=device, 
        num_file_train=opt['num_file_train'], 
        num_file_val=opt['num_file_val'], 
        size=opt['size'], 
        batch_size=opt['train']['batch_size'],
        num_workers=opt['num_workers']
    )
    num_train_image = opt['num_file_train']
   
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
    run_time_list = []
    num_org = len(opt['category'])
    for epoch in range(opt['train']['epoches']):
        DSCM.net_G.train()
        print('======================== training epoch %d ========================'%(epoch+1))    
        print(f'Epoch {epoch+1}, Learning Rate: {DSCM.optim_G.param_groups[0]["lr"]}')
        bar = tqdm(total=num_train_image//opt['train']['batch_size']*opt['train']['num_iter'])
        # list for training loss 
        curr_list = [[] for _ in range(cols)]
        for iter in range(opt['train']['num_iter']):
            # enumerate in train Dataloader
            for batch_index, data in enumerate(train_loader):  
                HR_list, confocal_list = data        
                Input, GT_DS, fake_main, std = DSCM(HR_list)
                loss_main = DSCM.calculate_loss(batch_index=batch_index, stage="train")
                DSCM.update_net(loss_list=loss_main)
                # save for every last image
                if (batch_index+1) == opt['num_file_train'] and opt['train']['epoches_per_val']%(epoch+1) == 0: toolbox.save_last_train_image(Input, GT_DS, fake_main, epoch+1)
                for col in range(cols):
                    curr_list[col].append(loss_main[col].item()) if loss_main[col] != 0 else curr_list[col].append(0)
                bar.set_description_str(
                    f'=== pixel: {np.mean(curr_list[0]):.3f}, fea: {np.mean(curr_list[1]):.3f}, SSIM: {1 - (np.mean(curr_list[2])/num_org):.3f}, grad: {np.mean(curr_list[3]):.3f}, corr: {np.mean(curr_list[4]):.3f}, G: {np.mean(curr_list[5]):.3f}, D: {np.mean(curr_list[6]):.3f}'
                )
                bar.update(1)
        # update scheduler
        lr_list.append(round(DSCM.optim_G.param_groups[0]["lr"], 10))
        if opt['train']['scheduler'] != "None":
            DSCM.update_scheduler()
        # append loss to the train list
        epoch_list_train.append(epoch+1)
        for col in range(cols):
            loss_list_train[col].append(np.mean(curr_list[col]))
        # validate per "epoches_per_test"
        if (epoch+1) % opt['train']['epoches_per_val'] == 0:
            with torch.no_grad():
                #DSCM.net_G.eval()
                Input_list = []
                GT_list = []
                sorted_list = []
                std_list = []
                # list for validation loss
                curr_list = [[] for _ in range(cols)]
                pearson_coef_list = []
                # enumerate in test Dataloader 
                for batch_index, data in enumerate(val_loader):
                    HR_list, confocal_list = data        
                    Input, GT_DS, fake_main, std = DSCM(HR_list)
                    loss_main, pearson_coef = DSCM.validation(mask=None)
                    # append to list for epoches to save
                    if (epoch+1) % opt['train']['epoches_per_save'] == 0:
                        Input_list.append(Input)
                        GT_list.append([fake_main, GT_DS])
                        if opt['FLIM']: sorted_list.append(sorted_image)
                        std_list.append(std)
                    for col in range(7):
                        curr_list[col].append(loss_main[col].item()) if loss_main[col] != 0 else curr_list[col].append(0)
                    # PCC loss
                    pearson_coef_list.append(pearson_coef.item())
                # append loss to the val list
                epoch_list_val.append(epoch+1)
                for col in range(cols):
                    loss_list_val[col].append(np.mean(curr_list[col]))
                pearson_aver = np.mean(pearson_coef_list)
                pearson_list.append(pearson_aver)
                # save val stack and model
                if (epoch+1) % opt['train']['epoches_per_save'] == 0:
                    if opt['FLIM']:
                        toolbox.gen_validation_images_FLIM(data_list=[Input_list, GT_list, sorted_list, std_list])
                    else:
                        toolbox.gen_validation_images_in_model(data_list=[Input_list, GT_list, std_list])
                    toolbox.save_val_list(name="main")
                    #toolbox.save_val_list(name="{}".format(epoch+1))
                    toolbox.save_model(model=[DSCM.net_G], name=["main_G"])
                    #toolbox.save_model(model=[DSCM.net_G], name=["{}".format(epoch+1)])
                    if opt['net_G']['weight_decouple'][5] > 0:
                        toolbox.save_model(model=[DSCM.net_D_1], name=["main_D"])
                        #if opt['net_G']['weight_decouple'][4] > 0:
                        #    toolbox.save_model(model=[DSCM.net_D_2], name=["main_D"])
                    pixel_aver = np.mean(curr_list[0])
                    if epoch+1 == opt['train']['epoches_per_save']: 
                        best_score_pixel = pixel_aver
                        best_score_pear = pearson_aver
                    else:
                        if pixel_aver < best_score_pixel:
                            best_score_pixel = pixel_aver
                            toolbox.save_model(model=[DSCM.net_G], name=["best_G_pixel"])
                            toolbox.save_val_list(name="best_pixel")
                        if pearson_aver > best_score_pear:
                            best_score_pear = pearson_aver
                            toolbox.save_model(model=[DSCM.net_G], name=["best_G_pear"])
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
    #time.sleep(600)
    main()
    if 0:
        #options of .yml format in "options" folder
        opt_path = 'options/5_train_1_stage_STED2.yml'
        # read options
        opt = parse(opt_path=os.path.join(cwd, opt_path))
        main()