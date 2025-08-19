import os
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from openpyxl import Workbook
from utils import to_cpu, resize, write2Yaml


class ToolBox(nn.Module):
    """
    ToolBox for validation
    """
    def __init__(self, opt):
        super(ToolBox, self).__init__()
        self.opt = opt
    def make_folders(self):
        upper_dir = os.path.join(os.getcwd(), self.opt['validation_dir'])
        name_dir = os.path.join(upper_dir, '{}'.format(self.opt['validation_date']))
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        write2Yaml(self.opt, os.path.join(name_dir, 'option.yml'))
        self.save_dir_list = []
        for tag in self.opt['validation_list']:
            tag_dir = os.path.join(name_dir, tag)
            if not os.path.exists(tag_dir):
                os.mkdir(tag_dir)
                target_dir = os.path.join(tag_dir, '{}'.format(1))
                os.mkdir(target_dir)                
            else:
                num_folder = len(os.listdir(tag_dir))            
                target_dir = os.path.join(tag_dir, '{}'.format(num_folder)) if len(os.listdir(os.path.join(tag_dir, '{}'.format(num_folder)))) == 0 else os.path.join(tag_dir, '{}'.format(num_folder+1))           
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir) 
            self.save_dir_list.append(target_dir)
        return self.save_dir_list
    def save_model(self, model, name):      
        for i in range(len(name)):
            save_dir = os.path.join(self.save_dir_list[1], f'{name[i]}.pth')
            torch.save(model[i], save_dir)

    def reverse_map_values(self, image, min_val, max_val, new_min=0, new_max=255):
        image = (image - new_min) * (max_val - min_val) / (new_max - new_min) + min_val
        return image

    def gen_validation_images_in_model(self, data_list, epoch=None, batch_index=None):
        self.val_list = []
        Input_list, decouple_list, std = data_list
        for i in range(len(data_list[0])):
            Input = to_cpu((Input_list[i]*std[i]).squeeze(0).permute(1,2,0))            
            fake_main = to_cpu((decouple_list[i][0]*std[i]).squeeze(0).permute(1,2,0))
            GT_main = to_cpu((decouple_list[i][1]*std[i]).squeeze(0).permute(1,2,0))           
            #Input = to_cpu(self.reverse_map_values(Input_list[i], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            #fake_main = to_cpu(self.reverse_map_values(decouple_list[i][0], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            #GT_main = to_cpu(self.reverse_map_values(decouple_list[i][1], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            fake_main[fake_main<0] = 0            
            if self.opt['up_factor'] != 1: Input = resize(Input, (self.opt['size'], self.opt['size']))
            plain = np.zeros_like(Input)
            plot = np.vstack((Input, plain))
            for i in range(fake_main.shape[-1]):
                fake_temp_DS = np.hstack((fake_temp_DS, fake_main[:,:,i:i+1])) if i > 0 else fake_main[:,:,0:1]
                GT_temp_DS = np.hstack((GT_temp_DS, GT_main[:,:,i:i+1])) if i > 0 else GT_main[:,:,0:1]
            col_temp = np.vstack((GT_temp_DS, fake_temp_DS))
            plot = np.hstack((plot, col_temp))
            plot[plot < 0] = 0
            #if epoch != None:                
            #    temp_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSRM_Microtubes_Mitochondria_noise_level_200_corr_0_Unet_test\temp"
            #    cv2.imencode('.tif', plot)[1].tofile(os.path.join(temp_dir, f"{epoch}_batch_index_{batch_index}.tif"))
            self.val_list.append(np.uint16(plot))

    def gen_validation_images_with_dataset(self, data_list, epoch=None, batch_index=None):
        self.val_list = []
        Input_list, decouple_list, sta_list = data_list
        for i in range(len(data_list[0])):
            Input = to_cpu((Input_list[i]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))            
            fake_main = to_cpu((decouple_list[i][0]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))
            GT_main = to_cpu((decouple_list[i][1]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))    
            #Input = to_cpu(self.reverse_map_values(Input_list[i], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            #fake_main = to_cpu(self.reverse_map_values(decouple_list[i][0], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            #GT_main = to_cpu(self.reverse_map_values(decouple_list[i][1], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            fake_main[fake_main<0] = 0            
            if self.opt['up_factor'] != 1: Input = resize(Input, (self.opt['size'], self.opt['size']))
            plain = np.zeros_like(Input)
            plot = np.vstack((Input, plain))
            for i in range(fake_main.shape[-1]):
                fake_temp_DS = np.hstack((fake_temp_DS, fake_main[:,:,i:i+1])) if i > 0 else fake_main[:,:,0:1]
                GT_temp_DS = np.hstack((GT_temp_DS, GT_main[:,:,i:i+1])) if i > 0 else GT_main[:,:,0:1]
            col_temp = np.vstack((GT_temp_DS, fake_temp_DS))
            plot = np.hstack((plot, col_temp))
            plot[plot < 0] = 0
            #if epoch != None:                
            #    temp_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSRM_Microtubes_Mitochondria_noise_level_200_corr_0_Unet_test\temp"
            #    cv2.imencode('.tif', plot)[1].tofile(os.path.join(temp_dir, f"{epoch}_batch_index_{batch_index}.tif"))
            self.val_list.append(np.uint16(plot))

    def gen_validation_images_with_dn(self, data_list, epoch=None, batch_index=None):
        self.val_list = []
        Input_list, decouple_list, dn_list, sta_list = data_list
        for i in range(len(data_list[0])):
            Input = to_cpu((Input_list[i]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))            
            fake_main = to_cpu((decouple_list[i][0]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))
            GT_main = to_cpu((decouple_list[i][1]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))    
            fake_dn = to_cpu((dn_list[i][0]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))       
            GT_dn = to_cpu((dn_list[i][1]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))       
            fake_main[fake_main<0] = 0            
            fake_dn[fake_dn<0] = 0
            if self.opt['up_factor'] != 1: Input = resize(Input, (self.opt['size'], self.opt['size']))
            plain = np.zeros_like(Input)
            plot = np.vstack((Input, plain))
            fake_temp_DS = fake_dn
            GT_temp_DS = GT_dn
            for i in range(fake_main.shape[-1]):
                fake_temp_DS = np.hstack((fake_temp_DS, fake_main[:,:,i:i+1]))
                GT_temp_DS = np.hstack((GT_temp_DS, GT_main[:,:,i:i+1]))
            col_temp = np.vstack((GT_temp_DS, fake_temp_DS))
            plot = np.hstack((plot, col_temp))
            plot[plot < 0] = 0
            #if epoch != None:                
            #    temp_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSRM_Microtubes_Mitochondria_noise_level_200_corr_0_Unet_test\temp"
            #    cv2.imencode('.tif', plot)[1].tofile(os.path.join(temp_dir, f"{epoch}_batch_index_{batch_index}.tif"))
            self.val_list.append(np.uint16(plot))

    def gen_validation_images_FLIM(self, data_list, epoch=None, batch_index=None):
        self.val_list = []
        Input_list, decouple_list, lifetime_list, sta_list = data_list
        for i in range(len(data_list[0])):
            Input = to_cpu((Input_list[i]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))            
            fake_main = to_cpu((decouple_list[i][0]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))
            GT_main = to_cpu((decouple_list[i][1]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))           
            lifetime = to_cpu((lifetime_list[i]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))
            lifetime = lifetime / np.max(lifetime) * 128
            #Input = to_cpu(self.reverse_map_values(Input_list[i], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            #fake_main = to_cpu(self.reverse_map_values(decouple_list[i][0], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            #GT_main = to_cpu(self.reverse_map_values(decouple_list[i][1], min_val=sta_list[i]['Input_min_val'], max_val=sta_list[i]['Input_max_val']).squeeze(0).permute(1,2,0))
            fake_main[fake_main<0] = 0
            if self.opt['up_factor'] != 1: Input = resize(Input, (self.opt['size'], self.opt['size']))
            plot = np.vstack((Input, lifetime))
            for i in range(fake_main.shape[-1]):
                fake_temp_DS = np.hstack((fake_temp_DS, fake_main[:,:,i:i+1])) if i > 0 else fake_main[:,:,0:1]
                GT_temp_DS = np.hstack((GT_temp_DS, GT_main[:,:,i:i+1])) if i > 0 else GT_main[:,:,0:1]
            col_temp = np.vstack((GT_temp_DS, fake_temp_DS))
            plot = np.hstack((plot, col_temp))
            plot[plot < 0] = 0
            #if epoch != None:
            #    temp_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSRM_Microtubes_Mitochondria_noise_level_200_corr_0_Unet_test\temp"
            #    cv2.imencode('.tif', plot)[1].tofile(os.path.join(temp_dir, f"{epoch}_batch_index_{batch_index}.tif"))
            self.val_list.append(np.uint16(plot))
   
    def save_last_train_image(self, Input, GT, Output, epoch):
        Input = to_cpu(Input)
        GT = to_cpu(GT)
        Output = to_cpu(Output)
        plot = np.hstack((Input[0,0,:,:], GT[0,0,:,:]))
        plot = np.hstack((plot, Output[0,0,:,:]))
        if GT.shape[1] > 1:
            for i in range(GT.shape[1] - 1):
                plot = np.hstack((plot, GT[0,i+1,:,:]))
                plot = np.hstack((plot, Output[0,i+1,:,:]))
        tifffile.imwrite(os.path.join(self.save_dir_list[4], '{}.tif'.format(epoch)), plot)

    def save_last_train_image_dn(self, Input, GT, Output, denoised, dn_output, epoch):
        Input = to_cpu(Input)
        GT = to_cpu(GT)
        Output = to_cpu(Output)
        denoised = to_cpu(denoised)
        dn_output = to_cpu(dn_output)
        plot = np.hstack((Input[0,0,:,:], denoised[0,0,:,:]))
        plot = np.hstack((plot, dn_output[0,0,:,:]))
        for i in range(GT.shape[1]):
            plot = np.hstack((plot, GT[0,i,:,:]))
            plot = np.hstack((plot, Output[0,i,:,:]))
        plot -= np.min(plot)
        plot = plot / np.max(plot) * 255
        tifffile.imwrite(os.path.join(self.save_dir_list[4], '{}.tif'.format(epoch)), np.uint8(plot))
    def save_val_list(self, name):
        for i in range(len(self.val_list)):
            val_data = np.expand_dims(self.val_list[i], axis=0)
            val_stack = val_data if i == 0 else np.concatenate((val_stack, val_data), 0)
        save_dir_val_list = os.path.join(self.save_dir_list[3], '{}.tif'.format(name))
        tifffile.imwrite(save_dir_val_list, np.array(val_stack))

    def make_loss_plots(self, epoch_list_train, epoch_list_val, train_list, val_list, 
                        model_name_list, loss_name_list, lr_list, pearson_list):
        #loss_denoise_train, loss_1_train, loss_2_train = train_list
        #loss_denoise_test, loss_1_test, loss_2_test = val_list        
        for i in range(len(train_list)):            
            for j in range(len(train_list[0])-2):
                plt.figure()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(epoch_list_train, train_list[i][j], label=r'train')                
                plt.plot(epoch_list_val, val_list[i][j], label=r'val')
                plt.legend()
                plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_{loss_name_list[j]}.png'))
                plt.close()
            if np.mean(train_list[i][-2]) != 0:
                plt.figure()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(epoch_list_train, train_list[i][-2], label=r'G')                
                plt.plot(epoch_list_train, train_list[i][-1], label=r'D')
                plt.legend()
                plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_GAN.png'))
                plt.close()
            else:
                plt.figure()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(epoch_list_train, train_list[i][-2], label=r'train')                
                plt.plot(epoch_list_val, val_list[i][-1], label=r'val')
                plt.legend()
                plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_degen.png'))
                plt.close()
            plt.figure()
            plt.xlabel('epoch')
            plt.ylabel('lr')
            plt.plot(epoch_list_train, lr_list, label=r'train')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_lr.png'))
            plt.close()

            plt.figure()
            plt.xlabel('epoch')
            plt.ylabel('pearson_coef')
            plt.plot(epoch_list_val, pearson_list, label=r'val')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_pearson_coef.png'))
            plt.close()
                
    def make_plots_(self, opt, epoch_list_train, epoch_list_test, loss_list_train, loss_list_test, 
                   loss_list_pixel_train, loss_list_pixel_test, loss_list_SSIM_train, loss_list_SSIM_test, 
                   loss_list_fea_train, loss_list_fea_test, loss_list_grad_train, loss_list_grad_test, 
                   loss_list_corr_train, loss_list_corr_test, loss_list_denoise_train, loss_list_denoise_test, 
                   loss_list_train_GAN_G, loss_list_train_GAN_D):
        train_loss_workbook = Workbook()
        test_loss_workbook = Workbook()
        train_loss_sheet = train_loss_workbook.active
        test_loss_sheet = test_loss_workbook.active
        for i in range(len(loss_list_train)):
            train_loss_sheet['A%d'%(i+1)] = loss_list_train[i]
            train_loss_sheet['B%d'%(i+1)] = epoch_list_train[i]
        for i in range(len(loss_list_test)):
            test_loss_sheet['A%d'%(i+1)] = loss_list_test[i]
            test_loss_sheet['B%d'%(i+1)] = epoch_list_test[i]    
        train_loss_workbook.save(os.path.join(self.save_dir_list[2], 'train_%s.xlsx'%self.opt['validation_date']))
        test_loss_workbook.save(os.path.join(self.save_dir_list[2], 'test_%s.xlsx'%self.opt['validation_date']))
        plt.figure(1)
        plt.title('loss')
        plt.xlabel('epoches')
        plt.ylabel('loss')
        plt.plot(epoch_list_train, loss_list_train, label=r'train')
        plt.plot(epoch_list_test, loss_list_test, label=r'test')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir_list[0], '%s.png'%self.opt['validation_date']))
        plt.close()

        plt.figure(2)
        plt.title('pixel_loss')
        plt.xlabel('epoches')
        plt.ylabel('loss')
        plt.plot(epoch_list_train, loss_list_pixel_train, label=r'train')
        plt.plot(epoch_list_test, loss_list_pixel_test, label=r'test')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir_list[0], '%s_pixel.png'%self.opt['validation_date']))
        plt.close()

        plt.figure(3)
        plt.title('SSIM_loss')
        plt.xlabel('epoches')
        plt.ylabel('loss')
        plt.plot(epoch_list_train, loss_list_SSIM_train, label=r'train')
        plt.plot(epoch_list_test, loss_list_SSIM_test, label=r'test')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir_list[0], '%s_SSIM.png'%self.opt['validation_date']))
        plt.close()
        if opt['mode'] == 'GAN':
            plt.figure(4)
            plt.title('GAN_loss')
            plt.xlabel('epoches')
            plt.ylabel('GAN_loss')
            plt.plot(epoch_list_train, loss_list_train_GAN_G, label=r'G')
            plt.plot(epoch_list_train, loss_list_train_GAN_D, label=r'D')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], '%s_GAN.png'%self.opt['validation_date']))
            plt.close()
        if loss_list_fea_train:
            plt.figure(5)
            plt.title('fea_loss')
            plt.xlabel('epoches')
            plt.ylabel('fea_loss')
            plt.plot(epoch_list_train, loss_list_fea_train, label=r'fea_train')
            plt.plot(epoch_list_test, loss_list_fea_test, label=r'fea_test')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], '%s_fea.png'%self.opt['validation_date']))
            plt.close()
        if loss_list_grad_train:
            plt.figure(6)
            plt.title('grad_loss')
            plt.xlabel('epoches')
            plt.ylabel('grad_loss')
            plt.plot(epoch_list_train, loss_list_grad_train, label=r'grad_train')
            plt.plot(epoch_list_test, loss_list_grad_test, label=r'grad_test')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], '%s_grad.png'%self.opt['validation_date']))
            plt.close()
        if loss_list_corr_train:
            plt.figure(7)
            plt.title('corr_loss')
            plt.xlabel('epoches')
            plt.ylabel('corr_loss')
            plt.plot(epoch_list_train, loss_list_corr_train, label=r'corr_train')
            plt.plot(epoch_list_test, loss_list_corr_test, label=r'corr_test')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], '%s_corr.png'%self.opt['validation_date']))
            plt.close()
        if loss_list_denoise_train:
            plt.figure(8)
            plt.title('denoise_loss')
            plt.xlabel('epoches')
            plt.ylabel('denoise_loss')
            plt.plot(epoch_list_train, loss_list_denoise_train, label=r'denoise_train')
            plt.plot(epoch_list_test, loss_list_denoise_test, label=r'denoise_test')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], '%s_denoise.png'%self.opt['validation_date']))
            plt.close()
    #def save_state(self, optimizer, scheduler, path):
        

