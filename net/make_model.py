import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torchvision import transforms
#from torch.cuda.amp import autocast, GradScaler
#scaler = GradScaler()

#from joblib import Parallel, delayed

from net.generate_utils import *
from net.PatchGAN import *
from net.model_tools import *
from net.unet import *
from net.Unet_tiny import Unet_tiny
from net.Swin_Unet import *
from net.DFCAN_ import *
from net.SSFIN import *
from net.RRDBNet_arch import *
from net.AWAN import *
from net.Swinir import *
from net.Unet_3D import *
from net.Unet_FLIM import *
#from net.BasicVSRPP import *
#from net.DPATISR import *
#sys.path.append('./utils')
from utils import *
from dataset.degradation_model import *
#from SSIM_loss import *
from loss.SSIM_loss import *
#from loss.decouple_loss import *
from loss.gradient_loss import *
from loss.pearson_loss import *
from loss.FFT_loss import *
from loss.GAN_loss import *


class Basemodel():
    """
    Basemodel including functions for network structures, network forward and backward process
    """
    def __init__(self):
        super(Basemodel).__init__()
    def generate_G(self, model_name, in_channels, num_classes, upscale_factor=1):        
        if model_name == 'Unet':
            net = Unet(input_dim=in_channels, num_classes=num_classes)
        elif model_name == "Unet_tiny":
            net = Unet_tiny(in_channels=in_channels, num_classes=num_classes)
        elif model_name == "Unet_tri":
            net = Unet_tri(input_dim=in_channels, num_classes=num_classes)
        elif model_name == 'DFCAN' or model_name == 'DFGAN':
            net = DFCAN(in_channels=in_channels, out_channels=num_classes, n_channels=64, scale=upscale_factor)
        elif model_name == 'SSFIN':
            net = SpatialSpectralSRNet(in_channels=in_channels, out_channels=num_classes, upscale_factor=upscale_factor)
        elif model_name == 'RRDB':
            net = RRDBNet(in_nc=in_channels, out_nc=num_classes, nf=64, nb=23, scale=1)
        elif model_name == 'Swin_Unet':
            size = self.opt['size']
            in_channels = in_channels
            num_classes = num_classes
            net = SwinTransformerSys(
                img_size=size, 
                patch_size=4, 
                in_chans=in_channels,
                num_classes=num_classes,  
                embed_dim=96, 
                depths=[2,2,2,2],
                num_heads=[3,6,12,24],
                window_size=7, 
                mlp_ratio=4.0, 
                qkv_bias=True, 
                qk_scale=None, 
                drop_rate=0.0, 
                drop_path_rate=0.02, 
                ape=False, 
                patch_norm=True, 
                use_checkpoint=False
            )
        elif model_name == 'Swinir':
            size = 512
            height = size // 1
            width = size // 1
            window_size = 8
            embed_dim = 60
            num_heads = [6, 6, 6, 6]
            depths = [6, 6, 6, 6]
            in_channels = num_classes
            net = SwinIR(upscale=1, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=depths, in_chans=in_channels, 
                    embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2, upsampler='pixelshuffledirect')
        
        elif model_name == 'Unet3D':
            net = UNet3D(in_channels=in_channels, out_channels=num_classes)
        elif model_name == 'Unet_FLIM':
            net = Unet(input_dim=in_channels+1, num_classes=num_classes)
        elif model_name == 'Unet_FLIM_att':
            net = Unet_FLIM_att(input_dim=in_channels, num_classes=num_classes)
        elif model_name == "Unet_SR":
            net = Unet(input_dim=in_channels, num_classes=1)
        elif model_name == "BasicVSRPP":
            net = BasicVSRNet(num_classes=num_classes)
        elif model_name == "DPATISR":
            net = DPATISR(num_classes=num_classes)
        return net
    def generate_D(self, model_name, in_channels):
        if model_name == 'UnetD':
            net_D = UnetD(in_channels=in_channels)
        elif model_name == "Unet_for_D":
            net_D = Unet_for_D(input_dim=in_channels, num_classes=1)
        elif model_name == "PatchGAN":
            net_D = PatchGAN_dis(input_nc=in_channels)
        return net_D
    def generate_optim(self, model, lr_G, lr_D):
        if self.optimizer_name == 'Adam':
            self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=lr_G)
            if self.weight_list[5] > 0: self.optim_D = torch.optim.Adam(self.net_D.parameters(), lr=lr_D)
    # adjust the importance among main output and auxiliary outputs in cosine manner
    def auxiliary_decay(self, epoch=0):
        return math.cos(np.pi * (epoch+1) / (self.opt['train']['epoches']))
    
    # calculate PatchGAN loss 
    def cal_PatchGAN_loss_G(self, fake, patch_size=64):
        GAN_loss_G = 0
        for p in self.net_D_1.parameters():
            p.requires_grad = False
        d_F = self.net_D_1(fake)
        T_df = torch.ones_like(d_F, device=self.device)
        h, w = fake.size()[2:4]
        num_row = h // patch_size
        num_col = w // patch_size
        for row in range(num_row):
            for col in range(num_col):
                d_F = self.net_D_1(fake[:,:,row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size])
                T_df = torch.zeros_like(d_F)
                T_df = True
                GAN_loss_G += self.GAN_criterion(d_F, T_df)
        return GAN_loss_G / 100
    def cal_PatchGAN_loss_D(self, batch_index, fake, GT, patch_size=64):
        if (batch_index+1) % self.index_per_D == 0:            
            GAN_loss_D = 0
            for p in self.net_D_1.parameters():
                p.requires_grad = True
            d_F = self.net_D_1(fake.detach())
            d_T = self.net_D_1(GT.detach())
            T_dT = torch.zeros_like(d_T, device=self.device)
            F_dF = torch.zeros_like(d_F, device=self.device)
            h, w = fake.size()[2:4]
            num_row = h // patch_size
            num_col = w // patch_size
            for row in range(num_row):
                for col in range(num_col):
                    d_F = self.net_D_1(fake[:,:,row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size].detach())
                    d_T = self.net_D_1(GT[:,:,row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size].detach())
                    F_dF = torch.zeros_like(d_F, device=self.device)
                    T_dT = torch.zeros_like(d_T, device=self.device)
                    F_dF = False
                    T_dT = True
                    GAN_loss_D += self.GAN_criterion(d_F, F_dF) + self.GAN_criterion(d_T, T_dT)
        return GAN_loss_D / 100
    # calculate GAN loss for generator
    def cal_GAN_loss_G_1(self, fake):
        for p in self.net_D_1.parameters():
            p.requires_grad = False
        e_F, d_F, _, _ = self.net_D_1(fake) 
        T_eF = torch.ones_like(e_F, device=self.device)
        T_dF = torch.ones_like(d_F, device=self.device)
        T_eF = True
        T_dF = True
        GAN_loss_G = self.GAN_criterion(e_F, T_eF) + self.GAN_criterion(d_F, T_dF)
        #GAN_loss_G = self.GAN_criterion(e_F, T_eF)
        return GAN_loss_G    
    # calculate GAN loss for the second generator if exists
    def cal_GAN_loss_G_2(self, fake):
        for p in self.net_D_2.parameters():
            p.requires_grad = False
        e_F, d_F, _, _ = self.net_D_2(fake) 
        T_eF = torch.ones_like(e_F, device=self.device)
        T_dF = torch.ones_like(d_F, device=self.device)
        T_eF = True
        T_dF = True
        GAN_loss_G = self.GAN_criterion(e_F, T_eF) + self.GAN_criterion(d_F, T_dF)
        return GAN_loss_G  
    # calculate GAN loss for discriminator
    def cal_GAN_loss_D_1(self, batch_index, fake, GT):
        if (batch_index+1) % self.index_per_D == 0:            
            for p in self.net_D_1.parameters():
                p.requires_grad = True
            e_F, d_F, _, _ = self.net_D_1(fake.detach())
            e_T, d_T, _, _ = self.net_D_1(GT.detach())
            T_eT = torch.ones_like(e_T, device=self.device)
            T_dT = torch.ones_like(d_T, device=self.device)
            F_eF = torch.zeros_like(e_F, device=self.device)
            F_dF = torch.zeros_like(d_F, device=self.device)

            T_eT = True
            T_dT = True
            F_eF = False
            F_dF = False
            GAN_loss_D = self.GAN_criterion(e_F, F_eF) + self.GAN_criterion(d_F, F_dF) + self.GAN_criterion(e_T, T_eT) + self.GAN_criterion(d_T, T_dT)
            #GAN_loss_D = self.GAN_criterion(e_F, F_eF) + self.GAN_criterion(e_T, T_eT)
        return GAN_loss_D
    # calculate GAN loss for the second discriminator if exists
    def cal_GAN_loss_D_2(self, batch_index, fake, GT):
        if (batch_index+1) % self.index_per_D == 0:            
            for p in self.net_D_2.parameters():
                p.requires_grad = True
            e_F, d_F, _, _ = self.net_D_2(fake.detach())
            e_T, d_T, _, _ = self.net_D_2(GT.detach())
            T_eT = torch.ones_like(e_T, device=self.device)
            T_dT = torch.ones_like(d_T, device=self.device)
            F_eF = torch.zeros_like(e_F, device=self.device)
            F_dF = torch.zeros_like(d_F, device=self.device)
            T_eT = True
            T_dT = True
            F_eF = False
            F_dF = False
            GAN_loss_D = self.GAN_criterion(e_F, F_eF) + self.GAN_criterion(d_F, F_dF) + self.GAN_criterion(e_T, T_eT) + self.GAN_criterion(d_T, T_dT)
        return GAN_loss_D
    # initialize the network (normal)
    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)
    # initialize the network
    def init_net(self, net, init_type='normal', init_gain=0.02):
        self.init_weights(net, init_type, gain=init_gain)
        return net


class Processing_model():
    """
    Processing model for simulating microscopy image degradation and reconstruction.
    Args:
        target_resolution (int/float): Target resolution used as baseline for degradation.
        STED_resolution_dict (dict): STED resolution settings for different channels.
        noise_level (float): Noise intensity to simulate poisson noise.
        average (int): Number of averages to simulate averaging in microscopy imaging.
        size (int): Image size H,W.
        factor_list (list): A list containing factors for each component.
        device (str): Device for computation, e.g., "cpu" or "cuda".
    """
    def __init__(self, target_resolution, STED_resolution_dict, noise_level, average, size, factor_list, device):
        super(Processing_model, self).__init__()
        self.target_resolution = target_resolution
        self.noise_level = noise_level
        self.average = average
        self.size = size
        self.factor_list = factor_list
        self.device = device
        # Instantiate the degradation model
        self.deg = Degradation_base_model(target_resolution=target_resolution, STED_resolution_dict=STED_resolution_dict, noise_level=noise_level, average=average, size=size, factor_list=factor_list)
        # Find w0_S for each component
        self.get_HR_resolution(resolution_dict=STED_resolution_dict)
    def get_HR_resolution(self, resolution_dict):
        self.w0_S_list = []
        self.psf_list = []
        self.w0_T = self.deg.find_psf_for_resolution(resolution=self.target_resolution)
        for key, value in resolution_dict.items():
            self.w0_S_org = self.deg.find_psf_for_resolution(resolution=value)
            self.w0_S_list.append(self.w0_S_org)
            self.psf_list.append(self.deg.generate_cal_psf(w0_S=self.w0_S_org, w0_T=self.w0_T))
    def map_values(self, x):
        for index in range(len(x)):
            x[index] = self.deg.map_values_numpy(x[index], new_max=255, new_min=0, percentile=99.9) * self.factor_list[index]
        return x
    def stack_images(self, x):
        for index in range(len(x)):
            if index != 0:
                temp_x = np.concatenate((temp_x, np.expand_dims(x[index], 0)), axis=0) 
            else:
                temp_x = np.expand_dims(x[index], 0)
        return temp_x
    def resolution_degeneration(self, x):
        for index in range(len(x)):
            x[index] = self.deg.degrade_resolution_numpy(np.expand_dims(x[index], axis=-1), self.psf_list[index])
        return x
    def composite_LR(self, x):
        x = self.deg.composition_LR(x)
        return x
    def composite_HR(self, x):
        x = self.deg.composition_HR(x)
        return x

    def noise_degeneration_single(self, x, noise_level, average):
        x = self.deg.degrade_noise(x, version="numpy", noise_scale=noise_level, average=average)
        return x

    def noise_degeneration_multi(self, x, noise_level, average):
        for index in range(len(x)):    
            x[index] = self.deg.degrade_noise(x[index], version="numpy", noise_scale=noise_level, average=average)
        return x
    def norm_statistic(self, Input, std=None):
        mean = torch.mean(Input).to(self.device)
        mean_zero = torch.zeros_like(mean).to(self.device)
        std = torch.std(Input).to(self.device) if std == None else std
        output = transforms.Normalize(mean_zero, std)(Input)
        return output, mean_zero, std
    # network's forward process
    def forward(self, raw_HR_images):
        with joblib.parallel_backend('loky', n_jobs=-1):
            # map to 0-255 for each component
            raw_HR_images = self.map_values(raw_HR_images)
            multi_HR = self.stack_images(raw_HR_images)
            multi_LR = self.resolution_degeneration(raw_HR_images) if self.target_resolution else raw_HR_images
            single_HR = self.composite_HR(raw_HR_images)
            single_LR = self.composite_LR(multi_LR)
            denoised = single_LR
            single_LR = self.noise_degeneration_single(single_LR, noise_level=self.noise_level, average=self.average) if self.noise_level else single_LR
            #multi_LR = self.noise_degeneration_multi(multi_LR, noise_level=self.noise_level, average=self.average) if self.noise_level else multi_LR
            multi_LR = self.stack_images(multi_LR)
            #print(single_LR.shape, multi_HR.shape, multi_LR.shape, single_HR.shape, denoised.shape)
            return single_LR, multi_HR, multi_LR, single_HR, denoised
    # convert the processing model's outputs to tensors
    def results_to_tensor(self, single_LR, multi_HR, multi_LR, single_HR, denoised):
        single_LR = torch.tensor(np.float32(single_LR), dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        multi_HR = torch.tensor(np.float32(multi_HR), dtype=torch.float32, device=self.device).unsqueeze(0)
        multi_LR = torch.tensor(np.float32(multi_LR), dtype=torch.float32, device=self.device).unsqueeze(0)
        single_HR = torch.tensor(np.float32(single_HR), dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        denoised = torch.tensor(np.float32(denoised), dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # normalization
        Input, Input_mean, Input_std = self.norm_statistic(single_LR, std=None)
        GT_DS, GT_DS_mean, GT_DS_std = self.norm_statistic(multi_HR, Input_std)
        GT_D, GT_D_mean, GT_D_std = self.norm_statistic(multi_LR, Input_std)
        GT_S, GT_S_mean, GT_S_std = self.norm_statistic(single_HR, Input_std)
        Denoised, _, _ = self.norm_statistic(denoised)
        return Input, GT_DS, GT_D, GT_S, Denoised, Input_std

class DSCM_in_model(Basemodel, nn.Module):
    """
    Proposed network for decoupling and super-resolution.
    Args:
        target_resolution (int/float): Target resolution used as baseline for degradation.
        STED_resolution_dict (dict): STED resolution settings for different channels.
        noise_level (float): Noise intensity to simulate poisson noise.
        average (int): Number of averages to simulate averaging in microscopy imaging.
        size (int): Image size H,W.
        factor_list (list): A list containing factors for each component.
        org_list (list): Containing components name. 
        device (str): Device for computation, e.g., "cpu" or "cuda".
    """
    def __init__(
            self, 
            target_resolution, 
            STED_resolution_dict, 
            noise_level, average, 
            size, 
            factor_list, 
            org_list, 
            device
            ):
        STED_resolution_dict = {k: v for k, v in STED_resolution_dict.items() if k in org_list}
        nn.Module.__init__(self)
        Basemodel.__init__(self)
        self.generate_degradation_model(target_resolution=target_resolution, STED_resolution_dict=STED_resolution_dict, noise_level=noise_level, average=average, size=size, factor_list=factor_list, device=device)
    def generate_degradation_model(self, target_resolution, STED_resolution_dict, noise_level, average, size, factor_list, device):
        self.degradation_model = Processing_model(target_resolution=target_resolution, STED_resolution_dict=STED_resolution_dict, noise_level=noise_level, average=average, size=size, factor_list=factor_list, device=device)
    def init_training(
            self, 
            opt, 
            in_channels, 
            num_classes, 
            model_name_G, 
            model_name_D, 
            initialize, 
            weight_list, 
            lr_G, 
            lr_D, 
            index_per_D, 
            optimizer_name, 
            scheduler_name, 
            device
            ):
        # get the option
        self.opt = opt
        # give the device
        self.device = device
        #generate generator
        print(f"Model name: {model_name_G}")
        self.net_G = self.generate_G(model_name=model_name_G, in_channels=in_channels, num_classes=num_classes, upscale_factor=1).to(device)
        # initialize the model before training 
        if initialize:
            self.net_G = self.init_net(self.net_G)
        # give the weight_list
        self.weight_list = weight_list
        # generate and initialize the discriminator when using GAN loss
        if self.weight_list[5] > 0: 
            self.net_D_1 = self.generate_D(model_name=model_name_D, in_channels=num_classes).to(device)
            self.net_D_1.train()
            if initialize:
                self.net_D_1 = self.init_net(self.net_D_1)
        # load pretrain models, currently the denoising model is not applied
        if opt['net_G']['pretrain_dir'] != "None":
            self.net_G = torch.load(r'{}'.format(opt['net_G']['pretrain_dir']), weights_only=False)
            for param in self.net_G.parameters():
                param.requires_grad = True
            self.net_G.train()
        if opt['net_D']['pretrain_dir_1'] != "None":
            self.net_D_1 = torch.load(r'{}'.format(opt['net_D']['pretrain_dir_1']), weights_only=False)
            for param in self.net_D_1.parameters():
                param.requires_grad = True
            self.net_D_1.train()
        # generate loss criterion
        self.pixel_criterion = nn.MSELoss().to(device)            
        self.vgg = VGGFeatureExtractor().to(device)
        self.feature_criterion = nn.MSELoss().to(device)           
        self.freq_criterion = FFTLoss().to(device)
        self.SSIM_criterion = SSIM().to(device)
        self.gradient_criterion = nn.MSELoss().to(device)        
        self.get_grad = Get_grad_std(device=device, num_classes=1, kernel_size=3, blur_kernel_size=7, blur_kernel_std=3)
        self.corr_criterion = nn.MSELoss().to(device)#Pearson_loss()
        self.pearson_criterion = Pearson_loss().to(device)
        self.GAN_criterion = GANLoss('gan', 1.0, 0.0).to(device)
        self.degen_criterion = nn.MSELoss().to(device)
        # generate optimizer
        if optimizer_name == 'Adam':
            self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(0.9, 0.999))
            if self.weight_list[5] > 0: 
                self.optim_D_1 = torch.optim.Adam(self.net_D_1.parameters(), lr=lr_D)
        # give number between G and D updates
        self.index_per_D = index_per_D
        # generate scheduler
        if scheduler_name == 'OneCycleLR':
            self.scheduler_G = OneCycleLR(self.optim_G, max_lr=lr_G, 
                total_steps=(self.opt['train']['epoches']), pct_start=0.1)        
            if self.weight_list[5] > 0: 
                self.scheduler_D_1 = OneCycleLR(self.optim_D_1, max_lr=lr_D, total_steps=(self.opt['train']['epoches']), pct_start=0.1)   
        elif scheduler_name == "CosineAnnealingLR":
            self.scheduler_G = CosineAnnealingLR(self.optim_G, T_max=self.opt['train']['epoches'], eta_min=self.opt['train']['lr_G']/100)
            if self.weight_list[5] > 0: 
                self.scheduler_D_1 = CosineAnnealingLR(self.optim_D_1, T_max=(self.opt['train']['epoches']), eta_min=self.opt['train']['lr_D']/100)
                
    def set_train(self, network):
        network.train()
    def set_eval(self, network):
        network.eval()
    # Network's forward process, includes the degradation model and neuron network
    def forward(self, x):
        Input, GT_DS, GT_D, GT_S, denoised = self.degradation_model.forward(raw_HR_images=x)
        Input, GT_DS, GT_D, GT_S, denoised, std = self.degradation_model.results_to_tensor(Input, GT_DS, GT_D, GT_S, denoised)
        self.Input, self.GT_main = (Input, GT_DS)
        self.fake_main = self.net_G(self.Input)
        return Input, self.GT_main, self.fake_main, std
    
    def calculate_loss(self, batch_index=0, stage="train", mask=None):
        pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, adv_loss = [0, 0, 0, 0, 0, 0]
        # MSE loss for pixel loss
        pixel_loss = self.pixel_criterion(self.fake_main, self.GT_main)
        # calculate PCC while validation
        pearson_coef = 0 if stage == "train" else self.pearson_criterion(self.fake_main, self.GT_main)
        # VGG-based feature loss
        if self.weight_list[1] > 0: 
            for i in range(self.fake_main.size()[1]):
                fea_fake = self.vgg(self.fake_main[:,i:i+1,:,:])
                fea_GT = self.vgg(self.GT_main[:,i:i+1,:,:])              
                feature_loss += self.feature_criterion(fea_fake, fea_GT)
        # SSIM loss
        if self.weight_list[2] > 0: 
            for i in range(self.fake_main.size()[1]):
                temp_fake = self.fake_main[:,i:i+1,:,:].detach()
                temp_GT = self.GT_main[:,i:i+1,:,:].detach()
                temp_fake = temp_fake / torch.max(temp_fake)
                temp_GT = temp_GT / torch.max(temp_GT)
                SSIM_value = self.SSIM_criterion(temp_fake, temp_GT)
                value_one = torch.ones_like(SSIM_value)
                SSIM_loss = SSIM_loss + (value_one - SSIM_value)
        # Gradient loss
        if self.weight_list[3] > 0:
            grad_fake = self.get_grad(self.fake_main[:,0:1,:,:])
            grad_real = self.get_grad(self.GT_main[:,0:1,:,:])
            grad_loss = self.gradient_criterion(grad_fake, grad_real)
        # correlation loss
        if self.weight_list[4] > 0:
            masked_GT = self.GT_main * mask
            masked_fake = self.fake_main * mask
            corr_loss = self.corr_criterion(masked_fake, masked_GT)
        # GAN loss
        if self.weight_list[5] > 0:
            if self.weight_list[4] > 0:
                mask = mask.clone()
                mask[mask == 0] = 1
                self.fake_main = self.fake_main * mask
                self.GT_main = self.GT_main * mask
            else:
                pass
            if self.opt["net_D"]["model_name"] == "PatchGAN":
                adv_loss = self.cal_PatchGAN_loss_G(fake=self.fake_main)
                dis_loss = self.cal_PatchGAN_loss_D(batch_index=batch_index, fake=self.fake_main, GT=self.GT_main) if (batch_index+1) % self.index_per_D == 0 else 0
            else:
                adv_loss = self.cal_GAN_loss_G_1(fake=self.fake_main)
                dis_loss = self.cal_GAN_loss_D_1(batch_index=batch_index, fake=self.fake_main, GT=self.GT_main) if (batch_index+1) % self.index_per_D == 0 else 0
            #if self.weight_list[4] > 0:
            #    adv_loss += self.weight_list[4] * self.cal_GAN_loss_G_2(fake=masked_fake)
            #    #dis_loss += self.cal_GAN_loss_D_2(batch_index=batch_index, fake=masked_fake, GT=masked_GT)
            #    self.dis_loss_2 = self.cal_GAN_loss_D_2(batch_index=batch_index, fake=masked_fake, GT=masked_GT)
            self.loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, adv_loss, dis_loss]
        else:
            self.loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, 0, 0]
        if stage == "train":
            return self.loss_list
        elif stage == "validation":
            return self.loss_list, pearson_coef
    # update network's parameter after loss calculation
    def update_net(self, loss_list):
        self.loss_list = loss_list
        self.total_loss_G = 0
        self.total_loss_D = 0
        for i in range(len(self.weight_list)):
            self.total_loss_G += self.weight_list[i] * self.loss_list[i]
        self.optim_G.zero_grad()
        self.total_loss_G.backward()
        self.optim_G.step()
        if self.weight_list[5] > 0 and self.loss_list[-1]:
            self.total_loss_D = self.loss_list[-1]
            self.optim_D_1.zero_grad()
            self.total_loss_D.backward()
            self.optim_D_1.step()
    def update_scheduler(self):
        if self.scheduler_G:
            self.scheduler_G.step()
            if self.weight_list[5] > 0:
                self.scheduler_D_1.step() 
    def validation(self, mask=None, save_image=False):        
        self.total_loss_G, pearson_coef = self.calculate_loss(stage="validation", mask=mask)
        if save_image:
            image_list = [self.Input, self.fake_main, self.GT_main]
            return self.total_loss_G, image_list, pearson_coef
        else:
            return self.total_loss_G, pearson_coef
    

class DSCM_with_dataset(Basemodel):
    def __init__(
            self, 
            opt, 
            model_name_G, 
            model_name_D, 
            in_channels, 
            num_classes, 
            device, 
            weight_list, 
            initialize=False, 
            upscale_factor=1, 
            optimizer_name='Adam', 
            lr_G=0.0001, 
            lr_D=0.00001, 
            index_per_D=1, 
            scheduler_name='None'
    ):
        super(DSCM_with_dataset, self).__init__()
        self.opt = opt
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = device
        self.weight_list = weight_list
        self.upscale_factor = upscale_factor        
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        #generate model
        print(f"Model name: {model_name_G}")
        self.net_G = self.generate_G(model_name=model_name_G, in_channels=in_channels, num_classes=num_classes, upscale_factor=upscale_factor).to(device)
        # initialize the model before training 
        if initialize:
            self.net_G = self.init_net(self.net_G)
        # generate and initialize the discriminator when using GAN loss
        if self.weight_list[5] > 0: 
            self.net_D_1 = self.generate_D(model_name=model_name_D, in_channels=num_classes).to(device)
            self.net_D_1.train()
            if initialize:
                self.net_D_1 = self.init_net(self.net_D_1)
        # load pretrain models, currently the denoising model is not applied
        if self.opt['net_G']['pretrain_dir'] != "None":
            self.net_G = torch.load(r'{}'.format(opt['net_G']['pretrain_dir']), weights_only=False)
            for param in self.net_G.parameters():
                param.requires_grad = True
            self.net_G.train()
        if self.opt['net_D']['pretrain_dir_1'] != "None":
            self.net_D_1 = torch.load(r'{}'.format(opt['net_D']['pretrain_dir_1']), weights_only=False)
            for param in self.net_D_1.parameters():
                param.requires_grad = True
            self.net_D_1.train()
        #if self.opt['net_D']['pretrain_dir_2'] != "None": 
        #    self.net_D_2 = torch.load(r'{}'.format(opt['net_D']['pretrain_dir_2']))
        #    for param in self.net_D_2.parameters():
        #        param.requires_grad = True
        #    self.net_D_2.train()
        # generate loss criterion
        self.pixel_criterion = nn.MSELoss().to(self.device)            
        self.vgg = VGGFeatureExtractor().to(self.device)
        self.feature_criterion = nn.MSELoss().to(self.device)           
        self.freq_criterion = FFTLoss().to(self.device)
        self.SSIM_criterion = SSIM().to(self.device)
        self.gradient_criterion = nn.MSELoss().to(self.device)        
        self.get_grad = Get_grad_std(device=self.device, num_classes=1, kernel_size=3, blur_kernel_size=7, blur_kernel_std=3)
        self.corr_criterion = nn.MSELoss().to(self.device)#Pearson_loss()
        self.pearson_criterion = Pearson_loss().to(self.device)
        self.GAN_criterion = GANLoss('gan', 1.0, 0.0).to(self.device)
        self.degen_criterion = nn.MSELoss().to(self.device)
        # generate optimizer
        if self.optimizer_name == 'Adam':
            self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(0.9, 0.999))
            if self.weight_list[5] > 0: 
                self.optim_D_1 = torch.optim.Adam(self.net_D_1.parameters(), lr=lr_D)
                #self.optim_D_2 = torch.optim.Adam(self.net_D_2.parameters(), lr=lr_D)
        self.index_per_D = index_per_D
        # generate scheduler
        if self.scheduler_name == 'OneCycleLR':
            self.scheduler_G = OneCycleLR(self.optim_G, max_lr=lr_G, 
                total_steps=(self.opt['train']['epoches']), pct_start=0.1)        
            if self.weight_list[5] > 0: 
                self.scheduler_D_1 = OneCycleLR(self.optim_D_1, max_lr=lr_D, total_steps=(self.opt['train']['epoches']), pct_start=0.1)   
                #self.scheduler_D_2 = OneCycleLR(self.optim_D_2, max_lr=lr_D, total_steps=(self.opt['train']['epoches']), pct_start=0.1)   
        elif self.scheduler_name == "CosineAnnealingLR":
            self.scheduler_G = CosineAnnealingLR(self.optim_G, T_max=self.opt['train']['epoches'], eta_min=self.opt['train']['lr_G']/100)
            if self.weight_list[5] > 0: 
                self.scheduler_D_1 = CosineAnnealingLR(self.optim_D_1, T_max=(self.opt['train']['epoches']), eta_min=self.opt['train']['lr_D']/100)
                #self.scheduler_D_2 = CosineAnnealingLR(self.optim_D_2, T_max=(self.opt['train']['epoches']), eta_min=self.opt['train']['lr_D']/100)
        
        self.noise_level = opt['noise_level']
    def feed_data(self, Input, GT, epoch=0):
        self.Input = Input.to(self.device)
        self.GT_main = GT.to(self.device)
        self.fake_main = self.net_G(self.Input).to(self.device)
        return self.fake_main
    
    def calculate_loss(self, batch_index=0, stage="train", mask=None):
        pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, adv_loss = [0, 0, 0, 0, 0, 0]
        # MSE for the pixel loss
        pixel_loss = self.pixel_criterion(self.fake_main, self.GT_main)
        # Calculate PCC while validation
        pearson_coef = 0 if stage == "train" else self.pearson_criterion(self.fake_main, self.GT_main)
        # VGG-19 based feature loss
        if self.weight_list[1] > 0: 
            for i in range(self.fake_main.size()[1]):
                fea_fake = self.vgg(self.fake_main[:,i:i+1,:,:])
                fea_GT = self.vgg(self.GT_main[:,i:i+1,:,:])              
                feature_loss += self.feature_criterion(fea_fake, fea_GT)
        # SSIM loss
        if self.weight_list[2] > 0: 
            #freq_loss = self.freq_criterion(self.fake, self.GT)
            for i in range(self.fake_main.size()[1]):
                temp_fake = self.fake_main[:,i:i+1,:,:].detach()
                temp_GT = self.GT_main[:,i:i+1,:,:].detach()
                temp_fake = temp_fake / torch.max(temp_fake)
                temp_GT = temp_GT / torch.max(temp_GT)
                #print(torch.max(temp_fake), torch.min(temp_fake), torch.max(temp_GT), torch.min(temp_GT))
                SSIM_value = self.SSIM_criterion(temp_fake, temp_GT)
                value_one = torch.ones_like(SSIM_value)
                #print((value_one - SSIM_value))
                SSIM_loss = SSIM_loss + (value_one - SSIM_value)
        # Gradient loss
        if self.weight_list[3] > 0:
            grad_fake = self.get_grad(self.fake_main[:,0:1,:,:])
            grad_real = self.get_grad(self.GT_main[:,0:1,:,:])
            grad_loss = self.gradient_criterion(grad_fake, grad_real)
        # Correlation loss
        if self.weight_list[4] > 0:
            masked_GT = self.GT_main * mask
            masked_fake = self.fake_main * mask
            corr_loss = self.corr_criterion(masked_fake, masked_GT)
        # GAN loss
        if self.weight_list[5] > 0:
            if self.weight_list[4] > 0:
                mask = mask.clone()
                mask[mask == 0] = 1
                self.fake_main = self.fake_main * mask
                self.GT_main = self.GT_main * mask
            else:
                pass
            if self.opt["net_D"]["model_name"] == "PatchGAN":
                adv_loss = self.cal_PatchGAN_loss_G(fake=self.fake_main)
                dis_loss = self.cal_PatchGAN_loss_D(batch_index=batch_index, fake=self.fake_main, GT=self.GT_main)
            else:
                adv_loss = self.cal_GAN_loss_G_1(fake=self.fake_main)
                dis_loss = self.cal_GAN_loss_D_1(batch_index=batch_index, fake=self.fake_main, GT=self.GT_main)
            #if self.weight_list[4] > 0:
            #    adv_loss += self.weight_list[4] * self.cal_GAN_loss_G_2(fake=masked_fake)
            #    #dis_loss += self.cal_GAN_loss_D_2(batch_index=batch_index, fake=masked_fake, GT=masked_GT)
            #    self.dis_loss_2 = self.cal_GAN_loss_D_2(batch_index=batch_index, fake=masked_fake, GT=masked_GT)
            self.loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, adv_loss, dis_loss]
        else:
            self.loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, 0, 0]
        if stage == "train":
            return self.loss_list
        elif stage == "validation":
            return self.loss_list, pearson_coef
    # update the network after calculating loss
    def update_net(self, loss_list):
        self.loss_list = loss_list
        self.total_loss_G = 0
        self.total_loss_D = 0
        for i in range(len(self.weight_list)):
            self.total_loss_G += self.weight_list[i] * self.loss_list[i]
        self.optim_G.zero_grad()
        self.total_loss_G.backward()
        self.optim_G.step()
        if self.weight_list[5] > 0:
            self.total_loss_D = self.loss_list[-1]
            self.optim_D_1.zero_grad()
            self.total_loss_D.backward()
            self.optim_D_1.step()
            #if self.weight_list[4] > 0:
            #    self.optim_D_2.zero_grad()
            #    self.dis_loss_2.backward()
            #    self.optim_D_2.step()

    def update_scheduler(self):
        if self.scheduler_name != None:
            self.scheduler_G.step()
            if self.weight_list[5] > 0:
                self.scheduler_D_1.step()        
    def validation(self, mask=None, save_image=False):        
        self.total_loss_G, pearson_coef = self.calculate_loss(stage="validation", mask=mask)
        if save_image:
            image_list = [self.Input, self.fake_main, self.GT_main]
            return self.total_loss_G, image_list, pearson_coef
        else:
            return self.total_loss_G, pearson_coef
        