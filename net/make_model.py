import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

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
#sys.path.append('./utils')

from utils import *
#from SSIM_loss import *
from loss.MS_SSIM_L1 import *
from loss.decouple_loss import *
from loss.gradient_loss import *
from loss.pearson_loss import *
from loss.FFT_loss import *


class Basemodel():
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
            #Input = torch.ones((1,in_channels,size,size), dtype=torch.float, device=device)
            #Output = model(Input)
            #print(Output.size())
        elif model_name == 'Swinir':
            size = 512
            height = size // upscale
            width = size // upscale
            window_size = 8
            embed_dim = 60
            num_heads = [6, 6, 6, 6]
            depths = [6, 6, 6, 6]
            in_channels = num_classes
            net = SwinIR(upscale=upscale, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=depths, in_chans=in_channels, 
                    embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2, upsampler='pixelshuffledirect')
        
        elif model_name == 'Unet3D':
            net = UNet3D(in_channels=in_channels, out_channels=num_classes)
        elif model_name == 'Unet_FLIM':
            net = Unet(input_dim=in_channels+1, num_classes=num_classes)
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
        I = torch.FloatTensor(I).to(self.device).unsqueeze(0)
        I = nn.Parameter(data=I, requires_grad=False).to(self.device)
        return I
    def add_poisson_pytorch(self, Input, intensity=1.0):
        Input_max = torch.max(Input)
        Input = Input / Input_max
        Input = torch.poisson(Input * intensity) / intensity * Input_max
        return Input
    def add_poisson_numpy(self, Input, intensity):        
        Input_max = torch.max(Input)
        Input = to_cpu(Input / Input_max)
        Input = np.random.poisson(Input * intensity) / intensity
        Input = torch.tensor(Input, dtype=torch.float, device=self.device) * Input_max
        return Input
    def add_noise(self, Input):   
        Input = self.add_poisson_pytorch(Input, intensity=self.noise_level)
        #Input = self.add_poisson_numpy(Input, intensity=self.noise_level)
        return Input
    def sr_degeneration(self, Input):
        Input = F.conv2d(input=Input, weight=self.psf_gaussian, padding=64, stride=1)
        return Input
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
            
            '''plt.figure(1)
            plt.subplot(131)
            plt.imshow(to_cpu(self.fake[0,0,:,:]))
            plt.subplot(132)
            plt.imshow(to_cpu(self.fake[0,1,:,:]))
            plt.subplot(133)
            plt.imshow(to_cpu(self.fake[0,2,:,:]))
            plt.figure(2)
            plt.subplot(131)
            plt.imshow(to_cpu(self.GT[0,0,:,:]))
            plt.subplot(132)
            plt.imshow(to_cpu(self.GT[0,1,:,:]))
            plt.subplot(133)
            plt.imshow(to_cpu(self.GT[0,2,:,:]))

            plt.figure(3)
            plt.subplot(121)
            plt.imshow(to_cpu(e_F[0,0,:,:]))
            plt.subplot(122)
            plt.imshow(to_cpu(e_T[0,0,:,:]))
            plt.figure(4)
            plt.subplot(121)
            plt.imshow(to_cpu(d_F[0,0,:,:]))
            plt.subplot(122)
            plt.imshow(to_cpu(d_T[0,0,:,:]))

            plt.figure(5)
            plt.subplot(121)
            plt.imshow(to_cpu(F_dF[0,0,:,:]))
            plt.subplot(122)
            plt.imshow(to_cpu(T_dT[0,0,:,:]))

            cv2.imencode('.tif', to_cpu(self.Input[0,0,:,:]))[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\temp\Input')
            cv2.imencode('.tif', to_cpu(self.fake[0,0,:,:]))[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\temp\fake_0')
            cv2.imencode('.tif', to_cpu(self.fake[0,1,:,:]))[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\temp\fake_1')
            cv2.imencode('.tif', to_cpu(self.fake[0,2,:,:]))[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\temp\fake_2')
            cv2.imencode('.tif', to_cpu(self.GT[0,0,:,:]))[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\temp\GT_0')
            cv2.imencode('.tif', to_cpu(self.GT[0,1,:,:]))[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\temp\GT_1')
            cv2.imencode('.tif', to_cpu(self.GT[0,2,:,:]))[1].tofile(r'D:\CQL\codes\microscopy_decouple\visualization\temp\GT_2')

            plt.show()'''

            T_eT = True
            T_dT = True
            F_eF = False
            F_dF = False
            GAN_loss_D = self.GAN_criterion(e_F, F_eF) + self.GAN_criterion(d_F, F_dF) + self.GAN_criterion(e_T, T_eT) + self.GAN_criterion(d_T, T_dT)
            #GAN_loss_D = self.GAN_criterion(e_F, F_eF) + self.GAN_criterion(e_T, T_eT)
        return GAN_loss_D
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


    def init_net(self, net, init_type='normal', init_gain=0.02):
        self.init_weights(net, init_type, gain=init_gain)
        return net

    def initialize_weights(self, net_l, scale=1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)
    

class make_model(Basemodel):
    def __init__(self, opt, model_name_G, model_name_D, in_channels, num_classes, mode, device, weight_list, initialize=False, upscale_factor=1, optimizer_name='Adam', lr_G=0.0001, lr_D=0.00001, index_per_D=1, scheduler_name='None'):
        super(make_model, self).__init__()
        self.opt = opt
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mode = mode
        self.device = device
        self.weight_list = weight_list
        self.upscale_factor = upscale_factor        
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        #generate model
        print(f"Model name: {model_name_G}")
        self.net_G = self.generate_G(model_name=model_name_G, in_channels=in_channels, num_classes=num_classes, upscale_factor=upscale_factor).to(device)
        if initialize:
            self.net_G = self.init_net(self.net_G)
        if self.weight_list[5] > 0: 
            self.net_D_1 = self.generate_D(model_name=model_name_D, in_channels=num_classes).to(device)
            self.net_D_1.train()
            #self.net_D_2 = self.generate_D(model_name=model_name_D, in_channels=num_classes).to(device)
            #self.net_D_2.train()
            if initialize:
                self.net_D_1 = self.init_net(self.net_D_1)
                #self.net_D_2 = self.init_net(self.net_D_2)
        #self.net_G = torch.load(r"D:\CQL\codes\microscopy_decouple\validation\6.3_Microtubes_Mitochondria_Lysosome_noise_level_1_pixel_one_stage\weights\1\main.pth")
        if self.opt['net_G']['pretrain_dir'] != "None":
            self.net_G = torch.load(r'{}'.format(opt['net_G']['pretrain_dir']))
            for param in self.net_G.parameters():
                param.requires_grad = True
            self.net_G.train()
        if self.opt['net_D']['pretrain_dir_1'] != "None":
            self.net_D_1 = torch.load(r'{}'.format(opt['net_D']['pretrain_dir_1']))
            for param in self.net_D_1.parameters():
                param.requires_grad = True
            self.net_D_1.train()
        #if self.opt['net_D']['pretrain_dir_2'] != "None": 
        #    self.net_D_2 = torch.load(r'{}'.format(opt['net_D']['pretrain_dir_2']))
        #    for param in self.net_D_2.parameters():
        #        param.requires_grad = True
        #    self.net_D_2.train()
        #generate loss criterion
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
        N = 129
        w0 = 2.05
        span = 12
        self.psf_gaussian = self.generate_psf(m=0, N=N, w0=w0, span=span).unsqueeze(0)
        self.noise_level = opt['noise_level']        
    def feed_data(self, Input, GT, epoch=0):
        self.Input = Input.to(self.device)
        self.GT = GT.to(self.device)
        self.fake = self.net_G(self.Input).to(self.device)
        return self.fake
    def calculate_loss(self, batch_index=0, stage="train", mask=None):
        pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, adv_loss = [0, 0, 0, 0, 0, 0]
        pixel_loss = self.pixel_criterion(self.fake, self.GT)
        pearson_coef = 0 if stage == "train" else self.pearson_criterion(self.fake, self.GT)
        if self.weight_list[1] > 0: 
            for i in range(self.fake.size()[1]):
                fea_fake = self.vgg(self.fake[:,i:i+1,:,:])
                fea_GT = self.vgg(self.GT[:,i:i+1,:,:])              
                feature_loss += self.feature_criterion(fea_fake, fea_GT)
        if self.weight_list[2] > 0: 
            #freq_loss = self.freq_criterion(self.fake, self.GT)
            for i in range(self.fake.size()[1]):
                temp_fake = self.fake[:,i:i+1,:,:].detach()
                temp_GT = self.GT[:,i:i+1,:,:].detach()
                temp_fake = temp_fake / torch.max(temp_fake)
                temp_GT = temp_GT / torch.max(temp_GT)
                #print(torch.max(temp_fake), torch.min(temp_fake), torch.max(temp_GT), torch.min(temp_GT))
                SSIM_value = self.SSIM_criterion(temp_fake, temp_GT)
                value_one = torch.ones_like(SSIM_value)
                #print((value_one - SSIM_value))
                SSIM_loss = SSIM_loss + (value_one - SSIM_value)
        if self.weight_list[3] > 0:
            grad_fake = self.get_grad(self.fake[:,0:1,:,:])
            grad_real = self.get_grad(self.GT[:,0:1,:,:])
            grad_loss = self.gradient_criterion(grad_fake, grad_real)
        if self.weight_list[4] > 0:
            masked_GT = self.GT * mask
            masked_fake = self.fake * mask
            corr_loss = self.corr_criterion(masked_fake, masked_GT)
        if self.weight_list[5] > 0:
            if self.weight_list[4] > 0:
                mask = mask.clone()
                mask[mask == 0] = 1
                self.fake = self.fake * mask
                self.GT = self.GT * mask
            else:
                pass
            if self.opt["net_D"]["model_name"] == "PatchGAN":
                adv_loss = self.cal_PatchGAN_loss_G(fake=self.fake)
                dis_loss = self.cal_PatchGAN_loss_D(batch_index=batch_index, fake=self.fake, GT=self.GT)
            else:
                adv_loss = self.cal_GAN_loss_G_1(fake=self.fake)
                dis_loss = self.cal_GAN_loss_D_1(batch_index=batch_index, fake=self.fake, GT=self.GT)
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

    def update_net(self, loss_list):
        self.loss_list = loss_list
        self.total_loss_G = 0
        self.total_loss_D = 0
        for i in range(len(self.weight_list)):
            self.total_loss_G += self.weight_list[i] * self.loss_list[i]
        #    self.total_loss_G += self.total_loss_list[i]
        #self.total_loss_G = sum(self.loss_list[:-1])
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
            image_list = [self.Input, self.fake, self.GT]
            return self.total_loss_G, image_list, pearson_coef
        else:
            return self.total_loss_G, pearson_coef
        
'''
class make_model_three_stage(Basemodel):
    def __init__(self, opt, model_name_G, model_name_D, in_channels, num_classes, mode, device, weight_list, initialize=False, upscale_factor=1, optimizer_name='Adam', lr_G=0.0001, lr_D=0.00001, index_per_D=1, scheduler_name='None'):
        super(make_model_three_stage, self).__init__()
        self.opt = opt
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mode = mode
        self.device = device
        self.weight_list = weight_list
        self.upscale_factor = upscale_factor
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        #generate model
        print(f"Model name: {model_name_G}")
        self.net_G = self.generate_G(model_name=model_name_G, in_channels=in_channels, num_classes=num_classes, upscale_factor=upscale_factor).to(device)
        if initialize:
            self.net_G = self.init_net(self.net_G)
        if self.mode == "GAN": 
            self.net_D = self.generate_D(model_name=model_name_D, in_channels=num_classes).to(device)
            self.net_D.train()
            if initialize:
                self.net_D = self.init_net(self.net_D)
        #self.net_G = torch.load(r'D:\CQL\codes\microscopy_decouple\validation\6.3_Microtubes_Mitochondria_Lysosome_noise_level_1_pixel_one_stage\weights\1\main.pth')
        #self.net_G = torch.load(r'D:\CQL\codes\microscopy_decouple\validation\6.3_Microtubes_Mitochondria_inner_Lysosome_noise_level_1_pixel_three_stage\weights\1\main.pth')
        #generate loss criterion
        self.pixel_criterion = nn.MSELoss().to(self.device)            
        self.vgg = VGGFeatureExtractor().to(self.device)
        self.feature_criterion = nn.MSELoss().to(self.device)           
        self.SSIM_criterion = SSIML1_Loss().to(self.device)
        #self.SSIM_criterion = SSIM(channel=num_classes).to(self.device)        
        self.gradient_criterion = nn.MSELoss().to(self.device)        
        self.get_grad = Get_grad_std(device=self.device, num_classes=1, kernel_size=3, blur_kernel_size=7, blur_kernel_std=3)
        self.corr_criterion = Pearson_loss()        
        self.GAN_criterion = GANLoss('ragan', 1.0, 0.0).to(self.device)
        self.degen_criterion = nn.MSELoss().to(self.device)
        # generate optimizer
        if self.optimizer_name == 'Adam':
            self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=lr_G)
            if self.mode == "GAN": self.optim_D = torch.optim.Adam(self.net_D.parameters(), lr=lr_D)               
        self.index_per_D = index_per_D        
        # generate scheduler
        if self.scheduler_name == 'OneCycleLR':
            self.scheduler_G = OneCycleLR(self.optim_G, max_lr=lr_G, 
                total_steps=(self.opt['train']['epoches']*self.opt['train']['num_iter']*self.opt['num_train'])//self.opt['train']['batch_size'], pct_start=0.3)        
            if self.mode == "GAN": self.scheduler_D = OneCycleLR(self.optim_D, max_lr=lr_D, 
                total_steps=(self.opt['train']['epoches']*self.opt['train']['num_iter']*self.opt['num_train'])//self.opt['train']['batch_size'], pct_start=0.3)   
        elif self.scheduler_name == "CosineAnnealingLR":
            self.scheduler_G = CosineAnnealingLR(self.optim_G, T_max=(self.opt['train']['epoches']*self.opt['train']['num_iter']*self.opt['num_train'])//self.opt['train']['batch_size'])
            if self.mode == "GAN": self.scheduler_D = CosineAnnealingLR(self.optim_D, T_max=(self.opt['train']['epoches']*self.opt['train']['num_iter']*self.opt['num_train'])//self.opt['train']['batch_size'])
        N = 129
        w0 = 2.05
        span = 12
        self.psf_gaussian = self.generate_psf(m=0, N=N, w0=w0, span=span).unsqueeze(0)
        self.noise_level = opt['noise_level']        
    def feed_data(self, Input, epoch=0):
        self.Input = Input.to(self.device)     
        self.fake_1, self.fake_2, self.fake_3 = self.net_G(self.Input)
        return self.fake_1.to(self.device), self.fake_2.to(self.device), self.fake_3.to(self.device)
    
    def calculate_loss(self, Output, GT, batch_index=0, mode="train", std=1):
        if mode == "degen_denoise_only":        
            GT = self.add_noise(GT*std, self.noise_level) / std
            Output = self.add_noise(Output*std, self.noise_level) / std            
        elif mode == "degen_sr_only":
            Output = F.conv2d(Output*std, self.psf_gaussian, padding=64, stride=1) / std
            GT = F.conv2d(GT*std, self.psf_gaussian, padding=64, stride=1) / std
        pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss = [0, 0, 0, 0, 0]
        pixel_loss = self.pixel_criterion(Output, GT)
        if self.weight_list[1] > 0:
             for i in range(Output.size()[1]):
                fea_fake = self.vgg(Output[:,i:i+1,:,:])
                fea_GT = self.vgg(GT[:,i:i+1,:,:])              
                feature_loss += self.feature_criterion(fea_fake, fea_GT)
        if self.weight_list[2] > 0: 
            for i in range(Output.size()[1]):
                SSIM_loss += self.SSIM_criterion(Output[:,i:i+1,:,:], GT[:,i:i+1,:,:])                      
        if self.weight_list[3] > 0:
            for i in range(Output.size()[1]):
                grad_fake = self.get_grad(Output[:,i:i+1,:,:])
                grad_real = self.get_grad(GT[:,i:i+1,:,:])
                grad_loss += self.gradient_criterion(grad_fake, grad_real)
        if self.weight_list[4] > 0:
            corr_loss = self.corr_criterion(Output, GT)
            torch.ones_like(corr_loss, dtype=torch.float, device=self.device) - corr_loss 
        if mode == "GAN":
            loss_G = self.cal_GAN_loss_G(fake=Output)
            loss_D = self.cal_GAN_loss_D(fake=Output, GT=GT, batch_index=batch_index) 
        if mode.find("degen") != -1 and mode.find("only") == -1:
            if mode == "degen_denoise":                              
                self.degenerated = self.add_noise(Output*std, self.noise_level)
            elif mode == "degen_sr":
                self.degenerated = F.conv2d(Output*std, self.psf_gaussian, padding=64, stride=1)
            elif mode == "degen_decouple":                
                for i in range(Output.size()[1]):
                    if i == 0:
                        self.degenerated = Output[:,i:i+1,:,:]*std
                    else:
                        self.degenerated += Output[:,i:i+1,:,:]*std
            self.degen_loss = self.degen_criterion(self.degenerated/std, self.Input)
            self.total_loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, self.degen_loss, 0]
        elif mode == "GAN":
            self.total_loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, loss_G, loss_D]
        else:
            self.total_loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, 0, 0]
        return self.total_loss_list
    def cal_GAN_loss_G(self, fake):
        for p in self.net_D.parameters():
            p.requires_grad = False
        e_F, d_F, _, _ = self.net_D(fake) 
        T_eF = torch.ones_like(e_F, device=self.device)
        T_dF = torch.ones_like(d_F, device=self.device)
        T_eF = True
        T_dF = True
        GAN_loss_G = self.GAN_criterion(e_F, T_eF) + self.GAN_criterion(d_F, T_dF)
        return GAN_loss_G 
    def cal_GAN_loss_D(self, fake, GT, batch_index):
        if (batch_index+1) % self.index_per_D == 0:            
            for p in self.net_D.parameters():
                p.requires_grad = True
            e_F, d_F, _, _ = self.net_D(fake.detach())
            e_T, d_T, _, _ = self.net_D(GT.detach())
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
    
    def update_net(self, loss_list_1, loss_list_2, loss_list_3):
        if self.mode == "GAN":            
            loss_1_G = sum(loss_list_1[:-1])
            #loss_1_D = loss_list_1[-1]
            loss_2_G = sum(loss_list_2[:-1])
            #loss_2_D = loss_list_2[-1]
            loss_3_G = sum(loss_list_3[:-1])
            loss_D = loss_list_3[-1]
        else:
            loss_1_G = sum(loss_list_1)
            loss_2_G = sum(loss_list_2)
            loss_3_G = sum(loss_list_3)
        total_loss_G = loss_1_G + loss_2_G + loss_3_G
        self.optim_G.zero_grad()
        total_loss_G.backward()
        self.optim_G.step()
        if self.mode == "GAN":
            self.optim_D.zero_grad()
            loss_D.backward()
            self.optim_D.step()

    def update_scheduler(self):
        if self.scheduler_name != None:
            self.scheduler_G.step()
            if self.mode == "GAN":
                self.scheduler_D.step()        
        
    def validation(self, std=1, save_image=False):        
        self.total_loss_G = self.calculate_loss(mode="validation", std=std)
        if save_image:
            image_list = [self.Input, self.fake_3, self.GT]
            return self.total_loss_G, image_list
        else:
            return self.total_loss_G   
'''

class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
            #self.loss = nn.BCELoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))
    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        #print('#######target_label.shape#########')
        #print(target_label.shape)
        loss = self.loss(input, target_label)
        return loss