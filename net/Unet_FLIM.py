import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class Unet_FLIM_att(nn.Module):
    def __init__(self, input_dim, n_channels=64, num_classes=2, upscale_factor=1):
        super(Unet_FLIM_att, self).__init__()
        self.upscale_factor = upscale_factor
        self.block_1 = self.make_conv_block(inchannels=1, outchannels=n_channels, kernel_size=3, padding=1)
        self.block_2 = self.make_conv_block(inchannels=n_channels, outchannels=n_channels*2, kernel_size=3, padding=1)
        self.block_3 = self.make_conv_block(inchannels=n_channels*2, outchannels=n_channels*4, kernel_size=3, padding=1)
        self.block_4 = self.make_conv_block(inchannels=n_channels*4, outchannels=n_channels*8, kernel_size=3, padding=1)
        self.block_5 = self.make_conv_block(inchannels=n_channels*8, outchannels=n_channels*16, kernel_size=3, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.up_block_4 = self.make_up_block(method='Upsample', inchannels=n_channels*16, outchannels=n_channels*8)
        self.up_block_3 = self.make_up_block(method='Upsample', inchannels=n_channels*8, outchannels=n_channels*4)
        self.up_block_2 = self.make_up_block(method='Upsample', inchannels=n_channels*4, outchannels=n_channels*2)
        self.up_block_1 = self.make_up_block(method='Upsample', inchannels=n_channels*2, outchannels=n_channels)

        self.att_4 = self.make_att_gate(n_channels=n_channels*8)
        self.att_3 = self.make_att_gate(n_channels=n_channels*4)
        self.att_2 = self.make_att_gate(n_channels=n_channels*2)
        self.att_1 = self.make_att_gate(n_channels=n_channels*1)

        self.pool_5_ = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_4_ = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_3_ = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_2_ = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.att_1_FLIM = self.make_att_FLIM(gate_in_channels=1, n_channels=n_channels* 2)
        self.att_2_FLIM = self.make_att_FLIM(gate_in_channels=1, n_channels=n_channels* 4)
        self.att_3_FLIM = self.make_att_FLIM(gate_in_channels=1, n_channels=n_channels* 8)
        self.att_4_FLIM = self.make_att_FLIM(gate_in_channels=1, n_channels=n_channels* 16)
        self.att_5_FLIM = self.make_att_FLIM(gate_in_channels=1, n_channels=n_channels* 16)
        #self.up_block_4 = nn.ConvTranspose2d(in_channels=n_channels*16, out_channels=n_channels*8, kernel_size=2, padding=0, stride=2, bias=False)
        #self.up_block_3 = nn.ConvTranspose2d(in_channels=n_channels*8, out_channels=n_channels*4, kernel_size=2, padding=0, stride=2, bias=False)
        #self.up_block_2 = nn.ConvTranspose2d(in_channels=n_channels*4, out_channels=n_channels*2, kernel_size=2, padding=0, stride=2, bias=False)
        #self.up_block_1 = nn.ConvTranspose2d(in_channels=n_channels*2, out_channels=n_channels, kernel_size=2, padding=0, stride=2, bias=False)
        
        if upscale_factor == 2:
            self.up_block_0 = nn.Sequential(
                nn.PixelShuffle(upscale_factor=2), 
                nn.Conv2d(in_channels=n_channels//4, out_channels=n_channels, kernel_size=1, padding=0, bias=True)
            )
        
        self.block_4_ = self.make_conv_block(inchannels=n_channels*16, outchannels=n_channels*8, kernel_size=3, padding=1)
        self.block_3_ = self.make_conv_block(inchannels=n_channels*8, outchannels=n_channels*4, kernel_size=3, padding=1)
        self.block_2_ = self.make_conv_block(inchannels=n_channels*4, outchannels=n_channels*2, kernel_size=3, padding=1)
        self.block_1_ = self.make_conv_block(inchannels=n_channels*2, outchannels=n_channels, kernel_size=3, padding=1)
      
        self.pred_block = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=n_channels, out_channels=num_classes, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )
        #self.pred_conv = nn.Conv2d(in_channels=n_channels, out_channels=num_classes, kernel_size=1, padding=0, bias=True)
    def make_att_gate(self, n_channels):
        block = Attention_block(F_g=n_channels, F_l=n_channels, F_int=n_channels // 2)
        return block
    
    def make_att_FLIM(self, gate_in_channels, n_channels):
        block = Attention_block(F_g=gate_in_channels, F_l=n_channels, F_int=n_channels // 2)
        return block

    def make_up_block(self, method, inchannels, outchannels, kernel_size=1, padding=0, stride=1):
        if method == "Upsample":
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'), 
                #F.interpolate()
                nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, padding=0)
            )
        elif method == "TransposeConv":
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=2, padding=0, stride=2, bias=False)
            )
        return block

    def make_conv_block(self, inchannels, outchannels,kernel_size, padding, stride=1, dilation=1, use_bias=False):
        block = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.1),
        )
        return block
        
    def forward(self, input, FLIM_map=None):
        # Downsample FLIM_map
        FLIM_map = input[:,1:2,:,:]
        input = input[:,0:1,:,:]
        FLIM_map_2 = self.pool_2_(FLIM_map)
        FLIM_map_3 = self.pool_3_(FLIM_map_2)
        FLIM_map_4 = self.pool_4_(FLIM_map_3)
        FLIM_map_5 = self.pool_5_(FLIM_map_4)
        # ===============================================
        conv_1 = self.block_1(input)
        pool_1 = self.pool_1(conv_1)
        # ===============================================
        conv_2 = self.block_2(pool_1)
        pool_2 = self.pool_2(conv_2)
        # ===============================================
        conv_3 = self.block_3(pool_2)
        pool_3 = self.pool_3(conv_3)
        # ===============================================
        conv_4 = self.block_4(pool_3)
        pool_4 = self.pool_4(conv_4)
        # ===============================================
        conv_5 = self.block_5(pool_4)
        att_FLIM_5 = self.att_5_FLIM(g=FLIM_map_5, x=conv_5)
        # ===============================================
        # ===============================================
        up_4 = self.up_block_4(att_FLIM_5)
        #conv_4 = self.att_4(g=up_4, x=conv_4)
        concat_4 = torch.concat([up_4, conv_4], 1)
        att_FLIM_4 = self.att_4_FLIM(g=FLIM_map_4, x=concat_4)
        conv_4_ = self.block_4_(att_FLIM_4)
        # ===============================================
        up_3 = self.up_block_3(conv_4_)  
        #conv_3 = self.att_3(g=up_3, x=conv_3)
        concat_3 = torch.concat([up_3, conv_3], 1)
        att_FLIM_3 = self.att_3_FLIM(g=FLIM_map_3, x=concat_3)
        conv_3_ = self.block_3_(att_FLIM_3)
        # ===============================================
        up_2 = self.up_block_2(conv_3_)
        #conv_2 = self.att_2(g=up_2, x=conv_2)
        concat_2 = torch.concat([up_2, conv_2], 1)
        att_FLIM_2 = self.att_2_FLIM(g=FLIM_map_2, x=concat_2)
        conv_2_ = self.block_2_(att_FLIM_2)
        # ===============================================
        up_1 = self.up_block_1(conv_2_)
        #conv_1 = self.att_1(g=up_1, x=conv_1)
        concat_1 = torch.concat([up_1, conv_1], 1)
        att_FLIM_1 = self.att_1_FLIM(g=FLIM_map, x=concat_1)
        conv_1_ = self.block_1_(att_FLIM_1)
        #if self.upscale_factor == 2: conv_1_ = self.up_block_0(conv_1_)
        # ===============================================
        pred = self.pred_block(conv_1_)
        return pred

  
if __name__ == "__main__":
    import os, sys
    import tifffile
    import numpy as np
    import matplotlib.pyplot as plt

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    from utils import *
    input = torch.randn((1,1,1024,1024), dtype=torch.float32).cuda()
    FLIM_distribution = tifffile.imread(r"C:\Users\18923\Desktop\FLIM_decouple\asc_map_3\1_FLIM_distribution.tif")
    FLIM_distribution = torch.tensor(np.float64(FLIM_distribution), dtype=torch.float, device='cuda').unsqueeze(0).unsqueeze(0)
    model = Unet_FLIM_att(input_dim=1, num_classes=2, n_channels=64, upscale_factor=1).cuda()
    input = torch.concat([input, FLIM_distribution], dim=1)
    output= model(input)
    print(output.size())