import os
import torch
    
import torch.nn as nn

from torch.functional import Tensor
from unet import *
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     


class CALayer(nn.Module):
    def __init__(self, n_channels, kernel_size=3, padding=1, reduction=16):
        super(CALayer, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=1, stride=1, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels//reduction, kernel_size=kernel_size, padding=1, stride=1, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=n_channels//reduction, out_channels=n_channels, kernel_size=kernel_size, padding=1, stride=1, bias=False)
        self.average_pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, input):
        conv = self.conv_1(input)
        conv = self.conv_2(conv)
        conv = self.conv_3(conv)
        mul = input * conv
        return mul


class RCAB(nn.Module):
    def __init__(self, n_channels):
        super(RCAB, self).__init__()
        self.conv_1 = self.make_conv(n_channels)                        
        self.attention_layer = CALayer(n_channels=n_channels)
    def make_conv(self, channel):
        block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.GELU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.GELU()            
        )
        return block    
    def forward(self, input):
        conv = self.conv_1(input)
        att = self.attention_layer(conv)
        output = att + input
        return output
    

class ResidualGroup(nn.Module):
    def __init__(self, channels=64, kernel_size=3, padding=1, n_RCAB=2):
        super(ResidualGroup, self).__init__()
        RCAB_list = []        
        for i in range(n_RCAB):
            RCAB_list.append(RCAB(n_channels=channels))
        self.Res_RCAB = nn.Sequential(*RCAB_list)
    def forward(self, input):
        res_RCAB = self.Res_RCAB(input)
        return res_RCAB
        #output = torch.concat([res_RCAB, input], dim=1)
        #print(output.size())
        #return output
            



class Autoencoder(nn.Module):
    def __init__(self, in_channels=1, n_channels=64, out_channels=1, n_ResGroup=1, scale=2):
        super(Autoencoder, self).__init__()

        #self.unet = Unet(input_dim=1, num_class=2)
        #self.unet.load_state_dict(torch.load(r'C:\Users\20151\Desktop\460.pth', map_location='cpu'))
        #for param in self.unet.parameters():
        #    param.requires_grad = False

        self.conv_1 = self.make_conv_block(inchannels=in_channels, outchannels=n_channels, kernel_size=3, padding='same')
        self.conv_2 = self.make_conv_block(inchannels=n_channels, outchannels=n_channels, kernel_size=3, padding='same')
        self.conv_3 = self.make_conv_block(inchannels=n_channels, outchannels=n_channels, kernel_size=3, padding='same')
        self.conv_4 = self.make_conv_block(inchannels=n_channels, outchannels=n_channels * (scale ** 2), kernel_size=3, padding='same')
        self.pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)        
        ResGroup_list = []
        for _ in range(n_ResGroup):
            ResGroup_list.append(ResidualGroup(channels=64))
        self.ResGroups_1 = nn.Sequential(*ResGroup_list)
        self.ResGroups_2 = nn.Sequential(*ResGroup_list)
        self.ResGroups_3 = nn.Sequential(*ResGroup_list)
        self.layer = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2), 
            self.make_conv_block(inchannels=n_channels, outchannels=n_channels * (scale ** 2), kernel_size=3, padding='same'), 
            nn.PixelShuffle(upscale_factor=2), 
            self.make_conv_block(inchannels=n_channels, outchannels=n_channels, kernel_size=3, padding='same'), 
            nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=True)
        )
        

    def make_conv_block(self, inchannels, outchannels, kernel_size, padding, stride=1, dilation=1, use_bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            #nn.ReLU(inplace=False),
            nn.GELU(),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            #nn.ReLU(inplace=False)
            nn.GELU()
        )
        return block    

    def forward(self, input):
        conv_1 = self.conv_1(input)
        res_1 = self.ResGroups_1(conv_1)
        pool_1 = self.pooling_1(res_1)
        conv_2 = self.conv_2(pool_1)
        res_2 = self.ResGroups_2(conv_2)
        pool_2 = self.pooling_2(res_2)
        conv_3 = self.conv_3(pool_2)
        res_3 =self.ResGroups_3(conv_3)
        conv_4 = self.conv_4(res_3)
        output = self.layer(conv_4)
        '''up_1 = self.ps_1(conv_4)
        up_conv_1 = self.conv_up_1(up_1)
        up_2 = self.ps_2(up_conv_1)
        up_conv_2 = self.conv_up_2(up_2)
        output = self.pred_conv(up_conv_2)'''
        return output
    

class fusion_block(nn.Module):
    def __init__(self, in_channels, n_channels=256):
        super(fusion_block, self).__init__()
        #self.conv_1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=True),
        #self.conv_2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=True),
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(in_channels, n_channels*2 , kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels*2, n_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, in_channels//2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )        
        self.ReLU = nn.ReLU(inplace=True)
    def forward(self, input_1, input_2):        
        fusion = self.fusion_layers(torch.concat([input_1, input_2], dim=1))
        output_1 = self.ReLU(input_1 + fusion)
        output_2 = self.ReLU(input_2 + fusion)
        return output_1, output_2
    

class fusion_block_4(nn.Module):
    def __init__(self, n_channels=256, num_species=4):
        super(fusion_block_4, self).__init__()
        #self.conv_1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=True),
        #self.conv_2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=True),
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(n_channels*num_species*4, (n_channels*num_species*4)//2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d((n_channels*num_species*4)//2, (n_channels*num_species*4)//4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d((n_channels*num_species*4)//4, (n_channels*4), kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(n_channels*4),
            nn.ReLU(inplace=True)
        )                
        self.bn = nn.BatchNorm2d(num_features=n_channels*4)
        self.ReLU = nn.ReLU(inplace=True)
    def forward(self, input_list):       
        #print(input_1.size(), input_2.size())         
        fusion = self.fusion_layers(torch.concat(input_list, dim=1))
        fusion = self.bn(fusion)
        output_list = []
        for i in range(len(input_list)):
            #print(input_list[i].size(), fusion.size())
            out = torch.concat([input_list[i], fusion], dim=1)
            #output_list.append(self.ReLU(input_list[i] + fusion))
            output_list.append(out)
        return output_list
    

class Channel_pool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self):
        super(spatial_attn_layer, self).__init__()
        self.compress = Channel_pool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(1, 1, eps=1e-5, momentum=0.01, affine=True)
            nn.ReLU()
        )
    def forward(self, input):
        compressed = self.compress(input)
        spatial = self.spatial(compressed)
        scale = torch.sigmoid(spatial)
        return input * scale


class DAB(nn.Module):
    def __init__(self, n_channels):
        super(DAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=True)
        )
        self.CA = CALayer(n_channels=n_channels)
        self.SA = spatial_attn_layer()           
        self.conv_1x1 = nn.Conv2d(n_channels*2, n_channels, kernel_size=1, padding=0, bias=True)     
    def forward(self, input):
        res = self.body(input)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.concat([sa_branch, ca_branch], dim=1)
        res = self.conv_1x1(res)
        res += input
        return res



class RRG(nn.Module):
    def __init__(self, n_channels=256, out_channels=256, num_DAB=5):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [DAB(n_channels=n_channels) for i in range(num_DAB)]
        self.body = nn.Sequential(*modules_body)
        self.conv_1 = nn.Conv2d(n_channels, out_channels, kernel_size=3, padding=1, bias=True)
    def forward(self, input):        
        res = self.body(input)
        res = self.conv_1(res)
        res += input
        return res
        
        

class demo_net(nn.Module):
    def __init__(self, n_channels=64, device='cuda', pretrain=True, num_species=2, weights_list=None):
        super(demo_net, self).__init__()
        self.num_species = num_species
        self.Encoder_list = []
        for i in range(num_species):
            self.Encoder_list.append(Autoencoder().to(device))                        
        if pretrain:
            for i in range(num_species):
                self.Encoder_list[i].load_state_dict(torch.load(weights_list[i]))
        for i in range(num_species):
            self.Encoder_list[i].layer = nn.Sequential()

        self.fusion_block_4 = fusion_block_4(n_channels=n_channels, num_species=num_species)

        self.ps_list_fused = []
        self.ps_list_res = []
        self.fusion_net_list = []
        self.pred_conv_list = []        
        for i in range(num_species):            
            self.ps_list_fused.append(self.make_ps(in_channels=n_channels*2, n_channels=n_channels).to(device))
            self.ps_list_res.append(self.make_ps(in_channels=n_channels, n_channels=n_channels).to(device))
            self.fusion_net_list.append(RRG(n_channels=n_channels // 2, out_channels=n_channels // 2, num_DAB=5).to(device))
            self.pred_conv_list.append(nn.Conv2d(n_channels // 2, 1, kernel_size=3, padding=1, bias=True).to(device))

            
    def make_ps(self, in_channels, n_channels, upscale_factor=2):
        ps = nn.Sequential(
            self.make_conv(n_channels=in_channels*4, out_channels=n_channels*4),
            nn.PixelShuffle(upscale_factor=upscale_factor),
            self.make_conv(n_channels=n_channels, out_channels=n_channels*2),
            nn.PixelShuffle(upscale_factor=upscale_factor)
        )
        return ps
    
    #def make_up_conv(self, n_channels, upscale_factor=2)
    
    def make_conv(self, n_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same'),
            nn.GELU(),
            nn.Conv2d(n_channels, out_channels, kernel_size=3, padding='same'),
            nn.GELU()            
        )
        return block
        
    def forward(self, input):        
        res_feature_list = []
        for i in range(self.num_species):
            res_feature_list.append(self.Encoder_list[i](input))            
        fused_feature_list = list(self.fusion_block_4(res_feature_list))                    
        up_fused_list = []        
        up_res_list = []
        out_list = []
        pred_list = []
        for i in range(self.num_species):
            up_fused_list.append(self.ps_list_fused[i](fused_feature_list[i]))
            up_res_list.append(self.ps_list_res[i](res_feature_list[i]))
            out_list.append(self.fusion_net_list[i](up_fused_list[i]))     
            pred_list.append(self.pred_conv_list[i](out_list[i]))
        output = torch.concat(pred_list, 1)
        return output

            


if __name__ == "__main__":      
    device = 'cuda'      
    if 1:
        a = torch.ones(size=(1,1,256,256), device=device, dtype=torch.float)
        model = demo_net(n_channels=64, pretrain=False, num_species=3).to(device)
        output = model(a)
        print(output.size())

    if 0:
        a = torch.ones(size=(1,1,512,512), device=device, dtype=torch.float)
        model = Autoencoder().to(device)
        output = model(a)
        print(output.size())
        #print(model.modules())
        
        '''for index, module in enumerate(model.modules()):
            if isinstance(module, nn.PixelShuffle):
                print(1)            
                break'''
        empty = nn.Sequential()
        model.layer = empty
        output = model(a)
        print(output.size())


    

    
    
            
    