import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class FCALayer(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, padding=1, reduction=16):
        super(FCALayer, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=1, stride=1, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels//reduction, kernel_size=kernel_size, padding=1, stride=1, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=n_channels//reduction, out_channels=n_channels, kernel_size=kernel_size, padding=1, stride=1, bias=False)
        self.average_pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, input):
        fft_origin = torch.fft.fft2(input)
        fft_shift = torch.fft.fftshift(fft_origin)
        fft_shift = torch.abs(fft_shift)
        conv = self.conv_1(fft_shift)
        conv = self.average_pooling(conv)
        conv = self.conv_2(conv)
        conv = self.conv_3(conv)
        mul = input * conv
        return mul

class FCAB(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, padding=1):
        super(FCAB, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU()
        )
        self.attention_layer = FCALayer(n_channels=n_channels)
        self.bn = nn.BatchNorm2d(n_channels)
    def forward(self, input):
        conv = self.conv_block_1(input)
        att = self.attention_layer(conv)
        output = att + input
        output = self.bn(output)
        return output


class ResidualGroup(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, padding=1, n_RCAB=4):
        super(ResidualGroup, self).__init__()
        RCAB_list = []
        #RCAB_list.append(FCAB(n_channels=n_channels))
        for i in range(n_RCAB):            
            RCAB_list.append(FCAB(n_channels=n_channels))            
        #RCAB_list.append(FCAB(n_channels=n_channels))
        self.Res_FCAB = nn.Sequential(*RCAB_list)
    def forward(self, input):
        #print(input.size(), 'RCAB')
        return self.Res_FCAB(input)


class DFCAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_channels=64, kernel_size=3, padding=1, n_RCAB=4, n_ResGroup=4, scale=2):
        super(DFCAN, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU()
            )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding),# * (scale ** 2)
            nn.GELU()
            )
        ResGroup_list = []
        for i in range(n_ResGroup):
            ResGroup_list.append(ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB))
        self.ResGroups = nn.Sequential(*ResGroup_list)
        self.ps = nn.PixelShuffle(upscale_factor=scale)
        #self.pred_conv = nn.Conv2d(in_channels=n_channels//4, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.pred_conv = nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, input):
        conv = self.conv_block_1(input)
        conv_DFCAN = self.ResGroups(conv)
        conv = self.conv_block_2(conv_DFCAN)
        #print(conv.size())
        #up = self.ps(conv)
        #print(up.size())
        out = self.pred_conv(conv)
        #out = self.pred_conv(up)
        #print(out.size(), conv_DFCAN.size())        
        return out#, conv_DFCAN


if __name__ == "__main__":
    input = torch.randn((1,1,32,32), dtype=torch.float).cuda()
    model = DFCAN(in_channels=1, out_channels=2, n_channels=32).cuda()
    output= model(input)
    #print(output.size())


class F_block(nn.Module):
    def __init__(self, out_channels, n_channels=64, kernel_size=3, padding=1, n_RCAB=4, n_ResGroup=4, scale=2):
        super(F_block, self).__init__()
        ResGroup_list = []        
        ResGroup_list.append(ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB))     
        ResGroup_list.append(self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2))
        n_channels = n_channels // 2  
        for i in range(n_ResGroup-2):            
            ResGroup_list.append(ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB))            
            ResGroup_list.append(self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2))
            n_channels = n_channels // 2
        ResGroup_list.append(ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB))            
        ResGroup_list.append(self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels))
        self.ResGroups = nn.Sequential(*ResGroup_list)        
        self.pred_conv = nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=True)

    def make_up_conv_block(self, inchannels, outchannels, kernel_size=3, padding=1, stride=1, dilation=1, use_bias=False):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=2, padding=0, stride=2, bias=False),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=False)            
        )
        return block  

    def forward(self, x):
        #x =         
        fea_S = self.ResGroups(x)
        #print(fea_S.size())
        Output = self.pred_conv(fea_S)
        #print(Output.size())
        return Output, fea_S
    

class F_blocks(nn.Module):
    def __init__(self, out_channels, n_channels=64, kernel_size=3, padding=1, n_RCAB=4, n_ResGroup=4, scale=2):
        super(F_blocks, self).__init__()
        self.Res_1 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_1 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2)
        n_channels = n_channels // 2
        self.Res_2 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_2 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2)
        n_channels = n_channels // 2
        self.Res_3 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_3 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2)
        n_channels = n_channels // 2
        self.Res_4 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_4 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2)
        n_channels = n_channels // 2
        self.Res_5 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)
        self.up_5 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels)

        self.pred_conv = nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=True)

    def make_up_conv_block(self, inchannels, outchannels, kernel_size=3, padding=1, stride=1, dilation=1, use_bias=False):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=2, padding=0, stride=2, bias=False),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=False)            
        )
        return block  

    def forward(self, x):
        res_list = []
        res_1 = self.Res_1(x)
        res_list.append(res_1)
        res_1 = self.up_1(res_1)
        res_2 = self.Res_2(res_1)
        res_list.append(res_2)
        res_2 = self.up_2(res_2)
        res_3 = self.Res_3(res_2)
        res_list.append(res_3)
        res_3 = self.up_3(res_3)
        res_4 = self.Res_4(res_3)
        res_list.append(res_4)
        res_4 = self.up_4(res_4)
        res_5 = self.Res_5(res_4)
        res_5 = self.up_5(res_5)
        Output = self.pred_conv(res_4)
        #print(Output.size())
        return Output, res_list
        

class F_block_3(nn.Module):
    def __init__(self, out_channels, n_channels=64, kernel_size=3, padding=1, n_RCAB=4, n_ResGroup=4, scale=2):
        super(F_block_3, self).__init__()
        self.Res_1 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_1 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2)
        n_channels = n_channels // 2
        self.Res_2 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_2 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2)
        n_channels = n_channels // 2
        self.Res_3 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_3 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2)
        n_channels = n_channels // 2
        self.Res_4 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_4 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels//2)        
        n_channels = n_channels // 2
        self.Res_5 = ResidualGroup(n_channels=n_channels, n_RCAB=n_RCAB)   
        self.up_5 = self.make_up_conv_block(inchannels=n_channels, outchannels=n_channels) 

        self.pred_conv = nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=True)

    def make_up_conv_block(self, inchannels, outchannels, kernel_size=3, padding=1, stride=1, dilation=1, use_bias=False):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=2, padding=0, stride=2, bias=False),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=False)            
        )
        return block  

    def forward(self, x):
        res_list = []
        res_1 = self.Res_1(x)
        res_list.append(res_1)
        res_1 = self.up_1(res_1)
        res_2 = self.Res_2(res_1)
        res_list.append(res_2)
        res_2 = self.up_2(res_2)
        res_3 = self.Res_3(res_2)
        res_list.append(res_3)
        res_3 = self.up_3(res_3)
        res_4 = self.Res_4(res_3)
        res_list.append(res_4)
        res_4 = self.up_4(res_4)
        res_5 = self.Res_5(res_4)
        res_list.append(res_5)
        #res_5 = self.up_5(res_5)
        #print(fea_S.size())
        Output = self.pred_conv(res_5)
        #print(Output.size())
        return Output, res_list





