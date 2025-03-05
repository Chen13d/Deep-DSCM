# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.fft

def fftshift2d(img, size_psc=128):
    bs,ch, h, w = img.shape
    fs11 = img[:,:, h//2:, w//2:]
    fs12 = img[:,:, h//2:, :w//2]
    fs21 = img[:,:, :h//2, w//2:]
    fs22 = img[:,:, :h//2, :w//2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    # output = tf.image.resize_images(output, (size_psc, size_psc), 0)
    return output


class RCAB(nn.Module):
    def __init__(self): #size_psc：crop_size input_shape：depth
        super().__init__()
        self.conv_gelu1=nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                               nn.GELU())
        self.conv_gelu2=nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                       nn.GELU())


        self.conv_relu1=nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                               nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_relu2=nn.Sequential(nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0),
                               nn.ReLU())
        self.conv_sigmoid=nn.Sequential(nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0),
                               nn.Sigmoid())

    def forward(self,x,gamma=0.8):
        x0=x
        x  = self.conv_gelu1(x)
        x  = self.conv_gelu2(x)
        x1 = x
        x  = torch.fft.fftn(x,dim=(2,3))
        x  = torch.pow(torch.abs(x)+1e-8, gamma) #abs
        x  = fftshift2d(x)
        x  = self.conv_relu1(x)
        x  = self.avg_pool(x)
        x  = self.conv_relu2(x)
        x  = self.conv_sigmoid(x)
        x  = x1*x
        x  = x0+x
        return x


class ResGroup(nn.Module):
    def __init__(self, n_RCAB=4): #size_psc：crop_size input_shape：depth
        super().__init__()
        RCABs=[]
        for _ in range(n_RCAB):
            RCABs.append(RCAB())
        self.RCABs=nn.Sequential(*RCABs)

    def forward(self,x):
        x0=x
        x=self.RCABs(x)
        x=x0+x
        return x


class DFCAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_channels=64, n_ResGroup=4, n_RCAB=4, scale=2): #size_psc：crop_size input_shape：depth
        super().__init__()
        self.input=nn.Sequential(nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
                                       nn.GELU(),)
        ResGroups=[]
        for _ in range(n_ResGroup):
            ResGroups.append(ResGroup(n_RCAB=n_RCAB))
        self.RGs  =nn.Sequential(*ResGroups)
        self.conv_gelu=nn.Sequential(nn.Conv2d(n_channels, n_channels*(scale ** 2), kernel_size=3, stride=1, padding=1),
                                       nn.GELU())
        self.scale = scale
        if scale == 2:
            self.pixel_shuffle = nn.PixelShuffle(scale)        
        self.conv_sigmoid=nn.Sequential(nn.Conv2d(n_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
                                       #nn.Sigmoid())
    def forward(self,x):
        x=self.input(x)
        x=self.RGs(x)
        x=self.conv_gelu(x)
        if self.scale==2: x = self.pixel_shuffle(x)        
        x=self.conv_sigmoid(x)
        return x


if __name__ == '__main__':
    #x = Variable(torch.rand(2,1,64,64)).cuda()
    device = 'cuda'
    x=torch.rand(1,1,512,512).to(device)

    #model = UNet().cuda()
    model = DFCAN(in_channels=1, out_channels=1, scale=1).to(device)
    # model.eval()
    y = model(x)
    print('Output shape:',y.size())
    # import hiddenlayer as hl
