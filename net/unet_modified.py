import os
import torch
    
import torch.nn as nn

from torch.functional import Tensor
from net.unet import Unet


class Unet_tri(nn.Module):
    def __init__(self, input_dim, n_channels=64, num_classes=2, upscale_factor=1):
        super(Unet_tri, self).__init__()
        self.net_1 = Unet(input_dim=input_dim, n_channels=n_channels, num_classes=1, upscale_factor=upscale_factor)
        self.net_2 = Unet(input_dim=input_dim, n_channels=n_channels, num_classes=1, upscale_factor=upscale_factor)
        self.net_3 = Unet(input_dim=input_dim, n_channels=n_channels, num_classes=num_classes, upscale_factor=upscale_factor)        
    def forward(self, input):
        output_1 = self.net_1(input)
        output_2 = self.net_2(output_1)
        output_3 = self.net_3(output_2)
        return output_1, output_2, output_3

    
if __name__ == "__main__":
    input = torch.randn((1,1,32,32), dtype=torch.float).cuda()
    model = Unet_tri(input_dim=1, num_classes=2, n_channels=32, upscale_factor=2).cuda()
    output= model(input)