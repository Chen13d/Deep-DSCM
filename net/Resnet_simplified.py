import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)            
        )    
        self.ReLU = nn.ReLU()
        self.downsample = downsample
    def forward(self, input):
        res = input
        if self.downsample is not None:
            res = self.downsample(input)
        out = self.res_block(input)
        out += res
        out = self.ReLU(out)
        return out

class Res_net_gen(nn.Module):
    def __init__(self, in_channels, num_classes, block_num_list, n_channels=64, top=None, block=BasicBlock):
        super(Res_net_gen, self).__init__()    
        self.top = top
        self.in_channel = n_channels        
        self.extractor = nn.Sequential(
            nn.Sequential(
                #nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=7, padding=3, stride=2, bias=False),
                nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=False)  
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.make_layer(channel=n_channels, block=block, block_num=block_num_list[0]),
            self.make_layer(channel=n_channels*4, block=block, block_num=block_num_list[2], stride=1),
            self.make_layer(channel=n_channels*4, block=block, block_num=block_num_list[2], stride=2),
            self.make_layer(channel=n_channels*8, block=block, block_num=block_num_list[3], stride=2)
        )
        if self.top == 'reconstruction':
            self.head = nn.Sequential(
                self.make_up_conv_block(inchannels=n_channels*8, outchannels=n_channels*4),
                self.make_up_conv_block(inchannels=n_channels*4, outchannels=n_channels*2),
                self.make_up_conv_block(inchannels=n_channels*2, outchannels=n_channels*1),
                self.make_up_conv_block(inchannels=n_channels*1, outchannels=n_channels*1),
                nn.Conv2d(in_channels=n_channels*1, out_channels=num_classes, kernel_size=1, padding=0, bias=True)
            )
        elif self.top == 'classification':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
    def make_layer(self, channel, block, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample=downsample))
        #print(self.in_channel, channel)
        self.in_channel = channel * block.expansion 
        #print(self.in_channel, channel)
        for _ in range(1, block_num):
            layers.append(
                block(self.in_channel, channel)
            )
        return nn.Sequential(*layers)
    
    def make_up_conv_block(self, inchannels, outchannels, kernel_size=3, padding=1, stride=1, dilation=1, use_bias=False):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=2, padding=0, stride=2, bias=False),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=False)
        )
        return block   
    def forward(self, x):
        x = self.extractor(x)     
        #print(x.size())
        #if self.top != None:
        #    x = self.head(x)
        if self.top == 'classification':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            #x = nn.Sigmoid()(x)
        elif self.top == 'reconstruction':
            x = self.head
        return x

if __name__ == "__main__":
    device = 'cuda'
    a = torch.ones(size=(1,1,256,256), device=device, dtype=torch.float)
    #model = Decouple_Spatial_net_11_10(inchannels=1, outchannels=3, n_channels=64).to(device)
    model = Res_net_gen(in_channels=1, num_classes=6, block=BasicBlock, block_num_list=[3, 4, 6, 3], top='classification').to(device)
    b = model(a)

    #print(b.size())
