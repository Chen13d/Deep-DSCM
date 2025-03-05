import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # 中间层
        self.middle = self.conv_block(512, 1024)
        
        # 解码器部分
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)

        self.block4 = self.conv_block(1024, 512)
        self.block3 = self.conv_block(512, 256)
        self.block2 = self.conv_block(256, 128)
        self.block1 = self.conv_block(128, 64)
        
        # 输出层
        self.output = nn.Conv3d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, 2))
        enc3 = self.encoder3(F.max_pool3d(enc2, 2))
        enc4 = self.encoder4(F.max_pool3d(enc3, 2))
        # 中间层
        middle = self.middle(F.max_pool3d(enc4, 2))
        # 解码器
        dec4 = self.decoder4(F.interpolate(middle, scale_factor=2, mode='trilinear', align_corners=True))
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.block4(dec4)
        dec3 = self.decoder3(F.interpolate(dec4, scale_factor=2, mode='trilinear', align_corners=True))
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.block3(dec3)
        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2, mode='trilinear', align_corners=True))
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.block2(dec2)
        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2, mode='trilinear', align_corners=True))
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.block1(dec1)
        # 输出
        out = self.output(dec1)
        return out


if __name__ == "__main__":
    device = "cuda"
    Input = torch.rand((1, 1, 16, 512, 512), device=device)
    model = UNet3D(in_channels=1, out_channels=3).to(device)
    output = model(Input)
    print(Input.size(), output.size())

