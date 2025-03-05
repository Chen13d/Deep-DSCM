import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import Tensor


class Unetplusplus(nn.Module):
    def __init__(self, input_dim, num_classes=1) -> None:
        super(Unetplusplus, self).__init__()
        self.num_classes = num_classes

        self.conv_block_11 = self.make_block(inchannels=input_dim, outchannels=32, kernel_size=3, padding=1)
        self.conv_block_21 = self.make_block(inchannels=32, outchannels=64, kernel_size=3, padding=1)
        self.conv_block_31 = self.make_block(inchannels=64, outchannels=128, kernel_size=3, padding=1)
        self.conv_block_41 = self.make_block(inchannels=128, outchannels=256, kernel_size=3, padding=1)
        self.conv_block_51 = self.make_block(inchannels=256, outchannels=512, kernel_size=3, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.conv_T_42 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_T_33 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_T_24 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_T_15 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, padding=0, stride=2, bias=False)

        self.conv_T_32 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_T_23 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_T_14 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_T_22 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_T_13 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_T_12 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, padding=0, stride=2, bias=False)
        
        self.conv_block_42 = self.make_block(inchannels=512, outchannels=256, kernel_size=3, padding=1)
        self.conv_block_32 = self.make_block(inchannels=256, outchannels=128, kernel_size=3, padding=1)
        self.conv_block_33 = self.make_block(inchannels=384, outchannels=128, kernel_size=3, padding=1)
        self.conv_block_22 = self.make_block(inchannels=128, outchannels=64, kernel_size=3, padding=1)
        self.conv_block_23 = self.make_block(inchannels=192, outchannels=64, kernel_size=3, padding=1)
        self.conv_block_24 = self.make_block(inchannels=256, outchannels=64, kernel_size=3, padding=1)
        self.conv_block_12 = self.make_block(inchannels=64, outchannels=32, kernel_size=3, padding=1)
        self.conv_block_13 = self.make_block(inchannels=96, outchannels=32, kernel_size=3, padding=1)
        self.conv_block_14 = self.make_block(inchannels=128, outchannels=32, kernel_size=3, padding=1)
        self.conv_block_15 = self.make_block(inchannels=160, outchannels=32, kernel_size=3, padding=1)
        
        
        self.pred_conv = nn.Conv2d(in_channels=32, out_channels=self.num_classes, kernel_size=1, padding=0, bias=True)

    def make_block(self, inchannels, outchannels, kernel_size, padding, use_bias = False):
        block = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=None),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=False)
        )


        return block
    
    def forward(self, input):
        conv_11 = self.conv_block_11(input)
        pool_12 = self.pool_1(conv_11)
        # ===============================================
        conv_21 = self.conv_block_21(pool_12)
        pool_23 = self.pool_2(conv_21)
        # ===============================================
        conv_31 = self.conv_block_31(pool_23)
        pool_34 = self.pool_3(conv_31)
        # ===============================================
        conv_41 = self.conv_block_41(pool_34)
        pool_45 = self.pool_4(conv_41)
        # ===============================================
        conv_51 = self.conv_block_51(pool_45)
        # ===============================================
        up_42 = self.conv_T_42(conv_51)
        concat_42 = torch.concat([up_42, conv_41], 1)
        conv_42 = self.conv_block_42(concat_42)
        # ===============================================
        up_32 = self.conv_T_32(conv_41)
        concat_32 = torch.concat([conv_31, up_32], 1)
        conv_32 = self.conv_block_32(concat_32)
        up_33 = self.conv_T_33(conv_42)
        concat_33 = torch.concat([conv_31, conv_32, up_33], 1)
        conv_33 = self.conv_block_33(concat_33)
        # ===============================================
        up_22 = self.conv_T_22(conv_31)
        concat_22 = torch.concat([conv_21, up_22], 1)
        conv_22 = self.conv_block_22(concat_22)
        up_23 = self.conv_T_23(conv_33)
        concat_23 = torch.concat([conv_21, conv_22, up_23], 1)
        #print(concat_23.size())
        conv_23 = self.conv_block_23(concat_23)
        up_24 = self.conv_T_24(conv_33)
        concat_24 = torch.concat([conv_21, conv_22, conv_23, up_24], 1)
        conv_24 = self.conv_block_24(concat_24)
        # ===============================================
        up_12 = self.conv_T_12(conv_21)
        concat_12 = torch.concat([conv_11, up_12], 1)
        conv_12 = self.conv_block_12(concat_12)
        up_13 = self.conv_T_13(conv_22)
        concat_13 = torch.concat([conv_11, conv_12, up_13], 1)
        conv_13 = self.conv_block_13(concat_13)
        up_14 = self.conv_T_14(conv_23)
        concat_14 = torch.concat([conv_11, conv_12, conv_13, up_14], 1)
        conv_14 = self.conv_block_14(concat_14)
        up_15 = self.conv_T_15(conv_24)
        concat_15 = torch.concat([conv_11, conv_12, conv_13, conv_14, up_15], 1)
        conv_15 = self.conv_block_15(concat_15)
        #print(conv_15.size())
        # ===============================================
        pred_conv = self.pred_conv(conv_15)
        #output = nn.Softmax(dim=1)(pred_conv)
        #print(output.size())
        # ===============================================

        # ===============================================

        # ===============================================

        # ===============================================

        # ===============================================

        # ===============================================

        #print(output.size())
        return pred_conv

     

    
    
if __name__ == "__main__":

    #train()
    #n = 30
    #data_predict_and_storage(weight_name=r'D:\codes\aggregation\weights\weight_Unetplusplus_%d.pth'%n, size=576)

    #train_()

    #data_predict_and_storage(weight_name=r'D:\codes\aggregation\data\test\weights_\weight_Unetplusplus_%d.pth'%8, size=576)

    print("test")