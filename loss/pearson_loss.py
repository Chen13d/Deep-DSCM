import torch 
from torch import nn


class Pearson_loss(nn.Module):
    def __init__(self):
        super(Pearson_loss, self).__init__()
    def forward(self, img_1, img_2):
        mean_1 = torch.mean(img_1)
        mean_1 = torch.mean(img_2)
        std_1 = torch.std(img_1)
        std_2 = torch.std(img_2)
        covariance = torch.mean((img_1 - mean_1) * (img_2 - mean_1))
        pearson_corr = covariance / (std_1 * std_2)
        return pearson_corr


