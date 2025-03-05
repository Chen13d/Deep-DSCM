import torch
from torch import nn

class decouple_loss(nn.Module):
    def __init__(self, opt):
        super(decouple_loss, self).__init__()
        self.opt = opt
        if self.opt['train']['decouple_criterion'] == 'l2': self.decouple_criterion = nn.MSELoss().to(self.opt['device'])        
    def forward(self, generated, GT):               
        _, C, H, W = generated.size()
        #total_loss = torch.zeros((_, 1), device=self.opt['device'])
        total_loss = 0
        #overlapped_img_GT = torch.zeros((_, 1, H, W), device=self.opt['device'])
        overlapped_img_generated = torch.zeros((_, 1, H, W), device=self.opt['device'])
        for i in range(C):
            #overlapped_img_GT += GT[:,i,:,:]
            overlapped_img_generated += generated[:,i:i+1,:,:]
        for i in range(C):
            temp = overlapped_img_generated - generated[:,i:i+1,:,:]
            GT_temp = GT[:,-(i+1):-(i),:,:] if i+1 == 2 else GT[:,0:1,:,:]
            #print(temp.size(), GT_temp.size())
            loss = self.decouple_criterion(temp, GT_temp)
            total_loss += loss               
        return total_loss
        
