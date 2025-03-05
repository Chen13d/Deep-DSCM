import torch
from torch import nn
import torch.nn.functional as F


class Get_gradient(nn.Module):
    def __init__(self, device, num_species=2, kernel_size=5):
        super(Get_gradient, self).__init__()
        self.num_species = num_species
        if kernel_size == 3:
            self.padding = 1
            kernel_v = [[0, -1, 0], 
                        [0, 0, 0], 
                        [0, 1, 0]]
            kernel_h = [[0, 0, 0], 
                        [-1, 0, 1], 
                        [0, 0, 0]]
        elif kernel_size == 5:
            self.padding = 2
            kernel_v = [[0, 0, -10, 0, 0],
                        [0, 0, -1, 0, 0], 
                        [0, 0, 0, 0, 0], 
                        [0, 0, 1, 0, 0], 
                        [0, 0, 10, 0, 0]]
            kernel_h = [[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0], 
                        [-10, -1, 0, 1, 10], 
                        [0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0]]           
        elif kernel_size == 7:
            self.padding = 3
            kernel_v = [[0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, -10, 0, 0, 0],
                        [0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0]]
            kernel_h = [[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [-1, -10, -1, 0, 1, 10, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]

        
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).to(device)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).to(device)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).to(device)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).to(device)
    def forward(self, x):
        if self.num_species == 1:
            x_v = F.conv2d(x, self.weight_v, padding=self.padding)
            x_h = F.conv2d(x, self.weight_h, padding=self.padding)
            grad = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        elif self.num_species == 2:
            x0 = x[:,0:1,:,:]
            x1 = x[:,1:2,:,:]
            #print(x.size(), x0.size())
            x0_v = F.conv2d(x0, self.weight_v, padding=self.padding)
            x0_h = F.conv2d(x0, self.weight_h, padding=self.padding)
            x1_v = F.conv2d(x1, self.weight_v, padding=self.padding)
            x1_h = F.conv2d(x1, self.weight_h, padding=self.padding)
            x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
            x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
            grad = torch.cat([x0, x1], dim=1)
        elif self.num_species == 3:
            x0 = x[:,0:1,:,:]
            x1 = x[:,1:2,:,:]
            x2 = x[:,2:3,:,:]
            x0_v = F.conv2d(x0, self.weight_v, padding=self.padding)
            x0_h = F.conv2d(x0, self.weight_h, padding=self.padding)
            x1_v = F.conv2d(x1, self.weight_v, padding=self.padding)
            x1_h = F.conv2d(x1, self.weight_h, padding=self.padding)
            x2_v = F.conv2d(x2, self.weight_v, padding=self.padding)
            x2_h = F.conv2d(x2, self.weight_h, padding=self.padding)
            x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
            x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
            x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)
            grad = torch.cat([x0, x1, x2], dim=1)
        #grad = x*grad
        return grad
    
class Get_grad_std(nn.Module):
    def __init__(self, device, num_classes, kernel_size=3, blur_kernel_size=16, blur_kernel_std=3):
        super(Get_grad_std, self).__init__()        
        self.padding = kernel_size // 2
        self.blur_padding = blur_kernel_size // 2
        self.num_classes = num_classes
        if kernel_size == 3:
            kernel_f = [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
            kernel_b = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0]
            ]
            kernel_l = [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0]
            ]
            kernel_r = [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]
            ] 
            kernel_fl = [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
            kernel_fr = [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]
            ]
            kernel_bl = [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0]
            ]
            kernel_br = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]
            ]
        elif kernel_size == 5:
            kernel_f = [
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
            kernel_b = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0]
            ]
            kernel_l = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
            kernel_r = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
            kernel_fl = [
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
            kernel_fr = [
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
            kernel_bl = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0]
            ]
            kernel_br = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0]
            ]
        kernel_f = torch.FloatTensor(kernel_f).unsqueeze(0).unsqueeze(0).to(device)
        kernel_b = torch.FloatTensor(kernel_b).unsqueeze(0).unsqueeze(0).to(device)
        kernel_l = torch.FloatTensor(kernel_l).unsqueeze(0).unsqueeze(0).to(device)
        kernel_r = torch.FloatTensor(kernel_r).unsqueeze(0).unsqueeze(0).to(device)
        kernel_fl = torch.FloatTensor(kernel_fl).unsqueeze(0).unsqueeze(0).to(device)
        kernel_fr = torch.FloatTensor(kernel_fr).unsqueeze(0).unsqueeze(0).to(device)
        kernel_bl = torch.FloatTensor(kernel_bl).unsqueeze(0).unsqueeze(0).to(device)
        kernel_br = torch.FloatTensor(kernel_br).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_f = nn.Parameter(data = kernel_f, requires_grad = False).to(device)
        self.weight_b = nn.Parameter(data = kernel_b, requires_grad = False).to(device)
        self.weight_l = nn.Parameter(data = kernel_l, requires_grad = False).to(device)
        self.weight_r = nn.Parameter(data = kernel_r, requires_grad = False).to(device)
        self.weight_fl = nn.Parameter(data = kernel_fl, requires_grad = False).to(device)
        self.weight_fr = nn.Parameter(data = kernel_fr, requires_grad = False).to(device)
        self.weight_bl = nn.Parameter(data = kernel_bl, requires_grad = False).to(device)
        self.weight_br = nn.Parameter(data = kernel_br, requires_grad = False).to(device)

        self.blur_kernel = nn.Parameter(data=self.gaussian_kernel(kernel_size=blur_kernel_size, sigma=blur_kernel_std), requires_grad=False).to(device)

    def gaussian_kernel(self, kernel_size, sigma):
        # 创建一个以中心为原点的二维网格
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        grid = torch.meshgrid(x, y)
        grid = torch.stack(grid)

        # 计算高斯分布的权重
        variance = sigma**2
        gaussian = torch.exp(-torch.sum(grid**2, dim=0) / (2*variance))
        gaussian = gaussian / gaussian.sum()  # 归一化

        # 转换为卷积核的形状
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        gaussian = gaussian.repeat(1, self.num_classes, 1, 1)
        return gaussian
        
    def forward(self, x):
        #res = x.detach()
        #print(x.size())
        #x = F.conv2d(x, self.blur_kernel, padding=self.blur_padding)
        x_f = F.conv2d(x, self.weight_f, padding=self.padding)
        x_b = F.conv2d(x, self.weight_b, padding=self.padding)
        x_l = F.conv2d(x, self.weight_l, padding=self.padding)
        x_r = F.conv2d(x, self.weight_r, padding=self.padding)
        x_fl = F.conv2d(x, self.weight_fl, padding=self.padding)
        x_fr = F.conv2d(x, self.weight_fr, padding=self.padding)
        x_bl = F.conv2d(x, self.weight_bl, padding=self.padding)
        x_br = F.conv2d(x, self.weight_br, padding=self.padding)
        x_grad = torch.concat([x_f, x_b, x_l, x_r, x_fl, x_fr, x_bl, x_br], 1)
        #print(x_grad.size())
        x_grad_std = torch.std(x_grad, 1)
        #print(torch.max(x_grad_std))
        return x_grad_std


    
if __name__ == "__main__":
    from utils import *
    import tifffile
    device = 'cuda'
    image = np.array(Image.open(r'D:\CQL\codes\microscopy_decouple\data\STED_data\simulation\Lysosome\STED_0.tif'))
    img = torch.tensor(np.float64(image), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
    get_gradstd = Get_grad_std(kernel_size=3, blur_kernel_size=7, blur_kernel_std=3)
    #get_gradstd(img)
    size = 512
    gt_list = []
    pred_list = []
    stack = tifffile.imread(r"C:\Users\20151\Desktop\新建文件夹 (3)\700-1.tif")
    stack = torch.tensor(np.float64(stack), dtype=torch.float, device=device)
    for index_frame in range(stack.shape[0]):
        img = stack[index_frame,:,:]
        for i in range(3):
            gt_list.append(img[0:size, (i+1)*size:(i+2)*size].unsqueeze(0).unsqueeze(0))
            pred_list.append(img[size:2*size, (i+1)*size:(i+2)*size].unsqueeze(0).unsqueeze(0))
    
    index = 0
    get_gradstd(pred_list[index])
    get_gradstd(gt_list[index])

    plt.show()
    """get_grad = Get_gradient(device='cuda', num_species=1, kernel_size=3)
    grad = get_grad(img)
    #img = to_cpu(img.squeeze(0).squeeze(0))
    grad = to_cpu(grad.squeeze(0).squeeze(0))
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(grad)
    plt.show()"""


