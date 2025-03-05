from torchvision import transforms
from dataset.prepare_data import *

def get_crop_params(img_size, output_size):
    h, w = img_size
    th = output_size
    tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w
    i = randint(0, h - th)
    j = randint(0, w - tw)
    #print(h, w, h - th, w - tw, i, j)
    #print(i, j, th, tw)
    return i, j, th, tw


def rand_crop_single(Input, GT, size):
    i,j,height,width = get_crop_params(img_size=Input.size()[1:3], output_size=size) 
    Input = Input[:,i:i+height, j:j+width]
    GT = GT[:,i:i+height, j:j+width]
    return Input, GT


class synthetic_dataset(Dataset):
    def __init__(self, read_dir, num_file, num_org, org_list, device):
        super(synthetic_dataset, self).__init__()
        self.read_dir = read_dir
        self.num_file = num_file
        self.num_org = num_org
        self.org_list = org_list
        self.device = device
        self.generate_read_dir()

        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.to(torch.float32))
        ])

    def generate_read_dir(self):
        self.Input_dir = os.path.join(self.read_dir, "Input")
        self.GT_DS_dir = os.path.join(self.read_dir, "GT_DS")
        self.GT_D_dir = os.path.join(self.read_dir, "GT_D")
        self.GT_S_dir = os.path.join(self.read_dir, "GT_S")

    def __len__(self):
        return self.num_file
    
    
    def map_values(self, image, new_min=0, new_max=1, min_val=None, max_val=None, percentile=100, index=0):
        if index == 0:
            # 计算指定百分位数的最小值和最大值
            min_val = torch.quantile(image, (100 - percentile) / 100)
            max_val = torch.quantile(image, percentile / 100)
            # 避免除以零的情况
            if max_val == min_val:
                raise ValueError("最大值和最小值相等，无法进行归一化。")
            
        # 将图像值缩放到新范围
        scaled = (image - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
        
        # 可选：将值限制在新范围内
        #scaled = torch.clamp(scaled, min=new_min, max=new_max)
        
        return scaled, min_val, max_val
    
    def norm_statistic(self, Input, std=None):        
        mean = torch.mean(Input).to(self.device)
        mean_zero = torch.zeros_like(mean).to(self.device)
        std = torch.std(Input).to(self.device) if std == None else std
        output = transforms.Normalize(mean_zero, std)(Input)
        return output, mean_zero, std
    
    def __getitem__(self, index):
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=int(random()>0.5))
        self.vertical_flip = transforms.RandomVerticalFlip(p=int(random()>0.5))   
        Input = self.transform(self.vertical_flip(self.horizontal_flip(Image.open(os.path.join(self.Input_dir, f"{index+1}.tif"))))).to(self.device)
        GT_DS_list = []
        GT_D_list = []
        for i in range(self.num_org):
            GT_DS_list.append(self.transform(self.vertical_flip(self.horizontal_flip(Image.open(os.path.join(self.GT_DS_dir, f"{index+1}_{self.org_list[i]}.tif"))))).to(self.device))
            #GT_D_list.append(self.transform(Image.open(os.path.join(self.GT_D_dir, f"{index+1}_{self.org_list[i]}.tif"))).to(self.device))
        #GT_S = self.transform(Image.open(os.path.join(self.GT_S_dir, f"{index+1}.tif"))).to(self.device)

        GT_DS = torch.concatenate([*GT_DS_list], dim=0)
        #GT_D = torch.concatenate([*GT_D_list], dim=0)
        #print(Input.size(), GT_DS.size(), GT_D.size(), GT_S.size())
        Input, GT_DS = rand_crop_single(Input=Input, GT=GT_DS, size=512)
        # normalizations
        Input, Input_val_min, Input_val_max = self.map_values(Input)
        GT_DS, _, _ = self.map_values(GT_DS, min_val=Input_val_min, max_val=Input_val_max, index=1)
        Input, Input_mean, Input_std = self.norm_statistic(Input)
        GT_DS, GT_DS_mean, GT_DS_std = self.norm_statistic(GT_DS, Input_std)
        # generate statistic dict for validation
        statistic_dict = {
            "Input_mean":Input_mean, "Input_std":Input_std
            }
        #return Input, GT_DS, GT_D, GT_S, statistic_dict
        return Input, GT_DS, 0, 0, statistic_dict
    

def gen_prepared_dataloader(read_dir_train, read_dir_val, num_file_train, num_file_val, num_org, org_list, batch_size, device):
    train_dataset = synthetic_dataset(read_dir=read_dir_train, num_file=num_file_train, num_org=num_org, org_list=org_list, device=device)
    val_dataset = synthetic_dataset(read_dir=read_dir_val, num_file=num_file_val, num_org=num_org, org_list=org_list, device=device)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1)
    return train_dataloader, val_dataloader, num_file_train
