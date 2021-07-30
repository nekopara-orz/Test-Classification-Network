import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms,models,utils
from tqdm.notebook import tqdm
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim

train_path = '../kaggle/train'

test_path = '../kaggle/test'

class MyDataset(Dataset):
    def __init__(self, data_path:str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx: int):
        # img to tensor and label to tensor
        img_path = self.path_list[idx]
        if self.train_flag is True:
            if img_path.split('.')[0] == 'dog' :
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0]) # split 的是str类型要转换为int
        label = torch.as_tensor(label, dtype=torch.int64) # 必须使用long 类型数据，否则后面训练会报错 expect long
        img_path = os.path.join(self.data_path, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.path_list)

# train_ds = MyDataset(train_path)
# test_ds = MyDataset(test_path,train=False)
# for i, item in enumerate(tqdm(train_ds)):
# #     pass
#     print(item)
#     break