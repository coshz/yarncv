import torch
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import pandas as pd 
from PIL import Image


class YARN_Dataset(Dataset):
    def __init__(self, data_dir, ds_csv, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.data = pd.read_csv(ds_csv, header=0)
        # self.data = self.prepare_data(data_dir, train)
        # self.transform = transform if transform else self.simple_transform_
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, img_label = self.data.loc[index].iloc[0], self.data.loc[index].iloc[1]
        
        # grayout, resize, toTensor
        img = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])(Image.open(os.path.join(self.data_dir, img_path)))
        if img_label > 3: img_label -= 1
        label = torch.tensor(img_label, dtype=torch.long)

        # apply transform 
        if self.transform: img = self.transform(img)

        return img, label