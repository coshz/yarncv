import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd 
from PIL import Image
from .utils import img_transform


class YarnDataset(Dataset):
    def __init__(self, data_dir, ds_csv, transform=img_transform()):
        super().__init__()
        self.data_dir = data_dir
        self.data = pd.read_csv(ds_csv, header=0)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, img_label = self.data.loc[index].iloc[0], self.data.loc[index].iloc[1]
        
        # apply transform
        img = self.transform(Image.open(os.path.join(self.data_dir, img_path)))

        # skip label-3
        if img_label > 3: img_label -= 1
        label = torch.tensor(img_label, dtype=torch.long)

        return img, label


def create_yarn_dataloaders(data_dir, train_csv, test_csv, batch_size):
    dataset_train = YarnDataset(data_dir, train_csv)
    dataset_test = YarnDataset(data_dir, test_csv)

    train_set, val_set = random_split(
        dataset_train, 
        [int(len(dataset_train)*0.9), len(dataset_train) - int(len(dataset_train)*0.9)],
        torch.Generator().manual_seed(42)
    )
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        ,'val': DataLoader(val_set,batch_size=batch_size, num_workers=2)
        ,'test': DataLoader(dataset_test,batch_size=1)
    }
    return dataloaders