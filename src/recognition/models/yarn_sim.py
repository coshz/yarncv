import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_simple_model(in_channel=1, out_dim=5):
    return YarnModel(in_channel=in_channel, num_classes=out_dim)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, bias=False) 
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, bias=False) 
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x 
        x = self.bn1(self.conv1(x)) 
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity) 
        return x
    

class YarnModel(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,4,5,2)
        self.bn1 = nn.BatchNorm2d(4)
        self.resb1 = ResidualBlock(4)
        self.conv2 = nn.Conv2d(4,8,3,1)
        self.bn2 = nn.BatchNorm2d(8)
        self.resb2 = ResidualBlock(8)
        self.conv3 = nn.Conv2d(8,16,3,1)
        self.bn3 = nn.BatchNorm2d(16)
        self.resb3 = ResidualBlock(16)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(179776, 24)
        self.fc2 = nn.Linear(24, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resb1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resb2(x)
        x = F.relu(self.bn3(self.conv3(x)))  
        x = self.resb3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x