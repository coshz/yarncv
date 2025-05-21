import torch
import torch.nn as nn
import torch.nn.functional as F


def make_simple_model2(in_channel=1, out_dim=5):
    return YarnModel2(in_channel=in_channel, num_classes=out_dim)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x 
        x = self.bn1(self.conv1(x)) 
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity) 
        return x
    

class YarnModel2(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,16,5,2)
        self.bn1 = nn.BatchNorm2d(16)
        self.resb1 = ResidualBlock(16)
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.bn2 = nn.BatchNorm2d(32)
        self.resb2 = ResidualBlock(32)
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.bn3 = nn.BatchNorm2d(64)
        self.resb3 = ResidualBlock(64)
        self.global_pool = nn.AdaptiveAvgPool2d((16,16))
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64*16*16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resb1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resb2(x)
        x = F.relu(self.bn3(self.conv3(x)))  
        x = self.resb3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x