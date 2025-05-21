import torchvision
import torch.nn as nn
import torch.nn.functional as F


def net_from_resnet18(in_channel=1, out_dim=5):
    # modify in_dim and out_dim
    net = torchvision.models.resnet18()
    net.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(net.fc.in_features, out_dim)
    return net


def net_from_resnet18_2(in_channel=1, out_dim=5):
    # modify in_dim and out_dim
    net = torchvision.models.resnet18()
    net.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(net.fc.in_features, out_dim)
    )
    return net