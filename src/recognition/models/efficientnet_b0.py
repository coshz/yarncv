import torchvision
import torch.nn as nn
import torch.nn.functional as F


def net_from_efficientnet_b0(in_channel=1, out_dim=5):
    # modify in_dim and out_dim
    net = torchvision.models.efficientnet_b0()
    net.features[0][0] = nn.Conv2d(in_channels=in_channel,out_channels=32,kernel_size=3)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, out_dim)
    return net