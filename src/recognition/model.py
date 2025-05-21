import torch.nn as nn
from models import models_builder


class YARN_Model(nn.Module):
    def __init__(self, model_name, out_dim=5):
        super().__init__()
        self.net = self.net_from_name(model_name, in_out_dim=(1, out_dim))
        self.out_dim_ = out_dim
    
    def forward(self, Xs):
        Xs = self.net(Xs)
        return Xs
    
    def num_classes(self):
        return self.out_dim_
    
    @staticmethod
    def net_from_name(name, in_out_dim):
        if name not in models_builder.keys():
            raise NotImplementedError(f"Model {name} not implemented.")
        net = models_builder[name](in_channel=in_out_dim[0], out_dim=in_out_dim[1])
        return net 