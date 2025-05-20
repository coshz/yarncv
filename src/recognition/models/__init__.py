from .resnet18 import *
from .efficientnet_b0 import *

__all__ = [ 'models_builder' ]


models_builder = {
    'efficientnet': net_from_efficientnet_b0,
    'resnet': net_from_resnet18
}