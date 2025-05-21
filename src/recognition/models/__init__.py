from .resnet18 import *
from .efficientnet_b0 import *
from .yarn_sim import *
from .yarn_sim2 import *


__all__ = [ 'models_builder' ]


models_builder = {
    'efficientnet': net_from_efficientnet_b0,
    'resnet': net_from_resnet18,
    'resnet_2': net_from_resnet18_2,
    'yarn_sim': make_simple_model,
    'yarn_sim2': make_simple_model2,
}