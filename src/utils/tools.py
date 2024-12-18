import random

import numpy as np
import torch
from torch import nn
from torch.nn.init import trunc_normal_


def init_parameters(model):
    if isinstance(model, nn.Linear):
        trunc_normal_(model.weight, std=0.02)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.LayerNorm):
        nn.init.constant_(model.bias, 0)
        nn.init.constant_(model.weight, 1.0)
    elif isinstance(model, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(model.weight, 1)
        if model.bias is not None:
            nn.init.normal_(model.bias, mean=0, std=0.02)

def initialize(model, gain=1, std=0.02):
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)