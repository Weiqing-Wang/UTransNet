from torch import nn
import torch.nn.functional as F

from .layer import *

def create_encoder(in_channels,filters,kernel_size,u_activation=nn.LeakyReLU,convolution=nn.Conv2d):
    encoder=[]
    for i in range(len(filters)):
        if i == 0:
            encoder_layer = create_encoder_depthpoint_layer(in_channels,filters[i],kernel_size,u_activation,convolution)
        else:
            encoder_layer = create_encoder_depthpoint_layer(filters[i - 1],filters[i],kernel_size,u_activation,convolution)
        encoder = encoder + [encoder_layer]
    return nn.Sequential(*encoder)
def encoder_forward(x,encoders):
    tensors = []
    indices = []
    sizes = []
    for encode in encoders:
        x = encode(x)
        sizes.append(x.size())
        tensors.append(x)
        x, indice = F.max_pool2d(x, 2, 2, return_indices=True)
        indices.append(indice)
    return x, tensors, indices, sizes

