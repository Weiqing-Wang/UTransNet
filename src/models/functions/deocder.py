import torch
from torch import nn
import torch.nn.functional as F
from .layer import *

def create_decoder(out_channels,filters,kernel_size,u_activation=nn.LeakyReLU,convolution=nn.ConvTranspose2d,res_type=True):
    decoder=[]
    for i in range(len(filters)):
        if i ==0:
            decoder_layer=create_decoder_comm_layer(filters[i],out_channels,kernel_size,u_activation,convolution,res=res_type,final_layer=True)
        else:
            decoder_layer = create_decoder_comm_layer(filters[i],filters[i-1],kernel_size,u_activation,convolution,res=res_type,final_layer=False)
        decoder = [decoder_layer] + decoder
    return nn.Sequential(*decoder)
def decode_forwrd(x,tensors,indices,sizes,decoders,res_type=True):
    for decoder in decoders:
        tensor = tensors.pop()
        size = sizes.pop()
        ind = indices.pop()
        x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
        if res_type==True:
            x = torch.cat([tensor, x], dim=1)
        x = decoder(x)
    return x
#three out_channels
def decode_forward_ex(x, tensors, indices, sizes, decoders,res_type=True):
    y = []
    for _decoder in decoders:
        _x = x
        _tensors = tensors[:]
        _indices = indices[:]
        _sizes = sizes[:]
        for decoder in _decoder.children():
            tensor = _tensors.pop()
            size = _sizes.pop()
            ind = _indices.pop()
            _x = F.max_unpool2d(_x, ind, 2, 2, output_size=size)
            if res_type==True:
                _x = torch.cat([tensor, _x], dim=1)
            _x = decoder(_x)
        y.append(_x)
    out=torch.cat(y, dim=1)
    return out


