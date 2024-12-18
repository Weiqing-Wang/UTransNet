
from torch import nn


def create_layer(in_channels,out_channels,kernel_size,u_activation=None,convolution=nn.Conv2d,groups=1):
    layer=[]
    conv=convolution(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=kernel_size//2,groups=groups)
    layer.append(conv)
    if u_activation is not  None:
        layer.append(u_activation())
    return nn.Sequential(*layer)

def create_depthconv_layer(in_channels,kernel_size,u_activation=None,convolution=nn.Conv2d):
    layer=[]
    conv=create_layer(in_channels,in_channels,kernel_size,u_activation,convolution,groups=in_channels)
    layer.append(conv)
    return nn.Sequential(*layer)

def create_pointconv_layer(in_channels,out_channels,kernel_size,u_activation=None,convolution=nn.Conv2d):
    layer=[]
    conv=create_layer(in_channels,out_channels,kernel_size,u_activation,convolution)
    layer.append(conv)
    return nn.Sequential(*layer)

#depthpoint encoder
def create_encoder_depthpoint_layer(in_channels,out_channels,kernel_size,u_activation=None,convolution=nn.Conv2d):
    layer=[]
    conv1=create_depthconv_layer(in_channels,kernel_size,u_activation,convolution)
    layer.append(conv1)
    conv2=create_pointconv_layer(in_channels,out_channels,1,u_activation,convolution)
    layer.append(conv2)
    return nn.Sequential(*layer)
#common decoder
def create_decoder_comm_layer(in_channels,out_channels,kernel_size,u_activation=None,convolution=nn.Conv2d,res=None,final_layer=None):
    layer=[]
    if res:
        conv1=create_layer(in_channels*2,in_channels,kernel_size,u_activation,convolution)
    else:
        conv1=create_layer(in_channels,in_channels,kernel_size,u_activation,convolution)
    layer.append(conv1)
    if final_layer:
        u_activation = None
    conv2 = create_layer(in_channels, out_channels, kernel_size, u_activation, convolution)
    layer.append(conv2)

    return nn.Sequential(*layer)