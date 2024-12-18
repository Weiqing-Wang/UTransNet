from torch import nn

from .functions.attention_stage import Stage
from .functions.deocder import create_decoder, decode_forward_ex, decode_forwrd
from .functions.encoder import encoder_forward, create_encoder
from ..utils.tools import init_parameters

class UTransNet(nn.Module):
    def __init__(self,in_channels,out_channels,filters,kernel_size,num_heads,
                 mlp_ratio,depths,a_activation,a_norm,u_activation=nn.LeakyReLU,multi_out_channels=True,atten_tye='convolution',res_type=True):
        super().__init__()
        self.encoders=create_encoder(in_channels,filters,kernel_size,u_activation)
        self.attention_stage=Stage(filters[-1],num_heads,mlp_ratio,depths,atten_tye,a_activation, a_norm)
        self.res_type=res_type
        if multi_out_channels:
            decoders=[]
            for i in range(out_channels):
                decoders.append(create_decoder(1,filters,kernel_size,u_activation,res_type=res_type))
            self.decoders=nn.Sequential(*decoders)
        else:
            self.decoders=create_decoder(out_channels,filters,kernel_size,u_activation,res_type=res_type)
        self.multi_out_channels=multi_out_channels
        self.apply(self.init_para)
    def init_para(self, model):
        init_parameters(model)
    def forward(self,x):
        x,tensors,indices,sizes=encoder_forward(x,self.encoders)
        x=self.attention_stage(x)
        if self.multi_out_channels:
            x=decode_forward_ex(x,tensors,indices,sizes,self.decoders,self.res_type)
        else:
            x=decode_forwrd(x,tensors,indices,sizes,self.decoders,self.res_type)
        return x