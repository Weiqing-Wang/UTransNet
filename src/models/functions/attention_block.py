from torch import nn

from .attention import Attention
from .mlp import MLP


class Block(nn.Module):
    def __init__(self,embed_dim,num_heads,mlp_ratio,atten_type='convolution',a_activation=nn.GELU,a_norm=nn.LayerNorm,):
        super().__init__()
        self.norm=a_norm(embed_dim)
        self.attention=Attention(embed_dim,num_heads,atten_type)
        hidden_feature=int(embed_dim*mlp_ratio)
        self.mlp=MLP(embed_dim,hidden_feature,embed_dim,a_activation)
    def forward(self,x,h,w):
        #x.size=(batch_size,h*w,channels)
        res=x
        x=self.norm(x)
        attention=self.attention(x,h,w)
        x=res+attention
        x=x+self.mlp(self.norm(x))
        return x