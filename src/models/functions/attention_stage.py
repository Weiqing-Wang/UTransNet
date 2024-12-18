from einops import rearrange
from torch import nn

from .attention_block import Block


class Stage(nn.Module):
    def __init__(self,embed_dim,num_heads,mlp_ratio,depths,atten_type,a_activation=nn.GELU,a_norm=nn.LayerNorm):
        super().__init__()
        blocks=[]
        for i in range(depths):
            block=Block(embed_dim,num_heads,mlp_ratio,atten_type,a_activation,a_norm)
            blocks.append(block)
        self.blocks=nn.Sequential(*blocks)
    def forward(self,x):
        B,C,H,W=x.shape
        x=rearrange(x,"b c h w -> b (h w) c")
        for block in self.blocks:
            x=block(x,H,W)
        x=rearrange(x,'b (h w) c -> b c h w',h=H,w=W)
        return x
