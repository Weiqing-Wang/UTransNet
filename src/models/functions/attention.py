from collections import OrderedDict

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads,atten_type='convolution'):
        super().__init__()
        self.scale=(embed_dim//num_heads)**(-0.5)
        self.num_heads=num_heads
        self.atten_type=atten_type
        self.embed_dim=embed_dim
        if self.atten_type=='convolution':
            self.atten_proj_q=self.build_conv_proj(self.embed_dim)
            self.atten_proj_kv = self.build_conv_proj(self.embed_dim,stride=1)
        elif self.atten_type=='linear':
            self.atten_proj=self.build_linear_proj(self.embed_dim)

    def build_conv_proj(self,embed_dim,kernel_size=3,padding=1,stride=1):
        qkv = nn.Sequential(OrderedDict([
            (
            'depthwidth_conv', nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False,groups=embed_dim)),
            ('bn',nn.BatchNorm2d(embed_dim)),
            ('rearrage', Rearrange('b c h w -> b (h w) c'))
        ]))
        return qkv
    def build_linear_proj(self,embed_dim):
        qkv=nn.Linear(embed_dim,embed_dim*3,bias=False)
        return qkv

    def forward(self,x,h,w):
        #x.size=(batch_size,h*w,channels)
        if self.atten_type=="convolution":
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            q = self.atten_proj_q(x)
            k = self.atten_proj_kv(x)
            v = self.atten_proj_kv(x)
            q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
            k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
            v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
            attention_score = (q @ k.permute([0, 1, 3, 2])) * self.scale
            atten = F.softmax(attention_score, dim=-1)
            x = atten @ v

            x = rearrange(x, 'b h t d -> b t (h d)')
            return x
        elif self.atten_type=="linear":
            B, N, C = x.shape
            qkv = self.atten_proj(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1))*self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return x