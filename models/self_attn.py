# Adapted from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .spectral_normalization import SpectralNorm
import numpy as np

class Self_Attn(nn.module):
    """Self attention layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels=in_dim//8, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
            inputs:
                x: input feature maps (B, C, W, H)
            outputs:
                out: self-attention value + input x
                attention: (B, N, N) aka (B, W*H, W*H)
        """
        batch_size, C, width, height = x.size()
        N = width * height
        proj_query = self.query_conv(x).view(batch_size, -1, N).permute(0,2,1) # (B, C, N) or (B, N, C)?
        proj_key = self.key_conv(x).view(batch_size, -1, N) # (B, C, N)
        energy = torch.bmm(proj_query, proj_key) # transpose check?
        attention = self.softmax(energy) # (B, N, N)
        proj_value = self.value_conv(x).view(batch_size, -1, N).permute(0,2,1) # (B, C, N)?

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out, attention