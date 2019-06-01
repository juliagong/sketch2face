# Adapted from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
import torch
import torch.nn as nn

class SelfAttn(nn.Module):
    """Self attention layer.
        The output_attention parameter sets whether forward() returns just the activation of the layer, 
        or an (activation, attention) tuple. Worth setting to False if using it with nn.Sequential 
        (TODO: have instance variable point to where to store it?) and True otherwise.
    """
    def __init__(self, in_dim, activation, forward_outputs_attention=True, where_to_log_attention=None):
        super(SelfAttn, self).__init__()
        self.channel_in = in_dim
        self.activation = activation
        self.forward_outputs_attention = forward_outputs_attention
        self.where_to_log_attention = where_to_log_attention

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels=in_dim, kernel_size=1)
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
        proj_query = self.query_conv(x).view(batch_size, -1, N).permute(0,2,1) # (B, N, C//8)
        proj_query = self.activation(proj_query)
        proj_key = self.key_conv(x).view(batch_size, -1, N) # (B, C//8, N)
        proj_key = self.activation(proj_key)
        energy = torch.bmm(proj_query, proj_key) # transpose check?
        attention = self.softmax(energy) # (B, N, N)
        proj_value = self.value_conv(x).view(batch_size, -1, N) # (B, C, N)
        proj_value = self.activation(proj_value)

        out = torch.bmm(proj_value, attention.permute(0,2,1))

        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        if self.where_to_log_attention is not None:
            self.where_to_log_attention['attn'] = attention 

        if not self.forward_outputs_attention:
            return out
        return out, attention
