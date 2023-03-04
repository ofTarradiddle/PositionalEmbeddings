import math
import torch
import torch.nn as nn
from einops import rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # create learnable parameters for sinusoidal positional encoding
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.weights = nn.Parameter(emb, requires_grad=True)

    def forward(self, x):
        # add positional encoding to input tensor
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).float()
        positions /= self.max_seq_len
        positions = positions[:, None] * self.weights[None, :]
        positions = torch.cat([positions.sin(), positions.cos()], dim=-1)
        return x + positions


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.angles = nn.Parameter(torch.zeros(dim // 2))

    def forward(self, x):
        # apply rotary embedding to input tensor
        sin_angles = torch.sin(self.angles)
        cos_angles = torch.cos(self.angles)
        sin_x = x * cos_angles[..., None] + rearrange(sin_angles, 'd -> () () d')
        cos_x = x * sin_angles[..., None] + rearrange(cos_angles, 'd -> () () d')
        return torch.cat([sin_x, cos_x], dim=-1)


class PosRotEmbedding(nn.Module):
    def __init__(self, input_dim, max_seq_len):
        super().__init__()
        self.positional_emb = PositionalEmbedding(input_dim, max_seq_len)
        self.rotary_emb = RotaryEmbedding(input_dim)

    def forward(self, x):
        x = self.positional_emb(x)
        x = self.rotary_emb(x)
        return x


import torch

def circular_shift(x, shift):
    shift = shift % x.size(-1)
    return torch.cat((x[..., -shift:], x[..., :-shift]), dim=-1)

shifted_angle_emb = circular_shift(angle_emb, shift)
rotary_emb = shifted_angle_emb * sine_cosine_emb
