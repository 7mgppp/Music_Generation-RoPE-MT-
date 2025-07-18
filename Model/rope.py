import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0/(10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum('i, j->ij', t, self.inv_freq)
        emb = torch.cat((freqs,freqs), dim = -1)


        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len = None):
        if seq_len is None:
            seq_len = x.shape[1]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        return (x * cos) + (self.rotate_half(x) * sin)








