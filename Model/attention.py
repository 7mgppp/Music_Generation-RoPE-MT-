import math
import torch
import torch.nn as nn
from copy import deepcopy
from Model.rope import RotaryPositionalEmbedding

def clones(module, n):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.rope = RotaryPositionalEmbedding(self.d_k)
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)


    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        query = self.rope(query)
        key = self.rope(key)

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):


    def __init__(self, d_model, d_ff, dropout=0.1):
            super(PositionwiseFeedForward, self).__init__()
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
            return self.w_2(self.dropout(self.w_1(x).relu()))





