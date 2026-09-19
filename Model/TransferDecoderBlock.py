"""
Transformer Decoder Blocks with Pre-LayerNorm and Residual Connections
"""

import torch
import torch.nn as nn
from typing import Optional
from Model.attention import MultiHeadedAttention, PositionwiseFeedForward


class TransformerDecoderBlock(nn.Module):
    """
    Pre-LayerNorm Decoder Block with RoPE Self-Attention and FeedForward network.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadedAttention(
            n_heads=num_heads,
            d_model=d_model,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_start_pos: int = 0
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, mask=mask, seq_start_pos=seq_start_pos)
        x = x + self.dropout1(attn_out)

        # Pre-LN FeedForward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)

        return x


class TransformerDecoder(nn.Module):
    """
    Stack of Transformer Decoder Blocks followed by final Layer Normalization.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_start_pos: int = 0
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, seq_start_pos=seq_start_pos)
        return self.norm(x)