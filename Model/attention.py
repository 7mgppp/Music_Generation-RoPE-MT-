"""
Multi-Head Attention with Rotary Positional Embeddings (RoPE) and FeedForward Network
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from Model.rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> torch.Tensor:
    """
    Compute Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -1e9)
        else:
            scores = scores.masked_fill(mask == 0, -1e9)

    attn_probs = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn_probs = dropout(attn_probs)

    return torch.matmul(attn_probs, value)


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module with Rotary Positional Embeddings (RoPE).
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1, max_seq_len: int = 4096):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_start_pos: int = 0
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (1, 1, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
            seq_start_pos: Starting index for position embedding during incremental generation
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections & reshape to (batch, n_heads, seq_len, d_k)
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = self.rope(q, seq_len=seq_len + seq_start_pos)
        if seq_start_pos > 0:
            q = q[:, :, seq_start_pos:]
        k = self.rope(k, seq_len=seq_len + seq_start_pos)
        if seq_start_pos > 0:
            k = k[:, :, seq_start_pos:]

        # Attention computation
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # Reshape back to (batch, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_out)


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer FeedForward Network with activation and dropout.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.act(self.w_1(x))))
