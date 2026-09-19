"""
Rotary Positional Embedding (RoPE) for Multi-Head Attention
Applies relative rotary positional encodings to query and key states.
"""

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as described in RoFormer / LLaMA.
    Rotates query and key representations based on their sequence positions.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # inv_freq: (dim // 2,)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, n_heads, seq_len, d_k)
            seq_len: Optional sequence length override
        Returns:
            RoPE-transformed tensor with same shape as x
        """
        if seq_len is None:
            seq_len = x.shape[2]

        if seq_len > self.max_seq_len:
            self._build_cache(max(seq_len, self.max_seq_len * 2))

        # Shape: (1, 1, seq_len, dim) for broadcasting with (batch, n_heads, seq_len, dim)
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(1)

        return (x * cos) + (self.rotate_half(x) * sin)
