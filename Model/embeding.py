"""
Token Embedding Layer with sqrt(d_model) scaling
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learned token embedding scaled by sqrt(d_model).
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)
