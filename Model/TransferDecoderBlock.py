from Model.attention import MultiHeadedAttention
from Model.attention import PositionwiseFeedForward
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention block
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attn_out)

        # Feedforward block
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout2(ff_out)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)