import torch
import torch.nn as nn
from Model.TransferDecoderBlock import TransformerDecoder

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.decoder = TransformerDecoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embedding(x) + self.pos_embedding[:, :x.size(1)]
        x = self.decoder(x, mask=mask)
        return self.lm_head(x)



