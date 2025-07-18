import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model


    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)





