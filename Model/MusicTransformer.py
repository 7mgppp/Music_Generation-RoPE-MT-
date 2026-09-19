"""
Music Transformer with Rotary Position Embeddings (RoPE-MT)
Autoregressive decoder-only Transformer for symbolic music generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from Model.embeding import TokenEmbedding
from Model.TransferDecoderBlock import TransformerDecoder


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Generate lower-triangular causal attention mask of shape (1, 1, seq_len, seq_len).
    True indicates allowed attention positions, False indicates masked future positions.
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf")
) -> torch.Tensor:
    """
    Filter logits using top-k and/or nucleus (top-p) filtering.
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


class MusicTransformer(nn.Module):
    """
    RoPE Music Transformer: Decoder-only autoregressive language model for symbolic music.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=nhead,
            d_ff=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        custom_causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            mask: Optional explicit attention mask
            custom_causal_mask: If True and mask is None, automatically creates causal triangular mask
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        if mask is None and custom_causal_mask and seq_len > 1:
            mask = generate_causal_mask(seq_len, device=x.device)

        hidden_states = self.token_embedding(x)
        decoded = self.decoder(hidden_states, mask=mask)
        logits = self.lm_head(decoded)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_generate_len: int = 512,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_id: Optional[int] = 2,
        device: Optional[torch.device] = None
    ) -> List[int]:
        """
        Autoregressive generation with temperature, top-k, and top-p (nucleus) sampling.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        generated = list(prompt_ids)

        for _ in range(max_generate_len):
            # Crop to context window if needed
            context = generated[-self.max_seq_len:]
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)

            logits = self.forward(input_tensor)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

            filtered_logits = top_k_top_p_filtering(
                next_token_logits.unsqueeze(0),
                top_k=top_k,
                top_p=top_p
            ).squeeze(0)

            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            if eos_id is not None and next_token == eos_id:
                break

        return generated
