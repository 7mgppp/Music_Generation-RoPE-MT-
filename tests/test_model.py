"""
Automated Model Tests: Music Transformer (RoPE-MT)
Tests Rotary Positional Embeddings, Multi-Head Attention, Causal Mask Invariance,
Loss Backpropagation, and Autoregressive Token Generation.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Model.rope import RotaryPositionalEmbedding
from Model.attention import MultiHeadedAttention
from Model.MusicTransformer import MusicTransformer, generate_causal_mask


def test_rope_dimensions_and_transformation():
    print("\n[Test 1] Testing Rotary Positional Embedding (RoPE)...")
    d_k = 64
    seq_len = 128
    batch_size = 2
    n_heads = 4

    rope = RotaryPositionalEmbedding(dim=d_k, max_seq_len=256)
    x = torch.randn(batch_size, n_heads, seq_len, d_k)
    x_rot = rope(x)

    assert x_rot.shape == x.shape, f"RoPE shape mismatch: {x_rot.shape} vs {x.shape}"
    assert not torch.allclose(x, x_rot), "RoPE did not transform the input tensor!"
    print("✅ RoPE tensor transformation passed!")


def test_causal_mask_invariance():
    print("\n[Test 2] Testing Causal Mask Invariance (Future-Token Isolation)...")
    vocab_size = 100
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_ff = 128

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=0.0
    )
    model.eval()

    # Create two sequences that are identical up to the last token
    seq_len = 16
    prefix = torch.randint(4, vocab_size, (1, seq_len - 1))
    token_a = torch.tensor([[10]])
    token_b = torch.tensor([[20]])

    seq_a = torch.cat([prefix, token_a], dim=1)
    seq_b = torch.cat([prefix, token_b], dim=1)

    with torch.no_grad():
        out_a = model(seq_a)
        out_b = model(seq_b)

    # In a truly causal model, logits for prefix tokens (0 to seq_len-2) must be strictly identical!
    prefix_diff = (out_a[:, :-1, :] - out_b[:, :-1, :]).abs().max().item()
    assert prefix_diff < 1e-5, f"Causal mask leaked future token! Max prefix diff: {prefix_diff}"
    print(f"✅ Causal Masking strictly prevents future token leakage (Max prefix diff: {prefix_diff:.2e})")


def test_model_forward_and_backward():
    print("\n[Test 3] Testing MusicTransformer Forward & Loss Backward...")
    vocab_size = 320
    batch_size = 2
    seq_len = 64

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    model.train()

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(dummy_input)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Unexpected logits shape: {logits.shape}"

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        dummy_target.view(-1)
    )
    loss.backward()

    # Verify gradients computed across all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter: {name}"

    print(f"✅ Model forward pass and backward pass passed! (Loss: {loss.item():.4f})")


def test_autoregressive_generation():
    print("\n[Test 4] Testing Autoregressive Token Generation...")
    vocab_size = 50
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0
    )
    model.eval()

    prompt = [1, 5, 12]  # [BOS], token 5, token 12
    generated = model.generate(
        prompt_ids=prompt,
        max_generate_len=20,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        eos_id=2
    )

    assert len(generated) > len(prompt), "Generation failed to produce new tokens!"
    assert generated[:len(prompt)] == prompt, "Generated sequence did not preserve prompt prefix!"
    print(f"✅ Generated {len(generated)} tokens starting from prompt: {generated[:10]}...")


def run_all_tests():
    test_rope_dimensions_and_transformation()
    test_causal_mask_invariance()
    test_model_forward_and_backward()
    test_autoregressive_generation()
    print("\n🎉 ALL MODEL ARCHITECTURE TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_all_tests()
