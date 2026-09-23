"""
Standalone Music Generation CLI for MidiTok (REMI + BPE)
Loads a trained model checkpoint, generates symbolic music tokens with plain top-k / top-p sampling,
and exports a standard playable MIDI (.mid) file via tokenizer.decode().
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from miditok import REMI
import symusic
from Model.MusicTransformer import MusicTransformer, top_k_top_p_filtering


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_from_checkpoint(checkpoint_path: str, vocab_size: int, device: torch.device) -> MusicTransformer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    d_model = config.get("d_model", 512)
    nhead = config.get("n_heads", 8)
    num_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("d_ff", 2048)
    dropout = config.get("dropout", 0.1)
    max_seq_len = config.get("seq_len", 1024)

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"🧠 Loaded model checkpoint: {checkpoint_path} (step {checkpoint.get('step', 'unknown')})")
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate music with trained RoPE-MT and MidiTok")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt", help="Path to model checkpoint .pt")
    parser.add_argument("--vocab_path", type=str, default="vocab/miditok_remi_bpe.json", help="Path to MidiTok tokenizer JSON")
    parser.add_argument("--output_midi", type=str, default="generated_miditok.mid", help="Output path for generated MIDI file")
    parser.add_argument("--output_tokens", type=str, default="generated_miditok_tokens.txt", help="Output path to save token list")
    parser.add_argument("--prompt_midi", type=str, default=None, help="Optional MIDI file to use as priming prompt")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) filtering threshold")

    args = parser.parse_args()
    device = get_device()

    # Load MidiTok Tokenizer
    vocab_p = Path(args.vocab_path)
    if not vocab_p.exists():
        raise FileNotFoundError(f"Tokenizer not found at {vocab_p}")
    tokenizer = REMI(params=vocab_p)
    vocab_size = len(tokenizer)
    print(f"📖 Loaded MidiTok REMI BPE tokenizer of size {vocab_size}")

    # Load Model
    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_p}")
    model = load_model_from_checkpoint(str(ckpt_p), vocab_size, device)

    # Determine prompt token IDs
    bos_id = tokenizer["BOS_None"]
    eos_id = tokenizer["EOS_None"]

    if args.prompt_midi and Path(args.prompt_midi).exists():
        print(f"🎹 Priming generation with prompt MIDI: {args.prompt_midi}")
        score = symusic.Score(str(args.prompt_midi))
        tok_seq = tokenizer.encode(score)[0]
        prompt_ids = tok_seq.ids
        
        # Remove trailing EOS if present in prompt
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]
            
        # Ensure BOS is at start
        if not prompt_ids or prompt_ids[0] != bos_id:
            prompt_ids = [bos_id] + prompt_ids
            
        # Cap prompt to reasonable length (e.g. max half of max_len)
        max_prompt = min(len(prompt_ids), args.max_len // 2)
        prompt_ids = prompt_ids[:max_prompt]
        print(f"✂️ Prompt length set to {len(prompt_ids)} tokens.")
    else:
        prompt_ids = [bos_id]

    print(f"🎼 Starting plain autoregressive generation ({len(prompt_ids)} prompt tokens) ...")
    generated = list(prompt_ids)
    generated_only = []

    model.eval()
    with torch.no_grad():
        for _ in range(args.max_len):
            context = generated[-model.max_seq_len:]
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)

            logits = model(input_tensor)[0, -1, :].clone()

            # Plain temperature scaling
            logits = logits / max(args.temperature, 1e-5)

            # Plain top-k & top-p (nucleus) filtering
            filtered_logits = top_k_top_p_filtering(
                logits.unsqueeze(0),
                top_k=args.top_k,
                top_p=args.top_p
            ).squeeze(0)

            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            generated_only.append(next_token)

            if next_token == eos_id:
                print("🛑 Received EOS token.")
                break

    print(f"✨ Generation finished. Total sequence length: {len(generated)} tokens ({len(generated_only)} newly generated).")

    # Save tokens
    if args.output_tokens:
        with open(args.output_tokens, "w", encoding="utf-8") as f:
            f.write("\n".join(str(tid) for tid in generated))
        print(f"📝 Saved token IDs to: {args.output_tokens}")

    # Decode with MidiTok to standard playable MIDI
    out_midi_p = Path(args.output_midi)
    score = tokenizer.decode([generated])
    score.dump_midi(str(out_midi_p))
    print(f"🎉 Successfully exported MIDI to: {out_midi_p}")


if __name__ == "__main__":
    main()
