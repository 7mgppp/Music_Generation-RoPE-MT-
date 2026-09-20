"""
Standalone Music Generation CLI (RoPE-MT)
Loads a trained model checkpoint, generates symbolic music tokens with top-k / top-p sampling,
and exports a standard playable MIDI (.mid) file.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import argparse
from pathlib import Path
from typing import List, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Data.vocab import MusicVocab, BOS_TOKEN, EOS_TOKEN
from Data.tokenizer import midi_to_tokens, save_tokens
from Model.MusicTransformer import MusicTransformer
from Postprocessing.to_midi import ids_to_midi, tokens_to_midi


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
    max_seq_len = config.get("seq_len", 512)

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
    parser = argparse.ArgumentParser(description="Generate music with trained RoPE Music Transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt", help="Path to model checkpoint .pt")
    parser.add_argument("--vocab_path", type=str, default="vocab/vocab.json", help="Path to vocab.json")
    parser.add_argument("--output_midi", type=str, default="generated.mid", help="Output path for generated MIDI file")
    parser.add_argument("--output_tokens", type=str, default="generated_tokens.txt", help="Output path to save token list")
    parser.add_argument("--prompt_midi", type=str, default=None, help="Optional MIDI file to use as priming prompt")
    parser.add_argument("--prompt_tokens", type=str, default=None, help="Optional whitespace-separated token string prompt")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) filtering threshold")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo BPM for generated MIDI")

    args = parser.parse_args()
    device = get_device()

    # Load Vocabulary
    vocab_p = Path(args.vocab_path)
    if not vocab_p.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_p}")
    vocab = MusicVocab.load(vocab_p)
    print(f"📖 Loaded vocabulary of size {len(vocab)}")

    # Load Model
    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_p}")
    model = load_model_from_checkpoint(str(ckpt_p), len(vocab), device)

    # Determine prompt token IDs
    if args.prompt_midi and Path(args.prompt_midi).exists():
        print(f"🎹 Priming generation with prompt MIDI: {args.prompt_midi}")
        prompt_tokens = midi_to_tokens(args.prompt_midi)
        if prompt_tokens and prompt_tokens[-1] == EOS_TOKEN:
            prompt_tokens = prompt_tokens[:-1]
        prompt_ids = vocab.encode(prompt_tokens)
    elif args.prompt_tokens:
        prompt_tokens = args.prompt_tokens.strip().split()
        prompt_ids = vocab.encode(prompt_tokens)
    else:
        prompt_ids = [vocab.bos_id]

    print(f"🎼 Starting autoregressive generation ({len(prompt_ids)} prompt tokens) ...")
    generated_ids = model.generate(
        prompt_ids=prompt_ids,
        max_generate_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=vocab.eos_id,
        device=device
    )

    generated_tokens = vocab.decode(generated_ids)
    print(f"✨ Generated {len(generated_tokens)} total tokens.")

    # Save tokens
    if args.output_tokens:
        save_tokens(generated_tokens, args.output_tokens)
        print(f"📝 Saved tokens to: {args.output_tokens}")

    # Save MIDI
    out_midi_p = Path(args.output_midi)
    ids_to_midi(generated_ids, vocab, output_path=out_midi_p, bpm=args.bpm)
    print(f"🎉 Successfully exported MIDI to: {out_midi_p}")


if __name__ == "__main__":
    main()
