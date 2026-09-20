"""
Standalone Music Generation CLI (RoPE-MT)
Loads a trained model checkpoint, generates symbolic music tokens with top-k / top-p sampling,
constrained decoding (pedal, note state, no-repeat n-grams), and exports a standard playable MIDI (.mid) file.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import argparse
from pathlib import Path
from typing import List, Optional, Set, Dict, Tuple

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Data.vocab import MusicVocab, BOS_TOKEN, EOS_TOKEN
from Data.tokenizer import midi_to_tokens, save_tokens
from Model.MusicTransformer import MusicTransformer, top_k_top_p_filtering
from Postprocessing.to_midi import ids_to_midi, tokens_to_midi


class FastMidiConstraintTracker:
    """
    Vectorized constraint tracker for autoregressive generation:
    - Enforces pedal state (forbids duplicate SUSTAIN_ON/OFF)
    - Enforces monophonic pitch state (forbids duplicate NOTE_ON for active notes)
    - Forbids invalid NOTE_OFF for silent notes (using boolean masked_fill)
    - Applies strict no-repeat n-gram ban on generated tokens
    - Closes dangling sounding notes and active pedal at sequence end
    """
    def __init__(self, vocab: MusicVocab, device: torch.device):
        self.vocab = vocab
        self.device = device
        self.sustain_on_id = vocab.token_to_id.get("SUSTAIN_ON")
        self.sustain_off_id = vocab.token_to_id.get("SUSTAIN_OFF")

        # Map pitch <-> token ID
        self.pitch_to_on_id: Dict[int, int] = {}
        self.pitch_to_off_id: Dict[int, int] = {}
        self.all_off_ids: List[int] = []

        for tok_str, tok_id in vocab.token_to_id.items():
            parts = tok_str.split("_")
            if len(parts) == 3 and parts[0] == "NOTE":
                try:
                    pitch = int(parts[1])
                    if parts[2] == "ON":
                        self.pitch_to_on_id[pitch] = tok_id
                    elif parts[2] == "OFF":
                        self.pitch_to_off_id[pitch] = tok_id
                        self.all_off_ids.append(tok_id)
                except ValueError:
                    pass

        self.all_off_ids_tensor = torch.tensor(self.all_off_ids, dtype=torch.long, device=device)
        self.pedal_active: bool = False
        self.active_notes: Set[int] = set()

    def update_state(self, token_id: int):
        if token_id == self.sustain_on_id:
            self.pedal_active = True
        elif token_id == self.sustain_off_id:
            self.pedal_active = False
        else:
            tok_str = self.vocab.decode_id(token_id)
            if tok_str.startswith("NOTE_"):
                parts = tok_str.split("_")
                if len(parts) == 3:
                    try:
                        pitch = int(parts[1])
                        if parts[2] == "ON":
                            self.active_notes.add(pitch)
                        elif parts[2] == "OFF":
                            self.active_notes.discard(pitch)
                    except ValueError:
                        pass

    def replay_prompt(self, prompt_token_ids: List[int]):
        """Replay prompt tokens to synchronize active note and pedal state."""
        for tid in prompt_token_ids:
            self.update_state(tid)

    def apply_constraints(
        self,
        logits: torch.Tensor,
        generated_only_tokens: List[int],
        no_repeat_ngram_size: int = 8
    ) -> torch.Tensor:
        """
        Builds a boolean banned mask and applies masked_fill(-inf) to preserve
        original model logits for all valid tokens.
        """
        banned_mask = torch.zeros(len(self.vocab), dtype=torch.bool, device=self.device)

        # 1. Always mask non-musical special tokens
        banned_mask[self.vocab.pad_id] = True
        banned_mask[self.vocab.bos_id] = True
        banned_mask[self.vocab.unk_id] = True

        # 2. Mask EOS for the first 64 generated tokens to ensure musical phrase length
        if len(generated_only_tokens) < 64:
            banned_mask[self.vocab.eos_id] = True

        # 3. Pedal constraints
        if self.pedal_active and self.sustain_on_id is not None:
            banned_mask[self.sustain_on_id] = True
        elif not self.pedal_active and self.sustain_off_id is not None:
            banned_mask[self.sustain_off_id] = True

        # 4. Note-on constraints: forbid re-triggering already sounding pitches
        for pitch in self.active_notes:
            on_id = self.pitch_to_on_id.get(pitch)
            if on_id is not None:
                banned_mask[on_id] = True

        # 5. Note-off constraints: forbid turning off notes that are NOT currently sounding
        if len(self.all_off_ids_tensor) > 0:
            banned_mask[self.all_off_ids_tensor] = True
            for pitch in self.active_notes:
                off_id = self.pitch_to_off_id.get(pitch)
                if off_id is not None:
                    banned_mask[off_id] = False

        # 6. Strict No-Repeat N-Gram Ban (applied strictly on generated tokens)
        if no_repeat_ngram_size > 1 and len(generated_only_tokens) >= no_repeat_ngram_size - 1:
            ngram_prefix = tuple(generated_only_tokens[-(no_repeat_ngram_size - 1):])
            for i in range(len(generated_only_tokens) - no_repeat_ngram_size + 1):
                if tuple(generated_only_tokens[i : i + no_repeat_ngram_size - 1]) == ngram_prefix:
                    banned_tok = generated_only_tokens[i + no_repeat_ngram_size - 1]
                    banned_mask[banned_tok] = True

        return logits.masked_fill(banned_mask, -float("inf"))

    def get_closing_tokens(self) -> List[int]:
        """Returns closing NOTE_OFF tokens for all active pitches and SUSTAIN_OFF if pedal is down."""
        closing_ids = []
        for pitch in sorted(list(self.active_notes)):
            off_id = self.pitch_to_off_id.get(pitch)
            if off_id is not None:
                closing_ids.append(off_id)
        if self.pedal_active and self.sustain_off_id is not None:
            closing_ids.append(self.sustain_off_id)
        return closing_ids


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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint .pt")
    parser.add_argument("--vocab_path", type=str, default="vocab/vocab.json", help="Path to vocab.json")
    parser.add_argument("--output_midi", type=str, default="generated.mid", help="Output path for generated MIDI file")
    parser.add_argument("--output_tokens", type=str, default="generated_tokens.txt", help="Output path to save token list")
    parser.add_argument("--prompt_midi", type=str, default=None, help="Optional MIDI file to use as priming prompt")
    parser.add_argument("--prompt_tokens", type=str, default=None, help="Optional whitespace-separated token string prompt")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=8, help="No-repeat n-gram size ban (n >= 8)")
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
        
        # Cut prompt cleanly right after a VEL_* or SHIFT_* token (max half of max_len)
        max_prompt_len = min(len(prompt_tokens), max(64, args.max_len // 2))
        if len(prompt_tokens) > max_prompt_len:
            valid_cut = -1
            for idx in range(max_prompt_len, 0, -1):
                tok = prompt_tokens[idx - 1]
                if tok.startswith("VEL_") or tok.startswith("SHIFT_"):
                    valid_cut = idx
                    break
            if valid_cut > 0:
                prompt_tokens = prompt_tokens[:valid_cut]
            else:
                prompt_tokens = prompt_tokens[:max_prompt_len]

        print(f"✂️ Prompt cleanly trimmed to {len(prompt_tokens)} tokens (ends with '{prompt_tokens[-1]}').")
        prompt_ids = vocab.encode(prompt_tokens)
    elif args.prompt_tokens:
        prompt_tokens = args.prompt_tokens.strip().split()
        prompt_ids = vocab.encode(prompt_tokens)
    else:
        prompt_ids = [vocab.bos_id]

    tracker = FastMidiConstraintTracker(vocab, device)
    tracker.replay_prompt(prompt_ids)

    print(f"🎼 Starting autoregressive generation ({len(prompt_ids)} prompt tokens) ...")
    generated = list(prompt_ids)
    generated_only = []

    model.eval()
    with torch.no_grad():
        for _ in range(args.max_len):
            context = generated[-model.max_seq_len:]
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)

            logits = model(input_tensor)[0, -1, :].clone()

            # Apply constrained decoding with boolean masked_fill
            logits = tracker.apply_constraints(
                logits=logits,
                generated_only_tokens=generated_only,
                no_repeat_ngram_size=args.no_repeat_ngram_size
            )

            # Apply temperature scaling
            logits = logits / max(args.temperature, 1e-5)

            # Top-k & Top-p filtering
            filtered_logits = top_k_top_p_filtering(
                logits.unsqueeze(0),
                top_k=args.top_k,
                top_p=args.top_p
            ).squeeze(0)

            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            generated_only.append(next_token)
            tracker.update_state(next_token)

            if next_token == vocab.eos_id:
                print("🛑 Received EOS token.")
                break

    # Close all dangling active notes and pedal
    closing = tracker.get_closing_tokens()
    if closing:
        print(f"🔒 Appending {len(closing)} closing tokens for dangling notes/pedal: {[vocab.decode_id(c) for c in closing]}")
        generated.extend(closing)

    if generated[-1] != vocab.eos_id:
        generated.append(vocab.eos_id)

    generated_tokens = vocab.decode(generated)
    print(f"✨ Total sequence length: {len(generated_tokens)} tokens ({len(generated_only)} newly generated).")

    # Save tokens
    if args.output_tokens:
        save_tokens(generated_tokens, args.output_tokens)
        print(f"📝 Saved tokens to: {args.output_tokens}")

    # Save MIDI
    out_midi_p = Path(args.output_midi)
    ids_to_midi(generated, vocab, output_path=out_midi_p, bpm=args.bpm)
    print(f"🎉 Successfully exported MIDI to: {out_midi_p}")


if __name__ == "__main__":
    main()
