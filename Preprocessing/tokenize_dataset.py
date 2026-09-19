"""
Dataset Tokenization Script for MAESTRO and MIDI datasets.
Reads raw MIDI files, converts them to event token sequences using shared MusicVocab,
and organizes them into tokenized train/validation/test splits.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Data.vocab import MusicVocab
from Data.tokenizer import midi_to_tokens, save_tokens
from Preprocessing.token_id_maps import create_reverse_vocab


def find_maestro_metadata(midi_dir: Path) -> Optional[Path]:
    """Find MAESTRO metadata json/csv if present in midi_dir."""
    json_candidates = list(midi_dir.rglob("*maestro*.json"))
    if json_candidates:
        return json_candidates[0]
    csv_candidates = list(midi_dir.rglob("*maestro*.csv"))
    if csv_candidates:
        return csv_candidates[0]
    return None


def preprocess_and_tokenize(
    midi_dir: str,
    out_dir: str,
    vocab_out: Optional[str] = "vocab/vocab.json",
    build_vocab: bool = False
):
    midi_path = Path(midi_dir).resolve()
    out_path = Path(out_dir).resolve()

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI directory does not exist: {midi_path}")

    # Find all MIDI files
    midi_files = sorted(list(midi_path.rglob("*.mid")) + list(midi_path.rglob("*.midi")))
    if not midi_files:
        print(f"No MIDI files found in {midi_path}")
        return

    print(f"🎵 Found {len(midi_files)} MIDI files in {midi_path}")

    # Check for MAESTRO metadata for automatic train/validation/test splitting
    meta_path = find_maestro_metadata(midi_path)
    file_to_split = {}
    if meta_path:
        print(f"📋 Found MAESTRO metadata at: {meta_path.name}")
        try:
            import pandas as pd
            if meta_path.suffix == ".json":
                df = pd.read_json(meta_path)
            else:
                df = pd.read_csv(meta_path)

            for _, row in df.iterrows():
                fname = Path(row["midi_filename"]).name
                file_to_split[fname] = str(row.get("split", "train")).strip().lower()
            print(f"Mapped {len(file_to_split)} files to dataset splits (train/val/test).")
        except Exception as e:
            print(f"Warning: Could not parse metadata ({e}). Splitting will be skipped.")


    # Create output directories
    out_path.mkdir(parents=True, exist_ok=True)
    if file_to_split:
        for s in ["train", "validation", "test"]:
            (out_path / s).mkdir(parents=True, exist_ok=True)

    token_file_paths = []
    success_count = 0
    fail_count = 0

    for f in tqdm(midi_files, desc="Tokenizing MIDIs", unit="file"):
        try:
            tokens = midi_to_tokens(f)
            split = file_to_split.get(f.name)
            if split:
                target_file = out_path / split / f"{f.stem}.txt"
            else:
                target_file = out_path / f"{f.stem}.txt"

            save_tokens(tokens, target_file)
            token_file_paths.append(target_file)
            success_count += 1
        except Exception as e:
            fail_count += 1
            # print(f"Error processing {f.name}: {e}")

    print(f"\n✅ Tokenization Complete: {success_count} succeeded, {fail_count} failed.")

    # Handle vocabulary
    if build_vocab or not (Path(vocab_out).exists() if vocab_out else False):
        print("🔨 Building vocabulary from tokenized files...")
        vocab = MusicVocab.build_from_files(token_file_paths, min_freq=1)
        if vocab_out:
            vocab_p = Path(vocab_out).resolve()
            vocab.save(vocab_p)
            print(f"Saved vocabulary ({len(vocab)} tokens) to {vocab_p}")
            create_reverse_vocab(vocab_p)
    else:
        if vocab_out and Path(vocab_out).exists():
            vocab_p = Path(vocab_out).resolve()
            create_reverse_vocab(vocab_p)
            print(f"Using existing vocabulary at {vocab_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize MIDI dataset")
    parser.add_argument("--midi_dir", type=str, default="Data/raw", help="Path to raw MIDI dataset directory")
    parser.add_argument("--out_dir", type=str, default="Data/tokenized", help="Output directory for tokenized text files")
    parser.add_argument("--vocab_out", type=str, default="vocab/vocab.json", help="Path to save vocabulary JSON")
    parser.add_argument("--build_vocab", action="store_true", help="Rebuild vocabulary from tokenized files")
    args = parser.parse_args()

    preprocess_and_tokenize(
        midi_dir=args.midi_dir,
        out_dir=args.out_dir,
        vocab_out=args.vocab_out,
        build_vocab=args.build_vocab
    )
