"""
MidiTok PyTorch Dataset Pipeline for Symbolic Music Generation (RoPE-MT)
Pre-encodes and disk-caches pitch shift variants [-5, +5] on the train split,
builds dedicated sliding-window chunks per canonical source position,
and provides O(1) in-memory chunk sampling with zero runtime BPE overhead.
"""

from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from miditok import REMI
import symusic


class MidiTokDataset(Dataset):
    def __init__(
        self,
        data_source: Union[str, Path, List[Union[str, Path]]],
        tokenizer: REMI,
        max_seq_len: int = 1024,
        stride: Optional[int] = None,
        is_train: bool = False,
        pitch_shifts: Tuple[int, ...] = tuple(range(-5, 6)),
        raw_midi_dir: Optional[Union[str, Path]] = None,
        min_tokens: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len
        self.is_train = is_train
        self.pitch_shifts = pitch_shifts if is_train else (0,)
        
        self.pad_id = tokenizer.pad_token_id  # 0
        self.bos_id = tokenizer["BOS_None"]   # 1
        self.eos_id = tokenizer["EOS_None"]   # 2

        data_path = Path(data_source)
        if data_path.is_dir():
            self.token_files = sorted(list(data_path.glob("*.json")))
            self.cache_path = data_path / "variants_cache.pt"
        else:
            self.token_files = [data_path]
            self.cache_path = None

        # self.variants: List[List[List[int]]] -> [file_idx][variant_idx] -> token_ids
        self.variants: List[List[List[int]]] = []
        self.canonical_chunk_indices: List[Tuple[int, int]] = []
        
        self._load_and_cache_variants(raw_midi_dir, min_tokens)
        self._build_canonical_indices(min_tokens)

    def _load_and_cache_variants(self, raw_midi_dir: Optional[Union[str, Path]], min_tokens: int):
        # 1. Validation & Test: load base pre-tokenized BPE tokens directly (instant)
        if not self.is_train or len(self.pitch_shifts) <= 1:
            for token_file in self.token_files:
                with open(token_file, "r") as fh:
                    base_ids = json.load(fh)
                base_ids = self._ensure_special_tokens(base_ids)
                self.variants.append([base_ids])
            return

        # 2. Train split: check if pre-encoded variants cache exists on disk
        if self.cache_path and self.cache_path.exists():
            print(f"⚡ Loading pre-encoded shift variants from disk cache: {self.cache_path}")
            self.variants = torch.load(self.cache_path)
            print(f"✅ Loaded {len(self.variants)} training pieces with {len(self.pitch_shifts)} variants each in memory.")
            return

        # 3. If cache does not exist, pre-encode all variants once and save to disk
        print(f"🔨 Pre-encoding {len(self.pitch_shifts)} pitch shift variants across {len(self.token_files)} training pieces...")
        for token_file in self.token_files:
            file_variants = []
            
            # Base shift 0
            with open(token_file, "r") as fh:
                base_ids = json.load(fh)
            base_ids = self._ensure_special_tokens(base_ids)
            file_variants.append(base_ids)

            # Pre-encode non-zero shift variants from raw MIDI
            if raw_midi_dir:
                raw_path = Path(raw_midi_dir) / f"{token_file.stem}.midi"
                if not raw_path.exists():
                    raw_path = Path(raw_midi_dir) / f"{token_file.stem}.mid"

                if raw_path.exists():
                    score = symusic.Score(str(raw_path))
                    for s in self.pitch_shifts:
                        if s != 0:
                            shifted_score = score.copy().shift_pitch(s)
                            tok_seq = self.tokenizer.encode(shifted_score)[0]
                            shifted_ids = self._ensure_special_tokens(tok_seq.ids)
                            file_variants.append(shifted_ids)

            self.variants.append(file_variants)

        # Save to disk cache for instant reuse on subsequent launches
        if self.cache_path:
            print(f"💾 Saving pre-encoded variants cache to: {self.cache_path}")
            torch.save(self.variants, self.cache_path)
            print(f"✅ Disk cache created ({self.cache_path.stat().st_size / (1024*1024):.1f} MB).")

    def _ensure_special_tokens(self, ids: List[int]) -> List[int]:
        if not ids or ids[0] != self.bos_id:
            ids = [self.bos_id] + ids
        if ids[-1] != self.eos_id:
            ids = ids + [self.eos_id]
        return ids

    def _build_canonical_indices(self, min_tokens: int):
        target_chunk_len = self.max_seq_len + 1

        for file_idx, file_variants in enumerate(self.variants):
            canonical_ids = file_variants[0]  # Shift 0
            if len(canonical_ids) < min_tokens:
                continue

            if len(canonical_ids) <= target_chunk_len:
                self.canonical_chunk_indices.append((file_idx, 0))
            else:
                for start_pos in range(0, len(canonical_ids) - 1, self.stride):
                    chunk = canonical_ids[start_pos : start_pos + target_chunk_len]
                    if len(chunk) < 2:
                        break
                    self.canonical_chunk_indices.append((file_idx, start_pos))

    def __len__(self) -> int:
        return len(self.canonical_chunk_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start_pos = self.canonical_chunk_indices[idx]

        # Sample ONE random shift variant per source chunk during training
        if self.is_train and len(self.variants[file_idx]) > 1:
            variant_idx = random.randint(0, len(self.variants[file_idx]) - 1)
            token_ids = self.variants[file_idx][variant_idx]
        else:
            token_ids = self.variants[file_idx][0]  # Shift 0 (canonical)

        target_len = self.max_seq_len + 1
        chunk = token_ids[start_pos : start_pos + target_len]

        # Dynamic padding if chunk length < target_len
        if len(chunk) < target_len:
            pad_len = target_len - len(chunk)
            chunk = chunk + [self.pad_id] * pad_len

        input_tensor = torch.tensor(chunk[:self.max_seq_len], dtype=torch.long)
        target_tensor = torch.tensor(chunk[1:target_len], dtype=torch.long)

        # Attention mask: True for valid tokens, False for padding (pad_id = 0)
        attention_mask = input_tensor != self.pad_id

        # Mask target padding with -100 for CrossEntropyLoss ignore_index
        target_tensor = target_tensor.masked_fill(target_tensor == self.pad_id, -100)

        return {
            "input_ids": input_tensor,
            "target_ids": target_tensor,
            "attention_mask": attention_mask
        }


def collate_miditok_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "target_ids": torch.stack([b["target_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


def create_miditok_dataloaders(
    data_dir: Union[str, Path],
    tokenizer: REMI,
    max_seq_len: int = 1024,
    stride: Optional[int] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    raw_midi_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, DataLoader]:
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "validation"
    test_dir = data_path / "test"

    # Train: is_train=True, pitch_shifts=[-5, ..., +5]
    train_ds = MidiTokDataset(
        data_source=train_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        is_train=True,
        pitch_shifts=tuple(range(-5, 6)),
        raw_midi_dir=raw_midi_dir,
    )

    # Validation: is_train=False, pitch_shifts=(0,) [strictly unshifted]
    val_ds = MidiTokDataset(
        data_source=val_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        is_train=False,
        pitch_shifts=(0,),
        raw_midi_dir=None,
    )

    # Test: is_train=False, pitch_shifts=(0,) [strictly unshifted]
    test_ds = MidiTokDataset(
        data_source=test_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        is_train=False,
        pitch_shifts=(0,),
        raw_midi_dir=None,
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_miditok_batch),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_miditok_batch),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_miditok_batch),
    }
