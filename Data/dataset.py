"""
PyTorch Dataset and DataLoader Pipeline for Symbolic Music Generation (RoPE-MT)
Handles token sequence loading, sliding-window chunking, autoregressive next-token (x, y) pairs,
dynamic padding, and train/val/test data loaders.
"""

from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader

from Data.vocab import MusicVocab, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from Data.tokenizer import load_tokens


class MusicDataset(Dataset):
    """
    Music token dataset that chunks tokenized musical compositions into fixed-length
    autoregressive training pairs (x = tokens[t : t+L], y = tokens[t+1 : t+L+1]).
    """

    def __init__(
        self,
        data_source: Union[str, Path, List[Union[str, Path]]],
        vocab: Union[MusicVocab, str, Path],
        max_seq_len: int = 512,
        stride: Optional[int] = None,
        cache_in_memory: bool = True,
        ignore_pad_in_loss: bool = True,
        min_tokens: int = 4
    ):

        super().__init__()
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len
        self.ignore_pad_in_loss = ignore_pad_in_loss
        self.min_tokens = min_tokens

        # Load vocabulary
        if isinstance(vocab, (str, Path)):
            self.vocab = MusicVocab.load(vocab)
        else:
            self.vocab = vocab

        self.pad_id = self.vocab.pad_id
        self.bos_id = self.vocab.bos_id
        self.eos_id = self.vocab.eos_id

        # Collect file paths
        if isinstance(data_source, (str, Path)):
            data_path = Path(data_source)
            if data_path.is_dir():
                self.files = sorted(list(data_path.rglob("*.txt")))
            elif data_path.is_file():
                self.files = [data_path]
            else:
                self.files = []
        else:
            self.files = [Path(f) for f in data_source if Path(f).exists()]

        self.samples: List[Tuple[List[int], List[int]]] = []
        self._build_samples(cache_in_memory)

    def _build_samples(self, cache_in_memory: bool):
        total_tokens = 0
        for file_path in self.files:
            try:
                tokens = load_tokens(file_path)
                if len(tokens) < self.min_tokens:
                    continue

                token_ids = self.vocab.encode(tokens)
                total_tokens += len(token_ids)

                # Ensure BOS and EOS are present
                if not token_ids or token_ids[0] != self.bos_id:
                    token_ids = [self.bos_id] + token_ids
                if token_ids[-1] != self.eos_id:
                    token_ids = token_ids + [self.eos_id]

                # Chunk using sliding window
                # Need sequence length of max_seq_len + 1 to form input and target pairs
                target_chunk_len = self.max_seq_len + 1

                if len(token_ids) <= target_chunk_len:
                    # Pad short sequence
                    pad_len = target_chunk_len - len(token_ids)
                    padded_ids = token_ids + [self.pad_id] * pad_len
                    inp = padded_ids[:self.max_seq_len]
                    tgt = padded_ids[1:target_chunk_len]
                    self.samples.append((inp, tgt))
                else:
                    for i in range(0, len(token_ids) - 1, self.stride):
                        chunk = token_ids[i : i + target_chunk_len]
                        if len(chunk) < 2:
                            break
                        if len(chunk) < target_chunk_len:
                            # Pad remaining tail
                            pad_len = target_chunk_len - len(chunk)
                            chunk = chunk + [self.pad_id] * pad_len

                        inp = chunk[:self.max_seq_len]
                        tgt = chunk[1:target_chunk_len]
                        self.samples.append((inp, tgt))

            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inp, tgt = self.samples[idx]

        input_tensor = torch.tensor(inp, dtype=torch.long)
        target_tensor = torch.tensor(tgt, dtype=torch.long)

        # Attention mask: True for valid tokens, False for [PAD] tokens
        attention_mask = input_tensor != self.pad_id

        # Replace target padding with -100 so CrossEntropyLoss ignores it
        if self.ignore_pad_in_loss:
            target_tensor = target_tensor.masked_fill(target_tensor == self.pad_id, -100)

        return {
            "input_ids": input_tensor,
            "target_ids": target_tensor,
            "attention_mask": attention_mask
        }


def collate_music_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates a list of sample dicts into batch tensors."""
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    target_ids = torch.stack([item["target_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask
    }


def create_music_dataloaders(
    data_dir: Union[str, Path],
    vocab: Union[MusicVocab, str, Path],
    max_seq_len: int = 512,
    stride: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    val_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    shuffle_train: bool = True
) -> Dict[str, DataLoader]:
    """
    Builds and returns a dictionary of DataLoaders: {'train': ..., 'val': ..., 'test': ...}
    Automatically inspects subdirectories 'train/', 'validation/' (or 'val/'), 'test/'.
    If not split into subdirectories, splits the files automatically by ratios.
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "validation" if (data_path / "validation").exists() else data_path / "val"
    test_dir = data_path / "test"

    loaders = {}

    if train_dir.exists() and len(list(train_dir.glob("*.txt"))) > 0:
        train_ds = MusicDataset(train_dir, vocab=vocab, max_seq_len=max_seq_len, stride=stride)
        loaders["train"] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle_train,
            num_workers=num_workers, collate_fn=collate_music_batch
        )

        if val_dir.exists() and len(list(val_dir.glob("*.txt"))) > 0:
            val_ds = MusicDataset(val_dir, vocab=vocab, max_seq_len=max_seq_len, stride=stride)
            loaders["val"] = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate_music_batch
            )

        if test_dir.exists() and len(list(test_dir.glob("*.txt"))) > 0:
            test_ds = MusicDataset(test_dir, vocab=vocab, max_seq_len=max_seq_len, stride=stride)
            loaders["test"] = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate_music_batch
            )
    else:
        # Auto-split files
        all_files = sorted(list(data_path.rglob("*.txt")))
        n_total = len(all_files)
        n_val = int(n_total * val_split_ratio)
        n_test = int(n_total * test_split_ratio)
        n_train = n_total - n_val - n_test

        train_files = all_files[:n_train]
        val_files = all_files[n_train : n_train + n_val]
        test_files = all_files[n_train + n_val:]

        if train_files:
            loaders["train"] = DataLoader(
                MusicDataset(train_files, vocab=vocab, max_seq_len=max_seq_len, stride=stride),
                batch_size=batch_size, shuffle=shuffle_train,
                num_workers=num_workers, collate_fn=collate_music_batch
            )
        if val_files:
            loaders["val"] = DataLoader(
                MusicDataset(val_files, vocab=vocab, max_seq_len=max_seq_len, stride=stride),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate_music_batch
            )
        if test_files:
            loaders["test"] = DataLoader(
                MusicDataset(test_files, vocab=vocab, max_seq_len=max_seq_len, stride=stride),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate_music_batch
            )

    return loaders
