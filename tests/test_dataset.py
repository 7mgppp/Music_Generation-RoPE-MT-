"""
Automated Dataset Tests: MusicDataset & DataLoader Pipeline (RoPE-MT)
Verifies sequence chunking, (x, y) autoregressive target alignment, padding, and batch loading.
"""

import sys
import tempfile
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Data.vocab import MusicVocab, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from Data.dataset import MusicDataset, collate_music_batch, create_music_dataloaders


def test_music_dataset_chunking_and_targets():
    print("\n[Test 1] Testing MusicDataset chunking and (x, y) target alignment...")

    vocab = MusicVocab.build_standard_vocab()
    max_seq_len = 32

    # Create dummy token file
    dummy_tokens = [
        BOS_TOKEN,
        "NOTE_60_ON", "VEL_16", "SHIFT_100", "NOTE_60_OFF",
        "NOTE_64_ON", "VEL_20", "SHIFT_200", "NOTE_64_OFF",
        "NOTE_67_ON", "VEL_24", "SHIFT_300", "NOTE_67_OFF",
        "SUSTAIN_ON", "SHIFT_500", "SUSTAIN_OFF",
        EOS_TOKEN
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "sample_piece.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(dummy_tokens))

        dataset = MusicDataset(
            data_source=tmp_dir,
            vocab=vocab,
            max_seq_len=max_seq_len,
            stride=max_seq_len
        )

        assert len(dataset) > 0, "Dataset contains 0 samples!"
        sample = dataset[0]

        input_ids = sample["input_ids"]
        target_ids = sample["target_ids"]
        attn_mask = sample["attention_mask"]

        assert input_ids.shape == (max_seq_len,), f"Input shape mismatch: {input_ids.shape}"
        assert target_ids.shape == (max_seq_len,), f"Target shape mismatch: {target_ids.shape}"
        assert attn_mask.shape == (max_seq_len,), f"Mask shape mismatch: {attn_mask.shape}"

        # Verify next-token prediction alignment for valid tokens
        valid_len = attn_mask.sum().item()
        print(f"Sample contains {valid_len} valid tokens and {max_seq_len - valid_len} pad tokens.")

        for i in range(valid_len - 1):
            if target_ids[i].item() != -100:
                assert target_ids[i].item() == input_ids[i + 1].item(), (
                    f"Target token at index {i} ({target_ids[i]}) does not match next input token ({input_ids[i+1]})"
                )

        # Verify padding positions in targets are masked to -100
        pad_positions = (input_ids == vocab.pad_id)
        if pad_positions.sum() > 0:
            assert (target_ids[pad_positions] == -100).all(), "Padding in target was not masked to -100!"

    print("✅ Autoregressive (x, y) alignment and padding verified!")


def test_music_dataloader_batching():
    print("\n[Test 2] Testing DataLoader batch generation...")

    vocab = MusicVocab.build_standard_vocab()
    max_seq_len = 16
    batch_size = 4

    tokens_list = [
        [BOS_TOKEN, "NOTE_60_ON", "VEL_10", "SHIFT_50", "NOTE_60_OFF", EOS_TOKEN],
        [BOS_TOKEN, "NOTE_62_ON", "VEL_12", "SHIFT_100", "NOTE_62_OFF", EOS_TOKEN],
        [BOS_TOKEN, "NOTE_64_ON", "VEL_14", "SHIFT_150", "NOTE_64_OFF", EOS_TOKEN],
        [BOS_TOKEN, "NOTE_65_ON", "VEL_16", "SHIFT_200", "NOTE_65_OFF", EOS_TOKEN],
        [BOS_TOKEN, "NOTE_67_ON", "VEL_18", "SHIFT_250", "NOTE_67_OFF", EOS_TOKEN],
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_dir = Path(tmp_dir) / "train"
        train_dir.mkdir()
        val_dir = Path(tmp_dir) / "validation"
        val_dir.mkdir()

        for i, toks in enumerate(tokens_list[:4]):
            with open(train_dir / f"track_{i}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(toks))

        with open(val_dir / "track_val.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(tokens_list[4]))

        loaders = create_music_dataloaders(
            data_dir=tmp_dir,
            vocab=vocab,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            shuffle_train=True
        )

        assert "train" in loaders, "Train loader missing!"
        assert "val" in loaders, "Val loader missing!"

        for batch in loaders["train"]:
            assert batch["input_ids"].shape[0] <= batch_size
            assert batch["input_ids"].shape[1] == max_seq_len
            assert batch["target_ids"].shape[1] == max_seq_len
            assert batch["attention_mask"].shape[1] == max_seq_len
            print(f"✅ Loaded train batch: input_ids shape = {batch['input_ids'].shape}")
            break

    print("✅ DataLoader batch generation passed!")


def run_all_tests():
    test_music_dataset_chunking_and_targets()
    test_music_dataloader_batching()
    print("\n🎉 ALL DATASET PIPELINE TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_all_tests()
