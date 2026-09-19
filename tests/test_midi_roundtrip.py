"""
Unit and Integration Test: MIDI & Token Round-Trip Verification
Tests vocabulary mapping, token encoding/decoding, and MIDI event round-trip integrity.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Data.vocab import (
    MusicVocab,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    velocity_to_bin,
    bin_to_velocity,
    quantize_time,
)
from Data.tokenizer import extract_events_from_notes, events_to_tokens
from Postprocessing.to_midi import tokens_to_events


def test_vocab_encoding_decoding():
    print("\n[Test 1] Testing MusicVocab encode / decode roundtrip...")
    vocab = MusicVocab.build_standard_vocab()
    assert len(vocab) > 0, "Vocab is empty!"

    sample_tokens = [
        BOS_TOKEN,
        "SHIFT_100",
        "NOTE_60_ON",
        "VEL_16",
        "NOTE_64_ON",
        "VEL_20",
        "SUSTAIN_ON",
        "SHIFT_500",
        "NOTE_60_OFF",
        "NOTE_64_OFF",
        "SUSTAIN_OFF",
        EOS_TOKEN,
    ]

    ids = vocab.encode(sample_tokens)
    recovered_tokens = vocab.decode(ids)

    assert sample_tokens == recovered_tokens, (
        f"Vocab roundtrip failed!\nOriginal: {sample_tokens}\nRecovered: {recovered_tokens}"
    )
    print("✅ Vocab encoding/decoding passed!")


def test_vocab_serialization(tmp_path: Path):
    print("\n[Test 2] Testing Vocab save & load...")
    vocab = MusicVocab.build_standard_vocab()
    save_file = tmp_path / "test_vocab.json"
    vocab.save(save_file)

    loaded_vocab = MusicVocab.load(save_file)
    assert len(vocab) == len(loaded_vocab), "Saved and loaded vocab sizes differ!"
    assert vocab.token_to_id == loaded_vocab.token_to_id, "Vocab dictionaries differ!"
    print("✅ Vocab serialization passed!")


def test_velocity_and_time_quantization():
    print("\n[Test 3] Testing velocity & time quantization...")
    # Velocity round-trip within bin size tolerance
    for vel in [10, 32, 64, 96, 127]:
        bin_idx = velocity_to_bin(vel)
        reconstructed_vel = bin_to_velocity(bin_idx)
        assert abs(vel - reconstructed_vel) <= 4, f"Velocity {vel} -> bin {bin_idx} -> {reconstructed_vel} exceeds tolerance"

    # Time quantization
    assert quantize_time(48) == 50
    assert quantize_time(52) == 50
    assert quantize_time(103) == 100
    print("✅ Velocity and time quantization passed!")


def test_events_to_tokens_to_events_roundtrip():
    print("\n[Test 4] Testing MIDI Event -> Tokens -> MIDI Event roundtrip...")

    # Define synthetic musical score: C Major chord then melody notes with sustain pedal
    original_notes = [
        {'pitch': 60, 'velocity': 64, 'start': 0, 'end': 500},
        {'pitch': 64, 'velocity': 80, 'start': 0, 'end': 500},
        {'pitch': 67, 'velocity': 72, 'start': 0, 'end': 500},
        {'pitch': 72, 'velocity': 90, 'start': 600, 'end': 1000},
        {'pitch': 71, 'velocity': 85, 'start': 1000, 'end': 1400},
    ]

    original_pedals = [
        {'time': 0, 'value': 127},
        {'time': 500, 'value': 0},
        {'time': 600, 'value': 127},
        {'time': 1400, 'value': 0},
    ]

    # 1. Convert to raw events
    events = extract_events_from_notes(original_notes, original_pedals)

    # 2. Convert events to tokens
    tokens = events_to_tokens(events)
    print(f"Generated {len(tokens)} tokens:")
    print(" ->", " ".join(tokens[:15]), "...")

    # 3. Convert tokens back to events
    recovered_notes, recovered_pedals = tokens_to_events(tokens)

    # 4. Verify notes count and properties
    assert len(recovered_notes) == len(original_notes), (
        f"Note count mismatch! Original: {len(original_notes)}, Recovered: {len(recovered_notes)}"
    )

    for orig, rec in zip(original_notes, recovered_notes):
        assert orig['pitch'] == rec['pitch'], f"Pitch mismatch: orig {orig['pitch']} vs rec {rec['pitch']}"
        assert abs(orig['start'] - rec['start']) <= 50, (
            f"Start time mismatch for pitch {orig['pitch']}: orig {orig['start']} vs rec {rec['start']}"
        )
        assert abs(orig['end'] - rec['end']) <= 50, (
            f"End time mismatch for pitch {orig['pitch']}: orig {orig['end']} vs rec {rec['end']}"
        )
        assert abs(orig['velocity'] - rec['velocity']) <= 4, (
            f"Velocity mismatch for pitch {orig['pitch']}: orig {orig['velocity']} vs rec {rec['velocity']}"
        )

    # 5. Verify pedals
    assert len(recovered_pedals) == len(original_pedals), (
        f"Pedal count mismatch! Original {len(original_pedals)}, Recovered {len(recovered_pedals)}"
    )

    for orig, rec in zip(original_pedals, recovered_pedals):
        assert abs(orig['time'] - rec['time']) <= 50, f"Pedal time mismatch: orig {orig['time']} vs rec {rec['time']}"
        assert orig['value'] == rec['value'], f"Pedal value mismatch: orig {orig['value']} vs rec {rec['value']}"

    print("✅ Note and Pedal round-trip reconstruction matched perfectly within quantization bounds!")


def run_all_tests():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_vocab_encoding_decoding()
        test_vocab_serialization(tmp_path)
        test_velocity_and_time_quantization()
        test_events_to_tokens_to_events_roundtrip()

    print("\n🎉 ALL ROUND-TRIP TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_all_tests()
