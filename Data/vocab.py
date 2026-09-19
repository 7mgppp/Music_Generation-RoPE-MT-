"""
Shared Music Vocabulary for Music Generation (RoPE-MT)
Provides bidirectional token <-> id mapping, token encoding/decoding,
and consistent vocabulary definitions across data processing and inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterable

# Special Tokens
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

# Musical Constants
VELOCITY_BINS = 32
MAX_SHIFT_MS = 10000
SHIFT_STEP_MS = 50
MIN_PITCH = 0
MAX_PITCH = 127


def velocity_to_bin(vel: int, bins: int = VELOCITY_BINS) -> int:
    """Map MIDI velocity (0-127) to discrete velocity bin (0 to bins-1)."""
    bin_size = 128 // bins
    return min(bins - 1, max(0, int(vel) // bin_size))


def bin_to_velocity(bin_idx: int, bins: int = VELOCITY_BINS) -> int:
    """Map discrete velocity bin back to MIDI velocity (center of bin)."""
    bin_size = 128 // bins
    mid_offset = bin_size // 2
    return min(127, max(1, int(bin_idx) * bin_size + mid_offset))


def quantize_time(delta_ms: Union[int, float], step_ms: int = SHIFT_STEP_MS, max_ms: int = MAX_SHIFT_MS) -> int:
    """Quantize millisecond time delta to discrete step increments."""
    steps = max(1, round(delta_ms / step_ms))
    quantized = steps * step_ms
    return min(max_ms, int(quantized))


class MusicVocab:
    """
    Music vocabulary managing token-to-id and id-to-token mappings.
    """

    def __init__(self, token_to_id: Optional[Dict[str, int]] = None):
        if token_to_id is not None:
            self.token_to_id = dict(token_to_id)
            self._rebuild_id_to_token()
        else:
            self.token_to_id = {}
            self.id_to_token = []

    def _rebuild_id_to_token(self):
        max_id = max(self.token_to_id.values()) if self.token_to_id else -1
        self.id_to_token = [UNK_TOKEN] * (max_id + 1)
        for token, idx in self.token_to_id.items():
            if idx >= len(self.id_to_token):
                self.id_to_token.extend([UNK_TOKEN] * (idx - len(self.id_to_token) + 1))
            self.id_to_token[idx] = token

    @property
    def pad_id(self) -> int:
        return self.token_to_id.get(PAD_TOKEN, 0)

    @property
    def bos_id(self) -> int:
        return self.token_to_id.get(BOS_TOKEN, 1)

    @property
    def eos_id(self) -> int:
        return self.token_to_id.get(EOS_TOKEN, 2)

    @property
    def unk_id(self) -> int:
        return self.token_to_id.get(UNK_TOKEN, 3)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def decode_id(self, token_id: int) -> str:
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]
        return UNK_TOKEN

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.encode_token(tok) for tok in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.decode_id(i) for i in ids]

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MusicVocab":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        return cls(token_to_id)

    @classmethod
    def build_standard_vocab(
        cls,
        min_pitch: int = MIN_PITCH,
        max_pitch: int = MAX_PITCH,
        vel_bins: int = VELOCITY_BINS,
        shift_step_ms: int = SHIFT_STEP_MS,
        max_shift_ms: int = MAX_SHIFT_MS,
    ) -> "MusicVocab":
        """
        Build a complete, standard music vocabulary covering all pitches,
        quantized time shifts, velocity bins, and pedal/control events.
        """
        tokens = [
            PAD_TOKEN,
            BOS_TOKEN,
            EOS_TOKEN,
            UNK_TOKEN,
            "SUSTAIN_ON",
            "SUSTAIN_OFF",
        ]

        # Time shift tokens (e.g. SHIFT_50, SHIFT_100, ..., SHIFT_10000)
        for shift in range(shift_step_ms, max_shift_ms + 1, shift_step_ms):
            tokens.append(f"SHIFT_{shift}")

        # Note ON and OFF tokens
        for pitch in range(min_pitch, max_pitch + 1):
            tokens.append(f"NOTE_{pitch}_ON")
            tokens.append(f"NOTE_{pitch}_OFF")

        # Velocity bin tokens
        for v in range(vel_bins):
            tokens.append(f"VEL_{v}")

        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id)

    @classmethod
    def build_from_files(cls, token_files: Iterable[Union[str, Path]], min_freq: int = 1) -> "MusicVocab":
        """Build vocabulary from tokenized text files with frequency thresholding."""
        counter: Dict[str, int] = {}
        for file_path in token_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    tok = line.strip()
                    if tok:
                        counter[tok] = counter.get(tok, 0) + 1

        token_to_id = {
            PAD_TOKEN: 0,
            BOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        idx = 4
        for tok, count in counter.items():
            if tok not in token_to_id and count >= min_freq:
                token_to_id[tok] = idx
                idx += 1

        return cls(token_to_id)
