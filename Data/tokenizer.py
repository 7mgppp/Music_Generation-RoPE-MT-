"""
MIDI Tokenizer for Music Generation (RoPE-MT)
Converts MIDI files into event token sequences and vice versa.
"""

from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict
import json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


from Data.vocab import (
    MusicVocab,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    VELOCITY_BINS,
    MAX_SHIFT_MS,
    SHIFT_STEP_MS,
    velocity_to_bin,
    quantize_time,
)


def extract_events_from_notes(
    notes: List[Dict[str, int]],
    sustain_events: Optional[List[Dict[str, int]]] = None
) -> List[Tuple[int, str, int, int]]:
    """
    Given a list of note dicts {'pitch': int, 'start': int, 'end': int, 'velocity': int}
    and optional sustain pedal events {'time': int, 'value': int},
    return sorted unified time events: (time_ms, event_type, pitch/value, velocity)
    """
    events = []
    if sustain_events:
        for cc in sustain_events:
            val = 127 if cc.get('value', 0) >= 64 else 0
            events.append((int(cc['time']), 'SUSTAIN', val, 0))

    for note in notes:
        pitch = int(note['pitch'])
        vel = int(note['velocity'])
        start = int(note['start'])
        end = max(start + SHIFT_STEP_MS, int(note['end']))

        events.append((start, 'NOTE_ON', pitch, vel))
        events.append((end, 'NOTE_OFF', pitch, 0))

    # Priority at identical timestamp: SUSTAIN (0) -> NOTE_OFF (1) -> NOTE_ON (2)
    priority = {'SUSTAIN': 0, 'NOTE_OFF': 1, 'NOTE_ON': 2}
    events.sort(key=lambda x: (x[0], priority.get(x[1], 3)))
    return events


def events_to_tokens(events: List[Tuple[int, str, int, int]]) -> List[str]:
    """Convert sorted raw time events into token representation."""
    tokens = [BOS_TOKEN]
    last_time = 0
    active_notes: Dict[int, bool] = {}

    for time_ms, event_type, value1, value2 in events:
        delta = time_ms - last_time
        if delta > 0:
            while delta > MAX_SHIFT_MS:
                tokens.append(f"SHIFT_{MAX_SHIFT_MS}")
                delta -= MAX_SHIFT_MS
            if delta > 0:
                q_delta = quantize_time(delta)
                tokens.append(f"SHIFT_{q_delta}")
            last_time = time_ms

        if event_type == 'NOTE_ON':
            pitch, velocity = value1, value2
            if active_notes.get(pitch, False):
                tokens.append(f"NOTE_{pitch}_OFF")
            vel_bin = velocity_to_bin(velocity)
            tokens.append(f"NOTE_{pitch}_ON")
            tokens.append(f"VEL_{vel_bin}")
            active_notes[pitch] = True

        elif event_type == 'NOTE_OFF':
            pitch = value1
            if active_notes.get(pitch, False):
                tokens.append(f"NOTE_{pitch}_OFF")
                active_notes[pitch] = False

        elif event_type == 'SUSTAIN':
            if value1 >= 64 or value1 == 127:
                tokens.append("SUSTAIN_ON")
            else:
                tokens.append("SUSTAIN_OFF")

    # Close remaining open notes
    for pitch, is_active in active_notes.items():
        if is_active:
            tokens.append(f"NOTE_{pitch}_OFF")

    tokens.append(EOS_TOKEN)
    return tokens


def midi_to_tokens(midi_source: Union[str, Path, object]) -> List[str]:
    """
    Parse a MIDI file or miditoolkit MidiFile object into tokens.
    """
    notes = []
    pedals = []

    # If it's a miditoolkit MidiFile instance or file path
    if isinstance(midi_source, (str, Path)):
        try:
            from miditoolkit import MidiFile
            midi = MidiFile(str(midi_source))
        except ImportError:
            # Fallback or raise informative error
            raise ImportError("miditoolkit is required to load MIDI files. Install with: pip install miditoolkit")
    else:
        midi = midi_source

    # Extract notes and control changes
    for inst in midi.instruments:
        if getattr(inst, 'is_drum', False):
            continue
        for note in inst.notes:
            notes.append({
                'pitch': note.pitch,
                'velocity': note.velocity,
                'start': note.start,
                'end': note.end
            })
        for cc in getattr(inst, 'control_changes', []):
            if cc.number == 64:
                pedals.append({
                    'time': cc.time,
                    'value': cc.value
                })

    events = extract_events_from_notes(notes, pedals)
    return events_to_tokens(events)


def save_tokens(tokens: List[str], output_path: Union[str, Path]):
    """Save token list to text file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))


def load_tokens(input_path: Union[str, Path]) -> List[str]:
    """Load token list from text file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tokenize_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    vocab_output_path: Optional[Union[str, Path]] = None
) -> MusicVocab:
    """Tokenize all MIDI files in input_dir and optionally build vocabulary."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi")))
    print(f"Found {len(midi_files)} MIDI files in {input_dir}")

    token_files = []
    for f in tqdm(midi_files, desc="Tokenizing"):
        try:
            tokens = midi_to_tokens(f)
            save_path = output_dir / f"{f.stem}.txt"
            save_tokens(tokens, save_path)
            token_files.append(save_path)
        except Exception as e:
            print(f"Failed on {f.name}: {e}")

    vocab = MusicVocab.build_from_files(token_files)
    if vocab_output_path:
        vocab.save(vocab_output_path)
        print(f"Saved vocabulary to {vocab_output_path} (size: {len(vocab)})")

    return vocab