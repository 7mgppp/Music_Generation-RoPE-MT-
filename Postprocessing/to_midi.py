"""
Token to MIDI Converter for Music Generation (RoPE-MT)
Converts token sequences back into playable standard MIDI files.
"""

from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple
import argparse

from Data.vocab import (
    MusicVocab,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    SHIFT_STEP_MS,
    bin_to_velocity,
)


def tokens_to_events(tokens: List[str]) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    """
    Parse event tokens into notes and sustain pedal control changes.
    Returns (notes, sustain_pedals).
    """
    current_time = 0
    active_notes: Dict[int, Tuple[int, int]] = {}  # pitch -> (start_time, velocity)
    notes: List[Dict[str, int]] = []
    pedals: List[Dict[str, int]] = []

    pending_pitch: Optional[int] = None
    default_velocity = 64

    for token in tokens:
        token = token.strip()
        if not token or token in (PAD_TOKEN, BOS_TOKEN, UNK_TOKEN):
            continue

        if token == EOS_TOKEN:
            break

        # Time shifts
        if token.startswith("SHIFT_"):
            try:
                shift_ms = int(token.split("_")[1])
                current_time += shift_ms
            except (IndexError, ValueError):
                pass

        # Note ON
        elif token.startswith("NOTE_") and token.endswith("_ON"):
            try:
                pitch = int(token.split("_")[1])
                # If there was a pending note waiting for velocity, start it with default vel
                if pending_pitch is not None:
                    if pending_pitch in active_notes:
                        p_start, p_vel = active_notes.pop(pending_pitch)
                        notes.append({
                            'pitch': pending_pitch,
                            'velocity': p_vel,
                            'start': p_start,
                            'end': max(p_start + SHIFT_STEP_MS, current_time)
                        })
                    active_notes[pending_pitch] = (current_time, default_velocity)

                pending_pitch = pitch
            except (IndexError, ValueError):
                pass

        # Velocity specification for the preceding NOTE_ON
        elif token.startswith("VEL_"):
            try:
                vel_bin = int(token.split("_")[1])
                velocity = bin_to_velocity(vel_bin)
                if pending_pitch is not None:
                    if pending_pitch in active_notes:
                        p_start, _ = active_notes.pop(pending_pitch)
                        notes.append({
                            'pitch': pending_pitch,
                            'velocity': velocity,
                            'start': p_start,
                            'end': max(p_start + SHIFT_STEP_MS, current_time)
                        })
                    active_notes[pending_pitch] = (current_time, velocity)
                    pending_pitch = None
            except (IndexError, ValueError):
                pass

        # Note OFF
        elif token.startswith("NOTE_") and token.endswith("_OFF"):
            try:
                pitch = int(token.split("_")[1])
                if pending_pitch == pitch:
                    # Pitch was turned on with no velocity token, start and immediately close
                    active_notes[pitch] = (current_time, default_velocity)
                    pending_pitch = None

                if pitch in active_notes:
                    start_time, vel = active_notes.pop(pitch)
                    end_time = max(start_time + SHIFT_STEP_MS, current_time)
                    notes.append({
                        'pitch': pitch,
                        'velocity': vel,
                        'start': start_time,
                        'end': end_time
                    })
            except (IndexError, ValueError):
                pass

        # Sustain Pedal
        elif token == "SUSTAIN_ON":
            pedals.append({'time': current_time, 'value': 127})
        elif token == "SUSTAIN_OFF":
            pedals.append({'time': current_time, 'value': 0})

    # Flush any remaining pending pitch
    if pending_pitch is not None:
        active_notes[pending_pitch] = (current_time, default_velocity)

    # Close any unclosed notes at final timestamp
    final_time = current_time + SHIFT_STEP_MS
    for pitch, (start_time, vel) in active_notes.items():
        notes.append({
            'pitch': pitch,
            'velocity': vel,
            'start': start_time,
            'end': max(start_time + SHIFT_STEP_MS, final_time)
        })

    # Sort notes by start time
    notes.sort(key=lambda n: (n['start'], n['pitch']))
    return notes, pedals


def tokens_to_midi(
    tokens: List[str],
    output_path: Optional[Union[str, Path]] = None,
    bpm: int = 120,
    ticks_per_beat: int = 480
):
    """
    Convert a sequence of tokens into a MIDI file using miditoolkit.
    """
    try:
        from miditoolkit import MidiFile, Instrument, Note, TempoChange, ControlChange
    except ImportError:
        raise ImportError("miditoolkit is required to export MIDI files. Install with: pip install miditoolkit")

    notes, pedals = tokens_to_events(tokens)

    midi_obj = MidiFile(ticks_per_beat=ticks_per_beat)
    midi_obj.tempo_changes = [TempoChange(bpm=bpm, time=0)]

    inst = Instrument(program=0, is_drum=False, name="Piano")
    for n in notes:
        inst.notes.append(Note(
            pitch=n['pitch'],
            velocity=n['velocity'],
            start=n['start'],
            end=n['end']
        ))

    for p in pedals:
        inst.control_changes.append(ControlChange(
            number=64,
            value=p['value'],
            time=p['time']
        ))

    midi_obj.instruments.append(inst)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        midi_obj.dump(str(out_p))
        print(f"Saved generated MIDI to: {output_path}")

    return midi_obj


def ids_to_midi(
    token_ids: List[int],
    vocab: MusicVocab,
    output_path: Optional[Union[str, Path]] = None,
    bpm: int = 120
):
    """Convert token IDs back to a MIDI file using the vocabulary."""
    tokens = vocab.decode(token_ids)
    return tokens_to_midi(tokens, output_path=output_path, bpm=bpm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert token text file to MIDI")
    parser.add_argument("--input", type=str, default="Postprocessing/generated_tokens.txt", help="Input tokens text file")
    parser.add_argument("--output", type=str, default="Postprocessing/generated.mid", help="Output MIDI file path")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo BPM")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            toks = [line.strip() for line in f if line.strip()]
        tokens_to_midi(toks, output_path=args.output, bpm=args.bpm)
    else:
        print(f"Input token file not found: {args.input}")
