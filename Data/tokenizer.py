from miditoolkit import MidiFile
from pathlib import Path
from tqdm import tqdm
import json


# Constants
VELOCITY_BINS = 32
MAX_SHIFT_MS = 10000
SHIFT_STEP_MS = 50
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"


def velocity_to_bin(vel):
    bin_size = 128 // VELOCITY_BINS
    return min(VELOCITY_BINS - 1, max(0, vel // bin_size))


def quantize_time(delta):
    steps = max(1, round(delta / SHIFT_STEP_MS))
    quantized = steps * SHIFT_STEP_MS
    return min(MAX_SHIFT_MS, quantized)


# Fixed midi_to_tokens function
def midi_to_tokens(midi: MidiFile):
    events = []
    active_notes = {}  # active notes to prevent dangling NOTE_OFFs

    for inst in midi.instruments:
        if inst.is_drum:
            continue

        # Process notes
        for note in inst.notes:
            events.append((note.start, 'NOTE_ON', note.pitch, note.velocity))
            events.append((note.end, 'NOTE_OFF', note.pitch))
            active_notes[note.pitch] = False  # Initialize note state

        # Process sustain pedals (CC64)
        for cc in inst.control_changes:
            if cc.number == 64:  # Sustain pedal
                value = 127 if cc.value >= 64 else 0
                events.append((cc.time, 'SUSTAIN', value))

    # Sort by time, then by event type priority
    events.sort(key=lambda x: (x[0], {'SUSTAIN': 0, 'NOTE_OFF': 1, 'NOTE_ON': 2}.get(x[1], 3)))

    tokens = [BOS_TOKEN]
    last_time = 0
    sustain_active = False

    for event in events:
        time, event_type, *values = event

        # Handle time shift
        delta = time - last_time
        if delta > 0:
            q_delta = quantize_time(delta)
            tokens.append(f"SHIFT_{int(q_delta)}")  # Convert to int for consistent formatting
            last_time = time

        # Handle events
        if event_type == "NOTE_ON":
            pitch, velocity = values
            if active_notes.get(pitch, False):
                # Turn off previous instance of same pitch before turning on new one
                tokens.append(f"NOTE_{pitch}_OFF")
            vel_bin = velocity_to_bin(velocity)
            tokens.append(f"NOTE_{pitch}_ON")
            tokens.append(f"VEL_{vel_bin}")
            active_notes[pitch] = True

        elif event_type == "NOTE_OFF":
            pitch = values[0]
            if active_notes.get(pitch, False):
                tokens.append(f"NOTE_{pitch}_OFF")
                active_notes[pitch] = False
            # Else: note wasn't active, skip to avoid dangling NOTE_OFF

        elif event_type == "SUSTAIN":
            value = values[0]
            if value == 127:
                tokens.append("SUSTAIN_ON")
                sustain_active = True
            else:
                tokens.append("SUSTAIN_OFF")
                sustain_active = False

    # Turn off any remaining active notes before EOS
    for pitch, is_active in active_notes.items():
        if is_active:
            tokens.append(f"NOTE_{pitch}_OFF")

    tokens.append(EOS_TOKEN)
    return tokens


def save_tokens(tokens, path):
    with open(path, "w") as f:
        f.write("\n".join(tokens))


def build_vocabulary(token_dir):
    token_files = list(Path(token_dir).glob("*.txt"))
    if not token_files:
        print("No token files found!")
        return {}

    vocab_counter = {}

    for f in token_files:
        with open(f, 'r') as file:
            tokens = file.read().splitlines()
            for token in tokens:
                vocab_counter[token] = vocab_counter.get(token, 0) + 1

    # Create vocabulary with special tokens
    vocab = {
        PAD_TOKEN: 0,
        BOS_TOKEN: 1,
        EOS_TOKEN: 2,
        "[UNK]": 3
    }

    # Add musical tokens (filter rare tokens)
    idx = len(vocab)
    for token, count in vocab_counter.items():
        if token not in vocab and count >= 5:
            vocab[token] = idx
            idx += 1

    return vocab


if __name__ == "__main__":
    midi_dir = Path("/Users/miilee/Desktop/maestro-v3.0.0")
    out_dir = Path("../OutputFiles/google_tokens")
    vocab_dir = Path("../OutputFiles/vocab")

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi")))
    if not files:
        print("No MIDI files found.")
        exit()

    print(f"Found {len(files)} MIDI files.")
    for f in tqdm(files, desc="Tokenizing", unit="file"):
        try:
            midi = MidiFile(f)
            if not midi.instruments:
                continue
            tokens = midi_to_tokens(midi)
            save_path = out_dir / f"{f.stem}.txt"
            save_tokens(tokens, save_path)
        except Exception as e:
            print(f"Failed on {f.name}: {str(e)}")
            continue

    # Build vocabulary
    vocab = build_vocabulary(out_dir)
    vocab_path = vocab_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"Tokenized {len(list(out_dir.glob('*.txt')))} files")
    print(f"Vocabulary size: {len(vocab)}")