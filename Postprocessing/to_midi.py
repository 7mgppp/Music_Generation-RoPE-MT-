from miditoolkit import MidiFile, Instrument, Note, TempoChange, ControlChange

# Load generated tokens
with open("generated_tokens.txt") as f:
    tokens = [line.strip() for line in f.readlines()]

TIME = 0
notes = []
tempos = []
controls = []

pitch = None
dur = None
vel = 64  # default velocity

for token in tokens:
    if token.startswith("TIME_SHIFT_"):
        shift = int(token.split("_")[-1])
        TIME += shift
    elif token.startswith("NOTE_ON_"):
        pitch = int(token.split("_")[-1])
    elif token.startswith("DUR_"):
        dur = int(token.split("_")[-1])
    elif token.startswith("VEL_"):
        vel = int(token.split("_")[-1])
    elif token.startswith("TEMPO_"):
        bpm = int(token.split("_")[-1])
        tempos.append(TempoChange(bpm, TIME))
    elif token == "PEDAL_ON":
        controls.append(ControlChange(64, 127, TIME))
    elif token == "PEDAL_OFF":
        controls.append(ControlChange(64, 0, TIME))

    # Once pitch + dur are both ready, we create the note
    if pitch is not None and dur is not None:
        notes.append(Note(velocity=vel, pitch=pitch, start=TIME, end=TIME + dur))
        pitch = None
        dur = None

# Write to MIDI
inst = Instrument(program=0, is_drum=False, name="Piano")
inst.notes = notes
inst.control_changes = controls

midi = MidiFile()
midi.instruments.append(inst)
if tempos:
    midi.tempo_changes = tempos

midi.dump("generated.mid")
print("Saved 'generated.mid' — open it in MuseScore, GarageBand, or OnlineSequencer to listen.")
