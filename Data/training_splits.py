import os
import shutil
from tqdm import tqdm
import pandas as pd
from pathlib import Path


MAESTRO_JSON = "/Users/miilee/Desktop/maestro-v3.0.0/maestro-v3.0.0.json"
TOKENS_DIR = "/Users/miilee/PycharmProjects/Music Genration/OutputFiles/google_tokens"
OUTPUT_DIR = Path("../OutputFiles/Split")


for split in ["train", "validation", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


metadata = pd.read_json(MAESTRO_JSON)
print(f"Loaded metadata: {len(metadata)} entries")


copied = 0
missing = 0

for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Splitting"):
    midi_file = os.path.basename(row["midi_filename"])
    txt_name = os.path.splitext(midi_file)[0] + ".txt"
    split = row["split"]

    src = os.path.join(TOKENS_DIR, txt_name)
    dst = OUTPUT_DIR / split / txt_name

    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing += 1

print(f"Copied: {copied}   Missing: {missing}")
print(f"Output folders: {OUTPUT_DIR}/train, validation, test")
