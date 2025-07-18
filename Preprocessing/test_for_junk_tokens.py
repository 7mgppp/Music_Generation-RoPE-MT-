from collections import Counter
from pathlib import Path

path = Path("/Users/miilee/PycharmProjects/Music Genration/OutputFiles/google_tokens")
counter = Counter()

for file in path.glob("*.txt"):
    with open(file) as f:
        for line in f:
            token = line.strip()
            counter[token] += 1

for token, freq in counter.most_common():
    print(f"{token}: {freq}")

