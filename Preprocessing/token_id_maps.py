import json
from pathlib import Path

def create_reverse_vocab(vocab_path, output_path=None):
    # Set default output path if not provided
    if output_path is None:
        output_path = Path(vocab_path).parent / "id_to_token.json"

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)

    # Create reverse mapping using list for O(1) lookup by id
    max_id = max(token_to_id.values())
    id_to_token = [None] * (max_id + 1)

    for token, idx in token_to_id.items():
        # Ensure no duplicate IDs
        if id_to_token[idx] is not None:
            raise ValueError(f"Duplicate ID detected: {idx} for tokens '{id_to_token[idx]}' and '{token}'")
        id_to_token[idx] = token

    for i in range(len(id_to_token)):
        if id_to_token[i] is None:
            id_to_token[i] = "[UNK]"

    # Save reverse mapping
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(id_to_token, f, indent=2, ensure_ascii=False)

    print(f"Reverse vocabulary created with {len(id_to_token)} entries")
    print(f"Saved to: {output_path}")
    return id_to_token


if __name__ == "__main__":
    VOCAB_PATH = Path("/Users/miilee/PycharmProjects/Music Genration/OutputFiles/vocab/vocab.json")
    create_reverse_vocab(VOCAB_PATH)