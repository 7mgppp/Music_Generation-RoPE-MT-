import torch
import json
import torch.nn.functional as F
from tqdm import trange
from Model.MusicTransformer import MusicTransformer

# --- CONFIG ---
MAX_GEN_LEN = 128
MODEL_PATH = "/Users/miilee/PycharmProjects/Music Genration/music_transformer.pt"
VOCAB_PATH = "/Users/miilee/PycharmProjects/Music Genration/OutputFiles/vocab.json"
OUTPUT_TOKENS_PATH = "generated_tokens.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load vocab ---
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

# --- Load model ---
model = MusicTransformer(
    vocab_size=len(vocab),
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    max_seq_len=1024
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Top-k filtering ---
def top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)  # [batch, 1]
    return torch.where(logits < min_values, torch.full_like(logits, -float("Inf")), logits)

# --- Start generation with BOS ---
input_ids = [vocab["BOS"]]
generated = input_ids[:]

temperature = 1.0
top_k = 20

with torch.no_grad():
    for _ in trange(MAX_GEN_LEN, desc="Generating"):
        input_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, T]

        out = model(input_tensor)  # [1, T, V]
        next_token_logits = out[0, -1, :] / temperature  # [V]

        filtered_logits = top_k_logits(next_token_logits.unsqueeze(0), k=top_k).squeeze(0)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        if inv_vocab.get(next_token) == "EOS":
            break

# --- Save tokens ---
tokens = [inv_vocab[i] for i in generated]
with open(OUTPUT_TOKENS_PATH, "w") as f:
    f.write("\n".join(tokens))

print(f"✅ Generated tokens saved to {OUTPUT_TOKENS_PATH}")

