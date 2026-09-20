# RoPE Music Transformer (RoPE-MT)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autoregressive, decoder-only Transformer architecture equipped with **Rotary Position Embeddings (RoPE)** for expressive symbolic classical piano music generation. Trained on the [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) dataset using an event-based MIDI representation with stateful constrained decoding.

---

##  Key Features

- **RoPE Attention**: Incorporates Rotary Position Embeddings directly into query/key projections for enhanced long-range relative temporal modeling.
- **Event-Based Token Vocabulary (319 tokens)**:
  - `NOTE_{pitch}_ON` & `NOTE_{pitch}_OFF` (MIDI pitches 21–108)
  - `VEL_{0-31}` (32 discrete velocity bins)
  - `SHIFT_{50-10000}` (Quantized time shifts in 50 ms steps)
  - `SUSTAIN_ON` & `SUSTAIN_OFF` (State-deduplicated pedal transitions)
- **Constrained Inference (`generate.py`)**:
  - Vectorized boolean logit masking for legal note-on and note-off events.
  - Strict pedal state enforcement (prevents duplicate sustain events).
  - Repetition prevention via strict no-repeat $n$-gram ban ($n \ge 8$).
  - Priming with prompt MIDI files and automatic dangling note closure at EOF.
- **Hardware Agnostic**: Automatic compute acceleration selection across **NVIDIA CUDA**, **Apple Silicon MPS**, and **CPU**.

---

##  Repository Structure

```
Music_Generation-RoPE-MT-/
├── Data/
│   ├── raw/                 # Raw MAESTRO MIDI files
│   ├── tokenized_v2/        # Deduplicated event token sequences (train/val/test)
│   ├── dataset.py           # PyTorch Dataset & DataLoader with sliding-window chunking
│   ├── tokenizer.py         # State-tracked MIDI <-> Token event tokenizer
│   └── vocab.py             # MusicVocab mapping & quantization logic
├── Model/
│   ├── MusicTransformer.py  # Autoregressive Transformer with weight tying
│   ├── attention.py         # Multi-Head Attention with RoPE
│   ├── rope.py              # Rotary Position Embedding cache & rotation
│   └── TransferDecoderBlock.py # Pre-LayerNorm Transformer decoder stack
├── Preprocessing/
│   └── tokenize_dataset.py  # Dataset preprocessing & MAESTRO split parsing CLI
├── Postprocessing/
│   └── to_midi.py           # Token-to-playable-MIDI converter
├── vocab/
│   ├── vocab.json           # Shared 319-token vocabulary
│   └── id_to_token.json     # Reverse vocabulary mapping
├── checkpoints/             # Saved model weights & logs
├── tests/                   # Automated verification test suite
├── train.py                 # Training script with cosine decay & validation
├── generate.py              # Music generation CLI with constrained decoding
└── requirements.txt         # Project dependencies
```

---

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Music_Generation-RoPE-MT-.git
   cd Music_Generation-RoPE-MT-
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

##  Quick Start: Music Generation

Generate a piece using the trained model checkpoint:

### 1. Unconditional Generation (from Scratch)
```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --max_len 512 \
  --temperature 0.9 \
  --top_k 40 \
  --top_p 0.9 \
  --output_midi generated_song.mid
```

### 2. Prompt-Primed Generation (Seed with Existing MIDI)
```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt_midi path/to/prompt.mid \
  --max_len 512 \
  --temperature 0.9 \
  --output_midi primed_song.mid
```

---

##  Training the Model

### 1. Preprocess & Tokenize MIDI Data
```bash
python Preprocessing/tokenize_dataset.py \
  --midi_dir Data/raw \
  --out_dir Data/tokenized_v2 \
  --vocab_out vocab/vocab.json
```

### 2. Train with PyTorch
```bash
python train.py \
  --data_dir Data/tokenized_v2 \
  --vocab_path vocab/vocab.json \
  --batch_size 8 \
  --grad_accum 4 \
  --seq_len 512 \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --lr 3e-4 \
  --max_steps 50000 \
  --checkpoint_dir checkpoints
```

---

##  Running Automated Tests

Run the test suite to verify the tokenizer, vocabulary, and model architecture:

```bash
pytest tests/
```

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
