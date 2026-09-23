# Music Generation with RoPE Transformer (Music_Generation-RoPE-MT-)

A decoder-only Transformer for autoregressive symbolic music generation (classical solo piano MIDI), trained on the [MAESTRO v3.0.0 Dataset](https://magenta.tensorflow.org/datasets/maestro).

The model features custom **Rotary Position Embeddings (RoPE)**, a **MidiTok REMI + BPE** tokenization pipeline, and **offline pitch-shift data augmentation**, compressing musical representations while preserving long-range harmonic and temporal structure.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Journey & Engineering Key Decisions](#journey--engineering-key-decisions)
  - [Phase 1: Custom Tokenizer & The Pedal Degeneration Trap](#phase-1-custom-tokenizer--the-pedal-degeneration-trap)
  - [Phase 2: Deduplication & Short-Context Bottleneck](#phase-2-deduplication--short-context-bottleneck)
  - [Phase 3: MidiTok (REMI + BPE) Compression](#phase-3-miditok-remi--bpe-compression)
  - [Phase 4: Offline Pitch Augmentation & Scaled Training](#phase-4-offline-pitch-augmentation--scaled-training)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [1. Data Preparation & Tokenization](#1-data-preparation--tokenization)
  - [2. Model Training](#2-model-training)
  - [3. Music Generation](#3-music-generation)
- [Training & Evaluation Results](#training--evaluation-results)
- [Generated Music Samples](#generated-music-samples)
- [Recommended Generation Settings](#recommended-generation-settings)
- [Known Limitations](#known-limitations)
- [Acknowledgements & Citations](#acknowledgements--citations)

---

## Overview

Generating multi-voice symbolic piano music requires capturing both local polyphony (simultaneous chord notes, rapid arpeggios, sustain pedal states) and long-range structural coherence (phrasing, cadence, harmonic development). 

Standard absolute positional embeddings decay in relative timing perception across long token sequences. This project implements a from-scratch **Rotary Position Embedding (RoPE) Decoder-Only Transformer** paired with **Byte Pair Encoding (BPE)** over a **REMI** (Revamped MIDI) token vocabulary to generate expressive classical piano pieces.

---

## Architecture

Defined in `Model/MusicTransformer.py`, the model uses a causal decoder-only architecture with RoPE applied to query and key states at each attention head:

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Layers** (`N`) | `6` | Transformer Decoder Layers |
| **Attention Heads** (`H`) | `8` | Multi-Head Self-Attention with Causal Mask |
| **Embedding Dimension** (`d_model`) | `512` | Head dimension `d_k = 64` |
| **Feed-Forward Dimension** (`d_ff`) | `2048` | Expansion factor of 4 with GELU activation |
| **Max Context Length** (`L_max`) | `1024` | 1024 tokens (~550 notes under BPE) |
| **Positional Encoding** | **RoPE** | Rotary Position Embeddings (base &theta; = 10000) |
| **Vocabulary Size** (`V`) | `4096` | MidiTok REMI + BPE |
| **Total Parameters** | **21,000,192** | 21M trainable parameters (tied embedding/head) |
| **Regularization** | `0.1` | Dropout across attention and feed-forward projections |

---

## Journey & Engineering Key Decisions

The final architecture and pipeline are the result of iterative debugging and structural refinement across several distinct phases:

### Phase 1: Custom Tokenizer & The Pedal Degeneration Trap
- **Initial Setup**: Started with a custom event-based tokenizer (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`, `SUSTAIN_ON/OFF`).
- **Failure Mode**: The model consistently degenerated into infinite repetitive loops of `[SUSTAIN_ON, TIME_SHIFT_50, SUSTAIN_ON...]`.
- **Root Cause Analysis**: Analysis of raw MAESTRO MIDI files revealed massive continuous Control Change (CC64) polling by electronic pianos/recording rigs. **91.33% of all `SUSTAIN_ON` events in the dataset were redundant repetitions** sent without an intervening `SUSTAIN_OFF`. The model learned the superficial statistical frequency of these tokens and fell into an inescapable degenerate attractor state.

### Phase 2: Deduplication & Short-Context Bottleneck
- **The Fix**: Implemented state-tracked pedal deduplication, preserving sustain semantics while filtering redundant CC64 spam.
- **Outcome (`checkpoints_v2`)**: Eliminated the infinite pedal loop. However, uncompressed event representations consumed 3.80 tokens per note on average. At a context window of 512 tokens, the model could only see ~130 notes (~5–8 seconds of music), leading to rapid loss of harmonic direction and premature phrasing collapse.

### Phase 3: MidiTok (REMI + BPE) Compression
- **Migration**: Rebuilt the tokenization pipeline using **MidiTok** with **REMI** (Revamped MIDI) representation + **Byte Pair Encoding (BPE)**:
  - Time representation aligned to a musical grid (8/12 sub-beats per quarter note).
  - Velocity binned into 32 expressive levels; explicit note durations instead of decoupled `NOTE_OFF` events.
  - Trained a 4096-token BPE vocabulary on the MAESTRO training split (`vocab/miditok_remi_bpe.json`).
- **Impact**:
  - **Token density reduced by ~51%**: Dropped from **3.80 tokens/note** down to **1.86 tokens/note**.
  - **Context doubling**: Extended model context to `L_max = 1024`, allowing the model to attend across **~550 notes** (30–60+ seconds of dense classical music).

### Phase 4: Offline Pitch Augmentation & Scaled Training
- **Data Augmentation**: Classical repertoire is key-dependent and prone to overfitting. We introduced pitch shifting in the range [-5, +5] semitones (11 variants per piece).
- **Architecture Consideration**: Because BPE tokens represent merged n-grams, pitch shifting cannot be performed arithmetically on token IDs. Instead, MIDI files are transposed *prior* to base tokenization and BPE encoding.
- **Caching Mechanism**: All 11 pitch-shifted variants for the 962 training pieces were pre-encoded and cached to disk (`Data/tokenized_remi/train/variants_cache.pt`, ~245 MB). During training, each source chunk samples a random pitch variant on the fly (*O(1)* memory lookup), avoiding runtime CPU bottlenecks.
- **Validation Isolation**: Validation and test splits remain strictly un-augmented and unshifted.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- macOS (Apple Silicon MPS supported) or Linux with CUDA GPU

### Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/Music_Generation-RoPE-MT-.git
cd Music_Generation-RoPE-MT-

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies (`requirements.txt`)
- `torch == 2.2.2`
- `miditok == 3.0.6.post1`
- `symusic == 0.6.0`
- `miditoolkit == 0.1.16`
- `numpy == 1.26.4`

---

## Usage

### 1. Data Preparation & Tokenization

Tokenize MAESTRO v3.0.0 MIDI files into REMI + BPE representation:

```bash
# Train BPE vocabulary (MAESTRO train split only) and tokenize dataset
python3 Data/tokenize_remi.py \
    --midi_dir /path/to/maestro-v3.0.0 \
    --vocab_size 4096 \
    --output_dir Data/tokenized_remi \
    --vocab_out vocab/miditok_remi_bpe.json
```

### 2. Model Training

Train the RoPE Music Transformer with AdamW, Cosine LR scheduling with linear warmup, and gradient accumulation:

```bash
python3 train.py \
    --data_dir Data/tokenized_remi \
    --vocab_path vocab/miditok_remi_bpe.json \
    --checkpoint_dir checkpoints_remi \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 1024 \
    --lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_steps 170 \
    --max_steps 3400 \
    --eval_interval 170 \
    --save_interval 340
```

*Resume from existing checkpoint:*
```bash
python3 train.py --resume checkpoints_remi/best_model.pt
```

### 3. Music Generation

#### Primed Continuation (Recommended)
Provide an initial MIDI excerpt (e.g., the first 4–8 bars) to prime the model's key, tempo, and texture:

```bash
python3 generate_miditok.py \
    --checkpoint checkpoints_remi/best_model.pt \
    --vocab_path vocab/miditok_remi_bpe.json \
    --prompt_midi path/to/prompt.mid \
    --max_tokens 1024 \
    --temperature 0.85 \
    --top_k 30 \
    --top_p 0.9 \
    --output_path generated_primed.mid
```

#### Cold-Start Generation (Unprimed)
```bash
python3 generate_miditok.py \
    --checkpoint checkpoints_remi/best_model.pt \
    --vocab_path vocab/miditok_remi_bpe.json \
    --max_tokens 1024 \
    --temperature 0.75 \
    --top_k 20 \
    --top_p 0.9 \
    --output_path generated_unprimed.mid
```

---

## Training & Evaluation Results

The final model (`checkpoints_remi/best_model.pt`) was trained for 10 full epochs (3,400 optimizer steps, batch size 32 effective) on the MAESTRO train split with pitch augmentation.

| Metric | Initial (Step 1) | Midpoint (Step 1700) | Final (Step 3400) |
| :--- | :--- | :--- | :--- |
| **Train Loss** | `8.5024` (near ln 4096 = 8.32) | `4.2180` | **`3.8711`** |
| **Validation Loss** | `8.4110` | `4.3412` | **`3.9837`** |
| **Validation Perplexity** | `4500+` | `76.80` | **`53.72`** |

### Round-Trip MIDI Tokenization Fidelity
Tested over validation samples to verify lossless/near-lossless reconstruction:
- **Note Count Retention**: 100%
- **Pitch-Set Overlap**: 100%
- **Max Timing Error**: &le; 31.25 ms (sub-grid resolution boundary limit)

---

## Generated Music Samples

Listen to full MIDI compositions generated by the trained RoPE Transformer:

| Sample File | Mode | Sampling Parameters | Description |
| :--- | :--- | :--- | :--- |
| **[`remi_primed_t07.mid`](samples/remi_primed_t07.mid)** | **Primed Continuation** | `temperature = 0.70`<br>`top_k = 30`<br>`top_p = 0.90` | Conditioned on an initial classical piano motif; develops melodic themes with natural harmonic cadences and polyphonic accompaniment. |
| **[`remi_t09.mid`](samples/remi_t09.mid)** | **Cold-Start (Unprimed)** | `temperature = 0.90`<br>`top_k = 40`<br>`top_p = 0.90` | Generated entirely from `<BOS>` token with expressive velocity variation and dynamic sustain pedaling. |

> **Playback tip**: You can open `.mid` files directly in any digital audio workstation (DAW) like Logic Pro, Ableton, FL Studio, GarageBand, MuseScore, or VLC media player.

---

## Recommended Generation Settings

Autoregressive symbolic generation is sensitive to sampling temperature and sampling bounds:

- **Primed Continuation**: `temperature = 0.80 - 0.90`, `top_k = 25 - 40`, `top_p = 0.90`.
- **Unprimed (From Scratch)**: `temperature = 0.70 - 0.80`, `top_k = 15 - 30`, `top_p = 0.85`.
- *Note*: Setting `temperature > 1.0` or `top_k > 60` without priming increases the likelihood of out-of-scale pitch wandering.

---

## Known Limitations

1. **Cold-Start Harmonic Drift**: Without a prompt MIDI, unprimed generation can occasionally wander harmonically before establishing a consistent tonal center. Primed continuation (`--prompt_midi`) yields significantly better structural coherence.
2. **Global Sonata/Form Constraints**: While the 1024-token window covers 30–60 seconds of complex music, multi-movement structures or strict classical Sonata-Allegro recapitulations over 5+ minutes exceed causal attention span.
3. **Dataset Specificity**: The model is specialized for acoustic solo classical piano (MAESTRO) and does not handle multi-track orchestral or multi-instrument MIDI.

---

## Acknowledgements & Citations

- **Dataset**: [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) (MIDI and Audio Edited for Synchronous TRacks and Organization) by Google Magenta.
- **Tokenization**: [MidiTok](https://github.com/Natooz/MidiTok) (REMI and BPE implementations).
- **Positional Encoding**: [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864).
