# Project Development Logs: Music Generation (RoPE-MT)

This document tracks all changes, bug fixes, module implementations, architecture updates, and verification results across the development stages.

---

## 📅 Stage 1: Vocabulary & MIDI Event Synchronization
*Status: Completed & Pushed to GitHub (`commit 1943733`)*

### 1. Key Problems Addressed
- **Vocabulary & Postprocessing Mismatch**: `Data/tokenizer.py` produced event tokens (`SHIFT_...`, `NOTE_<pitch>_ON`, `VEL_<bin>`, `NOTE_<pitch>_OFF`, `SUSTAIN_ON/OFF`), whereas `Postprocessing/to_midi.py` expected mismatched tokens (`TIME_SHIFT_`, `DUR_`, `TEMPO_`, `PEDAL_ON`).
- **Missing Shared Vocab Module**: Token mapping was scattered and relied on hardcoded PyCharm absolute paths.
- **Quantization & Note Off Tracking**: Note-on and note-off events risked producing dangling notes or out-of-bounds velocities.

### 2. Implementations & Fixes
- **`Data/vocab.py`**:
  - Implemented `MusicVocab` class with bidirectional lookup (`token_to_id`, `id_to_token`).
  - Standardized special tokens: `[PAD]` (0), `[BOS]` (1), `[EOS]` (2), `[UNK]` (3).
  - Configured 32 velocity bins (`velocity_to_bin`, `bin_to_velocity`) and 50ms time shift quantization (`quantize_time`).
  - Added JSON serialization (`save`, `load`) and vocabulary generator (`build_standard_vocab`, `build_from_files`).
- **`Data/tokenizer.py`**:
  - Unified event extraction: `extract_events_from_notes()` and `events_to_tokens()`.
  - Added timestamp sorting with event priorities (`SUSTAIN` -> `NOTE_OFF` -> `NOTE_ON`).
  - Added support for large time-shift splitting (`MAX_SHIFT_MS = 10000`).
- **`Postprocessing/to_midi.py`**:
  - Rebuilt `tokens_to_events()` and `tokens_to_midi()` to parse the exact token vocabulary.
  - Added active note state tracking to accurately pair Note-On and Note-Off events with start/end millisecond timestamps.
  - Added CLI interface (`--input`, `--output`, `--bpm`).
- **`Preprocessing/token_id_maps.py`**:
  - Updated to use relative default paths pointing to `vocab/vocab.json`.
- **`.gitignore`**:
  - Added rules to ignore `.DS_Store`, `.idea/`, `.vscode/`, `checkpoints/`, `*.pt`, `*.mid`, and generated files.

### 3. Verification & Automated Tests
- **`tests/test_midi_roundtrip.py`**:
  - ✅ **Test 1**: `MusicVocab` encode/decode identity test.
  - ✅ **Test 2**: Vocabulary JSON serialization and reloading.
  - ✅ **Test 3**: Velocity (32 bins) and time delta (50ms step) quantization tolerance.
  - ✅ **Test 4**: Full synthetic MIDI (chords + melody + sustain pedal) -> Tokens -> Reconstructed MIDI round-trip.

---

## 📅 Stage 2: Model Architecture Fixes & Verification
*Status: Completed & Tested*

### 1. Key Problems Addressed
- **RoPE Dimension Mismatch**: `Model/rope.py` unsqueezed cosine/sine cache across incorrect dimensions for 4D multi-head attention queries and keys `(batch, heads, seq_len, d_k)`.
- **Redundant Learned Embeddings**: `Model/MusicTransformer.py` added static absolute position embeddings `pos_embedding` on top of Rotary Positional Embeddings.
- **Future Token Leakage / Masking**: Missing automatic lower-triangular causal attention masking in autoregressive decoding.
- **Weight Tying & Sampling**: Missing parameter weight tying between token embeddings and LM head, and missing nucleus ($p$) and top-$k$ sampling generation.

### 2. Implementations & Fixes
- **`Model/rope.py`**:
  - Aligned caching to broadcast properly with 4D attention tensors `(1, 1, seq_len, dim)`.
  - Added dynamic sequence length cache resizing when sequence exceeds initial cache length.
- **`Model/attention.py`**:
  - Refactored `MultiHeadedAttention` with explicit $W_q, W_k, W_v, W_o$ projections and RoPE rotation.
  - Implemented `scaled_dot_product_attention` with support for causal triangular masking.
  - Standardized `PositionwiseFeedForward` with GELU activation and dropout.
- **`Model/TransferDecoderBlock.py`**:
  - Implemented Pre-LayerNorm Transformer decoder blocks and deep stack decoder `TransformerDecoder`.
- **`Model/MusicTransformer.py`**:
  - Removed redundant absolute position parameter.
  - Added automatic causal mask generator (`generate_causal_mask`).
  - Added weight tying between input embedding and LM head projection.
  - Implemented built-in autoregressive `generate()` method with top-$k$, top-$p$ (nucleus) sampling and temperature scaling.
- **`Model/__init__.py`**:
  - Clean exports for all architecture components.

### 3. Verification & Automated Tests
- **`tests/test_model.py`**:
  - ✅ **Test 1 (RoPE Transformation)**: Verified 4D tensor transformation and dimension preservation.
  - ✅ **Test 2 (Causal Mask Invariance)**: Verified prefix logits differ by $0.00\times 10^0$ when modifying future tokens (strict zero-leakage guarantee).
  - ✅ **Test 3 (Forward & Backward)**: Verified output tensor shape `(batch, seq_len, vocab_size)` and gradient flow across all parameters.
  - ✅ **Test 4 (Autoregressive Generation)**: Verified multi-step token generation from prompt prefix.


---

## 📅 Stage 3: Dataset Tokenization & PyTorch Pipeline
*Status: Completed & Tested*

### 1. Key Objectives & Implementations
- **Preprocessing CLI (`Preprocessing/tokenize_dataset.py`)**:
  - Implemented command-line interface supporting `--midi_dir Data/raw --out_dir Data/tokenized --vocab_out vocab/vocab.json`.
  - Added automatic detection and parsing of MAESTRO metadata (`maestro-v3.0.0.json` / `maestro-v3.0.0.csv`).
  - Added NumPy 1.20+ compatibility shims (`np.int = int`, `np.float = float`) for `miditoolkit`.
  - Tokenized all **1,276 MAESTRO MIDI files** (0 failures):
    - `train/`: **962** pieces
    - `validation/`: **137** pieces
    - `test/`: **177** pieces
- **PyTorch Dataset (`Data/dataset.py`)**:
  - Implemented `MusicDataset` with sliding-window chunking into $(x, y)$ next-token pairs.
  - Automatically aligns targets ($y_t = x_{t+1}$) and masks `[PAD]` targets with `-100` for PyTorch cross-entropy loss.
  - Implemented `collate_music_batch()` and `create_music_dataloaders()`.
  - Validated DataLoader generation over the full tokenized dataset:
    - Train batches: **10,465** (seq_len=512, batch_size=8 $\approx 43\text{M}$ tokens)
    - Val batches: **1,192**
    - Test batches: **1,370**
- **Automated Tests (`tests/test_dataset.py`)**:
  - ✅ **Test 1**: Sequence chunking, target alignment, and padding mask verification.
  - ✅ **Test 2**: Multi-batch DataLoader iteration.

---

## 📅 Stage 4: Training Pipeline (`train.py`)
*Status: Completed & Verified*

### 1. Key Implementations & Features
- **Device Support**: Automatic selection across `CUDA -> MPS (Apple Silicon) -> CPU` with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Training Constraints & Defaults**:
  - FP32 precision by default without GradScaler or torch.compile.
  - `num_workers=0`, `pin_memory=False` for maximum stability and low overhead.
  - Default architecture config: `n_layers=6`, `d_model=512`, `n_heads=8`, `d_ff=2048`, `seq_len=512`, `batch_size=8`, `grad_accum=4`.
- **Optimization & Scheduling**:
  - Optimizer: `AdamW` with weight decay 0.01, betas (0.9, 0.98).
  - Scheduler: Linear Warmup (500 steps) followed by Cosine Annealing decay down to minimum learning rate.
  - Gradient clipping: Max norm $1.0$ (`torch.nn.utils.clip_grad_norm_`).
- **Loss Computation & Memory Optimization**:
  - Cross-Entropy Loss with `ignore_index=-100`.
  - Loss is accumulated on-device as a detached tensor without host `.item()` synchronization on every step.
- **Logging & Checkpointing**:
  - Logs training loss every 50 steps.
  - Evaluates validation loss and perplexity ($\text{PPL} = \exp(\text{val\_loss})$) every 500 steps and appends to `checkpoints/train_log.csv`.
  - Saves full training state checkpoint every 500 steps (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `step`, `config`) and tracks `best_model.pt`.
  - Generates a sample `.mid` composition every 2,000 steps using `ids_to_midi()`.

### 2. Verification & Overfit Test (`--overfit_batch`)
- **Initial Loss Sanity Check**:
  - Initial Loss on real batch: **5.8185**
  - Theoretical $\ln(\text{vocab\_size}) = \ln(319) = \mathbf{5.7652}$
  - Difference: **0.0533** (confirms uniform random initialization across vocabulary without bias).
- **Single Batch Memorization Convergence**:
  - Step 001: Loss = **5.8438**
  - Step 025: Loss = **1.4708**
  - Step 050: Loss = **0.5572**
  - Step 075: Loss = **0.0747**
  - Step 100: Loss = **0.0257**
  - Step 150: Loss = **0.0152**
  - Result: ✅ Overfit test passed, driving loss to $0.0152$.
  - Sample generated and saved to: `checkpoints/samples/sample_overfit.mid`.

---

## 📅 Stage 5: Inference CLI, Generation Script & Documentation
*Status: Pending*

