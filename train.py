"""
Training Pipeline for Symbolic Music Generation with RoPE Transformer (RoPE-MT)
Supports MidiTok (REMI + BPE), automatic device selection (CUDA -> MPS -> CPU),
gradient accumulation, warmup + cosine decay, periodic checkpointing, CSV logging,
and sample generation.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import math
import time
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from miditok import REMI
import symusic
from Data.dataset_miditok import create_miditok_dataloaders
from Model.MusicTransformer import MusicTransformer


def get_device() -> torch.device:
    """Select compute device automatically: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA Device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🍏 Using Apple Silicon MPS Acceleration")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """Linear warmup followed by cosine annealing decay down to min_lr at final step."""
    num_warmup_steps = max(1, num_warmup_steps)
    num_training_steps = max(num_warmup_steps + 1, num_training_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / float(num_training_steps - num_warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    vocab_size: int,
    device: torch.device,
    max_eval_batches: Optional[int] = 50
) -> Dict[str, float]:
    """Evaluate model loss and perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_evaluated = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_eval_batches and batch_idx >= max_eval_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=False)
        target_ids = batch["target_ids"].to(device, non_blocking=False)

        logits = model(input_ids)
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

        # Count non-padded tokens
        valid_mask = target_ids.view(-1) != -100
        num_valid = valid_mask.sum().item()

        if num_valid > 0:
            total_loss += loss.item() * num_valid
            total_tokens += num_valid
            batches_evaluated += 1

    model.train()
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(20.0, avg_loss)) if avg_loss < 20 else float("inf")
    return {"val_loss": avg_loss, "perplexity": perplexity}


def train(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load MidiTok Tokenizer
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {vocab_path}")
    tokenizer = REMI(params=vocab_path)
    vocab_size = len(tokenizer)
    print(f"📖 Loaded MidiTok REMI BPE tokenizer of size: {vocab_size} from {vocab_path}")

    # 2. Build Model
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        dim_feedforward=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.seq_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Music Transformer Architecture Initialized:")
    print(f"   - Total Parameters:     {total_params:,}")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    print(f"   - Context Window:       {args.seq_len} tokens")

    # 3. Data Loaders
    print(f"📂 Loading dataset from: {args.data_dir} ...")
    loaders = create_miditok_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        raw_midi_dir=args.raw_midi_dir,
    )

    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    print(f"📊 Training chunks: {len(train_loader.dataset):,} | Batches: {len(train_loader):,} (batch_size={args.batch_size})")
    if val_loader:
        print(f"📊 Validation chunks: {len(val_loader.dataset):,} | Batches: {len(val_loader):,}")

    # 4. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    # 5. Learning Rate Scheduler
    if args.warmup_steps is not None:
        warmup_steps = min(args.warmup_steps, args.max_steps)
    else:
        warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    print(f"📈 LR Schedule: Peak LR = {args.lr:.2e}, Min LR = {args.min_lr:.2e}, Warmup = {warmup_steps} steps, Total = {args.max_steps} steps")

    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
        min_lr_ratio=min_lr_ratio
    )

    # 6. Resume from Checkpoint if specified
    start_step = 0
    best_val_loss = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"🔄 Resumed training from step {start_step} (Best Val Loss: {best_val_loss:.4f})")

    # 7. CSV Logger setup
    log_file_path = checkpoint_dir / "train_log.csv"
    is_new_log = not log_file_path.exists()
    csv_file = open(log_file_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if is_new_log:
        csv_writer.writerow(["step", "epoch", "train_loss", "val_loss", "val_ppl", "lr", "tokens_per_sec", "time_elapsed_sec"])

    # 8. Training Loop
    print(f"\n🚀 Starting training for {args.max_steps} steps (Grad Accum={args.grad_accum}, Effective Batch={args.batch_size * args.grad_accum}) ...")
    model.train()

    step = start_step
    epoch = 0
    t0 = time.time()
    accum_loss_tracker = 0.0
    accum_steps_count = 0

    train_iter = iter(train_loader)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=False)
            target_ids = batch["target_ids"].to(device, non_blocking=False)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            # Divide loss by grad_accum for correct backward gradient scale
            (loss / args.grad_accum).backward()

            # Accumulate true per-token mean cross entropy loss for this optimizer step
            step_loss += loss.item() / args.grad_accum

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        step += 1

        accum_loss_tracker += step_loss
        accum_steps_count += 1

        # Log training loss: first 20 steps (and every log_interval thereafter)
        if step <= 20 or (step - start_step) <= 20 or step % args.log_interval == 0:
            avg_train_loss = accum_loss_tracker / max(1, accum_steps_count)
            accum_loss_tracker = 0.0
            accum_steps_count = 0
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            print(f"Step {step:06d}/{args.max_steps} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6e} | Elapsed: {elapsed:.1f}s")
            csv_writer.writerow([step, epoch, f"{avg_train_loss:.4f}", "", "", f"{current_lr:.6e}", "", f"{elapsed:.1f}"])
            csv_file.flush()

        # Validation Evaluation
        if val_loader and step % args.eval_interval == 0:
            print(f"\n🔍 Evaluating on Validation Set at step {step} ...")
            val_metrics = evaluate(model, val_loader, criterion, vocab_size, device)
            val_loss = val_metrics["val_loss"]
            val_ppl = val_metrics["perplexity"]

            print(f"   Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}\n")

            csv_writer.writerow([step, epoch, "", f"{val_loss:.4f}", f"{val_ppl:.2f}", f"{scheduler.get_last_lr()[0]:.6e}", "", f"{time.time() - t0:.1f}"])
            csv_file.flush()

            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": vars(args)
                }, best_path)
                print(f"🏆 New best model saved to: {best_path} (Val Loss: {best_val_loss:.4f})")

        # Save checkpoint periodically
        if step % args.save_interval == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": vars(args)
            }, ckpt_path)
            print(f"💾 Checkpoint saved to: {ckpt_path}")

        # Sample generation periodically
        if step % args.sample_interval == 0:
            sample_path = samples_dir / f"sample_step_{step}.mid"
            print(f"🎼 Generating periodic sample to: {sample_path} ...")
            model.eval()
            with torch.no_grad():
                sample_ids = model.generate(
                    prompt_ids=[tokenizer["BOS_None"]],
                    max_generate_len=args.seq_len,
                    temperature=0.9,
                    top_k=40,
                    top_p=0.9,
                    eos_id=tokenizer["EOS_None"],
                    device=device
                )
            score = tokenizer.decode([sample_ids])
            score.dump_midi(str(sample_path))
            print(f"✨ Periodic sample saved to {sample_path}")
            model.train()

    # Always save final and latest checkpoint at the end of training
    if step > start_step:
        final_ckpt = checkpoint_dir / f"checkpoint_step_{step}.pt"
        latest_ckpt = checkpoint_dir / "checkpoint_latest.pt"
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": vars(args)
        }
        torch.save(state, final_ckpt)
        torch.save(state, latest_ckpt)
        print(f"💾 Final checkpoint saved to: {final_ckpt} and {latest_ckpt}")

    csv_file.close()
    print("🎉 Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Music Transformer (RoPE-MT)")
    parser.add_argument("--data_dir", type=str, default="Data/tokenized_remi", help="Path to tokenized dataset directory")
    parser.add_argument("--raw_midi_dir", type=str, default="Data/raw/maestro-v3.0.0", help="Path to raw MAESTRO dataset directory")
    parser.add_argument("--vocab_path", type=str, default="vocab/miditok_remi_bpe.json", help="Path to MidiTok tokenizer JSON")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length (context window)")
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="FeedForward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Linear warmup steps (defaults to warmup_ratio * max_steps)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup fraction of total steps if warmup_steps is omitted")
    parser.add_argument("--max_steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_remi", help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between training loss logs")
    parser.add_argument("--eval_interval", type=int, default=500, help="Steps between validation evaluations")
    parser.add_argument("--save_interval", type=int, default=500, help="Steps between checkpoints")
    parser.add_argument("--sample_interval", type=int, default=2000, help="Steps between sample MIDI generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train(args)
