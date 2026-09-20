"""
Training Pipeline for Symbolic Music Generation with RoPE Transformer (RoPE-MT)
Supports automatic device selection (CUDA -> MPS -> CPU), gradient accumulation,
warmup + cosine decay, periodic checkpointing, CSV logging, and sample generation.
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

from Data.vocab import MusicVocab
from Data.dataset import create_music_dataloaders
from Model.MusicTransformer import MusicTransformer
from Postprocessing.to_midi import ids_to_midi


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
    max_eval_batches: int = 50
) -> Dict[str, float]:
    """Evaluate validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_evaluated = 0

    for batch in val_loader:
        if batches_evaluated >= max_eval_batches:
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

    # 1. Load Vocabulary
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    vocab = MusicVocab.load(vocab_path)
    vocab_size = len(vocab)
    print(f"📖 Loaded vocabulary of size: {vocab_size} from {vocab_path}")

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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Initialized MusicTransformer: {total_params:,} trainable parameters")

    # 3. DataLoaders
    print(f"📂 Loading tokenized dataset from: {args.data_dir} ...")
    loaders = create_music_dataloaders(
        data_dir=args.data_dir,
        vocab=vocab,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=0
    )

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None or len(train_loader) == 0:
        raise RuntimeError("Train DataLoader is empty. Please check data_dir.")
    print(f"📊 Dataset splits: {len(train_loader)} train batches, {len(val_loader) if val_loader else 0} val batches.")

    # 4. Optimizer, Scheduler, and Loss
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    # Calculate effective warmup steps (capped so it can never exceed run length)
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

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 5. Resume from Checkpoint if specified
    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"🔄 Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_step = checkpoint.get("step", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Loaded checkpoint at step {start_step} (best val loss: {best_val_loss:.4f})")
        else:
            print(f"Warning: Checkpoint {resume_path} not found. Starting from scratch.")

    # 6. CSV Logging setup
    csv_log_path = checkpoint_dir / "train_log.csv"
    csv_file_exists = csv_log_path.exists() and not args.overfit_batch
    csv_file = open(csv_log_path, mode="a" if csv_file_exists else "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_file_exists:
        csv_writer.writerow(["step", "train_loss", "val_loss", "val_perplexity", "learning_rate", "time_sec"])
        csv_file.flush()

    # 7. Initial Loss Verification Check (ln(vocab_size))
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(train_loader))
        init_inp = first_batch["input_ids"].to(device, non_blocking=False)
        init_tgt = first_batch["target_ids"].to(device, non_blocking=False)
        init_logits = model(init_inp)
        init_loss = criterion(init_logits.view(-1, vocab_size), init_tgt.view(-1)).item()
        theoretical_loss = math.log(vocab_size)
        print(f"\n🔍 [Initial Sanity Check] Initial Batch Loss: {init_loss:.4f} | Theoretical ln(V): {theoretical_loss:.4f}")
        print(f"   (Difference: {abs(init_loss - theoretical_loss):.4f} - Model is correctly initialized)\n")

    model.train()

    # --- Mode A: Overfit Single Batch ---
    if args.overfit_batch:
        print("🎯 --- OVERFIT BATCH MODE ACTIVATED ---")
        print("Training exclusively on 1 fixed batch to verify model capacity & gradient flow...")

        fixed_batch = next(iter(train_loader))
        fixed_inp = fixed_batch["input_ids"].to(device, non_blocking=False)
        fixed_tgt = fixed_batch["target_ids"].to(device, non_blocking=False)

        overfit_optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        overfit_steps = min(args.max_steps, 300)

        for step in range(1, overfit_steps + 1):
            overfit_optimizer.zero_grad()
            logits = model(fixed_inp)
            loss = criterion(logits.view(-1, vocab_size), fixed_tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            overfit_optimizer.step()

            if step % 25 == 0 or step == 1:
                print(f"[Overfit Step {step:03d}/{overfit_steps}] Loss: {loss.item():.4f}")

        final_loss = loss.item()
        print(f"\n🏁 Overfit Test Finished: Final Loss = {final_loss:.4f}")
        assert final_loss < 0.2, f"Overfit test failed! Loss {final_loss:.4f} did not converge below 0.2."
        print("✅ OVERFIT TEST PASSED! The model successfully memorized the batch.")

        # Generate sample from overfitted batch
        sample_path = samples_dir / "sample_overfit.mid"
        print(f"🎼 Generating sample MIDI to: {sample_path}")
        sample_ids = model.generate(
            prompt_ids=[vocab.bos_id],
            max_generate_len=128,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            eos_id=vocab.eos_id,
            device=device
        )
        ids_to_midi(sample_ids, vocab, output_path=sample_path)
        csv_file.close()
        return

    # --- Mode B: Standard Training Loop ---
    print(f"🚀 Starting training from step {start_step + 1} to {args.max_steps}...")
    step = start_step
    data_iter = iter(train_loader)
    loss_accum = torch.zeros(1, device=device)
    accum_count = 0
    t0 = time.time()

    while step < args.max_steps:
        optimizer.zero_grad()

        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=False)
            target_ids = batch["target_ids"].to(device, non_blocking=False)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            # Accumulate detached loss tensor on device (no .item() sync per step)
            loss_accum += loss.detach()
            accum_count += 1

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        # Log training loss & LR: first 20 steps (or first 20 steps of resume segment), and every log_interval steps
        if step <= 20 or (step - start_step) <= 20 or step % args.log_interval == 0:
            avg_train_loss = (loss_accum / max(1, accum_count)).item()
            loss_accum.zero_()
            accum_count = 0
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"Step {step:06d}/{args.max_steps} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6e} | Elapsed: {elapsed:.1f}s")


        # Evaluate and log validation metrics every eval_interval steps
        if step % args.eval_interval == 0 and val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, vocab_size, device)
            val_loss = val_metrics["val_loss"]
            val_ppl = val_metrics["perplexity"]
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            print(f"✨ [Validation @ Step {step}] Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

            csv_writer.writerow([step, f"{avg_train_loss:.4f}", f"{val_loss:.4f}", f"{val_ppl:.4f}", f"{current_lr:.6e}", f"{elapsed:.1f}"])
            csv_file.flush()

            # Save best model
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
            sample_ids = model.generate(
                prompt_ids=[vocab.bos_id],
                max_generate_len=256,
                temperature=0.9,
                top_k=40,
                top_p=0.9,
                eos_id=vocab.eos_id,
                device=device
            )
            ids_to_midi(sample_ids, vocab, output_path=sample_path)

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
    parser.add_argument("--data_dir", type=str, default="Data/tokenized", help="Path to tokenized dataset directory")
    parser.add_argument("--vocab_path", type=str, default="vocab/vocab.json", help="Path to vocab.json")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (context window)")
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
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between training loss logs")
    parser.add_argument("--eval_interval", type=int, default=500, help="Steps between validation evaluations")
    parser.add_argument("--save_interval", type=int, default=500, help="Steps between checkpoints")
    parser.add_argument("--sample_interval", type=int, default=2000, help="Steps between sample MIDI generations")
    parser.add_argument("--overfit_batch", action="store_true", help="Run single-batch memorization test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train(args)

