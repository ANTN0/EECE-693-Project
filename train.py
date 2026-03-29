"""
train.py - Unified training script for all 4 model variants.

Usage:
    python train.py --model 1               # Train Model 1 (RGB baseline)
    python train.py --model 2               # Train Model 2 (Lab color space)
    python train.py --model 3               # Train Model 3 (Lab + perceptual loss)
    python train.py --model 4               # Train Model 4 (reference-guided)
    python train.py --model 1 --epochs 2 --max_images 100   # Quick test
    python train.py --model 2 --resume checkpoints/model2_best.pt
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

import config
from dataset import (
    RestorationDataset, collect_image_paths, make_train_val_split,
    lab_to_rgb, rgb_to_lab,
)
from model import RestorationModel
from utils import (
    save_checkpoint, load_checkpoint, save_sample_grid,
    calculate_psnr, calculate_damage_accuracy, VGGPerceptualLoss,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# LAB <-> RGB helpers for perceptual loss (differentiable approximation)
# =============================================================================

def ab_to_rgb_batch(ab_pred, l_input):
    """
    Convert predicted ab + L input to RGB for perceptual loss computation.
    This is a simplified version — not perfectly differentiable but good enough
    for computing the VGG loss as a training signal.

    ab_pred: [B, 2, H, W] in [0, 1]
    l_input: [B, 1, H, W] in [0, 1]
    Returns: [B, 3, H, W] in [0, 1]
    """
    B, _, H, W = ab_pred.shape
    results = []
    for i in range(B):
        # Denormalize
        L = l_input[i, 0].detach().cpu().numpy() * 100.0
        a = ab_pred[i, 0].detach().cpu().numpy() * 255.0 - 128.0
        b = ab_pred[i, 1].detach().cpu().numpy() * 255.0 - 128.0
        lab = np.stack([L, a, b], axis=-1)
        rgb = lab_to_rgb(lab).astype(np.float32) / 255.0
        results.append(torch.from_numpy(rgb).permute(2, 0, 1))
    return torch.stack(results).to(ab_pred.device)


# =============================================================================
# LOSS COMPUTATION (changes per model variant)
# =============================================================================

def compute_loss(color_out, damage_logits, target, damage_labels,
                 model_variant, l1_fn, l2_fn, bce_fn,
                 perceptual_fn=None, l_input=None):
    """
    Compute the total loss for any model variant.

    Model 1: L1(RGB) + BCE(damage)
    Model 2: L1(ab) + L2(ab) + BCE(damage)
    Model 3: L1(ab) + L2(ab) + VGG_perceptual + BCE(damage)
    Model 4: same as Model 3 (reference guidance is architectural, not a loss)
    """
    # Damage classification loss (all models)
    cls_loss = bce_fn(damage_logits, damage_labels)

    if model_variant == 1:
        # Model 1: just L1 on RGB
        pixel_loss = l1_fn(color_out, target)
        total = (config.PIXEL_LOSS_WEIGHT * pixel_loss +
                 config.CLASSIFICATION_LOSS_WEIGHT * cls_loss)
        return total, pixel_loss.item(), cls_loss.item(), 0.0

    else:
        # Models 2-4: L1 + L2 on ab channels
        l1_loss = l1_fn(color_out, target)
        l2_loss = l2_fn(color_out, target)
        pixel_loss = config.PIXEL_LOSS_WEIGHT * l1_loss + config.L2_LOSS_WEIGHT * l2_loss

        perc_loss_val = 0.0

        # Models 3-4: add perceptual loss
        if model_variant >= 3 and perceptual_fn is not None and l_input is not None:
            # Convert ab predictions and targets to RGB for VGG
            pred_rgb = ab_to_rgb_batch(color_out, l_input)
            target_rgb = ab_to_rgb_batch(target, l_input)
            perc_loss = perceptual_fn(pred_rgb, target_rgb)
            perc_loss_val = perc_loss.item()
            pixel_loss = pixel_loss + config.PERCEPTUAL_LOSS_WEIGHT * perc_loss

        total = pixel_loss + config.CLASSIFICATION_LOSS_WEIGHT * cls_loss
        return total, l1_loss.item(), cls_loss.item(), perc_loss_val


# =============================================================================
# TRAINING AND VALIDATION
# =============================================================================

def train_one_epoch(model, loader, optimizer, scaler, device, epoch,
                    model_variant, l1_fn, l2_fn, bce_fn, perceptual_fn):
    model.train()
    totals = {"loss": 0, "pixel": 0, "cls": 0, "perc": 0}
    n = 0

    for batch_idx, batch in enumerate(loader):
        if model_variant == 4:
            degraded, target, damage_labels, refs = batch
            refs = refs.to(device)
        else:
            degraded, target, damage_labels = batch
            refs = None

        degraded = degraded.to(device)
        target = target.to(device)
        damage_labels = damage_labels.to(device)

        optimizer.zero_grad()

        with autocast(device.type, enabled=config.USE_AMP and device.type == "cuda"):
            color_out, damage_logits = model(degraded, refs)
            loss, px, cl, pc = compute_loss(
                color_out, damage_logits, target, damage_labels,
                model_variant, l1_fn, l2_fn, bce_fn, perceptual_fn, degraded
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += loss.item()
        totals["pixel"] += px
        totals["cls"] += cl
        totals["perc"] += pc
        n += 1

        if (batch_idx + 1) % config.LOG_EVERY == 1 or batch_idx == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} (pixel: {px:.4f}, cls: {cl:.4f}"
                  f"{f', perc: {pc:.4f}' if model_variant >= 3 else ''})")

    return {k: v / n for k, v in totals.items()}


def validate(model, loader, device, model_variant, l1_fn, l2_fn, bce_fn, perceptual_fn):
    model.eval()
    totals = {"loss": 0, "psnr": 0, "cls_acc": 0}
    n = 0
    sample_in = sample_out = sample_tgt = None

    with torch.no_grad():
        for batch in loader:
            if model_variant == 4:
                degraded, target, damage_labels, refs = batch
                refs = refs.to(device)
            else:
                degraded, target, damage_labels = batch
                refs = None

            degraded = degraded.to(device)
            target = target.to(device)
            damage_labels = damage_labels.to(device)

            with autocast(device.type, enabled=config.USE_AMP and device.type == "cuda"):
                color_out, damage_logits = model(degraded, refs)
                loss, _, _, _ = compute_loss(
                    color_out, damage_logits, target, damage_labels,
                    model_variant, l1_fn, l2_fn, bce_fn, perceptual_fn, degraded
                )

            totals["loss"] += loss.item()
            totals["psnr"] += calculate_psnr(color_out, target)
            acc, _ = calculate_damage_accuracy(damage_logits, damage_labels)
            totals["cls_acc"] += acc
            n += 1

            if sample_in is None:
                sample_in = degraded.cpu()
                sample_out = color_out.cpu()
                sample_tgt = target.cpu()

    return {k: v / n for k, v in totals.items()}, sample_in, sample_out, sample_tgt


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Restoration Model")
    parser.add_argument("--model", type=int, default=config.MODEL_VARIANT,
                        choices=[1, 2, 3, 4], help="Model variant (1-4)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max_images", type=int, default=config.MAX_IMAGES)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Update config with command-line model choice
    config.MODEL_VARIANT = args.model
    model_variant = args.model

    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model_names = {1: "RGB Baseline", 2: "Lab Color Space",
                   3: "Lab + Perceptual Loss", 4: "Reference-Guided"}
    print(f"=== Training Model {model_variant}: {model_names[model_variant]} ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # --- Data ---
    print("\n--- Loading Data ---")
    all_paths = collect_image_paths(max_images=args.max_images)
    train_paths, val_paths = make_train_val_split(all_paths)

    # For model 4, the training images also serve as the reference pool
    ref_paths = train_paths if model_variant == 4 else None

    train_dataset = RestorationDataset(
        train_paths, model_variant=model_variant, is_training=True, ref_paths=ref_paths
    )
    val_dataset = RestorationDataset(
        val_paths, model_variant=model_variant, is_training=False, ref_paths=ref_paths
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    print("\n--- Building Model ---")
    model = RestorationModel(model_variant=model_variant).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")

    # --- Loss functions ---
    l1_fn = nn.L1Loss()
    l2_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    # Perceptual loss (models 3-4 only)
    perceptual_fn = None
    if model_variant >= 3:
        print("Loading VGG-16 for perceptual loss...")
        perceptual_fn = VGGPerceptualLoss().to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = GradScaler("cuda", enabled=config.USE_AMP and device.type == "cuda")

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume) + 1

    # --- Training ---
    print(f"\n--- Starting Training ({args.epochs} epochs) ---\n")
    best_val_loss = float("inf")
    ckpt_prefix = f"model{model_variant}"

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            model_variant, l1_fn, l2_fn, bce_fn, perceptual_fn
        )

        val_stats, s_in, s_out, s_tgt = validate(
            model, val_loader, device, model_variant,
            l1_fn, l2_fn, bce_fn, perceptual_fn
        )

        scheduler.step(val_stats["loss"])
        dt = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} ({dt:.1f}s)")
        perc_str = f", perc: {train_stats['perc']:.4f}" if model_variant >= 3 else ""
        print(f"  Train Loss: {train_stats['loss']:.4f} "
              f"(pixel: {train_stats['pixel']:.4f}, cls: {train_stats['cls']:.4f}{perc_str})")
        print(f"  Val Loss:   {val_stats['loss']:.4f} | "
              f"PSNR: {val_stats['psnr']:.2f} dB | "
              f"Damage Acc: {val_stats['cls_acc']:.2%}\n")

        # Save samples
        if epoch % config.SAMPLE_EVERY == 0 and s_in is not None:
            save_sample_grid(s_in, s_out, s_tgt, epoch, model_variant)
            print(f"  Saved samples to {config.OUTPUT_DIR}/samples/")

        # Save checkpoint
        if epoch % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, val_stats["loss"],
                            os.path.join(config.CHECKPOINT_DIR,
                                         f"{ckpt_prefix}_epoch_{epoch}.pt"))

        # Save best model
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(model, optimizer, epoch, val_stats["loss"],
                            os.path.join(config.CHECKPOINT_DIR,
                                         f"{ckpt_prefix}_best.pt"))
            print(f"  New best model! Val loss: {val_stats['loss']:.4f}")

    print(f"\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: checkpoints/{ckpt_prefix}_best.pt")


if __name__ == "__main__":
    main()
