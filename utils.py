"""
utils.py - Helpers: checkpoints, metrics, image saving, VGG perceptual loss.
"""

import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import config
from dataset import lab_to_rgb


# =============================================================================
# CHECKPOINTS
# =============================================================================

def save_checkpoint(model, optimizer, epoch, loss, path=None):
    """Save model checkpoint."""
    if path is None:
        path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_variant": config.MODEL_VARIANT,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer=None, path=None):
    """Load model checkpoint. Returns the epoch number."""
    checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]


# =============================================================================
# TENSOR -> IMAGE CONVERSION
# =============================================================================

def tensor_to_image(tensor, model_variant=None):
    """
    Convert model output tensor to a PIL Image.

    Model 1: tensor is [3, H, W] RGB in [0, 1] -> RGB PIL Image
    Models 2-4: tensor is [2, H, W] ab channels in [0, 1] -> needs L channel to make RGB

    For grayscale input tensors [1, H, W] -> grayscale PIL Image.
    """
    variant = model_variant or config.MODEL_VARIANT
    t = tensor.detach().cpu().float()

    # Grayscale input
    if t.dim() == 3 and t.shape[0] == 1:
        arr = (t.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    # Model 1: RGB output
    if t.shape[0] == 3:
        arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    # Models 2-4: ab output (need to combine with L to get RGB)
    # This case is handled by ab_tensor_to_image() below
    return None


def ab_tensor_to_image(ab_tensor, l_tensor):
    """
    Convert predicted ab channels + input L channel back to an RGB PIL Image.

    ab_tensor: [2, H, W] in [0, 1] (normalized ab)
    l_tensor:  [1, H, W] in [0, 1] (normalized L)
    """
    ab = ab_tensor.detach().cpu().numpy()
    L = l_tensor.detach().cpu().numpy()

    # Denormalize
    L_denorm = L[0] * 100.0                    # [0, 1] -> [0, 100]
    a_denorm = ab[0] * 255.0 - 128.0           # [0, 1] -> [-128, 127]
    b_denorm = ab[1] * 255.0 - 128.0           # [0, 1] -> [-128, 127]

    lab_img = np.stack([L_denorm, a_denorm, b_denorm], axis=-1)
    rgb = lab_to_rgb(lab_img)
    return Image.fromarray(rgb)


# =============================================================================
# SAMPLE SAVING
# =============================================================================

def save_comparison(input_t, output_t, target_t, path, model_variant=None):
    """
    Save side-by-side: input (gray) | model output (color) | target (color).
    Handles both RGB and Lab modes.
    """
    variant = model_variant or config.MODEL_VARIANT

    # Input is always grayscale [1, H, W]
    in_img = tensor_to_image(input_t)
    if in_img.mode == "L":
        in_img = in_img.convert("RGB")

    # Output and target
    if variant == 1:
        out_img = tensor_to_image(output_t, variant)
        tgt_img = tensor_to_image(target_t, variant)
    else:
        out_img = ab_tensor_to_image(output_t, input_t)
        tgt_img = ab_tensor_to_image(target_t, input_t)

    h, w = in_img.height, in_img.width
    gap = 2
    combined = Image.new("RGB", (w * 3 + gap * 2, h), color=(80, 80, 80))
    combined.paste(in_img, (0, 0))
    combined.paste(out_img, (w + gap, 0))
    combined.paste(tgt_img, (2 * (w + gap), 0))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    combined.save(path)


def save_sample_grid(input_batch, output_batch, target_batch, epoch,
                     model_variant=None, output_dir=None):
    """Save sample comparisons during training."""
    output_dir = output_dir or os.path.join(config.OUTPUT_DIR, "samples")
    os.makedirs(output_dir, exist_ok=True)
    variant = model_variant or config.MODEL_VARIANT

    n = min(len(input_batch), config.NUM_SAMPLES)
    for i in range(n):
        path = os.path.join(output_dir, f"epoch{epoch:03d}_sample{i}.png")
        save_comparison(input_batch[i], output_batch[i], target_batch[i],
                        path, variant)


# =============================================================================
# METRICS
# =============================================================================

def calculate_psnr(pred, target):
    """Peak Signal-to-Noise Ratio between predicted and target tensors in [0, 1]."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def calculate_damage_accuracy(logits, labels, threshold=0.5):
    """Per-type and overall accuracy for damage classification."""
    preds = (torch.sigmoid(logits) > threshold).float()
    correct = (preds == labels).float()
    return correct.mean().item(), correct.mean(dim=0).tolist()


# =============================================================================
# VGG PERCEPTUAL LOSS (Models 3 and 4)
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    Compares intermediate VGG-16 features between predicted and target images.
    This encourages the model to match high-level texture/structure, not just pixels.
    Result: sharper, more vivid colors instead of blurry averages.

    The VGG is frozen (no gradients) — it's only used to compute the loss.

    Since VGG expects 3-channel RGB input, for Lab models (2-4) we need to
    convert ab predictions back to RGB before computing this loss. This is
    handled in train.py.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        # Load pretrained VGG-16 and extract feature layers
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = vgg.features

        # We compare at 3 different depths (early, mid, deep features)
        self.slice1 = nn.Sequential(*features[:5])    # relu1_2
        self.slice2 = nn.Sequential(*features[5:10])   # relu2_2
        self.slice3 = nn.Sequential(*features[10:17])  # relu3_3

        # Freeze all VGG weights
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization (VGG was trained with these values)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """Normalize input to match ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """
        pred, target: [B, 3, H, W] RGB images in [0, 1].
        Returns: scalar perceptual loss.
        """
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        x, y = pred, target
        for layer in [self.slice1, self.slice2, self.slice3]:
            x = layer(x)
            y = layer(y)
            loss += nn.functional.l1_loss(x, y)

        return loss
