"""
test.py - Inference and evaluation for all model variants.

Usage:
    python test.py --model 1 --image path/to/photo.jpg
    python test.py --model 2 --folder path/to/folder/
    python test.py --model 1 --test_set
    python test.py --model 3 --evaluate
    python test.py --model 4 --image photo.jpg --checkpoint checkpoints/model4_best.pt
"""

import argparse
import os
import glob

import torch
import numpy as np
from torch.amp import autocast
from PIL import Image

import config
from model import RestorationModel
from dataset import collect_test_paths, rgb_to_lab, lab_to_rgb
from utils import tensor_to_image, ab_tensor_to_image, calculate_psnr


def load_model(model_variant, checkpoint_path=None, device=None):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR,
                                        f"model{model_variant}_best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train first with: python train.py --model {model_variant}"
        )

    model = RestorationModel(model_variant=model_variant).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model {model_variant} loaded from: {checkpoint_path} "
          f"(epoch {checkpoint['epoch']})")
    return model, device


def preprocess_image(image_path, model_variant, target_size=None):
    """
    Load image, convert to model input format, resize for inference.
    Returns: (input_tensor [1,1,H,W], original PIL image, preprocessed PIL)
    """
    target_size = target_size or config.IMAGE_SIZE
    original = Image.open(image_path).convert("RGB")

    # Resize maintaining aspect ratio, pad to square
    w, h = original.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = original.resize((new_w, new_h), Image.BILINEAR)

    # Pad to target_size x target_size
    padded_rgb = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    padded_rgb.paste(resized, (left, top))

    if model_variant == 1:
        # Convert to grayscale
        gray = padded_rgb.convert("L")
        arr = np.array(gray, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:
        # Convert to Lab, use L channel
        lab = rgb_to_lab(np.array(padded_rgb))
        L = lab[:, :, 0] / 100.0  # normalize to [0, 1]
        tensor = torch.from_numpy(L.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    return tensor, original, padded_rgb


def restore_image(model, image_path, model_variant, device, output_path=None):
    """Restore and colorize a single image. Prints damage types."""
    input_tensor, original, padded = preprocess_image(image_path, model_variant)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        with autocast(device.type, enabled=config.USE_AMP and device.type == "cuda"):
            color_out, damage_logits = model(input_tensor)

    # Damage classification
    probs = torch.sigmoid(damage_logits).squeeze().cpu().numpy()
    print(f"\nImage: {os.path.basename(image_path)}")
    print("Detected damage:")
    for i, name in enumerate(config.DAMAGE_TYPES):
        marker = " <<<" if probs[i] > 0.5 else ""
        print(f"  {name:15s}: {probs[i]:.1%}{marker}")

    # Convert output to image
    if model_variant == 1:
        result_img = tensor_to_image(color_out.squeeze(0), model_variant)
    else:
        result_img = ab_tensor_to_image(color_out.squeeze(0), input_tensor.squeeze(0))

    # Save
    if output_path is None:
        os.makedirs(os.path.join(config.OUTPUT_DIR, "restored"), exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(config.OUTPUT_DIR, "restored",
                                    f"{base}_model{model_variant}_restored.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_img.save(output_path)

    # Side-by-side comparison
    comparison_path = output_path.replace("_restored.", "_comparison.")
    gray_rgb = padded.convert("L").convert("RGB")
    gap = 2
    w, h = gray_rgb.size
    combined = Image.new("RGB", (w * 2 + gap, h), color=(80, 80, 80))
    combined.paste(gray_rgb, (0, 0))
    combined.paste(result_img, (w + gap, 0))
    combined.save(comparison_path)

    print(f"Restored: {output_path}")
    print(f"Comparison: {comparison_path}")
    return result_img, probs


def restore_folder(model, folder_path, model_variant, device):
    """Restore all images in a folder."""
    paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
        paths.extend(glob.glob(os.path.join(folder_path, ext)))
    if not paths:
        print(f"No images found in {folder_path}")
        return
    print(f"Found {len(paths)} images")
    for p in paths:
        restore_image(model, p, model_variant, device)


def restore_test_set(model, model_variant, device):
    """Restore all test images (alfred archive, Studio)."""
    paths = collect_test_paths()
    if not paths:
        print("No test images found.")
        return
    print(f"\nRestoring {len(paths)} test images...")
    for p in paths:
        restore_image(model, p, model_variant, device)


def evaluate_validation(model, model_variant, device):
    """Run full evaluation on the validation set."""
    from dataset import collect_image_paths, make_train_val_split, RestorationDataset
    from torch.utils.data import DataLoader
    from utils import calculate_damage_accuracy

    print(f"\n--- Evaluating Model {model_variant} on Validation Set ---")

    all_paths = collect_image_paths()
    _, val_paths = make_train_val_split(all_paths)
    val_dataset = RestorationDataset(val_paths, model_variant=model_variant,
                                      is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)

    total_psnr = 0
    total_acc = 0
    per_type = np.zeros(config.NUM_DAMAGE_TYPES)
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            if model_variant == 4:
                degraded, target, damage_labels, refs = batch
            else:
                degraded, target, damage_labels = batch

            degraded = degraded.to(device)
            target = target.to(device)
            damage_labels = damage_labels.to(device)

            with autocast(device.type, enabled=config.USE_AMP and device.type == "cuda"):
                color_out, damage_logits = model(degraded)

            total_psnr += calculate_psnr(color_out, target)
            acc, type_accs = calculate_damage_accuracy(damage_logits, damage_labels)
            total_acc += acc
            per_type += np.array(type_accs)
            n += 1

    print(f"\nResults ({len(val_paths)} images):")
    print(f"  Average PSNR:            {total_psnr / n:.2f} dB")
    print(f"  Overall Damage Accuracy: {total_acc / n:.2%}")
    print(f"\n  Per-type accuracy:")
    for i, name in enumerate(config.DAMAGE_TYPES):
        print(f"    {name:15s}: {per_type[i] / n:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Test Restoration Model")
    parser.add_argument("--model", type=int, default=config.MODEL_VARIANT,
                        choices=[1, 2, 3, 4], help="Model variant")
    parser.add_argument("--image", type=str, help="Single image to restore")
    parser.add_argument("--folder", type=str, help="Folder of images to restore")
    parser.add_argument("--test_set", action="store_true",
                        help="Restore all test images")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on validation set")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config.MODEL_VARIANT = args.model
    model, device = load_model(args.model, args.checkpoint)

    if args.image:
        restore_image(model, args.image, args.model, device)
    elif args.folder:
        restore_folder(model, args.folder, args.model, device)
    elif args.test_set:
        restore_test_set(model, args.model, device)
    elif args.evaluate:
        evaluate_validation(model, args.model, device)
    else:
        print("Specify --image, --folder, --test_set, or --evaluate. See --help.")


if __name__ == "__main__":
    main()
