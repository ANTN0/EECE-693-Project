"""
dataset.py - Data loading and synthetic degradation pipeline.

Supports two color modes depending on the model variant:
    - Model 1 (RGB):  input = degraded grayscale [1,H,W], target = clean RGB [3,H,W]
    - Models 2-4 (Lab): input = L channel [1,H,W], target = ab channels [2,H,W]

For Model 4, also returns K=3 reference images (color images similar to the target).

Negative images (film negatives) are automatically filtered out by filename.
"""

import os
import random
import glob
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset

import config


# =============================================================================
# NEGATIVE IMAGE FILTERING
# =============================================================================
# Keywords that indicate a file is a film negative — we skip these
NEGATIVE_KEYWORDS = ["negat", "négatif", "negativ", "invert", "film_neg"]


def is_negative(filepath):
    """Check if a file path suggests it's a film negative."""
    lower = filepath.lower().replace("\\", "/")
    return any(kw in lower for kw in NEGATIVE_KEYWORDS)


# =============================================================================
# RGB <-> Lab CONVERSION (pure numpy, no extra dependencies)
# =============================================================================

def rgb_to_lab(rgb_array):
    """
    Convert RGB image (0-255 uint8 or 0-1 float) to Lab color space.
    Returns L in [0, 100], a in [-128, 127], b in [-128, 127].
    """
    # Make sure we're working with float [0, 1]
    arr = rgb_array.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Step 1: sRGB to linear RGB (remove gamma)
    mask = arr > 0.04045
    arr[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
    arr[~mask] = arr[~mask] / 12.92

    # Step 2: Linear RGB to XYZ (D65 illuminant)
    # Using the standard sRGB to XYZ matrix
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by D65 white point
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    # Step 3: XYZ to Lab
    def f(t):
        delta = 6.0 / 29.0
        mask = t > delta ** 3
        result = np.zeros_like(t)
        result[mask] = np.cbrt(t[mask])
        result[~mask] = t[~mask] / (3 * delta ** 2) + 4.0 / 29.0
        return result

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0       # [0, 100]
    a = 500.0 * (fx - fy)       # roughly [-128, 127]
    b_ch = 200.0 * (fy - fz)    # roughly [-128, 127]

    return np.stack([L, a, b_ch], axis=-1)


def lab_to_rgb(lab_array):
    """
    Convert Lab image back to RGB (0-255 uint8).
    L in [0, 100], a in [-128, 127], b in [-128, 127].
    """
    L, a, b_ch = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]

    # Step 1: Lab to XYZ
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_ch / 200.0

    delta = 6.0 / 29.0

    def f_inv(t):
        mask = t > delta
        result = np.zeros_like(t)
        result[mask] = t[mask] ** 3
        result[~mask] = 3 * delta ** 2 * (t[~mask] - 4.0 / 29.0)
        return result

    x = 0.95047 * f_inv(fx)
    y = 1.00000 * f_inv(fy)
    z = 1.08883 * f_inv(fz)

    # Step 2: XYZ to linear RGB
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Step 3: Linear RGB to sRGB (apply gamma)
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, None)
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1.0 / 2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


# =============================================================================
# SYNTHETIC DEGRADATION FUNCTIONS
# =============================================================================

def add_scratches(img):
    """Add random scratch lines — mimics physical handling damage."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(random.randint(2, 8)):
        if random.random() > 0.5:
            x1, x2 = random.randint(0, w // 3), random.randint(2 * w // 3, w)
            y1 = random.randint(0, h)
            y2 = y1 + random.randint(-h // 6, h // 6)
        else:
            y1, y2 = random.randint(0, h // 3), random.randint(2 * h // 3, h)
            x1 = random.randint(0, w)
            x2 = x1 + random.randint(-w // 6, w // 6)
        brightness = random.randint(180, 255)
        draw.line([(x1, y1), (x2, y2)], fill=(brightness,) * 3, width=random.randint(1, 2))
    return img


def add_noise(img):
    """Add Gaussian noise — mimics film grain and sensor degradation."""
    arr = np.array(img, dtype=np.float32)
    sigma = random.uniform(10, 35)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def add_blur(img):
    """Apply Gaussian blur — mimics focus loss or lens degradation."""
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 3.0)))


def add_aging(img):
    """Simulate natural aging: yellowing, contrast loss, faded colors."""
    arr = np.array(img, dtype=np.float32)
    gray = np.mean(arr, axis=2, keepdims=True)
    arr = gray + random.uniform(0.2, 0.6) * (arr - gray)
    yellow = np.array([random.uniform(15, 35), random.uniform(10, 25), random.uniform(-10, 0)])
    arr = arr + yellow
    mean_val = arr.mean()
    arr = mean_val + random.uniform(0.5, 0.8) * (arr - mean_val)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_water_damage(img):
    """Simulate water damage: irregular blotchy discoloration patches."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(random.randint(2, 6)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(min(h, w) // 8, min(h, w) // 3)
        y_c, x_c = np.ogrid[:h, :w]
        dist = np.sqrt((x_c - cx) ** 2 + (y_c - cy) ** 2)
        mask = np.maximum(mask, np.clip(1.0 - dist / radius, 0, 1))
    mask = mask[:, :, np.newaxis]
    stain = np.array([random.uniform(-30, -10), random.uniform(-20, -5), random.uniform(-40, -15)])
    intensity = random.uniform(0.3, 0.7)
    arr = arr + mask * stain * intensity * 2
    local_mean = arr.mean()
    arr = arr * (1 - mask * 0.2) + local_mean * mask * 0.2
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_fire_damage(img):
    """Simulate fire/heat damage: darkened edges, charring effect."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    y_c, x_c = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    max_dist = math.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((x_c - cx) ** 2 + (y_c - cy) ** 2)
    burn = (np.clip(dist / max_dist, 0, 1) ** random.uniform(1.5, 3.0))[:, :, np.newaxis]
    intensity = random.uniform(0.4, 0.8)
    burn_color = np.array([30, 15, 5], dtype=np.float32)
    arr = arr * (1 - burn * intensity) + burn_color * burn * intensity
    arr = arr + np.random.normal(0, 10, arr.shape) * burn
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_wear_and_tear(img):
    """Simulate physical wear: edge damage, corner wear, border fading."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    edge_width = random.randint(3, max(4, min(w, h) // 15))
    for _ in range(random.randint(5, 15)):
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            x1, y1 = random.randint(0, w), 0
            x2, y2 = x1 + random.randint(5, w // 4), random.randint(2, edge_width)
        elif side == "bottom":
            x1, y1 = random.randint(0, w), h - random.randint(2, edge_width)
            x2, y2 = x1 + random.randint(5, w // 4), h
        elif side == "left":
            x1, y1 = 0, random.randint(0, h)
            x2, y2 = random.randint(2, edge_width), y1 + random.randint(5, h // 4)
        else:
            x1, y1 = w - random.randint(2, edge_width), random.randint(0, h)
            x2, y2 = w, y1 + random.randint(5, h // 4)
        b = random.randint(160, 230)
        draw.rectangle([x1, y1, x2, y2], fill=(b, b, b))
    corner_size = random.randint(5, max(6, min(w, h) // 8))
    for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)]:
        if random.random() > 0.5:
            for r in range(corner_size, 0, -1):
                a = int(180 + 75 * (1 - r / corner_size))
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(a, a, a))
    return img


def add_mold_foxing(img):
    """Simulate mold/foxing spots — small dark brown organic-looking spots."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(random.randint(5, 25)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(1, max(2, min(w, h) // 30))
        r, g, b = random.randint(50, 110), random.randint(30, 80), random.randint(10, 50)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(r, g, b))
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def add_light_leak(img):
    """Simulate light leak — overexposed bright areas from damaged film."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    lx, ly = random.choice([0, w]), random.choice([0, h])
    y_c, x_c = np.ogrid[:h, :w]
    dist = np.sqrt((x_c - lx) ** 2 + (y_c - ly) ** 2)
    max_dist = math.sqrt(w ** 2 + h ** 2)
    leak = (np.clip(1.0 - dist / (max_dist * random.uniform(0.3, 0.6)), 0, 1)
            ** random.uniform(1.0, 2.0))[:, :, np.newaxis]
    color = np.array([255, random.uniform(200, 240), random.uniform(150, 200)])
    intensity = random.uniform(0.2, 0.6)
    arr = arr * (1 - leak * intensity) + color * leak * intensity
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_crease(img):
    """Simulate fold/crease lines — visible lines where a photo was folded."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(random.randint(1, 3)):
        if random.random() > 0.5:
            y = random.randint(h // 4, 3 * h // 4)
            b = random.randint(180, 230)
            draw.line([(0, y), (w, y)], fill=(b, b, b), width=random.randint(1, 3))
            off = random.choice([-2, 2])
            d = random.randint(40, 80)
            draw.line([(0, y + off), (w, y + off)], fill=(d, d, d), width=1)
        else:
            x = random.randint(w // 4, 3 * w // 4)
            b = random.randint(180, 230)
            draw.line([(x, 0), (x, h)], fill=(b, b, b), width=random.randint(1, 3))
            off = random.choice([-2, 2])
            d = random.randint(40, 80)
            draw.line([(x + off, 0), (x + off, h)], fill=(d, d, d), width=1)
    return img


# Map names to functions
DEGRADATION_FUNCTIONS = {
    "scratches": add_scratches,
    "noise": add_noise,
    "blur": add_blur,
    "aging": add_aging,
    "water_damage": add_water_damage,
    "fire_damage": add_fire_damage,
    "wear_and_tear": add_wear_and_tear,
    "mold_foxing": add_mold_foxing,
    "light_leak": add_light_leak,
    "crease": add_crease,
}


# =============================================================================
# DATASET CLASS
# =============================================================================

class RestorationDataset(Dataset):
    """
    Unified dataset for all model variants.

    Model 1 (RGB):
        input  = degraded grayscale [1, H, W]
        target = clean RGB [3, H, W]

    Models 2-4 (Lab):
        input  = L channel [1, H, W]  (lightness, normalized to [0, 1])
        target = ab channels [2, H, W] (color, normalized to [0, 1])

    Model 4 additionally returns reference color images [K, 3, H, W].
    """

    def __init__(self, image_paths, model_variant=None, image_size=None,
                 is_training=True, ref_paths=None):
        self.image_paths = image_paths
        self.model_variant = model_variant or config.MODEL_VARIANT
        self.image_size = image_size or config.IMAGE_SIZE
        self.is_training = is_training
        self.use_lab = self.model_variant >= 2
        # For model 4: pool of reference images to sample from
        self.ref_paths = ref_paths

    def __len__(self):
        return len(self.image_paths)

    def _load_and_crop(self, path):
        """Load image as RGB, resize so short side = image_size, then crop."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.image_size / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        w, h = img.size
        if self.is_training:
            left = random.randint(0, max(0, w - self.image_size))
            top = random.randint(0, max(0, h - self.image_size))
        else:
            left = max(0, (w - self.image_size) // 2)
            top = max(0, (h - self.image_size) // 2)
        img = img.crop((left, top, left + self.image_size, top + self.image_size))

        # Random horizontal flip
        if self.is_training and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Small brightness/contrast jitter
        if self.is_training and random.random() > 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))

        return img

    def _apply_degradation(self, color_img):
        """
        Apply random synthetic damage to a color image.
        Returns: (degraded_image, damage_labels)
        """
        damage_labels = np.zeros(config.NUM_DAMAGE_TYPES, dtype=np.float32)

        # 30% chance: no degradation (model just learns colorization)
        if random.random() < config.CLEAN_RATIO:
            return color_img, damage_labels

        # Pick 1 to MAX_DAMAGES_PER_IMAGE random damage types
        all_indices = list(range(config.NUM_DAMAGE_TYPES))
        random.shuffle(all_indices)
        num_to_apply = random.randint(1, config.MAX_DAMAGES_PER_IMAGE)

        selected = []
        for idx in all_indices:
            if len(selected) >= num_to_apply:
                break
            if random.random() < config.DEGRADATION_PROB:
                selected.append(idx)
        if not selected:
            selected = [random.choice(all_indices)]

        degraded = color_img
        for idx in selected:
            damage_name = config.DAMAGE_TYPES[idx]
            degraded = DEGRADATION_FUNCTIONS[damage_name](degraded)
            damage_labels[idx] = 1.0

        return degraded, damage_labels

    def _prepare_rgb(self, color_img, degraded_img):
        """Model 1: degraded grayscale -> clean RGB."""
        # Input: degraded image converted to grayscale
        gray = np.array(degraded_img.convert("L"), dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(gray).unsqueeze(0)  # [1, H, W]

        # Target: original clean color image
        rgb = np.array(color_img, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]

        return input_tensor, target_tensor

    def _prepare_lab(self, color_img, degraded_img):
        """Models 2-4: L channel -> ab channels."""
        # Convert clean color image to Lab
        lab = rgb_to_lab(np.array(color_img))

        # Input: L channel from the DEGRADED image (so model also learns restoration)
        degraded_lab = rgb_to_lab(np.array(degraded_img))
        L_input = degraded_lab[:, :, 0] / 100.0  # normalize L to [0, 1]
        input_tensor = torch.from_numpy(L_input.astype(np.float32)).unsqueeze(0)  # [1, H, W]

        # Target: ab channels from clean image, normalized to [0, 1]
        # a and b are in [-128, 127], shift to [0, 1]: (x + 128) / 255
        a_norm = (lab[:, :, 1] + 128.0) / 255.0
        b_norm = (lab[:, :, 2] + 128.0) / 255.0
        target_tensor = torch.from_numpy(
            np.stack([a_norm, b_norm], axis=0).astype(np.float32)
        )  # [2, H, W]

        return input_tensor, target_tensor

    def _load_references(self):
        """Model 4: load K random reference images as color tensors."""
        k = config.NUM_REFERENCES
        if self.ref_paths is None or len(self.ref_paths) == 0:
            # Return zeros if no references available
            return torch.zeros(k, 3, self.image_size, self.image_size)

        chosen = random.sample(self.ref_paths, min(k, len(self.ref_paths)))

        refs = []
        for p in chosen:
            try:
                ref_img = self._load_and_crop(p)
                arr = np.array(ref_img, dtype=np.float32) / 255.0
                refs.append(torch.from_numpy(arr).permute(2, 0, 1))  # [3, H, W]
            except Exception:
                refs.append(torch.zeros(3, self.image_size, self.image_size))

        # Pad if we got fewer than K
        while len(refs) < k:
            refs.append(torch.zeros(3, self.image_size, self.image_size))

        return torch.stack(refs)  # [K, 3, H, W]

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        try:
            color_img = self._load_and_crop(path)
        except Exception:
            color_img = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

        degraded_img, damage_labels = self._apply_degradation(color_img)

        # Prepare input/target based on model variant
        if self.use_lab:
            input_tensor, target_tensor = self._prepare_lab(color_img, degraded_img)
        else:
            input_tensor, target_tensor = self._prepare_rgb(color_img, degraded_img)

        damage_labels = torch.from_numpy(damage_labels)

        # Model 4: also return reference images
        if self.model_variant == 4:
            refs = self._load_references()
            return input_tensor, target_tensor, damage_labels, refs

        return input_tensor, target_tensor, damage_labels


# =============================================================================
# HELPER: Collect image paths (with negative filtering)
# =============================================================================

def collect_image_paths(data_root=None, subfolders=None, max_images=None):
    """Find all .jpg/.jpeg files under training folders, filtering out negatives."""
    data_root = data_root or config.DATA_ROOT
    subfolders = subfolders or config.TRAIN_SUBFOLDERS
    max_images = max_images or config.MAX_IMAGES

    all_paths = []
    for subfolder in subfolders:
        folder = os.path.join(data_root, subfolder)
        if not os.path.exists(folder):
            print(f"Warning: folder not found: {folder}")
            continue
        for pattern in ["**/*.jpg", "**/*.JPG", "**/*.jpeg", "**/*.JPEG"]:
            all_paths.extend(glob.glob(os.path.join(folder, pattern), recursive=True))

    # Remove duplicates
    all_paths = list(set(all_paths))

    # Filter out negatives
    before = len(all_paths)
    all_paths = [p for p in all_paths if not is_negative(p)]
    filtered = before - len(all_paths)
    if filtered > 0:
        print(f"Filtered out {filtered} negative images")

    random.shuffle(all_paths)
    if max_images and len(all_paths) > max_images:
        all_paths = all_paths[:max_images]

    print(f"Found {len(all_paths)} training images")
    return all_paths


def collect_test_paths(data_root=None, subfolders=None):
    """Collect paths for test/inference images (already damaged originals)."""
    data_root = data_root or config.DATA_ROOT
    subfolders = subfolders or config.TEST_SUBFOLDERS

    all_paths = []
    for subfolder in subfolders:
        folder = os.path.join(data_root, subfolder)
        if not os.path.exists(folder):
            continue
        for pattern in ["**/*.jpg", "**/*.JPG", "**/*.jpeg", "**/*.JPEG"]:
            all_paths.extend(glob.glob(os.path.join(folder, pattern), recursive=True))

    all_paths = list(set(all_paths))
    # Filter out negatives from test set too
    all_paths = [p for p in all_paths if not is_negative(p)]
    print(f"Found {len(all_paths)} test images")
    return all_paths


def make_train_val_split(image_paths, val_split=None):
    """Split image paths into training and validation sets."""
    val_split = val_split or config.VAL_SPLIT
    random.shuffle(image_paths)
    val_size = int(len(image_paths) * val_split)
    val_paths = image_paths[:val_size]
    train_paths = image_paths[val_size:]
    print(f"Training: {len(train_paths)} images, Validation: {len(val_paths)} images")
    return train_paths, val_paths
