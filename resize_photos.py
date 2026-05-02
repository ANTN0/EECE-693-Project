# ============================================================
# ONE-TIME DATA PREP: Resize images to 256px max for faster Drive reads
# Run this cell ONCE, then set DATA_ROOT in Cell 6 to the resized folder.
# Grayscale images are filtered out during resize so they never appear in training.
# Safe to interrupt and resume — already-resized images are skipped.
# ============================================================
from PIL import Image
import os, glob, json

SOURCE = r"G:\.shortcut-targets-by-id\1QoM2fXIon1-8KADDVEgwj_133vXKS_bL\Pictures for Project"
DEST   = r"G:\My Drive\Pictures for Project Resized"
MAX_SIZE  = 256
SUBFOLDERS = [
    "Arab and Lebanese Diaspora",
    "Baptism",
    "Studio",
    "Color",
    "Portraits_V2",
]

def _is_grayscale(img, threshold=5.0):
    img_small = img.copy()
    img_small.thumbnail((16, 16))
    import numpy as np
    arr = np.array(img_small, dtype=np.float32)
    return (abs(arr[:,:,0] - arr[:,:,1]).mean() < threshold and
            abs(arr[:,:,0] - arr[:,:,2]).mean() < threshold)

import time as _t

# Count total images first for upfront estimate
total_to_process = 0
all_src_paths = []
for subfolder in SUBFOLDERS:
    src_folder = os.path.join(SOURCE, subfolder)
    if not os.path.exists(src_folder):
        continue
    for pat in ["**/*.jpg", "**/*.JPG", "**/*.jpeg", "**/*.JPEG"]:
        found = glob.glob(os.path.join(src_folder, pat), recursive=True)
        all_src_paths.extend([(p, subfolder) for p in found])
total_to_process = len(all_src_paths)
print(f"Found {total_to_process} images to process.")
print(f"Rough estimate: {total_to_process * 2 / 3600:.1f}–{total_to_process * 4 / 3600:.1f} hours (depends on Drive speed).")
print("Progress printed every 200 images. Safe to interrupt and resume.")

count = 0
skipped_gray = 0
skipped_existing = 0
start_time = _t.time()

for src_path, subfolder in all_src_paths:
    rel      = os.path.relpath(src_path, SOURCE)
    dst_path = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        skipped_existing += 1
        continue
    try:
        img = Image.open(src_path).convert("RGB")
        if _is_grayscale(img):
            skipped_gray += 1
            continue
        img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=95)
        count += 1
        processed = count + skipped_gray
        if processed % 200 == 0:
            elapsed = _t.time() - start_time
            rate = processed / max(elapsed, 1)
            remaining = (total_to_process - skipped_existing - processed) / max(rate, 0.001)
            print(f"  [{processed}/{total_to_process - skipped_existing}] Resized: {count} | Grayscale skipped: {skipped_gray} | ~{remaining/3600:.1f}h remaining")
    except Exception as e:
        print(f"  Skipped {os.path.basename(src_path)}: {e}")

total_time = _t.time() - start_time
print(f"Done! Resized: {count} | Skipped grayscale: {skipped_gray} | Already done: {skipped_existing}")
print(f"Total time: {total_time/3600:.1f}h")
print(f"Resized folder: {DEST}")
print("Next step: change DATA_ROOT in Cell 6 to:")
print(f'  DATA_ROOT = "{DEST}"')
