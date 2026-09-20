#!/usr/bin/env python3

import os
import sys
import csv
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from patchify import patchify

# ============================================================
# ===================== CONFIGURATION =========================
# ============================================================

# Input dataset root
INPUT_MAIN_DIR = "/mnt/eye-datasets/DRIVE-2004/test"   # contains images/ and masks/

# Output dataset root
OUTPUT_MAIN_DIR = "/mnt/eye-datasets/DRIVE-2004-patches-256-64/test" # will create images/ and masks/

# Patch configuration
PATCH_SIZE = 256   # height = width = PATCH_SIZE
STRIDE     = 64

# Optional pre-resize (applied to both images and masks before patching)
# If RESIZE == 1, images/masks are resized to RESIZE_H x RESIZE_W.
# If RESIZE == 0, original sizes are used.
RESIZE   = 1
RESIZE_H = 1024
RESIZE_W = 1024

# Supported image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# CSV filenames
PATCH_CSV_NAME   = "patch_details.csv"
SUMMARY_CSV_NAME = "summary.csv"
TOTAL_CSV_NAME   = "total.csv"

# ============================================================
# ===================== PATH SETUP ============================
# ============================================================

INPUT_IMAGES_DIR = Path(INPUT_MAIN_DIR) / "images"
INPUT_MASKS_DIR  = Path(INPUT_MAIN_DIR) / "masks"

OUTPUT_IMAGES_DIR = Path(OUTPUT_MAIN_DIR) / "images"
OUTPUT_MASKS_DIR  = Path(OUTPUT_MAIN_DIR) / "masks"

PATCH_CSV_PATH   = Path(OUTPUT_MAIN_DIR) / PATCH_CSV_NAME
SUMMARY_CSV_PATH = Path(OUTPUT_MAIN_DIR) / SUMMARY_CSV_NAME
TOTAL_CSV_PATH   = Path(OUTPUT_MAIN_DIR) / TOTAL_CSV_NAME

# ============================================================
# ===================== UTILITIES =============================
# ============================================================

def fatal(msg):
    print(f"\n❌ ERROR: {msg}")
    sys.exit(1)

def clear_output_dir_contents(output_root: Path):
    if not output_root.exists():
        return
    if not output_root.is_dir():
        fatal(f"OUTPUT_MAIN_DIR is not a directory: {output_root}")


    for child in output_root.iterdir():
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except Exception as e:
            fatal(f"Failed to delete '{child}': {e}")

def load_image(path, resize_hw=None, is_mask=False):
    with Image.open(path) as img:
        orig_w, orig_h = img.size
        if resize_hw is not None:
            resize_h, resize_w = resize_hw
            resample = Image.NEAREST if is_mask else Image.BILINEAR
            img = img.resize((resize_w, resize_h), resample=resample)
        arr = np.array(img)
    return arr, orig_h, orig_w

def check_patch_condition(h, w):
    if (h - PATCH_SIZE) % STRIDE != 0 or (w - PATCH_SIZE) % STRIDE != 0:
        fatal(
            f"Patch condition failed for image size {h}x{w}.\n"
            f"Required: (size - {PATCH_SIZE}) % {STRIDE} == 0"
        )

def is_image_file(p):
    return p.suffix.lower() in IMAGE_EXTS

# ============================================================
# ===================== CSV SETUP ==============================
# ============================================================

Path(OUTPUT_MAIN_DIR).mkdir(parents=True, exist_ok=True)
clear_output_dir_contents(Path(OUTPUT_MAIN_DIR))

OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)

patch_csv_file = open(PATCH_CSV_PATH, "w", newline="")
summary_csv_file = open(SUMMARY_CSV_PATH, "w", newline="")
total_csv_file = open(TOTAL_CSV_PATH, "w", newline="")

patch_writer = csv.writer(patch_csv_file)
summary_writer = csv.writer(summary_csv_file)
total_writer = csv.writer(total_csv_file)

patch_writer.writerow([
    "original_file",
    "orig_h", "orig_w", "orig_c",
    "resized_h", "resized_w",
    "patch_file",
    "patch_h", "patch_w", "patch_c",
    "output_dir"
])

summary_writer.writerow([
    "original_file",
    "orig_h", "orig_w", "orig_c",
    "resized_h", "resized_w",
    "patch_size",
    "stride",
    "num_patches"
])

# totals.csv: dataset-level counts
total_writer.writerow([
    "num_original_images",
    "num_original_masks",
    "num_output_image_patches",
    "num_output_mask_patches",
])

# ============================================================
# ===================== MAIN LOOP =============================
# ============================================================

image_files = sorted([p for p in INPUT_IMAGES_DIR.iterdir() if is_image_file(p)])

if not image_files:
    fatal("No image files found in images/ directory.")

print(f"\nFound {len(image_files)} image files.")
print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}, Stride: {STRIDE}\n")

total_output_patches = 0

resize_hw = None
if RESIZE == 1:
    if not (isinstance(RESIZE_H, int) and isinstance(RESIZE_W, int) and RESIZE_H > 0 and RESIZE_W > 0):
        fatal("RESIZE is enabled but RESIZE_H/RESIZE_W are not valid positive integers.")
    resize_hw = (RESIZE_H, RESIZE_W)
    print(f"Resize enabled: {RESIZE_H}x{RESIZE_W}\n")
elif RESIZE != 0:
    fatal("RESIZE must be 0 or 1.")

for img_path in image_files:
    fname = img_path.name
    mask_path = INPUT_MASKS_DIR / fname

    if not mask_path.exists():
        fatal(f"Mask not found for image: {fname}")

    print(f"Processing: {fname}")

    img, img_orig_h, img_orig_w = load_image(img_path, resize_hw=resize_hw, is_mask=False)
    mask, mask_orig_h, mask_orig_w = load_image(mask_path, resize_hw=resize_hw, is_mask=True)

    # Normalize dimensions
    if img.ndim == 2:
        img = img[:, :, None]
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    H, W, C = img.shape
    mask_h, mask_w = mask.shape
    if (mask_h, mask_w) != (H, W):
        fatal(
            f"Image/mask size mismatch after resize for '{fname}': "
            f"image={H}x{W}, mask={mask_h}x{mask_w}"
        )
    check_patch_condition(H, W)

    # Patchify
    img_patches = patchify(
        img,
        (PATCH_SIZE, PATCH_SIZE, C),
        step=STRIDE
    )

    mask_patches = patchify(
        mask,
        (PATCH_SIZE, PATCH_SIZE),
        step=STRIDE
    )

    n_h, n_w = img_patches.shape[0], img_patches.shape[1]
    total_patches = n_h * n_w

    patch_idx = 1

    for i in range(n_h):
        for j in range(n_w):
            img_patch = img_patches[i, j, 0]
            mask_patch = mask_patches[i, j]

            patch_name = f"{img_path.stem}-patch-{patch_idx:04d}.png"

            Image.fromarray(img_patch).save(OUTPUT_IMAGES_DIR / patch_name)
            Image.fromarray(mask_patch).save(OUTPUT_MASKS_DIR / patch_name)

            patch_writer.writerow([
                fname,
                img_orig_h, img_orig_w, C,
                H, W,
                patch_name,
                PATCH_SIZE, PATCH_SIZE, C,
                str(OUTPUT_IMAGES_DIR)
            ])

            patch_writer.writerow([
                fname,
                mask_orig_h, mask_orig_w, 1,
                H, W,
                patch_name,
                PATCH_SIZE, PATCH_SIZE, 1,
                str(OUTPUT_MASKS_DIR)
            ])

            patch_idx += 1

    summary_writer.writerow([
        fname,
        img_orig_h, img_orig_w, C,
        H, W,
        PATCH_SIZE,
        STRIDE,
        total_patches
    ])

    total_output_patches += total_patches
    print(f"  → Created {total_patches} patches")

# ============================================================
# ===================== CLEANUP ===============================
# ============================================================

total_writer.writerow([
    len(image_files),
    len(image_files),  # masks are expected 1:1 with images
    total_output_patches,
    total_output_patches,
])

patch_csv_file.close()
summary_csv_file.close()
total_csv_file.close()

print("\n✅ Done.")
print(f"Patches saved to: {OUTPUT_MAIN_DIR}")
print(f"CSV files:")
print(f"  - {PATCH_CSV_PATH}")
print(f"  - {SUMMARY_CSV_PATH}")
print(f"  - {TOTAL_CSV_PATH}")
