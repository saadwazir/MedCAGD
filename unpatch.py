#!/usr/bin/env python3

import os
import sys
import csv
import shutil
import re
import numpy as np
from pathlib import Path
from PIL import Image
from patchify import unpatchify

# ============================================================
# ===================== CONFIGURATION =========================
# ============================================================

# INPUT_MAIN_DIR now points DIRECTLY to a patch directory
# Example:
#   /path/to/pred-imgs
#   /path/to/pred-imgs-edges
INPUT_MAIN_DIR = "output/test_eval_results/pred-imgs"

# OUTPUT directory (single stream output)
OUTPUT_MAIN_DIR = "output/test_eval_results/pred-imgs-unpatched"

# Patch configuration (must match patch extraction)
PATCH_SIZE = 256
STRIDE     = 64

# Canvas size patches were extracted from
PATCHED_H = 1024
PATCHED_W = 1024

# Optional resize after unpatch
RESIZE = 1
RESIZE_H = 584
RESIZE_W = 565

# Output extension
OUTPUT_EXT = ".png"

# CSV filenames
PATCH_CSV_NAME   = "patch_details.csv"
SUMMARY_CSV_NAME = "summary.csv"
TOTAL_CSV_NAME   = "total.csv"

# ============================================================
# ===================== REGEX & CONSTANTS =====================
# ============================================================

PATCH_RE = re.compile(r"-patch-(\d+)")
PATCH_NAME_RE = re.compile(r"^(.*)-patch-(\d+)$")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

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
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception as e:
            fatal(f"Failed to delete '{child}': {e}")

def patch_sort_key(name):
    m = PATCH_RE.search(name)
    return int(m.group(1)) if m else name

def compute_grid(h, w, patch_size, stride):
    if (h - patch_size) % stride != 0 or (w - patch_size) % stride != 0:
        fatal(
            f"Invalid grid: ({h}-{patch_size}) % {stride} != 0 "
            f"or ({w}-{patch_size}) % {stride} != 0"
        )
    n_h = (h - patch_size) // stride + 1
    n_w = (w - patch_size) // stride + 1
    return n_h, n_w

def to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    arr = np.rint(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)

def resize_array(arr, target_h, target_w, is_mask):
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    img = Image.fromarray(to_uint8(arr))
    img = img.resize((target_w, target_h), resample=resample)
    return np.array(img)

# ============================================================
# ===================== PATCH GROUPING ========================
# ============================================================

def group_patches_by_prefix(patches_dir: Path):
    groups = {}
    for p in patches_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = PATCH_NAME_RE.match(p.stem)
        if not m:
            continue
        prefix = m.group(1)
        groups.setdefault(prefix, []).append(p.name)
    return groups

def ensure_patch_indices(patch_files, expected_count, label):
    patch_files = sorted(patch_files, key=patch_sort_key)
    if len(patch_files) != expected_count:
        fatal(f"{label}: expected {expected_count} patches, found {len(patch_files)}")

    indices = []
    for name in patch_files:
        m = PATCH_NAME_RE.match(Path(name).stem)
        if not m:
            fatal(f"Invalid patch name: {name}")
        indices.append(int(m.group(2)))

    expected = list(range(1, expected_count + 1))
    if sorted(indices) != expected:
        fatal(f"{label}: patch indices are not continuous 1..{expected_count}")

    return patch_files

# ============================================================
# ===================== LOADING ===============================
# ============================================================

def detect_is_mask(sample_path: Path):
    with Image.open(sample_path) as img:
        arr = np.array(img)
    if arr.ndim == 2:
        return True
    if arr.ndim == 3 and arr.shape[2] == 1:
        return True
    return False

def load_patch(path, is_mask, expected_c):
    with Image.open(path) as img:
        arr = np.array(img)

    if is_mask:
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr

    if expected_c == 1:
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim == 3:
            arr = arr[:, :, :1]
        return arr

    if arr.ndim != 3 or arr.shape[2] != expected_c:
        fatal(f"Channel mismatch in patch: {path}")
    return arr

# ============================================================
# ===================== RECONSTRUCTION ========================
# ============================================================

def reconstruct_group(prefix, patch_files, is_mask):
    n_h, n_w = compute_grid(PATCHED_H, PATCHED_W, PATCH_SIZE, STRIDE)
    patch_files = ensure_patch_indices(
        patch_files, n_h * n_w, f"{prefix}"
    )

    first_patch_path = INPUT_DIR / patch_files[0]
    with Image.open(first_patch_path) as img:
        first_arr = np.array(img)

    if is_mask:
        patch_c = 1
    else:
        patch_c = 1 if first_arr.ndim == 2 else first_arr.shape[2]

    if is_mask:
        patches = np.zeros((n_h, n_w, PATCH_SIZE, PATCH_SIZE), dtype=first_arr.dtype)
    else:
        patches = np.zeros((n_h, n_w, 1, PATCH_SIZE, PATCH_SIZE, patch_c), dtype=first_arr.dtype)

    for idx, fname in enumerate(patch_files):
        patch = load_patch(INPUT_DIR / fname, is_mask, patch_c)
        i = idx // n_w
        j = idx % n_w
        if is_mask:
            patches[i, j] = patch
        else:
            patches[i, j, 0] = patch

    if is_mask:
        reconstructed = unpatchify(patches, (PATCHED_H, PATCHED_W))
    else:
        reconstructed = unpatchify(patches, (PATCHED_H, PATCHED_W, patch_c))

    if RESIZE == 1:
        reconstructed = resize_array(reconstructed, RESIZE_H, RESIZE_W, is_mask)

    reconstructed = to_uint8(reconstructed)
    if reconstructed.ndim == 3 and reconstructed.shape[2] == 1:
        reconstructed = reconstructed[:, :, 0]

    out_path = OUTPUT_DIR / f"{prefix}{OUTPUT_EXT}"
    Image.fromarray(reconstructed).save(out_path)

    return patch_c

# ============================================================
# ===================== MAIN ================================
# ============================================================

INPUT_DIR = Path(INPUT_MAIN_DIR)
OUTPUT_DIR = Path(OUTPUT_MAIN_DIR)

if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
    fatal(f"Invalid INPUT_MAIN_DIR: {INPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_dir_contents(OUTPUT_DIR)

groups = group_patches_by_prefix(INPUT_DIR)
if not groups:
    fatal("No valid patch files found")

sample_patch = INPUT_DIR / next(iter(next(iter(groups.values()))))
is_mask = detect_is_mask(sample_patch)

grid_h, grid_w = compute_grid(PATCHED_H, PATCHED_W, PATCH_SIZE, STRIDE)
expected_patches = grid_h * grid_w

PATCH_OUT = open(OUTPUT_DIR / PATCH_CSV_NAME, "w", newline="")
SUMMARY_OUT = open(OUTPUT_DIR / SUMMARY_CSV_NAME, "w", newline="")
TOTAL_OUT = open(OUTPUT_DIR / TOTAL_CSV_NAME, "w", newline="")

patch_writer = csv.writer(PATCH_OUT)
summary_writer = csv.writer(SUMMARY_OUT)
total_writer = csv.writer(TOTAL_OUT)

patch_writer.writerow([
    "output_file", "final_h", "final_w", "channels",
    "patch_file", "patch_h", "patch_w", "kind", "input_dir"
])

summary_writer.writerow([
    "output_file", "final_h", "final_w", "channels",
    "patch_size", "stride", "num_patches", "kind"
])

final_h = RESIZE_H if RESIZE == 1 else PATCHED_H
final_w = RESIZE_W if RESIZE == 1 else PATCHED_W

total_patch_count = 0

for prefix, patch_files in sorted(groups.items()):
    c = reconstruct_group(prefix, patch_files, is_mask)
    for p in patch_files:
        patch_writer.writerow([
            f"{prefix}{OUTPUT_EXT}", final_h, final_w, c,
            p, PATCH_SIZE, PATCH_SIZE,
            "mask" if is_mask else "image",
            str(INPUT_DIR)
        ])
    summary_writer.writerow([
        f"{prefix}{OUTPUT_EXT}", final_h, final_w, c,
        PATCH_SIZE, STRIDE, expected_patches,
        "mask" if is_mask else "image"
    ])
    total_patch_count += len(patch_files)

total_writer.writerow([
    total_patch_count,
    len(groups),
    "mask" if is_mask else "image"
])

PATCH_OUT.close()
SUMMARY_OUT.close()
TOTAL_OUT.close()

print("\n✅ Done.")
print(f"Input  : {INPUT_DIR}")
print(f"Output : {OUTPUT_DIR}")
print(f"Mode   : {'mask' if is_mask else 'image'}")
