import os
import csv
import time
import random
import shutil
import numpy as np
import cv2
import torch
import albumentations as A
from cfgs import *

# ============================================================
# SEEDING
# ============================================================

def seeding(seed):
    # XRUN override (if present)
    env_seed = os.environ.get("XRUN_SEED")
    if env_seed is not None:
        seed = int(env_seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = seed_val + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ============================================================
# DIRECTORY UTILITIES
# ============================================================

def create_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def remove_dir(path):
    """Remove directory if exists."""
    if os.path.exists(path):
        shutil.rmtree(path)


def reset_dir(path):
    """Delete and recreate directory."""
    remove_dir(path)
    create_dir(path)


# ============================================================
# TRAIN / EVAL DIRECTORY HELPERS
# ============================================================

def prepare_train_dirs():
    """
    Prepare training-related directories.
    SAFE: does not delete unless explicitly called.
    """
    create_dir("output")
    create_dir(checkpoint_dir)
    create_dir(sanity_check_logs)
    create_dir(train_samples)
    create_dir(train_logs)


def prepare_eval_dirs(eval_root):
    """
    Prepare evaluation directories under a given eval root.
    Used by test_eval.py and val_eval.py.
    """
    create_dir(eval_root)
    create_dir(os.path.join(eval_root, "pred-imgs"))
    create_dir(os.path.join(eval_root, "pred-csv"))


# ============================================================
# LOGGING
# ============================================================

def format_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def append_run_log(stage, message):
    """
    Unified run logger.
    Appends logs to output/run_log.csv

    Columns:
    timestamp, stage, message
    """
    log_file = "output/run_log.csv"
    create_dir("output")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    write_header = not os.path.exists(log_file)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "stage", "message"])
        writer.writerow([timestamp, stage, message])

    print(f"[{stage}] {message}")


# ============================================================
# SANITY & STATS LOGGING (UNCHANGED BEHAVIOR)
# ============================================================

def sanity_log(content: str, file_path: str = "log_output.txt"):
    """
    Print content and write it to a sanity log file.
    """
    file_path_final = os.path.join(sanity_check_logs, file_path)

    if os.path.exists(file_path_final):
        os.remove(file_path_final)

    print(content)
    with open(file_path_final, "a") as f:
        f.write(content + "\n")


def print_stats(dataset, log_file):
    image, mask = dataset[0]

    log_lines = []
    log_lines.append("=" * 50)
    log_lines.append("Image Stats")
    log_lines.append(f"Shape: {image.unsqueeze(0).shape}")
    log_lines.append(f"Type: {'RGB' if image.shape[0] == 3 else 'Grayscale'}")
    log_lines.append(f"Min: {image.min().item():.4f} | Max: {image.max().item():.4f}")
    log_lines.append(f"Dtype: {image.dtype}")
    log_lines.append("-" * 50)
    log_lines.append("Mask Stats")

    if num_Classes > 2:
        log_lines.append(f"Shape: {mask.shape}")
        log_lines.append(f"Unique class labels: {torch.unique(mask)}")
    else:
        log_lines.append(f"Shape: {mask.unsqueeze(0).shape}")
        log_lines.append(f"Unique values: {torch.unique(mask)}")

    log_lines.append(f"Min: {mask.min().item():.4f} | Max: {mask.max().item():.4f}")
    log_lines.append(f"Dtype: {mask.dtype}")
    log_lines.append("=" * 50)
    log_lines.append("\n")

    formatted_output = "\n".join(log_lines)
    sanity_log(formatted_output, log_file)


def all_stats(loader, model, optimizer, device, save_path):
    lines = []
    lines.append("=" * 50)
    lines.append("LOADER STATS")
    lines.append(f"Total dataset samples: {len(loader.dataset)}")
    lines.append(f"Batch size: {loader.batch_size}")
    lines.append(f"Num batches: {len(loader)}")

    try:
        first_batch = next(iter(loader))
        lines.append(f"Samples per batch (first batch): {len(first_batch[0])}")
    except Exception:
        lines.append("Samples per batch: could not retrieve")

    sampler_type = type(loader.sampler).__name__
    shuffle_status = 'True' if sampler_type == 'RandomSampler' else 'False'
    lines.append(f"Shuffle: {shuffle_status}")
    lines.append(f"Num workers: {loader.num_workers}")
    lines.append(f"Pin memory: {loader.pin_memory}")
    lines.append("")

    lines.append("=" * 50)
    lines.append("MODEL STATS")
    lines.append(f"Model class: {model.__class__.__name__}")
    lines.append(f"Model device: {device}")
    lines.append(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    lines.append(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    lines.append("")

    lines.append("=" * 50)
    lines.append("OPTIMIZER CONFIG")
    lines.append(str(optimizer))
    lines.append("=" * 50)

    with open(save_path, "w") as f:
        for line in lines:
            print(line)
            f.write(line + "\n")


def save_sample(dataset, path_var, num_samples=batch_size):
    os.makedirs(os.path.join(path_var, "images"), exist_ok=True)
    os.makedirs(os.path.join(path_var, "masks"), exist_ok=True)

    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        image_tensor, mask_tensor = dataset[idx]
        image = image_tensor.numpy().transpose(1, 2, 0)

        if num_Classes > 2:
            mask = mask_tensor.numpy().astype(np.uint8)
        else:
            mask = mask_tensor.numpy().squeeze(0) * 255.0

        image_path = os.path.join(path_var, "images", f"{idx}.png")
        mask_path = os.path.join(path_var, "masks", f"{idx}.png")

        cv2.imwrite(image_path, (image * 255).astype(np.uint8))
        cv2.imwrite(mask_path, mask.astype(np.uint8))
