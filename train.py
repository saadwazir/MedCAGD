import warnings
warnings.filterwarnings("ignore")

import os
import csv
import time
from glob import glob
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

from cfgs import *
from data import DriveDataset
from loss import get_loss_fn, get_edge_loss_fn
from utils import *
from model import build_model

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


# ============================================================
# RESUME CHECKPOINT RESOLUTION
# ============================================================

def get_resume_checkpoint():
    """
    Priority:
    1) RESUME_CKPT_OVERRIDE (xrun.py)
    2) resumed_chkpt from cfgs.py
    """
    env_ckpt = os.environ.get("RESUME_CKPT_OVERRIDE", "").strip()
    if env_ckpt:
        return env_ckpt

    if resume_train and resumed_chkpt:
        return resumed_chkpt

    return None


# ============================================================
# LOSS HELPERS (UNCHANGED)
# ============================================================

def deep_supervision_loss(main_pred, aux_preds, target, loss_fn, weights):
    assert len(weights) == 1 + len(aux_preds)
    total_loss = weights[0] * loss_fn(main_pred, target)
    for aux, w in zip(aux_preds, weights[1:]):
        total_loss += w * loss_fn(aux, target)
    return total_loss


def edge_supervision_loss(edge_preds, target, edge_loss_fn, edge_weights):
    assert len(edge_preds) == len(edge_weights)
    total_edge_loss = 0.0
    for edge_pred, w in zip(edge_preds, edge_weights):
        total_edge_loss += w * edge_loss_fn(edge_pred, target)
    return total_edge_loss


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def timestamp_ckpt_name():
    now = datetime.now()
    return now.strftime("%d-%m-%Y-%H-%M-%S-%f")[:-3] + ".pth"


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "lr": optimizer.param_groups[0]["lr"]
    }, path)


def load_checkpoint(model, optimizer, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state[k] = v

    model.load_state_dict(new_state)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    print(f"[INFO] Resumed from checkpoint: {ckpt_path}")
    return model, optimizer, start_epoch


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn, edge_loss_fn, device, epoch, scaler):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    weights = deep_supervision_weights
    if normalize_deep_weights:
        s = sum(weights)
        weights = [w / s for w in weights]

    eweights = edge_weights
    if normalize_edge_weights:
        s = sum(eweights)
        eweights = [w / s for w in eweights]

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long if num_Classes > 2 else torch.float32)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            y_main, y_aux_list, edge_preds = model(x)

            if deep_supervision:
                seg_loss = deep_supervision_loss(y_main, y_aux_list, y, loss_fn, weights)
            else:
                seg_loss = loss_fn(y_main, y)

            if edge_supervision:
                edge_loss = edge_supervision_loss(edge_preds, y, edge_loss_fn, eweights)
            else:
                edge_loss = 0.0

            loss = seg_loss + edge_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    elapsed = time.time() - start_time
    avg_loss = epoch_loss / len(loader)

    print(f"Epoch {epoch:03d} | Time {format_seconds_to_hhmmss(elapsed)} | Loss {avg_loss:.6f}")
    return avg_loss, elapsed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    seeding(seed_val)
    prepare_train_dirs()

    append_run_log("TRAIN", "Training started")

    # ---------------- Dataset ----------------
    train_x = sorted(glob(main_dataset_dir + train_images_dir))
    train_y = sorted(glob(main_dataset_dir + train_masks_dir))
    
    if fresh_training == 1:
        print(f"---------------- Dataset Size ----------------")
        print(f"Dataset Size: {len(train_x)}")
        sanity_log(f"Train: {len(train_x)}", "dataset_size.txt")

    dataset = DriveDataset(
        train_x,
        train_y,
        augment=train_augmentation_online,
        include_original=include_original_train
    )

    if fresh_training == 1:
        save_sample(dataset, train_samples)
        print_stats(dataset, "train_dataset.txt")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    # ---------------- Model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = get_loss_fn()
    edge_loss_fn = get_edge_loss_fn() if edge_supervision else None
    
    if fresh_training == 1:
        all_stats(loader,model,optimizer,device,os.path.join(sanity_check_logs, "all_stats.txt"))

    scheduler = None

    if use_lr_schedule:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_T0,
            T_mult=cosine_T_mult,
            eta_min=eta_min
        )

    scaler = GradScaler() if use_amp else None
    
    if use_amp:
        print("Mixed Precision Enabled")

    # ---------------- Resume Logic ----------------
    start_epoch = 0
    resume_ckpt = get_resume_checkpoint()

    if resume_ckpt:
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, resume_ckpt, device
        )

        append_run_log("TRAIN", f"Resumed training from checkpoint: {resume_ckpt}")

    # ================= OPTION A FIX =================
    end_epoch = start_epoch + num_epochs

    append_run_log(
        "TRAIN",
        f"Starting epochs {start_epoch + 1} → {end_epoch}"
    )

    # ---------------- Training Loop ----------------
    for epoch in range(start_epoch + 1, end_epoch + 1):

        loss, elapsed = train_one_epoch(
            model, loader, optimizer, loss_fn, edge_loss_fn, device, epoch, scaler
        )

        if scheduler:
            scheduler.step(epoch + 0.0)

        ckpt_name = timestamp_ckpt_name()
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        save_checkpoint(model, optimizer, epoch, loss, ckpt_path)

        with open(train_log_loss_file, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["checkpoint", "epoch", "loss", "time"])
            writer.writerow([ckpt_name, epoch, f"{loss:.6f}", format_seconds_to_hhmmss(elapsed)])

        append_run_log("TRAIN", f"Saved checkpoint {ckpt_name}")

    append_run_log("TRAIN", "Training finished")
