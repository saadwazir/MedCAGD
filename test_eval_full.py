# ============================================================
# test_eval_full.py
# Full test-time evaluation with:
# - PNG prediction saving
# - Edge map saving (if enabled)
# - Per-image metrics CSV
# - Summary metrics CSV
# - Safe re-run (auto cleanup of prediction folders + eval_status)
# ============================================================

import os
import csv
import shutil
import torch
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from cfgs import *
from data import DriveDataset
from model import build_model
from eval_metrics import compute_all_metrics
from utils import *
from loss import get_loss_fn, get_edge_loss_fn

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


# ============================================================
# PATHS
# ============================================================

PRED_IMG_DIR = os.path.join(eval_root_test, "pred-imgs")
PRED_EDGE_DIR = os.path.join(eval_root_test, "pred-imgs-edges")
PRED_CSV_DIR = os.path.join(eval_root_test, "pred-csv")
EVAL_STATUS_FILE = os.path.join(eval_root_test, "eval_status.csv")


# ============================================================
# UTILS
# ============================================================

def reset_prediction_dirs():
    shutil.rmtree(PRED_IMG_DIR, ignore_errors=True)
    shutil.rmtree(PRED_EDGE_DIR, ignore_errors=True)

    os.makedirs(PRED_IMG_DIR, exist_ok=True)
    if edge_supervision:
        os.makedirs(PRED_EDGE_DIR, exist_ok=True)


def reset_eval_status():
    if os.path.exists(EVAL_STATUS_FILE):
        os.remove(EVAL_STATUS_FILE)

    with open(EVAL_STATUS_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["checkpoint"])


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        clean_state[k] = v

    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()
    return model


def already_evaluated(ckpt_name):
    if not os.path.exists(EVAL_STATUS_FILE):
        return False
    with open(EVAL_STATUS_FILE, "r") as f:
        return ckpt_name in [row[0] for row in csv.reader(f)]


def record_eval_status(ckpt_name):
    with open(EVAL_STATUS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ckpt_name])


def save_seg_prediction(pred, image_name, save_dir):
    save_path = os.path.join(save_dir, f"{image_name}")

    if num_Classes > 2:
        label = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        img = label.squeeze().cpu().numpy().astype(np.uint8)
    else:
        img = (torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    cv2.imwrite(save_path, img)


def save_edge_prediction(edge_pred, image_name):
    save_path = os.path.join(PRED_EDGE_DIR, f"{image_name}")
    img = (torch.sigmoid(edge_pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    cv2.imwrite(save_path, img)


# ============================================================
# MAIN
# ============================================================

def main():

    seeding(seed_val)
    append_run_log("TEST_EVAL_FULL", "Full test evaluation started")

    prepare_eval_dirs(eval_root_test)
    create_dir(PRED_CSV_DIR)

    reset_prediction_dirs()
    reset_eval_status()

    test_x = sorted(glob(main_dataset_dir + test_images_dir))
    test_y = sorted(glob(main_dataset_dir + test_masks_dir))

    if len(test_x) == 0:
        raise RuntimeError("No test images found")

    dataset = DriveDataset(
        test_x,
        test_y,
        augment=False,
        include_original=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = get_loss_fn()
    edge_loss_fn = get_edge_loss_fn() if edge_supervision else None

    only_ckpt = os.environ.get("EVAL_ONLY_CKPT", "").strip()

    if only_ckpt:
        if not os.path.isfile(only_ckpt):
            raise RuntimeError(f"EVAL_ONLY_CKPT not found: {only_ckpt}")
        checkpoints = [only_ckpt]
    else:
        checkpoints = sorted(glob(os.path.join(checkpoint_dir, "*.pth")))

    if len(checkpoints) == 0:
        raise RuntimeError("No checkpoints found")

    summary_csv = os.path.join(PRED_CSV_DIR, "summary_metrics.csv")

    if not os.path.exists(summary_csv):
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Checkpoint", "Time/Image (s)", "Mean Time (s)", "Mean Seg Loss"]
            if edge_supervision:
                header.append("Mean Edge Loss")

            if num_Classes > 2:
                for m in ["dice", "precision", "recall", "hd95", "iou"]:
                    for c in range(num_Classes):
                        header.append(f"{m}_class_{c}")
                    header.append(f"{m}_macro_avg")
            else:
                header += ["Dice", "Precision", "Recall", "HD95", "IoU"]

            writer.writerow(header)

    # ---------------- Evaluation ----------------
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        ckpt_base = os.path.splitext(ckpt_name)[0]

        ckpt_pred_dir = os.path.join(PRED_IMG_DIR, ckpt_base)
        os.makedirs(ckpt_pred_dir, exist_ok=True)

        per_image_csv = os.path.join(
            PRED_CSV_DIR,
            f"{ckpt_base}_per_image_metrics.csv"
        )

        with open(per_image_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Image", "Time (s)", "Seg Loss"]
            if num_Classes > 2:
                for m in ["dice", "precision", "recall", "hd95", "iou"]:
                    for c in range(num_Classes):
                        header.append(f"{m}_class_{c}")
                    header.append(f"{m}_macro_avg")
            else:
                header += ["Dice", "Precision", "Recall", "HD95", "IoU"]
            writer.writerow(header)

        append_run_log("TEST_EVAL_FULL", f"Evaluating {ckpt_name}")

        model = build_model().to(device)
        model = load_checkpoint(model, ckpt_path, device)

        all_preds, all_gts = [], []
        per_image_times, seg_losses, edge_losses = [], [], []

        for idx, (imgs, masks) in tqdm(enumerate(loader), total=len(loader), ncols=80):
            imgs, masks = imgs.to(device), masks.to(device)

            image_name = os.path.basename(test_x[idx])

            with torch.no_grad():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                outputs = model(imgs)
                if len(outputs) == 3:
                    preds, _, edge_preds = outputs
                else:
                    preds, _ = outputs
                    edge_preds = None

                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end) / 1000.0

                seg_loss = loss_fn(preds, masks).item()
                per_image_times.append(elapsed)
                seg_losses.append(seg_loss)

                if edge_supervision and edge_preds is not None:
                    edge_loss = 0.0
                    for w, e in zip(edge_weights, edge_preds):
                        edge_loss += w * edge_loss_fn(e, masks).item()
                    edge_losses.append(edge_loss)

            save_seg_prediction(preds, image_name, ckpt_pred_dir)

            metrics_img = compute_all_metrics(
                preds.cpu(),
                masks.cpu(),
                threshold=0.5,
                selected_metrics=["dice", "precision", "recall", "hd95", "iou"]
            )

            with open(per_image_csv, "a", newline="") as f:
                writer = csv.writer(f)
                row = [image_name, f"{elapsed:.4f}", f"{seg_loss:.4f}"]

                if num_Classes > 2:
                    for m in ["dice", "precision", "recall", "hd95", "iou"]:
                        for c in range(num_Classes):
                            row.append(f"{metrics_img[m]['class_wise'][c]:.4f}")
                        row.append(f"{metrics_img[m]['macro_avg']:.4f}")
                else:
                    row.extend([
                        f"{metrics_img['dice']:.4f}",
                        f"{metrics_img['precision']:.4f}",
                        f"{metrics_img['recall']:.4f}",
                        f"{metrics_img['hd95']:.4f}",
                        f"{metrics_img['iou']:.4f}",
                    ])

                writer.writerow(row)

            all_preds.append(preds.cpu())
            all_gts.append(masks.cpu())

        record_eval_status(ckpt_name)

    append_run_log("TEST_EVAL_FULL", "Full test evaluation completed")


if __name__ == "__main__":
    main()