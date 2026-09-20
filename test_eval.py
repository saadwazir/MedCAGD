import os
import csv
import torch
import numpy as np
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
# EVAL PATHS (FROM CFGS)
# ============================================================

PRED_IMG_DIR = os.path.join(eval_root_test, "pred-imgs")
PRED_CSV_DIR = os.path.join(eval_root_test, "pred-csv")
EVAL_STATUS_FILE = os.path.join(eval_root_test, "eval_status.csv")


# ============================================================
# CHECKPOINT LOADER
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():

    seeding(seed_val)
    append_run_log("TEST_EVAL", "Test evaluation started")

    # Prepare directories
    prepare_eval_dirs(eval_root_test)
    create_dir(PRED_CSV_DIR)

    if not os.path.exists(EVAL_STATUS_FILE):
        with open(EVAL_STATUS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["checkpoint"])

    # ---------------- Dataset ----------------
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = get_loss_fn()
    edge_loss_fn = get_edge_loss_fn() if edge_supervision else None

    # ---------------- Checkpoint Selection ----------------
    only_ckpt = os.environ.get("EVAL_ONLY_CKPT", "").strip()

    if only_ckpt:
        if not os.path.isfile(only_ckpt):
            raise RuntimeError(f"EVAL_ONLY_CKPT not found: {only_ckpt}")

        checkpoints = [only_ckpt]

        append_run_log(
            "TEST_EVAL",
            f"Evaluating ONLY checkpoint: {only_ckpt}"
        )
    else:
        checkpoints = sorted(glob(os.path.join(checkpoint_dir, "*.pth")))

        append_run_log(
            "TEST_EVAL",
            f"Evaluating {len(checkpoints)} checkpoints"
        )

    if len(checkpoints) == 0:
        raise RuntimeError("No checkpoints found for test evaluation")

    # ---------------- Summary CSV ----------------
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

    # ---------------- Evaluation Loop ----------------
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)

        if already_evaluated(ckpt_name):
            continue

        append_run_log("TEST_EVAL", f"Evaluating {ckpt_name}")

        model = build_model().to(device)
        model = load_checkpoint(model, ckpt_path, device)

        all_preds, all_gts = [], []
        per_image_times, seg_losses, edge_losses = [], [], []

        for imgs, masks in tqdm(loader, desc=ckpt_name, ncols=80):
            imgs = imgs.to(device)
            masks = masks.to(device)

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

                per_image_times.append(elapsed)
                seg_losses.append(loss_fn(preds, masks).item())

                if edge_supervision and edge_preds is not None:
                    edge_loss = 0.0
                    for w, e in zip(edge_weights, edge_preds):
                        edge_loss += w * edge_loss_fn(e, masks).item()
                    edge_losses.append(edge_loss)

            all_preds.append(preds.cpu())
            all_gts.append(masks.cpu())

        preds_tensor = torch.cat(all_preds, dim=0)
        gts_tensor = torch.cat(all_gts, dim=0)

        metrics = compute_all_metrics(
            preds_tensor,
            gts_tensor,
            threshold=0.5,
            selected_metrics=["dice", "precision", "recall", "hd95", "iou"]
        )

        mean_time = float(np.mean(per_image_times))
        mean_seg_loss = float(np.mean(seg_losses))
        mean_edge_loss = float(np.mean(edge_losses)) if edge_losses else 0.0

        row = [
            ckpt_name,
            f"{per_image_times[0]:.4f}",
            f"{mean_time:.4f}",
            f"{mean_seg_loss:.4f}"
        ]

        if edge_supervision:
            row.append(f"{mean_edge_loss:.4f}")

        if num_Classes > 2:
            for m in ["dice", "precision", "recall", "hd95", "iou"]:
                for c in range(num_Classes):
                    row.append(f"{metrics[m]['class_wise'][c]:.4f}")
                row.append(f"{metrics[m]['macro_avg']:.4f}")
        else:
            row.extend([
                f"{metrics['dice']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['hd95']:.4f}",
                f"{metrics['iou']:.4f}",
            ])

        with open(summary_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

        record_eval_status(ckpt_name)

    append_run_log("TEST_EVAL", "Test evaluation completed")


if __name__ == "__main__":
    main()
