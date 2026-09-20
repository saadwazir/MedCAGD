# ============================================================
# test_eval_manual.py
# Standalone evaluation script for precomputed segmentation masks
#
# - No model inference
# - No project dependencies
# - Supports binary and multi-class segmentation
# - Computes per-image metrics + dataset summary
# - Background class (0) is ALWAYS excluded from averages
# ============================================================

import os
import csv
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

# ============================================================
# USER CONFIGURATION (EDIT THESE)
# ============================================================

# Ground-truth mask directory
GT_MASK_DIR = "../0-datasets/eye-datasets/DRIVE-2004/test/masks"

# Predicted mask directory
PRED_MASK_DIR = "output/test_eval_results/pred-imgs-unpatched"

# Output CSV paths
PER_IMAGE_CSV = "output/per_image_metrics_manual.csv"
SUMMARY_CSV = "output/summary_metrics_manual.csv"

# Number of classes (2 = binary, >2 = multiclass)
NUM_CLASSES = 2

# ============================================================
# CONSTANTS
# ============================================================

BACKGROUND_CLASS = 0
EPS = 1e-7

# ============================================================
# METRIC HELPERS
# ============================================================

def dice_score(pred, gt):
    inter = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return (2.0 * inter + EPS) / (union + EPS)

def iou_score(pred, gt):
    inter = np.sum(pred * gt)
    union = np.sum((pred + gt) > 0)
    return (inter + EPS) / (union + EPS)

def precision_score(pred, gt):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    return (tp + EPS) / (tp + fp + EPS)

def recall_score(pred, gt):
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    return (tp + EPS) / (tp + fn + EPS)

def hd95_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 95.0

    dt_pred = distance_transform_edt(~pred)
    dt_gt = distance_transform_edt(~gt)

    surface_pred = np.logical_xor(pred, binary_erosion(pred))
    surface_gt = np.logical_xor(gt, binary_erosion(gt))

    d_pred_gt = dt_gt[surface_pred]
    d_gt_pred = dt_pred[surface_gt]

    all_dist = np.concatenate([d_pred_gt, d_gt_pred])
    return np.percentile(all_dist, 95)

# ============================================================
# MASK LOADING
# ============================================================

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask

# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    gt_files = sorted(os.listdir(GT_MASK_DIR))
    if len(gt_files) == 0:
        raise RuntimeError("No GT masks found")

    # Prepare CSV headers
    if NUM_CLASSES == 2:
        per_image_header = ["Image", "Dice", "Precision", "Recall", "HD95", "IoU"]
    else:
        per_image_header = ["Image"]
        for m in ["Dice", "Precision", "Recall", "HD95", "IoU"]:
            for c in range(1, NUM_CLASSES):
                per_image_header.append(f"{m}_class_{c}")
            per_image_header.append(f"{m}_macro_avg")

    with open(PER_IMAGE_CSV, "w", newline="") as f:
        csv.writer(f).writerow(per_image_header)

    # Accumulators for summary
    summary_acc = {}

    for fname in gt_files:
        gt_path = os.path.join(GT_MASK_DIR, fname)
        pred_path = os.path.join(PRED_MASK_DIR, fname)

        if not os.path.exists(pred_path):
            raise RuntimeError(f"Missing prediction for {fname}")

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        if gt.shape != pred.shape:
            raise RuntimeError(f"Shape mismatch for {fname}")

        row = [fname]

        if NUM_CLASSES == 2:
            gt_bin = (gt > 0).astype(np.uint8)
            pred_bin = (pred > 0).astype(np.uint8)

            dice = dice_score(pred_bin, gt_bin)
            prec = precision_score(pred_bin, gt_bin)
            rec = recall_score(pred_bin, gt_bin)
            hd = hd95_score(pred_bin, gt_bin)
            iou = iou_score(pred_bin, gt_bin)

            metrics = [dice, prec, rec, hd, iou]
            row.extend([f"{m:.4f}" for m in metrics])

            for k, v in zip(["Dice", "Precision", "Recall", "HD95", "IoU"], metrics):
                summary_acc.setdefault(k, []).append(v)

        else:
            macro_vals = {k: [] for k in ["Dice", "Precision", "Recall", "HD95", "IoU"]}

            for c in range(1, NUM_CLASSES):
                gt_c = (gt == c).astype(np.uint8)
                pred_c = (pred == c).astype(np.uint8)

                d = dice_score(pred_c, gt_c)
                p = precision_score(pred_c, gt_c)
                r = recall_score(pred_c, gt_c)
                h = hd95_score(pred_c, gt_c)
                i = iou_score(pred_c, gt_c)

                for k, v in zip(["Dice", "Precision", "Recall", "HD95", "IoU"], [d, p, r, h, i]):
                    row.append(f"{v:.4f}")
                    macro_vals[k].append(v)
                    summary_acc.setdefault(f"{k}_class_{c}", []).append(v)

            for k in ["Dice", "Precision", "Recall", "HD95", "IoU"]:
                macro = float(np.mean(macro_vals[k]))
                row.append(f"{macro:.4f}")
                summary_acc.setdefault(f"{k}_macro_avg", []).append(macro)

        with open(PER_IMAGE_CSV, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ===================== SUMMARY CSV =====================

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary_acc.keys()))
        writer.writerow([f"{np.mean(v):.4f}" for v in summary_acc.values()])


if __name__ == "__main__":
    main()
