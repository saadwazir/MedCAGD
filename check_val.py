# ============================================================
# check_val.py — GREEDY GLOBAL BEST (DISK = TRUTH)
# ============================================================

import os
import csv
import pandas as pd

from cfgs import *
from utils import append_run_log


# ============================================================
# PATHS (FROM CFGS)
# ============================================================

VAL_ROOT = eval_root_val
PRED_CSV_DIR = os.path.join(VAL_ROOT, "pred-csv")
SUMMARY_CSV = os.path.join(PRED_CSV_DIR, "summary_metrics.csv")

BEST_CSV = os.path.join(VAL_ROOT, "best_val_weights.csv")
BEST_HISTORY_CSV = os.path.join(VAL_ROOT, "best_val_history.csv")


# ============================================================
# MAIN
# ============================================================

def main():

    append_run_log("CHECK_VAL", "Checking validation results")

    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(
            f"Validation summary CSV not found: {SUMMARY_CSV}"
        )

    # --------------------------------------------------------
    # Load validation summary
    # --------------------------------------------------------
    df = pd.read_csv(SUMMARY_CSV)

    if "Checkpoint" not in df.columns:
        raise RuntimeError(
            "Expected column 'Checkpoint' not found in validation CSV"
        )

    # --------------------------------------------------------
    # FILTER: ONLY EXISTING CHECKPOINTS (GREEDY INVARIANT)
    # --------------------------------------------------------
    def exists_on_disk(ckpt_name):
        return os.path.exists(os.path.join(checkpoint_dir, ckpt_name))

    df = df[df["Checkpoint"].apply(exists_on_disk)]

    if len(df) == 0:
        raise RuntimeError(
            "No existing checkpoints found on disk to select global best"
        )

    # --------------------------------------------------------
    # Detect task type & metric
    # --------------------------------------------------------
    if "Dice" in df.columns:
        metric_col = "Dice"
        task_type = "binary"
    elif "dice_macro_avg" in df.columns:
        metric_col = "dice_macro_avg"
        task_type = "multiclass"
    else:
        raise RuntimeError(
            "Neither 'Dice' nor 'dice_macro_avg' column found in validation CSV"
        )

    append_run_log(
        "CHECK_VAL",
        f"Detected task type: {task_type}, using metric: {metric_col}"
    )

    # --------------------------------------------------------
    # GLOBAL BEST SELECTION (ONLY SURVIVING CHECKPOINTS)
    # --------------------------------------------------------
    df_sorted = df.sort_values(by=metric_col, ascending=False)

    best_row = df_sorted.iloc[0]
    best_ckpt = best_row["Checkpoint"]
    best_score = float(best_row[metric_col])

    append_run_log(
        "CHECK_VAL",
        f"Global best checkpoint: {best_ckpt} | "
        f"{metric_col} = {best_score:.6f}"
    )

    # --------------------------------------------------------
    # Write best_val_weights.csv (SINGLE SOURCE OF TRUTH)
    # --------------------------------------------------------
    with open(BEST_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["weight_file", "dice_score"])
        writer.writerow([best_ckpt, f"{best_score:.6f}"])

    append_run_log(
        "CHECK_VAL",
        f"Saved current global best → {BEST_CSV}"
    )

    # --------------------------------------------------------
    # Maintain BEST HISTORY (SURVIVING ONLY)
    # --------------------------------------------------------
    if os.path.exists(BEST_HISTORY_CSV):
        hist_df = pd.read_csv(BEST_HISTORY_CSV)
        prev_best = float(hist_df["dice_score"].max())
    else:
        prev_best = float("-inf")

    if best_score > prev_best + 1e-8:
        write_header = not os.path.exists(BEST_HISTORY_CSV)
        with open(BEST_HISTORY_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["weight_file", "dice_score"])
            writer.writerow([best_ckpt, f"{best_score:.6f}"])

        append_run_log(
            "CHECK_VAL",
            "New GLOBAL BEST recorded in best_val_history.csv"
        )
    else:
        append_run_log(
            "CHECK_VAL",
            "No new global best (history unchanged)"
        )

    # --------------------------------------------------------
    # ALWAYS ENSURE BEST_VAL_WEIGHTS REFLECTS BEST-EVER
    # --------------------------------------------------------
    hist_df = pd.read_csv(BEST_HISTORY_CSV)
    best_ever_row = hist_df.sort_values(
        by="dice_score", ascending=False
    ).iloc[0]

    with open(BEST_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["weight_file", "dice_score"])
        writer.writerow([
            best_ever_row["weight_file"],
            f"{best_ever_row['dice_score']:.6f}"
        ])

    append_run_log(
        "CHECK_VAL",
        "best_val_weights.csv synchronized with best_val_history.csv"
    )


if __name__ == "__main__":
    main()
