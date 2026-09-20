# ============================================================
# stopearly.py — GLOBAL BEST EVER, MONOTONIC & SAFE
# ============================================================

import os
import csv

from cfgs import *
from utils import append_run_log


VAL_ROOT = eval_root_val

BEST_CSV = os.path.join(VAL_ROOT, "best_val_weights.csv")
STATE_FILE = os.path.join(VAL_ROOT, "early_stop_state.csv")
FLAG_FILE = os.path.join(VAL_ROOT, "early_stop_flag.txt")


# ============================================================
# LOADERS
# ============================================================

def load_current_best_score():
    """
    Load the current GLOBAL best score from best_val_weights.csv.
    This file is maintained by check_val.py.
    """
    if not os.path.exists(BEST_CSV):
        raise RuntimeError("best_val_weights.csv not found")

    with open(BEST_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0 or "dice_score" not in rows[0]:
        raise RuntimeError("Invalid best_val_weights.csv format")

    return float(rows[0]["dice_score"])


def load_state():
    """
    Load early stopping state.
    Returns:
        (prev_best_score, no_improve_count)
        or (None, 0) if first iteration
    """
    if not os.path.exists(STATE_FILE):
        return None, 0

    with open(STATE_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        return None, 0

    return float(rows[0]["best_score"]), int(rows[0]["no_improve_count"])


def save_state(best_score, no_improve_count):
    with open(STATE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["best_score", "no_improve_count"])
        writer.writerow([f"{best_score:.6f}", no_improve_count])


def write_flag(value):
    with open(FLAG_FILE, "w") as f:
        f.write(value)


# ============================================================
# MAIN
# ============================================================

def main():

    append_run_log("STOP_EARLY", "Early stop evaluation started")

    # --------------------------------------------------
    # CURRENT GLOBAL BEST (FROM CHECK_VAL)
    # --------------------------------------------------
    current_best = load_current_best_score()

    # --------------------------------------------------
    # PREVIOUS STATE
    # --------------------------------------------------
    prev_best, no_improve_count = load_state()

    # --------------------------------------------------
    # FIRST ITERATION — INITIALIZE BASELINE
    # --------------------------------------------------
    if prev_best is None:
        save_state(current_best, 0)
        write_flag("CONTINUE")
        append_run_log(
            "STOP_EARLY",
            f"Baseline initialized at {current_best:.6f}"
        )
        return

    # --------------------------------------------------
    # STRICT IMPROVEMENT (GLOBAL BEST EVER)
    # --------------------------------------------------
    if current_best > prev_best:
        save_state(current_best, 0)
        write_flag("CONTINUE")
        append_run_log(
            "STOP_EARLY",
            f"Improved global best {prev_best:.6f} → {current_best:.6f}"
        )
        return

    # --------------------------------------------------
    # NO IMPROVEMENT
    # --------------------------------------------------
    no_improve_count += 1
    save_state(prev_best, no_improve_count)

    append_run_log(
        "STOP_EARLY",
        f"No improvement {no_improve_count}/{val_eval_threshold}"
    )

    if no_improve_count >= val_eval_threshold:
        write_flag("STOP")
        append_run_log("STOP_EARLY", "Early stopping triggered")
    else:
        write_flag("CONTINUE")


if __name__ == "__main__":
    main()
