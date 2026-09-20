# ============================================================
# xrun.py — GREEDY GLOBAL BEST (DISK = TRUTH)
# ============================================================

import os
import sys
import shutil
import subprocess
import time
import csv
import pandas as pd
import argparse
from glob import glob
import numpy as np
import csv

from cfgs import *
from utils import append_run_log, create_dir


# ============================================================
# CHECKPOINT CSV CLEANUP (GREEDY MODE)
# ============================================================

def remove_checkpoint_from_all_csvs(ckpt_name):
    """
    Remove all CSV rows that reference a deleted checkpoint.

    Invariant:
    If a checkpoint does not exist on disk,
    it must not appear in any CSV.
    """

    # ---- Validation summary metrics ----
    summary_csv = os.path.join(eval_root_val, "pred-csv", "summary_metrics.csv")
    if os.path.exists(summary_csv):
        df = np.genfromtxt(summary_csv, delimiter=",", dtype=str)
        if df.ndim > 1:
            header, rows = df[0], df[1:]
            rows = rows[rows[:, 0] != ckpt_name]
            with open(summary_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    # ---- Evaluation status ----
    eval_status_csv = os.path.join(eval_root_val, "eval_status.csv")
    if os.path.exists(eval_status_csv):
        df = np.genfromtxt(eval_status_csv, delimiter=",", dtype=str)
        if df.ndim > 1:
            header, rows = df[0], df[1:]
            rows = rows[rows[:, 0] != ckpt_name]
            with open(eval_status_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    # ---- Best validation history ----
    best_hist_csv = os.path.join(eval_root_val, "best_val_history.csv")
    if os.path.exists(best_hist_csv):
        df = np.genfromtxt(best_hist_csv, delimiter=",", dtype=str)
        if df.ndim > 1:
            header, rows = df[0], df[1:]
            rows = rows[rows[:, 0] != ckpt_name]
            with open(best_hist_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    # ---- Current best pointer (safety) ----
    best_csv = os.path.join(eval_root_val, "best_val_weights.csv")
    if os.path.exists(best_csv):
        df = np.genfromtxt(best_csv, delimiter=",", dtype=str)
        if df.ndim > 1:
            header, row = df[0], df[1]
            if row[0] == ckpt_name:
                os.remove(best_csv)
    
    hist_csv = os.path.join(eval_root_val, "best_val_history.csv")
    if os.path.exists(hist_csv):
        df = pd.read_csv(hist_csv)
        df = df[df["weight_file"].apply(
            lambda x: os.path.exists(os.path.join(checkpoint_dir, x))
        )]
        df.to_csv(hist_csv, index=False)


# ============================================================
# TERMINAL / CACHE
# ============================================================

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def delete_pycache_once():
    for root, dirs, _ in os.walk("."):
        if "__pycache__" in dirs:
            p = os.path.join(root, "__pycache__")
            shutil.rmtree(p, ignore_errors=True)
            append_run_log("XRUN", f"Deleted {p}")


# ============================================================
# SCRIPT RUNNER
# ============================================================

def run_script(script_name):
    append_run_log("XRUN", f"Running {script_name}")
    proc = subprocess.Popen(
        [sys.executable, script_name],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{script_name} failed")


# ============================================================
# CFG REWRITE GUARDS (UNCHANGED)
# ============================================================

def rewrite_cfgs_fresh_training_zero():
    _rewrite_cfgs_fresh_training(0)


def rewrite_cfgs_fresh_training_one():
    _rewrite_cfgs_fresh_training(1)


def _rewrite_cfgs_fresh_training(value):
    cfg_file = "cfgs.py"
    with open(cfg_file) as f:
        lines = f.readlines()

    out = []
    found = False
    for l in lines:
        if l.strip().startswith("fresh_training"):
            out.append(f"fresh_training = {value}\n")
            found = True
        else:
            out.append(l)

    if not found:
        raise RuntimeError("fresh_training not found in cfgs.py")

    with open(cfg_file, "w") as f:
        f.writelines(out)

    append_run_log("XRUN", f"cfgs.py updated: fresh_training={value}")


# ============================================================
# OUTPUT RESET (FRESH TRAINING)
# ============================================================

def delete_output_once():
    if os.path.exists("output"):
        shutil.rmtree("output")
        append_run_log("XRUN", "Deleted output/ (fresh training)")

    create_dir("output")
    create_dir(checkpoint_dir)
    create_dir(eval_root_val)
    create_dir(eval_root_test)


# ============================================================
# CSV CLEANUP HELPERS (NEW — GREEDY CONSISTENCY)
# ============================================================

def _remove_ckpt_from_csv(csv_path, ckpt_name, col_name):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if col_name in df.columns:
        df = df[df[col_name] != ckpt_name]
        df.to_csv(csv_path, index=False)


def purge_checkpoint_from_all_csvs(ckpt_name):
    _remove_ckpt_from_csv(
        os.path.join(eval_root_val, "pred-csv", "summary_metrics.csv"),
        ckpt_name,
        "Checkpoint"
    )
    _remove_ckpt_from_csv(
        os.path.join(eval_root_val, "eval_status.csv"),
        ckpt_name,
        "checkpoint"
    )
    _remove_ckpt_from_csv(
        os.path.join(eval_root_val, "best_val_history.csv"),
        ckpt_name,
        "weight_file"
    )


# ============================================================
# GLOBAL BEST HANDLING (GREEDY)
# ============================================================

# ============================================================
# CONFIG COMBINATION CONTROL
# ============================================================

def select_combination(iter_idx):
    from cfg_switcher import load_combinations, rewrite_cfgs_with_combination

    combos = load_combinations()
    if not combos:
        raise RuntimeError("train_random_comb.csv is empty")

    # Iteration 1 → baseline cfgs.py (NO CHANGE)
    if iter_idx == 1:
        append_run_log("CFG_SWITCH", "Iteration 1: using baseline cfgs.py")
        return

    # Iteration 2 → combos[0], Iteration 3 → combos[1], ...
    combo_idx = (iter_idx - 2) % len(combos)
    combo = combos[combo_idx]

    rewrite_cfgs_with_combination(combo)

    append_run_log(
        "CFG_SWITCH",
        f"Iteration {iter_idx}: combo_index={combo_idx}"
    )



def mark_best_checkpoint(iter_idx):
    best_csv = os.path.join(eval_root_val, "best_val_weights.csv")
    df = pd.read_csv(best_csv)

    best_name = df.iloc[0]["weight_file"]
    best_score = df.iloc[0]["dice_score"]

    base, ext = os.path.splitext(best_name)

    # If already locked, do nothing
    if "_best_iteration_" in best_name:
        return os.path.join(checkpoint_dir, best_name)

    src = os.path.join(checkpoint_dir, best_name)
    if not os.path.exists(src):
        raise RuntimeError(f"Best checkpoint missing on disk: {best_name}")

    new_name = f"{base}_best_iteration_{iter_idx}_lock{ext}"
    dst = os.path.join(checkpoint_dir, new_name)

    # Create lock file
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        append_run_log("XRUN", f"🔒 Locked BEST CHECKPOINT: {new_name}")

    # -------------------------------------------------
    # UPDATE CSVs TO MATCH DISK (LOCKED NAMES ONLY)
    # -------------------------------------------------

    # Update best_val_weights.csv
    df.loc[0, "weight_file"] = new_name
    df.to_csv(best_csv, index=False)

    # Update best_val_history.csv (replace base with lock)
    hist_csv = os.path.join(eval_root_val, "best_val_history.csv")
    if os.path.exists(hist_csv):
        hdf = pd.read_csv(hist_csv)
        hdf["weight_file"] = hdf["weight_file"].replace(best_name, new_name)
        hdf.to_csv(hist_csv, index=False)

    return dst




def load_global_best_checkpoint():
    hist_csv = os.path.join(eval_root_val, "best_val_history.csv")
    if not os.path.exists(hist_csv):
        raise RuntimeError("best_val_history.csv missing")

    df = pd.read_csv(hist_csv)
    if len(df) == 0:
        raise RuntimeError("best_val_history.csv empty")

    best_row = df.sort_values(
        by="dice_score", ascending=False
    ).iloc[0]

    ckpt = best_row["weight_file"]
    path = os.path.join(checkpoint_dir, ckpt)

    if not os.path.exists(path):
        raise RuntimeError(f"Global best missing on disk: {ckpt}")

    return path



# ============================================================
# PRUNING (GREEDY + CSV SYNC)
# ============================================================

def prune_non_best_checkpoints():
    hist_csv = os.path.join(eval_root_val, "best_val_history.csv")
    if not os.path.exists(hist_csv):
        raise RuntimeError("best_val_history.csv missing")

    hist_df = pd.read_csv(hist_csv)
    locked = set(hist_df["weight_file"].tolist())

    removed = 0
    for f in os.listdir(checkpoint_dir):
        if not f.endswith((".pth", ".pt")):
            continue
        if f in locked:
            continue

        os.remove(os.path.join(checkpoint_dir, f))
        remove_checkpoint_from_all_csvs(f)
        removed += 1

    append_run_log(
        "XRUN",
        f"Pruned non-best checkpoints | removed={removed}"
    )



# ============================================================
# EARLY STOP
# ============================================================

def should_stop_early():
    f = os.path.join(eval_root_val, "early_stop_flag.txt")
    with open(f) as fh:
        return fh.read().strip() == "STOP"


# ============================================================
# SUMMARY
# ============================================================

def write_xrun_summary(total_iters, early_stop_iter):
    best_ckpt = load_global_best_checkpoint()

    best_csv = os.path.join(eval_root_val, "best_val_weights.csv")
    df = pd.read_csv(best_csv)
    row = df.iloc[0]

    best_dice = float(row["dice_score"])

    test_csv = os.path.join(eval_root_test, "pred-csv", "summary_metrics.csv")
    tdf = pd.read_csv(test_csv)
    final_test_dice = float(
        tdf.iloc[0]["Dice"]
        if "Dice" in tdf.columns else tdf.iloc[0]["dice_macro_avg"]
    )

    out = os.path.join("output", "summary.csv")
    write_header = not os.path.exists(out)

    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp",
                "iterations",
                "early_stop_iter",
                "best_dice",
                "best_checkpoint",
                "final_test_dice"
            ])
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            total_iters,
            early_stop_iter,
            f"{best_dice:.6f}",
            os.path.basename(best_ckpt),
            f"{final_test_dice:.6f}"
        ])

    append_run_log("XRUN", "Summary written")


def save_baseline_cfg():
    import csv
    from cfgs import (
        albumb_aug,
        complex_aug,
        image_only_aug,
        cutmix_augmnet,
        include_original_train,
        use_amp
    )

    path = "output/baseline_cfg.csv"
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "albumb_aug",
                "complex_aug",
                "image_only_aug",
                "cutmix_augmnet",
                "include_original_train",
                "use_amp"
                
            ])
            w.writerow([
                albumb_aug,
                complex_aug,
                image_only_aug,
                cutmix_augmnet,
                include_original_train,
                use_amp
            ])




# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int)
    args = parser.parse_args()

    clear_screen()
    append_run_log("XRUN", "XRUN started (GREEDY GLOBAL BEST MODE)")

    base_seed = seed_val

    # ---------------- Fresh / Resume ----------------
    if fresh_training == 1:
        # run_script("profile_model.py")
        delete_pycache_once()
        delete_output_once()
    else:
        if resume_train == 0:
            raise RuntimeError("Invalid config: fresh_training=0 AND resume_train=0")
        if not resumed_chkpt or not os.path.exists(resumed_chkpt):
            raise RuntimeError("resume_train=1 but resumed_chkpt invalid")

    actual_iters = 0
    early_stop_iter = None

    for t in range(1, tries + 1):
        actual_iters += 1
        
        if t == 1:
            save_baseline_cfg()

        seed = base_seed if random_seed == 0 else base_seed + t
        os.environ["XRUN_SEED"] = str(seed)
        append_run_log("XRUN", f"Iteration {t}/{tries} | seed={seed}")

        # ---------------- Resume ----------------
        if t == 1 and fresh_training == 0:
            os.environ["RESUME_CKPT_OVERRIDE"] = resumed_chkpt
        elif t > 1:
            best_ckpt = load_global_best_checkpoint()
            os.environ["RESUME_CKPT_OVERRIDE"] = best_ckpt
            append_run_log("XRUN", f"Resuming from GLOBAL BEST: {best_ckpt}")
        else:
            os.environ.pop("RESUME_CKPT_OVERRIDE", None)


        # ---------------- CONFIG SELECTION ----------------
        if random_train == 1:
            # Only change cfgs.py when NOT resuming
            if not (fresh_training == 0 and resume_train == 1):
                select_combination(t)
            else:
                append_run_log(
                    "CFG_SWITCH",
                    "Resume detected → using baseline cfgs.py"
                )
        else:
            append_run_log("CFG_SWITCH","random_train=0 → using cfgs.py as-is")
        
        # ---------------- CONFIG STATE (DEBUG / FORENSIC) ----------------
        append_run_log(
            "CFG_STATE",
            f"albumb_aug={albumb_aug}, "
            f"complex_aug={complex_aug}, "
            f"image_only_aug={image_only_aug}, "
            f"cutmix_augmnet={cutmix_augmnet}, "
            f"include_original_train={include_original_train}, "
            f"use_amp={use_amp}"
        )


        # ---------------- Train → Val → Check ----------------

        run_script("train.py")
        if fresh_training == 1:
            rewrite_cfgs_fresh_training_zero()

        run_script("val_eval.py")
        run_script("check_val.py")

        mark_best_checkpoint(t)

        run_script("stopearly.py")

        if should_stop_early():
            early_stop_iter = t
            prune_non_best_checkpoints()
            break

        prune_non_best_checkpoints()

    # ---------------- Final Test ----------------
    best_ckpt = load_global_best_checkpoint()
    os.environ["EVAL_ONLY_CKPT"] = best_ckpt
    run_script("test_eval.py")

    write_xrun_summary(actual_iters, early_stop_iter)
    rewrite_cfgs_fresh_training_one()

    append_run_log("XRUN", "XRUN completed successfully")


if __name__ == "__main__":
    main()



