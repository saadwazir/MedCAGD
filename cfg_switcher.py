# cfg_switcher.py
import csv
from utils import append_run_log

CFG_FILE = "cfgs.py"
COMB_FILE = "train_random_comb.csv"

TARGET_KEYS = [
    "albumb_aug",
    "complex_aug",
    "image_only_aug",
    "cutmix_augmnet",
    "include_original_train",
    "use_amp"
]

def load_combinations():
    with open(COMB_FILE) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({k: r[k] == "True" for k in TARGET_KEYS})
    return rows


def rewrite_cfgs_with_combination(combo: dict):
    with open(CFG_FILE) as f:
        lines = f.readlines()

    out = []
    for line in lines:
        stripped = line.strip()
        replaced = False
        for k, v in combo.items():
            if stripped.startswith(k + " ="):
                out.append(f"{k} = {v}\n")
                replaced = True
                break
        if not replaced:
            out.append(line)

    with open(CFG_FILE, "w") as f:
        f.writelines(out)

    append_run_log("CFG_SWITCH", f"Applied combination: {combo}")
