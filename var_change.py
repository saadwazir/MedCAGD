# var_change.py
# ----------------------------------------
# Replaces test dataset paths with val paths
# in cfgs.py (same directory).
# Only exact lines are modified.
# ----------------------------------------

import os
import sys

CFG_FILE = "cfgs.py"

OLD_LINES = {
    'test_images_dir = "test/images/*"':
        'test_images_dir = "val/images/*"',
    'test_masks_dir = "test/masks/*"':
        'test_masks_dir = "val/masks/*"',
}


def main():
    here = os.getcwd()
    cfg_path = os.path.join(here, CFG_FILE)

    if not os.path.isfile(cfg_path):
        print("ERROR: cfgs.py not found in current directory.")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    replaced = {k: False for k in OLD_LINES}

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in OLD_LINES:
            new_line = line.replace(stripped, OLD_LINES[stripped])
            new_lines.append(new_line)
            replaced[stripped] = True
        else:
            new_lines.append(line)

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Report status
    for k, v in replaced.items():
        if not v:
            print(f"WARNING: Line not found -> {k}")

    print("Done. cfgs.py updated safely.")


if __name__ == "__main__":
    main()
