gpu_list = "0"

random_train = 1
random_seed = 1
seed_val = 55


fresh_training = 0
resume_train = 0
resumed_chkpt = ""


# =============================
# TRAINING HYPERPARAMETERS
# =============================
batch_size = 24
num_epochs = 8
tries = 16
val_eval_threshold = 4


use_amp = False

num_Classes = 2


if num_Classes > 2:
    loss_name = "MulticlassBCEWithLogitsLoss"
else:
    loss_name = "BCEWithLogitsLoss"


edge_loss_name = "EdgeBCELoss"


# =============================
# IMAGE SIZE
# =============================
H = 256
W = 256
size = (H, W)



# =============================
# MODEL CONFIG
# =============================
encoder_name_str = "pvt_v2_b2"



# =============================
# AUGMENTATIONS
# =============================
train_augmentation_online = True
albumb_aug = True
complex_aug = True
image_only_aug = True
cutmix_augmnet = True
include_original_train = True


# =============================
# DEEP SUPERVISION
# =============================
deep_supervision = True
deep_supervision_weights = [1.0, 0.4, 0.3, 0.2]
normalize_deep_weights = True


# =============================
# EDGE SUPERVISION
# =============================
edge_supervision = True
edge_weights = [1.0, 1.0, 1.0]
normalize_edge_weights = True


# =============================
# OUTPUT PATHS (TRAINING)
# =============================

checkpoint_dir = "output/checkpoints/"
sanity_check_logs = "output/sanity-check/logs/"
train_samples = "output/sanity-check/training-samples"
train_logs = "output/train-logs/"
train_log_loss_file = "output/train-logs/train-loss.csv"


# =============================
# EVAL ROOTS (SINGLE SOURCE OF TRUTH)
# =============================

# Validation evaluation
eval_root_val = "output/val_eval_results"

# Test evaluation (used by run.py, test_eval.py, xrun.py)
eval_root_test = "output/test_eval_results"


# =============================
# DATASET ROOT
# =============================
main_dataset_dir = "../0-datasets/eye-datasets/DRIVE-2004-patches-256-128/"
# =============================
# DATASET SPLITS
# =============================

# ---- TRAIN ----
train_images_dir = "train/images/*"
train_masks_dir = "train/masks/*"

# ---- VALIDATION ----
val_images_dir = "val/images/*"
val_masks_dir = "val/masks/*"

# ---- TEST ----
test_images_dir = "test/images/*"
test_masks_dir = "test/masks/*"


# =============================
# LR SCHEDULER CONFIG
# =============================

use_lr_schedule = 1
eta_min = 1e-6
lr = 1e-4

def get_cosine_T0(num_epochs):
    """
    Rules:
    1 epoch   -> T0 = 3
    5 epochs  -> T0 = 5
    10 epochs -> T0 = 10
    >10       -> T0 = 20 (cap)
    """
    if num_epochs <= 1:
        return 2
    elif num_epochs <= 5:
        return 3
    elif num_epochs <= 10:
        return 5
    else:
        return 5


cosine_T0 = get_cosine_T0(num_epochs)
cosine_T_mult = 1
