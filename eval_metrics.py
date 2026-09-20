import cupy as cp
import torch
from cupyx.scipy.ndimage import distance_transform_edt, binary_erosion
from cfgs import num_Classes

# ==========================
# Helpers
# ==========================
def threshold_func(y_pred, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    return (y_pred > threshold).float()

def to_cupy(y_tensor):
    return cp.asarray(y_tensor.detach().cpu().numpy().astype(cp.uint8))

def dice_score(pred, gt):
    intersection = cp.sum(pred * gt)
    union = cp.sum(pred) + cp.sum(gt)
    return (2.0 * intersection + 1e-7) / (union + 1e-7)

def iou_score(pred, gt):
    intersection = cp.sum(pred * gt)
    union = cp.sum((pred + gt) > 0)
    return (intersection + 1e-7) / (union + 1e-7)

def precision_score(pred, gt):
    tp = cp.sum(pred * gt)
    fp = cp.sum(pred * (1 - gt))
    return (tp + 1e-7) / (tp + fp + 1e-7)

def recall_score(pred, gt):
    tp = cp.sum(pred * gt)
    fn = cp.sum((1 - pred) * gt)
    return (tp + 1e-7) / (tp + fn + 1e-7)

def hd95_score(pred, gt):
    pred = pred.astype(cp.bool_)
    gt = gt.astype(cp.bool_)

    if cp.sum(pred) == 0 and cp.sum(gt) == 0:
        return 0.0
    if cp.sum(pred) == 0 or cp.sum(gt) == 0:
        return 95.0  # max error

    dt_pred = distance_transform_edt(~pred)
    dt_gt = distance_transform_edt(~gt)

    surface_gt = cp.logical_xor(gt, binary_erosion(gt))
    surface_pred = cp.logical_xor(pred, binary_erosion(pred))

    distances_pred_to_gt = dt_gt[surface_pred]
    distances_gt_to_pred = dt_pred[surface_gt]

    all_distances = cp.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    return float(cp.percentile(all_distances, 95))

# ==========================
# Binary Metrics
# ==========================
def compute_binary_metrics(y_pred, y_true, threshold=0.5, selected_metrics=None):
    if selected_metrics is None:
        selected_metrics = ['dice', 'precision', 'recall', 'hd95', 'iou']

    y_pred = threshold_func(y_pred, threshold)
    y_true = y_true.float()
    results = {m: [] for m in selected_metrics}

    for i in range(y_pred.size(0)):
        pred = to_cupy(y_pred[i, 0])
        gt = to_cupy(y_true[i, 0])

        if 'dice' in selected_metrics:
            results['dice'].append(float(dice_score(pred, gt)))
        if 'iou' in selected_metrics:
            results['iou'].append(float(iou_score(pred, gt)))
        if 'hd95' in selected_metrics:
            results['hd95'].append(float(hd95_score(pred, gt)))
        if 'precision' in selected_metrics:
            results['precision'].append(float(precision_score(pred, gt)))
        if 'recall' in selected_metrics:
            results['recall'].append(float(recall_score(pred, gt)))

    return {k: float(cp.mean(cp.asarray(v))) for k, v in results.items()}

# ==========================
# Multiclass Metrics
# ==========================
def compute_multiclass_metrics(y_pred, y_true, selected_metrics=None):
    if selected_metrics is None:
        selected_metrics = ['dice', 'precision', 'recall', 'hd95', 'iou']

    # Softmax + argmax for class prediction
    y_pred_soft = torch.softmax(y_pred, dim=1)
    y_pred_labels = torch.argmax(y_pred_soft, dim=1)  # [B, H, W]
    y_true_labels = y_true.long()  # [B, H, W]

    results = {m: {cls: [] for cls in range(num_Classes)} for m in selected_metrics}

    for i in range(y_pred_labels.size(0)):
        for cls in range(num_Classes):
            pred = to_cupy((y_pred_labels[i] == cls).int())
            gt = to_cupy((y_true_labels[i] == cls).int())

            if 'dice' in selected_metrics:
                results['dice'][cls].append(float(dice_score(pred, gt)))
            if 'iou' in selected_metrics:
                results['iou'][cls].append(float(iou_score(pred, gt)))
            if 'hd95' in selected_metrics:
                results['hd95'][cls].append(float(hd95_score(pred, gt)))
            if 'precision' in selected_metrics:
                results['precision'][cls].append(float(precision_score(pred, gt)))
            if 'recall' in selected_metrics:
                results['recall'][cls].append(float(recall_score(pred, gt)))

    # Aggregate class-wise means
    final_results = {}
    for metric in selected_metrics:
        class_means = {cls: float(cp.mean(cp.asarray(scores))) for cls, scores in results[metric].items()}
        valid_classes = [cls for cls in class_means.keys() if cls != 0]
        macro_avg = float(cp.mean(cp.asarray([class_means[cls] for cls in valid_classes])))
        final_results[metric] = {"class_wise": class_means, "macro_avg": macro_avg}

    return final_results

# ==========================
# Unified Metrics Function
# ==========================
def compute_all_metrics(y_pred, y_true, threshold=0.5, selected_metrics=None):
    if num_Classes > 2:
        return compute_multiclass_metrics(y_pred, y_true, selected_metrics)
    else:
        return compute_binary_metrics(y_pred, y_true, threshold, selected_metrics)
