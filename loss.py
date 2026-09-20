import torch
import torch.nn as nn
import torch.nn.functional as F
from cfgs import *

from segmentation_models_pytorch.losses import LovaszLoss, MCCLoss

# ============================================================
# Core Segmentation Losses
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, smooth=1.0):
        # BCE (logits-safe, AMP-safe)
        bce = self.bce_logits(logits, targets)

        # Dice (apply sigmoid ONLY for dice)
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (
            probs.sum() + targets.sum() + smooth
        )

        return self.bce_weight * bce + self.dice_weight * dice_loss



class LovaszWrapper(nn.Module):
    def __init__(self):
        super(LovaszWrapper, self).__init__()
        self.loss = LovaszLoss(mode='binary', from_logits=False)

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return self.loss(inputs, targets)


class MCCWrapper(nn.Module):
    def __init__(self, eps=1e-5):
        super(MCCWrapper, self).__init__()
        self.loss = MCCLoss(eps=eps)

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return self.loss(inputs, targets)


class MulticlassBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        super(MulticlassBCEWithLogitsLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=-100,
            reduction='mean'
        )

    def forward(self, inputs, targets):
        return self.ce_loss(inputs, targets.long())


class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(MulticlassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        inputs = torch.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(inputs * targets_onehot, dims)
        cardinality = torch.sum(inputs + targets_onehot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss


class MulticlassBCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super(MulticlassBCEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.dice_loss = MulticlassDiceLoss()

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets.long())
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice



# ============================================================
# Calibration Losses
# ============================================================
class DECE(nn.Module):
    def __init__(self, n_bins=15, temperature=1.0):
        super(DECE, self).__init__()
        self.n_bins = n_bins
        self.temperature = temperature

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)
        probs = torch.sigmoid(logits / self.temperature)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        dece = torch.tensor(0.0, device=logits.device)
        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (probs > lower) & (probs <= upper)
            if mask.sum() > 0:
                avg_conf = probs[mask].mean()
                avg_acc = targets[mask].float().mean()
                dece += torch.abs(avg_conf - avg_acc) * mask.float().mean()
        return dece


class mL1ACE(nn.Module):
    def __init__(self, n_bins=15):
        super(mL1ACE, self).__init__()
        self.n_bins = n_bins

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)
        probs = torch.sigmoid(logits)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        ace = torch.tensor(0.0, device=logits.device)
        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (probs > lower) & (probs <= upper)
            if mask.sum() > 0:
                avg_conf = probs[mask].mean()
                avg_acc = targets[mask].float().mean()
                ace += torch.abs(avg_conf - avg_acc)
        return ace / self.n_bins


class CombinedLoss(nn.Module):
    def __init__(self, calibration_loss_fn, alpha=0.1):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.calibration_loss_fn = calibration_loss_fn
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        calibration = self.calibration_loss_fn(logits, targets)
        return bce + self.alpha * calibration





# ============================================================
# Focal Loss Variants
# ============================================================

class BinaryFocalLoss(nn.Module):
    """
    Standard Focal Loss for binary segmentation (logits-safe).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        # BCE-with-logits (AMP safe)
        bce_loss = self.bce_logits(logits, targets)

        # Compute pt
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()



class MulticlassFocalLoss(nn.Module):
    """
    Standard Focal Loss for multiclass segmentation.
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)

        # One-hot GT
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2).float()

        # pt = prob of true class
        pt = (probs * targets_onehot).sum(dim=1)

        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-7)
        return loss.mean()


class BinaryFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (binary).
    Best for tiny objects & Dice/IoU optimization.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        tversky = (TP + 1e-7) / (TP + self.alpha * FP + self.beta * FN + 1e-7)
        return (1 - tversky) ** self.gamma


class MulticlassFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (multiclass).
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)

        # One-hot encode GT
        targets_onehot = F.one_hot(
            targets, num_classes=num_classes
        ).permute(0,3,1,2).float()

        probs_flat = probs.reshape(num_classes, -1)
        targets_flat = targets_onehot.reshape(num_classes, -1)

        TP = (probs_flat * targets_flat).sum(dim=1)
        FP = ((1 - targets_flat) * probs_flat).sum(dim=1)
        FN = (targets_flat * (1 - probs_flat)).sum(dim=1)

        tversky = (TP + 1e-7) / (TP + self.alpha * FP + self.beta * FN + 1e-7)
        loss = (1 - tversky) ** self.gamma
        return loss.mean()



class BinaryAsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss.
    Boosts recall or precision via delta.
    AMP-safe using BCEWithLogitsLoss.
    """
    def __init__(self, delta=0.7, gamma=2.0):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        # BCE-with-logits
        ce = self.bce_logits(logits, targets)

        # Compute pt
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Asymmetric weighting
        asymmetric_weight = torch.where(
            targets == 1,
            (1 - pt) ** self.gamma,
            pt ** self.gamma * (1 - self.delta)
        )

        return (asymmetric_weight * ce).mean()


class MulticlassAsymmetricFocalLoss(nn.Module):
    """
    Multiclass Asymmetric Focal Loss.
    Biases FP vs FN for semantic segmentation.
    """
    def __init__(self, delta=0.7, gamma=2.0):
        super().__init__()
        self.delta = delta
        self.gamma = gamma

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)

        targets_onehot = F.one_hot(
            targets, num_classes=num_classes
        ).permute(0,3,1,2).float()

        pt = (probs * targets_onehot).sum(dim=1)

        pos_weight = (1 - pt) ** self.gamma
        neg_weight = pt ** self.gamma * (1 - self.delta)

        weights = torch.where(targets != 0, pos_weight, neg_weight)

        ce = -(targets_onehot * torch.log(probs + 1e-7)).sum(dim=1)
        return (weights * ce).mean()


































# ============================================================
# Edge Loss Utilities
# ============================================================
def generate_edge_targets(masks, radius=3):
    """
    Generate thick edge maps using distance transform approximation.
    Args:
        masks: [B, H, W] integer masks (class labels)
        radius: thickness of edge band
    Returns:
        edges: [B, 1, H, W] binary edge maps (edge band)
    """
    if masks.dim() == 4:
        masks = masks.squeeze(1)

    masks = masks.float().unsqueeze(1)  # [B,1,H,W]

    kernel = torch.ones((1,1,2*radius+1,2*radius+1), device=masks.device)

    # Dilation: expand object by radius
    dilated = F.conv2d(masks, kernel, padding=radius) > 0

    # Erosion: shrink object by radius
    eroded = F.conv2d(masks, kernel, padding=radius) == kernel.numel()

    # Edge band = dilated minus eroded region
    edges = (dilated & ~eroded).float()
    return edges



class EdgeBCELoss(nn.Module):
    def __init__(self):
        super(EdgeBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        edge_targets = generate_edge_targets(targets)
        return self.bce(inputs, edge_targets)


class EdgeDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(EdgeDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        edge_targets = generate_edge_targets(targets)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        edge_targets = edge_targets.view(-1)
        intersection = (inputs * edge_targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + edge_targets.sum() + self.smooth)
        return 1 - dice


class EdgeBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(EdgeBCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = EdgeDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce = self.bce(inputs, generate_edge_targets(targets))
        dice = self.dice(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================
# Loss Selectors
# ============================================================
def get_loss_fn():
    """Main segmentation loss selector."""
    if loss_name == "DiceBCELoss":
        return DiceBCELoss(dice_weight=0.7, bce_weight=0.3)
    elif loss_name == "DiceLoss":
        return DiceLoss()
    elif loss_name == "LovaszWrapper":
        return LovaszWrapper()
    elif loss_name == "MCCWrapper":
        return MCCWrapper()
    elif loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "MulticlassBCEWithLogitsLoss":
        return MulticlassBCEWithLogitsLoss()
    elif loss_name == "dece_bce":
        return CombinedLoss(DECE(n_bins=15), alpha=0.5)
    elif loss_name == "mL1ACE_bce":
        return CombinedLoss(mL1ACE(n_bins=15), alpha=0.5)
    elif loss_name == "MulticlassBCEDiceLoss":
        return MulticlassBCEDiceLoss(dice_weight=0.7, ce_weight=0.3)
    elif loss_name == "BinaryFocalLoss":
        return BinaryFocalLoss()
    elif loss_name == "MulticlassFocalLoss":
        return MulticlassFocalLoss()
    elif loss_name == "BinaryFocalTverskyLoss":
        return BinaryFocalTverskyLoss()
    elif loss_name == "MulticlassFocalTverskyLoss":
        return MulticlassFocalTverskyLoss()
    elif loss_name == "BinaryAsymmetricFocalLoss":
        return BinaryAsymmetricFocalLoss()
    elif loss_name == "MulticlassAsymmetricFocalLoss":
        return MulticlassAsymmetricFocalLoss()
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")


def get_edge_loss_fn():
    """Edge supervision loss selector."""
    if edge_loss_name == "EdgeBCELoss":
        return EdgeBCELoss()
    elif edge_loss_name == "EdgeDiceLoss":
        return EdgeDiceLoss()
    elif edge_loss_name == "EdgeBCEDiceLoss":
        return EdgeBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    else:
        raise ValueError(f"Unknown edge_loss_name: {edge_loss_name}")
