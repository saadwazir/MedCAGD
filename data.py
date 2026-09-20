import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from cfgs import *


def do_cutmix(image, mask, image2, mask2):
    _, H, W = image.shape
    lam = np.random.beta(1.0, 1.0)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]

    if mask.ndim == 3 and mask2.ndim == 3:
        mask[:, y1:y2, x1:x2] = mask2[:, y1:y2, x1:x2]
    else:
        mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

    return image, mask


class DriveDataset(Dataset):
    def __init__(
        self,
        images_path,
        masks_path,
        augment=True,
        include_original=True
    ):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.augment = augment
        self.include_original = include_original

        self.height, self.width = size  # from cfgs.py

        # --------------------------------------------------
        # Image-only augmentations (UPDATED)
        # --------------------------------------------------
        self.image_only_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.MultiplicativeNoise(p=0.2),
            A.GaussianBlur(p=0.2),
        ])

        # --------------------------------------------------
        # Joint (image + mask) augmentations
        # --------------------------------------------------
        
        if complex_aug:
            self.joint_transform = A.Compose([
                A.Rotate(limit=20, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.RandomRotate90(p=0.2),
                A.RandomCrop(self.height, self.width, p=0.3),
            ])
        else:
            self.joint_transform = A.Compose([
                A.Rotate(limit=20, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])

    def __len__(self):
        return self.n_samples * 2 if self.include_original else self.n_samples

    def __getitem__(self, index):
        real_index = index % self.n_samples

        image = cv2.imread(self.images_path[real_index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[real_index], cv2.IMREAD_GRAYSCALE)

        # --------------------------------------------------
        # Decide whether THIS index is an augmented copy
        # --------------------------------------------------
        is_augmented_copy = (
            self.include_original and index >= self.n_samples
        )

        # --------------------------------------------------
        # Albumentations (ONLY when allowed)
        # --------------------------------------------------
        if (
            self.augment
            and albumb_aug
            and (not self.include_original or is_augmented_copy)
        ):
            augmented = self.joint_transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            
            if image_only_aug:
                image = self.image_only_transform(image=image)["image"]

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------
        image = cv2.resize(image, (self.width, self.height))
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # --------------------------------------------------
        # Normalize image
        # --------------------------------------------------
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.from_numpy(image)

        if num_Classes > 2:
            mask = torch.from_numpy(mask.astype(np.int64))
        else:
            mask = mask / 255.0
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)
            mask = torch.from_numpy(mask)

        # --------------------------------------------------
        # CutMix (ONLY when allowed)
        # --------------------------------------------------
        if (
            self.augment
            and cutmix_augmnet
            and (not self.include_original or is_augmented_copy)
        ):
            if np.random.rand() < 0.6:
                idx2 = np.random.randint(0, self.n_samples)

                image2 = cv2.imread(self.images_path[idx2], cv2.IMREAD_COLOR)
                mask2 = cv2.imread(self.masks_path[idx2], cv2.IMREAD_GRAYSCALE)

                image2 = cv2.resize(image2, (self.width, self.height))
                mask2 = cv2.resize(mask2, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

                image2 = image2 / 255.0
                image2 = np.transpose(image2, (2, 0, 1)).astype(np.float32)
                image2 = torch.from_numpy(image2)

                if num_Classes > 2:
                    mask2 = torch.from_numpy(mask2.astype(np.int64))
                else:
                    mask2 = mask2 / 255.0
                    mask2 = (mask2 > 0).astype(np.float32)
                    mask2 = np.expand_dims(mask2, axis=0)
                    mask2 = torch.from_numpy(mask2)

                image, mask = do_cutmix(image, mask, image2, mask2)

        return image, mask
