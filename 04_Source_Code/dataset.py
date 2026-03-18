"""
PyTorch Dataset and DataLoader utilities for change detection.
"""

import os
import random
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob


class ChangeDetectionDataset(Dataset):
    """
    Loads (NDBI_t1, NDBI_t2, change_mask) triplets.

    All tensors are float32 with shape (1, 256, 256).
    NDBI values are clipped to [-1, 1]; change mask is binary {0, 1}.
    """

    def __init__(self, patches_dir, masks_dir, year_pairs, augment=False):
        self.augment = augment
        self.samples = []

        for t1, t2 in year_pairs:
            pair = f"{t1}_{t2}"
            mask_dir = os.path.join(masks_dir, pair)
            ndbi_t1 = os.path.join(patches_dir, str(t1), f"Jeddah_{t1}_NDBI_Raw")
            ndbi_t2 = os.path.join(patches_dir, str(t2), f"Jeddah_{t2}_NDBI_Raw")

            if not os.path.isdir(mask_dir):
                print(f"  [WARN] mask dir not found: {mask_dir}")
                continue

            for mp in sorted(glob(os.path.join(mask_dir, "patch_*.tif"))):
                fname = os.path.basename(mp)
                p1 = os.path.join(ndbi_t1, fname)
                p2 = os.path.join(ndbi_t2, fname)
                if os.path.exists(p1) and os.path.exists(p2):
                    self.samples.append((p1, p2, mp, pair, fname))

        print(f"  Dataset: {len(self.samples)} samples from "
              f"{len(year_pairs)} year pair(s)")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.clip(arr, -1.0, 1.0)
        return arr[np.newaxis]  # (1, H, W)

    def _apply_aug(self, t1, t2, m):
        if random.random() > 0.5:
            t1 = np.flip(t1, axis=2).copy()
            t2 = np.flip(t2, axis=2).copy()
            m = np.flip(m, axis=2).copy()
        if random.random() > 0.5:
            t1 = np.flip(t1, axis=1).copy()
            t2 = np.flip(t2, axis=1).copy()
            m = np.flip(m, axis=1).copy()
        k = random.randint(0, 3)
        if k:
            t1 = np.rot90(t1, k, axes=(1, 2)).copy()
            t2 = np.rot90(t2, k, axes=(1, 2)).copy()
            m = np.rot90(m, k, axes=(1, 2)).copy()
        return t1, t2, m

    def __getitem__(self, idx):
        p1, p2, mp, _, _ = self.samples[idx]
        t1 = self._read(p1)
        t2 = self._read(p2)
        with rasterio.open(mp) as src:
            mask = src.read(1).astype(np.float32)[np.newaxis]

        if self.augment:
            t1, t2, mask = self._apply_aug(t1, t2, mask)

        return torch.from_numpy(t1), torch.from_numpy(t2), torch.from_numpy(mask)


class _Subset(Dataset):
    """Thin wrapper that applies optional augmentation to a subset."""

    def __init__(self, base, indices, augment=False):
        self.base = base
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_aug = self.base.augment
        self.base.augment = self.augment
        item = self.base[self.indices[idx]]
        self.base.augment = orig_aug
        return item


def create_dataloaders(patches_dir, masks_dir, year_pairs,
                       batch_size=16, train_ratio=0.70, val_ratio=0.15,
                       num_workers=2, seed=42):
    """Return (train_loader, val_loader, test_loader)."""
    full = ChangeDetectionDataset(patches_dir, masks_dir, year_pairs, augment=False)
    n = len(full)

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = _Subset(full, train_idx, augment=True)
    val_ds = _Subset(full, val_idx, augment=False)
    test_ds = _Subset(full, test_idx, augment=False)

    kw = dict(pin_memory=True, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    print(f"  Splits → train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")
    return train_loader, val_loader, test_loader
