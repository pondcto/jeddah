"""
PyTorch Dataset and DataLoader utilities for change detection.
"""

import os  # Import OS module for file path operations
import random  # Import random module for shuffling and augmentation randomness
import numpy as np  # Import NumPy for array manipulation and numerical operations
import rasterio  # Import rasterio for reading georeferenced raster (GeoTIFF) files
import torch  # Import PyTorch for tensor creation and conversion
from torch.utils.data import Dataset, DataLoader  # Import Dataset base class and DataLoader for batching
from glob import glob  # Import glob for file pattern matching


class ChangeDetectionDataset(Dataset):  # Define a custom PyTorch Dataset for loading change detection triplets
    """
    Loads (NDBI_t1, NDBI_t2, change_mask) triplets.

    All tensors are float32 with shape (1, 256, 256).
    NDBI values are clipped to [-1, 1]; change mask is binary {0, 1}.
    """

    def __init__(self, patches_dir, masks_dir, year_pairs, augment=False):  # Constructor: scans directories to build the list of valid samples
        self.augment = augment  # Store whether data augmentation should be applied during loading
        self.samples = []  # Initialize an empty list to hold file path tuples for each valid sample

        for t1, t2 in year_pairs:  # Iterate over each year pair (e.g., 2018-2020, 2020-2022, 2022-2024)
            pair = f"{t1}_{t2}"  # Create a string identifier for this year pair (e.g., "2018_2020")
            mask_dir = os.path.join(masks_dir, pair)  # Build the path to the change mask directory for this pair
            ndbi_t1 = os.path.join(patches_dir, str(t1), f"Jeddah_{t1}_NDBI_Raw")  # Build the path to NDBI patches for year t1
            ndbi_t2 = os.path.join(patches_dir, str(t2), f"Jeddah_{t2}_NDBI_Raw")  # Build the path to NDBI patches for year t2

            if not os.path.isdir(mask_dir):  # Check if the mask directory for this pair exists
                print(f"  [WARN] mask dir not found: {mask_dir}")  # Print a warning if the directory is missing
                continue  # Skip this year pair and continue with the next one

            for mp in sorted(glob(os.path.join(mask_dir, "patch_*.tif"))):  # Iterate over all change mask files in sorted order
                fname = os.path.basename(mp)  # Extract the filename (e.g., "patch_256_512.tif")
                p1 = os.path.join(ndbi_t1, fname)  # Build the full path to the corresponding NDBI patch for year t1
                p2 = os.path.join(ndbi_t2, fname)  # Build the full path to the corresponding NDBI patch for year t2
                if os.path.exists(p1) and os.path.exists(p2):  # Only add the sample if both NDBI patches exist on disk
                    self.samples.append((p1, p2, mp, pair, fname))  # Store the triplet paths along with metadata

        print(f"  Dataset: {len(self.samples)} samples from "  # Print the total number of valid samples discovered
              f"{len(year_pairs)} year pair(s)")  # Print how many year pairs were scanned

    def __len__(self):  # Return the total number of samples in the dataset
        return len(self.samples)  # The length equals the number of valid triplets found

    @staticmethod  # Static method: does not depend on instance state
    def _read(path):  # Read a single GeoTIFF patch and return it as a preprocessed NumPy array
        with rasterio.open(path) as src:  # Open the GeoTIFF file for reading
            arr = src.read(1).astype(np.float32)  # Read band 1 as a float32 array
        arr = np.nan_to_num(arr, nan=0.0)  # Replace any NaN values with 0.0 to avoid errors during training
        arr = np.clip(arr, -1.0, 1.0)  # Clip NDBI values to the valid range [-1, 1]
        return arr[np.newaxis]  # Add a channel dimension, resulting in shape (1, H, W)

    def _apply_aug(self, t1, t2, m):  # Apply random spatial augmentations consistently to all three arrays
        if random.random() > 0.5:  # With 50% probability, apply a horizontal flip
            t1 = np.flip(t1, axis=2).copy()  # Flip the t1 image along the width axis and copy to make contiguous
            t2 = np.flip(t2, axis=2).copy()  # Flip the t2 image along the width axis
            m = np.flip(m, axis=2).copy()  # Flip the mask along the width axis to keep alignment
        if random.random() > 0.5:  # With 50% probability, apply a vertical flip
            t1 = np.flip(t1, axis=1).copy()  # Flip the t1 image along the height axis
            t2 = np.flip(t2, axis=1).copy()  # Flip the t2 image along the height axis
            m = np.flip(m, axis=1).copy()  # Flip the mask along the height axis
        k = random.randint(0, 3)  # Randomly choose a rotation: 0, 90, 180, or 270 degrees
        if k:  # If k is non-zero, apply the rotation
            t1 = np.rot90(t1, k, axes=(1, 2)).copy()  # Rotate t1 by k*90 degrees in the spatial plane
            t2 = np.rot90(t2, k, axes=(1, 2)).copy()  # Rotate t2 by the same angle
            m = np.rot90(m, k, axes=(1, 2)).copy()  # Rotate the mask by the same angle
        return t1, t2, m  # Return the augmented triplet

    def __getitem__(self, idx):  # Retrieve a single sample by index for the DataLoader
        p1, p2, mp, _, _ = self.samples[idx]  # Unpack the file paths for NDBI t1, NDBI t2, and the change mask
        t1 = self._read(p1)  # Read and preprocess the NDBI patch for year t1
        t2 = self._read(p2)  # Read and preprocess the NDBI patch for year t2
        with rasterio.open(mp) as src:  # Open the change mask GeoTIFF file
            mask = src.read(1).astype(np.float32)[np.newaxis]  # Read band 1 as float32 and add a channel dimension

        if self.augment:  # If augmentation is enabled (training mode)
            t1, t2, mask = self._apply_aug(t1, t2, mask)  # Apply random spatial augmentations to the triplet

        return torch.from_numpy(t1), torch.from_numpy(t2), torch.from_numpy(mask)  # Convert all arrays to PyTorch tensors and return


class _Subset(Dataset):  # A thin wrapper Dataset that provides a subset of the full dataset with optional augmentation
    """Thin wrapper that applies optional augmentation to a subset."""

    def __init__(self, base, indices, augment=False):  # Constructor takes the base dataset, index list, and augmentation flag
        self.base = base  # Store a reference to the full base dataset
        self.indices = indices  # Store the list of indices that belong to this subset (e.g., train indices)
        self.augment = augment  # Store whether augmentation should be applied for this subset

    def __len__(self):  # Return the number of samples in this subset
        return len(self.indices)  # Length equals the number of indices in the subset

    def __getitem__(self, idx):  # Retrieve a sample from the subset by local index
        orig_aug = self.base.augment  # Save the base dataset's current augmentation state
        self.base.augment = self.augment  # Temporarily set the base dataset's augmentation to this subset's setting
        item = self.base[self.indices[idx]]  # Fetch the sample from the base dataset using the mapped global index
        self.base.augment = orig_aug  # Restore the base dataset's original augmentation state
        return item  # Return the fetched sample (t1, t2, mask) tensors


def create_dataloaders(patches_dir, masks_dir, year_pairs,  # Factory function to create train, validation, and test DataLoaders
                       batch_size=16, train_ratio=0.70, val_ratio=0.15,
                       num_workers=2, seed=42):
    """Return (train_loader, val_loader, test_loader)."""
    full = ChangeDetectionDataset(patches_dir, masks_dir, year_pairs, augment=False)  # Create the full dataset without augmentation to scan all samples
    n = len(full)  # Get the total number of samples in the full dataset

    indices = list(range(n))  # Create a list of sequential indices [0, 1, 2, ..., n-1]
    random.seed(seed)  # Set the random seed for reproducible shuffling and splitting
    random.shuffle(indices)  # Randomly shuffle the indices to randomize the train/val/test assignment

    n_train = int(n * train_ratio)  # Calculate the number of training samples (70% of total)
    n_val = int(n * val_ratio)  # Calculate the number of validation samples (15% of total)

    train_idx = indices[:n_train]  # Slice the first 70% of shuffled indices for the training set
    val_idx = indices[n_train:n_train + n_val]  # Slice the next 15% for the validation set
    test_idx = indices[n_train + n_val:]  # Slice the remaining 15% for the test set

    train_ds = _Subset(full, train_idx, augment=True)  # Create the training subset with augmentation enabled
    val_ds = _Subset(full, val_idx, augment=False)  # Create the validation subset without augmentation
    test_ds = _Subset(full, test_idx, augment=False)  # Create the test subset without augmentation

    kw = dict(pin_memory=True, num_workers=num_workers)  # Common DataLoader keyword arguments: pin memory for faster GPU transfer, set worker count
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)  # Create the training DataLoader with shuffling enabled
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)  # Create the validation DataLoader without shuffling
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)  # Create the test DataLoader without shuffling

    print(f"  Splits -> train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")  # Print the number of samples in each split
    return train_loader, val_loader, test_loader  # Return the three DataLoaders
