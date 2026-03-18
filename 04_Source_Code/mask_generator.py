"""
Generate binary change masks from NDBI patches using Otsu thresholding.

For each year pair (e.g. 2018→2020):
  1. Load NDBI patch from both years
  2. Apply Otsu threshold → binary urban mask per year
  3. XOR the two masks → binary change mask
  4. Save as GeoTIFF preserving georeferencing
"""

import os
import numpy as np
import rasterio
from glob import glob
from tqdm import tqdm


def _otsu_threshold(values):
    """Compute Otsu's threshold for a 1-D array of finite values."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0

    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()

    w0 = np.cumsum(hist).astype(np.float64)
    w1 = total - w0
    sum0 = np.cumsum(hist * bin_centers)
    sum_total = (hist * bin_centers).sum()

    mean0 = np.divide(sum0, w0, out=np.zeros_like(sum0), where=w0 != 0)
    mean1 = np.divide(sum_total - sum0, w1, out=np.zeros_like(w1), where=w1 != 0)

    variance = w0 * w1 * (mean0 - mean1) ** 2
    idx = np.argmax(variance)
    return bin_centers[idx]


def _load_patch(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def _urban_mask(ndbi, threshold=None):
    valid = np.isfinite(ndbi) & (ndbi != 0)
    if valid.sum() < 100:
        return np.zeros_like(ndbi, dtype=np.uint8), 0.0

    if threshold is None:
        threshold = _otsu_threshold(ndbi[valid])

    mask = (ndbi > threshold).astype(np.uint8)
    mask[~valid] = 0
    return mask, threshold


def generate_change_masks(patches_dir, masks_dir, year_pairs,
                          min_valid=0.10, fixed_threshold=None):
    """
    Generate binary change masks for all year pairs.

    Parameters
    ----------
    patches_dir : str   – root of NDBI patches  (…/NDVI NDBI NDWI NaturalColor)
    masks_dir   : str   – where to save masks    (…/05_Results/change_masks)
    year_pairs  : list  – e.g. [(2018,2020), …]
    min_valid   : float – skip patches with < this fraction of valid pixels
    fixed_threshold : float or None – override Otsu with a fixed NDBI threshold

    Returns
    -------
    dict  {pair_key: {total, with_change, skipped}}
    """
    summary = {}

    for t1, t2 in year_pairs:
        pair = f"{t1}_{t2}"
        print(f"\n{'=' * 60}")
        print(f"  Change masks: {t1} → {t2}")
        print(f"{'=' * 60}")

        dir_t1 = os.path.join(patches_dir, str(t1), f"Jeddah_{t1}_NDBI_Raw")
        dir_t2 = os.path.join(patches_dir, str(t2), f"Jeddah_{t2}_NDBI_Raw")

        files_t1 = {os.path.basename(f): f for f in glob(os.path.join(dir_t1, "patch_*.tif"))}
        files_t2 = {os.path.basename(f): f for f in glob(os.path.join(dir_t2, "patch_*.tif"))}
        common = sorted(files_t1.keys() & files_t2.keys())
        print(f"  Common patches: {len(common)}")

        out_dir = os.path.join(masks_dir, pair)
        os.makedirs(out_dir, exist_ok=True)

        stats = {"total": 0, "with_change": 0, "skipped": 0}

        for fname in tqdm(common, desc=f"  {pair}", ncols=80):
            d1, profile = _load_patch(files_t1[fname])
            d2, _ = _load_patch(files_t2[fname])

            valid = np.isfinite(d1) & (d1 != 0) & np.isfinite(d2) & (d2 != 0)
            if valid.mean() < min_valid:
                stats["skipped"] += 1
                continue

            u1, _ = _urban_mask(d1, fixed_threshold)
            u2, _ = _urban_mask(d2, fixed_threshold)

            change = (u1 != u2).astype(np.uint8)
            change[~valid] = 0

            out_profile = profile.copy()
            out_profile.update(dtype="uint8", count=1, compress="lzw")
            with rasterio.open(os.path.join(out_dir, fname), "w", **out_profile) as dst:
                dst.write(change[np.newaxis])

            stats["total"] += 1
            if change.sum() > 0:
                stats["with_change"] += 1

        summary[pair] = stats
        print(f"  Saved {stats['total']} masks  "
              f"({stats['with_change']} contain change, "
              f"{stats['skipped']} skipped)")

    return summary
