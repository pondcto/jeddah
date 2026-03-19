"""
Generate binary change masks from NDBI patches using Otsu thresholding.

For each year pair (e.g. 2018→2020):
  1. Load NDBI patch from both years
  2. Apply Otsu threshold → binary urban mask per year
  3. XOR the two masks → binary change mask
  4. Save as GeoTIFF preserving georeferencing
"""

import os  # Import OS module for directory creation and path manipulation
import numpy as np  # Import NumPy for numerical array operations
import rasterio  # Import rasterio for reading and writing georeferenced raster (GeoTIFF) files
from glob import glob  # Import glob to find files matching a wildcard pattern
from tqdm import tqdm  # Import tqdm to display progress bars during long loops


def _otsu_threshold(values):  # Compute the optimal Otsu threshold that separates a 1-D array into two classes
    """Compute Otsu's threshold for a 1-D array of finite values."""
    values = values[np.isfinite(values)]  # Remove NaN and Inf values, keeping only finite numeric entries
    if len(values) < 2:  # If fewer than 2 valid values exist, thresholding is impossible
        return 0.0  # Return a default threshold of zero

    hist, bin_edges = np.histogram(values, bins=256)  # Compute a 256-bin histogram of the pixel intensity distribution
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the center value of each histogram bin
    total = hist.sum()  # Total number of pixels across all bins

    w0 = np.cumsum(hist).astype(np.float64)  # Cumulative sum of pixel counts for class 0 (background) at each threshold
    w1 = total - w0  # Pixel count for class 1 (foreground) is the complement of class 0
    sum0 = np.cumsum(hist * bin_centers)  # Cumulative weighted sum of intensities for class 0
    sum_total = (hist * bin_centers).sum()  # Total weighted sum of all pixel intensities

    mean0 = np.divide(sum0, w0, out=np.zeros_like(sum0), where=w0 != 0)  # Mean intensity of class 0; safe division avoids divide-by-zero
    mean1 = np.divide(sum_total - sum0, w1, out=np.zeros_like(w1), where=w1 != 0)  # Mean intensity of class 1; safe division avoids divide-by-zero

    variance = w0 * w1 * (mean0 - mean1) ** 2  # Inter-class variance at each candidate threshold (Otsu's criterion)
    idx = np.argmax(variance)  # Find the bin index that maximizes inter-class variance
    return bin_centers[idx]  # Return the intensity value at the optimal threshold bin


def _load_patch(path):  # Load a single GeoTIFF patch from disk and return its pixel data and metadata
    with rasterio.open(path) as src:  # Open the GeoTIFF file using rasterio
        data = src.read(1).astype(np.float32)  # Read band 1 (the only band) as a float32 NumPy array
        profile = src.profile.copy()  # Copy the raster metadata (CRS, transform, dimensions, dtype, etc.)
    return data, profile  # Return the pixel data array and the metadata profile


def _urban_mask(ndbi, threshold=None):  # Classify NDBI pixels into urban (1) and non-urban (0) using a threshold
    valid = np.isfinite(ndbi) & (ndbi != 0)  # Create a boolean mask identifying pixels that are finite and non-zero
    if valid.sum() < 100:  # If fewer than 100 valid pixels exist, the patch is too sparse to classify
        return np.zeros_like(ndbi, dtype=np.uint8), 0.0  # Return an all-zero mask and a threshold of 0.0

    if threshold is None:  # If no fixed threshold is provided, compute one automatically
        threshold = _otsu_threshold(ndbi[valid])  # Apply Otsu's method on the valid pixels to determine the optimal threshold

    mask = (ndbi > threshold).astype(np.uint8)  # Pixels with NDBI above the threshold are classified as urban (1)
    mask[~valid] = 0  # Set invalid pixel locations to 0 (non-urban) regardless of their value
    return mask, threshold  # Return the binary urban mask and the threshold value used


def generate_change_masks(patches_dir, masks_dir, year_pairs,  # Main function to generate binary change masks for all year pairs
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
    summary = {}  # Initialize an empty dictionary to store processing statistics per year pair

    for t1, t2 in year_pairs:  # Iterate over each consecutive year pair (e.g., 2018-2020)
        pair = f"{t1}_{t2}"  # Create a string key for this year pair (e.g., "2018_2020")
        print(f"\n{'=' * 60}")  # Print a visual separator line
        print(f"  Change masks: {t1} → {t2}")  # Print which year pair is currently being processed
        print(f"{'=' * 60}")  # Print a closing separator line

        dir_t1 = os.path.join(patches_dir, str(t1), f"Jeddah_{t1}_NDBI_Raw")  # Build the path to NDBI patches for year t1
        dir_t2 = os.path.join(patches_dir, str(t2), f"Jeddah_{t2}_NDBI_Raw")  # Build the path to NDBI patches for year t2

        files_t1 = {os.path.basename(f): f for f in glob(os.path.join(dir_t1, "patch_*.tif"))}  # Map filenames to full paths for year t1
        files_t2 = {os.path.basename(f): f for f in glob(os.path.join(dir_t2, "patch_*.tif"))}  # Map filenames to full paths for year t2
        common = sorted(files_t1.keys() & files_t2.keys())  # Find patch filenames that exist in both years and sort them
        print(f"  Common patches: {len(common)}")  # Print the number of spatially co-located patches found

        out_dir = os.path.join(masks_dir, pair)  # Build the output directory path for this year pair's change masks
        os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it does not already exist

        stats = {"total": 0, "with_change": 0, "skipped": 0}  # Initialize counters for tracking processing statistics

        for fname in tqdm(common, desc=f"  {pair}", ncols=80):  # Loop over each common patch filename with a progress bar
            d1, profile = _load_patch(files_t1[fname])  # Load the NDBI patch for year t1 and its geospatial metadata
            d2, _ = _load_patch(files_t2[fname])  # Load the NDBI patch for year t2 (metadata not needed again)

            valid = np.isfinite(d1) & (d1 != 0) & np.isfinite(d2) & (d2 != 0)  # Identify pixels that are valid in both years
            if valid.mean() < min_valid:  # If the fraction of valid pixels is below the threshold
                stats["skipped"] += 1  # Increment the skipped counter
                continue  # Skip this patch and move to the next one

            u1, _ = _urban_mask(d1, fixed_threshold)  # Generate the binary urban mask for year t1
            u2, _ = _urban_mask(d2, fixed_threshold)  # Generate the binary urban mask for year t2

            change = (u1 != u2).astype(np.uint8)  # XOR the two urban masks: 1 where urban status changed, 0 where it stayed the same
            change[~valid] = 0  # Set invalid pixel locations to 0 (no change) to avoid false detections

            out_profile = profile.copy()  # Copy the geospatial metadata from the source patch
            out_profile.update(dtype="uint8", count=1, compress="lzw")  # Update metadata: uint8 data type, single band, LZW compression
            with rasterio.open(os.path.join(out_dir, fname), "w", **out_profile) as dst:  # Open a new GeoTIFF file for writing
                dst.write(change[np.newaxis])  # Write the change mask as a single-band raster (adding a band dimension)

            stats["total"] += 1  # Increment the total processed patches counter
            if change.sum() > 0:  # If at least one pixel shows change
                stats["with_change"] += 1  # Increment the counter for patches that contain change

        summary[pair] = stats  # Store the statistics for this year pair in the summary dictionary
        print(f"  Saved {stats['total']} masks  "  # Print the number of masks saved
              f"({stats['with_change']} contain change, "  # Print how many of those contain change pixels
              f"{stats['skipped']} skipped)")  # Print how many patches were skipped due to insufficient valid data

    return summary  # Return the dictionary containing processing statistics for all year pairs
