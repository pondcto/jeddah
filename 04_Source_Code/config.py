"""
Configuration for Jeddah Urban Change Detection Pipeline.
Adjust DATA_ROOT to match your Google Drive mount path.
"""

import os  # Import the operating system module for file path operations

# ========================== PATHS ==========================
DATA_ROOT = "/content/drive/MyDrive/jeddah"  # Root directory of the project on Google Drive

PATCHES_DIR = os.path.join(DATA_ROOT, "03_ML_Ready_Patches", "NDVI NDBI NDWI NaturalColor")  # Path to the pre-sliced 256x256 spectral index patches
OUTPUT_DIR = os.path.join(DATA_ROOT, "05_Results")  # Path where all pipeline outputs will be saved
MASKS_DIR = os.path.join(OUTPUT_DIR, "change_masks")  # Path to store the generated binary change masks
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")  # Path to store saved model weight checkpoints
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")  # Path to store generated plots and visual outputs

# ========================== DATA ==========================
YEARS = [2018, 2020, 2022, 2024]  # List of Sentinel-2 imagery acquisition years
YEAR_PAIRS = [(2018, 2020), (2020, 2022), (2022, 2024)]  # Consecutive year pairs for change detection comparison
PATCH_SIZE = 256  # Spatial dimension of each image patch in pixels (256x256)
MIN_VALID_FRACTION = 0.10  # Minimum fraction of valid (non-zero, non-NaN) pixels required to keep a patch

# ======================== TRAINING ========================
BATCH_SIZE = 16  # Number of samples processed together in one forward/backward pass
NUM_EPOCHS = 50  # Maximum number of complete passes through the training dataset
LEARNING_RATE = 1e-4  # Initial step size for the Adam optimizer (0.0001)
WEIGHT_DECAY = 1e-5  # L2 regularization coefficient to prevent overfitting (0.00001)
EARLY_STOP_PATIENCE = 10  # Number of epochs to wait for validation loss improvement before stopping
TRAIN_RATIO = 0.70  # Fraction of the dataset allocated to training (70%)
VAL_RATIO = 0.15  # Fraction of the dataset allocated to validation (15%)
TEST_RATIO = 0.15  # Fraction of the dataset allocated to testing (15%)
NUM_WORKERS = 2  # Number of parallel subprocesses for data loading
SEED = 42  # Random seed for reproducible train/val/test splits

# ========================= MODEL =========================
IN_CHANNELS = 1  # Number of input channels per image (1 for single-band NDBI)
OUT_CHANNELS = 1  # Number of output channels (1 for binary change map)
ENCODER_CHANNELS = [64, 128, 256, 512]  # Feature map sizes at each encoder level


def get_ndbi_dir(year):  # Helper function to build the path to NDBI patches for a given year
    return os.path.join(PATCHES_DIR, str(year), f"Jeddah_{year}_NDBI_Raw")  # Constructs path like .../2018/Jeddah_2018_NDBI_Raw


def get_ndvi_dir(year):  # Helper function to build the path to NDVI patches for a given year
    return os.path.join(PATCHES_DIR, str(year), f"Jeddah_{year}_NDVI_Raw")  # Constructs path like .../2018/Jeddah_2018_NDVI_Raw


def get_mask_dir(year_t1, year_t2):  # Helper function to build the path to change masks for a year pair
    return os.path.join(MASKS_DIR, f"{year_t1}_{year_t2}")  # Constructs path like .../change_masks/2018_2020


def ensure_dirs():  # Creates all required output directories if they do not already exist
    for d in [OUTPUT_DIR, MASKS_DIR, CHECKPOINTS_DIR, VIS_DIR]:  # Iterate over each output directory path
        os.makedirs(d, exist_ok=True)  # Create the directory and all parent directories; no error if it already exists
