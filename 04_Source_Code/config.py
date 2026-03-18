"""
Configuration for Jeddah Urban Change Detection Pipeline.
Adjust DATA_ROOT to match your Google Drive mount path.
"""

import os

# ========================== PATHS ==========================
DATA_ROOT = "/content/drive/MyDrive/jeddah"

PATCHES_DIR = os.path.join(DATA_ROOT, "03_ML_Ready_Patches", "NDVI NDBI NDWI NaturalColor")
OUTPUT_DIR = os.path.join(DATA_ROOT, "05_Results")
MASKS_DIR = os.path.join(OUTPUT_DIR, "change_masks")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# ========================== DATA ==========================
YEARS = [2018, 2020, 2022, 2024]
YEAR_PAIRS = [(2018, 2020), (2020, 2022), (2022, 2024)]
PATCH_SIZE = 256
MIN_VALID_FRACTION = 0.10

# ======================== TRAINING ========================
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_WORKERS = 2
SEED = 42

# ========================= MODEL =========================
IN_CHANNELS = 1
OUT_CHANNELS = 1
ENCODER_CHANNELS = [64, 128, 256, 512]


def get_ndbi_dir(year):
    return os.path.join(PATCHES_DIR, str(year), f"Jeddah_{year}_NDBI_Raw")


def get_ndvi_dir(year):
    return os.path.join(PATCHES_DIR, str(year), f"Jeddah_{year}_NDVI_Raw")


def get_mask_dir(year_t1, year_t2):
    return os.path.join(MASKS_DIR, f"{year_t1}_{year_t2}")


def ensure_dirs():
    for d in [OUTPUT_DIR, MASKS_DIR, CHECKPOINTS_DIR, VIS_DIR]:
        os.makedirs(d, exist_ok=True)
