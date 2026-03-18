# Urban Change Detection in Jeddah

Detecting urban expansion in Jeddah, Saudi Arabia across **2018, 2020, 2022, and 2024** using Sentinel-2 satellite imagery and a **Siamese U-Net** deep learning model.

## Project Structure

```
jeddah/
├── 01_Full_Raw_Rasters/          # Full-resolution GeoTIFFs (NDBI, NDVI, NDWI, NaturalColor, Raw Bands)
├── 02_Visual_Previews/           # NaturalColor previews for each year
├── 03_ML_Ready_Patches/          # 256×256 patches sliced from full rasters
│   ├── NDVI NDBI NDWI NaturalColor/
│   │   ├── 2018/                 # Jeddah_2018_NDBI_Raw/, NDVI_Raw/, etc.
│   │   ├── 2020/
│   │   ├── 2022/
│   │   └── 2024/
│   └── Raw Bands Stack/
├── 04_Source_Code/
│   ├── slicer.py                 # Patch extraction from full rasters
│   ├── config.py                 # Configuration (paths, hyperparameters)
│   ├── mask_generator.py         # NDBI thresholding → binary change masks
│   ├── dataset.py                # PyTorch Dataset & DataLoader
│   ├── model.py                  # Siamese U-Net architecture
│   ├── train.py                  # Training loop (BCE+Dice, early stopping)
│   ├── evaluate.py               # Metrics (IoU, F1, etc.) & visualization
│   ├── pipeline.py               # End-to-end orchestration script
│   └── requirements.txt          # Python dependencies
├── 05_Results/                   # Generated after running the pipeline
│   ├── change_masks/             # NDBI-derived binary change masks
│   ├── predicted_masks/          # Model predictions per year pair
│   ├── checkpoints/              # Saved model weights
│   └── visualizations/           # Training curves, confusion matrix, etc.
└── Jeddah_Urban_Change_Detection.ipynb   # Main Colab notebook (self-contained)
```

## Quick Start (Google Colab)

1. Upload the `jeddah` folder to your Google Drive root
2. Open `Jeddah_Urban_Change_Detection.ipynb` in Google Colab
3. Set **Runtime → Change runtime type → T4 GPU**
4. Run all cells sequentially

The notebook is fully self-contained and will:
- Generate binary change masks from NDBI patches using Otsu thresholding
- Build train/val/test DataLoaders with augmentation
- Train a Siamese U-Net model
- Evaluate with IoU, F1, precision, recall, accuracy
- Generate visual change maps for each year pair

## Pipeline Overview

### Step 1: Mask Generation

Binary change masks are generated automatically from NDBI (Normalized Difference Built-up Index) patches:
- Otsu's threshold classifies each pixel as urban or non-urban
- XOR between consecutive years produces a binary change mask
- Year pairs: 2018→2020, 2020→2022, 2022→2024

### Step 2: Siamese U-Net Training

The model takes two NDBI patches from different years and predicts a binary change map:
- **Shared encoder**: Weight-tied encoder processes both temporal inputs
- **Feature fusion**: Skip connections from both branches are concatenated
- **Decoder**: Produces a pixel-wise change probability map
- **Loss**: Combined BCE + Dice loss (handles class imbalance)
- **Augmentation**: Random flips and 90° rotations

### Step 3: Evaluation

Metrics computed on a held-out test set:
- **IoU** (Intersection over Union)
- **F1-Score**
- **Precision & Recall**
- **Overall Accuracy**
- Confusion matrix and visual predictions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- rasterio
- numpy
- scikit-image
- matplotlib
- tqdm

Install via: `pip install -r 04_Source_Code/requirements.txt`

## Data

Sentinel-2 multi-temporal imagery for Jeddah, processed into spectral indices (NDBI, NDVI, NDWI) and sliced into 256×256 georeferenced patches.
