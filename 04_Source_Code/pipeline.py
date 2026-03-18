"""
End-to-end pipeline: mask generation → training → evaluation.

Usage (from Google Colab or command line):
    python pipeline.py
"""

import torch
from config import (
    PATCHES_DIR, MASKS_DIR, CHECKPOINTS_DIR, VIS_DIR,
    YEAR_PAIRS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, EARLY_STOP_PATIENCE, TRAIN_RATIO,
    VAL_RATIO, NUM_WORKERS, SEED, MIN_VALID_FRACTION,
    IN_CHANNELS, OUT_CHANNELS, ensure_dirs,
)
from mask_generator import generate_change_masks
from dataset import create_dataloaders
from model import SiameseUNet
from train import train_model
from evaluate import (
    compute_metrics, print_metrics,
    plot_training_curves, plot_predictions, plot_confusion_matrix,
)
import os


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ---- Step 1: Generate change masks ----
    print("=" * 60)
    print("  STEP 1 / 3 : Generating binary change masks (NDBI)")
    print("=" * 60)
    summary = generate_change_masks(
        PATCHES_DIR, MASKS_DIR, YEAR_PAIRS,
        min_valid=MIN_VALID_FRACTION,
    )
    for pair, s in summary.items():
        print(f"  {pair}: {s['total']} masks ({s['with_change']} with change)")

    # ---- Step 2: Train ----
    print("\n" + "=" * 60)
    print("  STEP 2 / 3 : Training Siamese U-Net")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        PATCHES_DIR, MASKS_DIR, YEAR_PAIRS,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    model = SiameseUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
    history = train_model(
        model, train_loader, val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=EARLY_STOP_PATIENCE,
        checkpoint_dir=CHECKPOINTS_DIR,
    )

    plot_training_curves(
        history,
        save_path=os.path.join(VIS_DIR, "training_curves.png"),
    )

    # ---- Step 3: Evaluate ----
    print("\n" + "=" * 60)
    print("  STEP 3 / 3 : Evaluating on test set")
    print("=" * 60)

    metrics = compute_metrics(model, test_loader, device=device)
    print_metrics(metrics, title="Test Set Results")

    plot_confusion_matrix(
        metrics,
        save_path=os.path.join(VIS_DIR, "confusion_matrix.png"),
    )

    plot_predictions(
        model, test_loader, device=device, num_samples=6,
        save_path=os.path.join(VIS_DIR, "sample_predictions.png"),
    )

    print("\nPipeline complete. Results saved to:", VIS_DIR)


if __name__ == "__main__":
    main()
