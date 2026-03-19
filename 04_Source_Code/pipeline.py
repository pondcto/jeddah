"""
End-to-end pipeline: mask generation → training → evaluation.

Usage (from Google Colab or command line):
    python pipeline.py
"""

import torch  # Import PyTorch for GPU detection and model operations
from config import (  # Import all configuration constants and helpers from the config module
    PATCHES_DIR, MASKS_DIR, CHECKPOINTS_DIR, VIS_DIR,  # Import directory paths for data, masks, checkpoints, and visualizations
    YEAR_PAIRS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,  # Import data year pairs and training hyperparameters
    WEIGHT_DECAY, EARLY_STOP_PATIENCE, TRAIN_RATIO,  # Import regularization, early stopping, and split ratio settings
    VAL_RATIO, NUM_WORKERS, SEED, MIN_VALID_FRACTION,  # Import validation ratio, data loading workers, seed, and valid pixel threshold
    IN_CHANNELS, OUT_CHANNELS, ensure_dirs,  # Import model channel configuration and the directory creation helper
)
from mask_generator import generate_change_masks  # Import the function that generates binary change masks from NDBI patches
from dataset import create_dataloaders  # Import the function that builds train/val/test DataLoaders
from model import SiameseUNet  # Import the Siamese U-Net model class
from train import train_model  # Import the training loop function
from evaluate import (  # Import evaluation and visualization functions
    compute_metrics, print_metrics,  # Import the metrics computation and formatted printing functions
    plot_training_curves, plot_predictions, plot_confusion_matrix,  # Import the plotting functions for curves, predictions, and confusion matrix
)
import os  # Import OS module for path operations


def main():  # Define the main pipeline function that orchestrates all steps
    ensure_dirs()  # Create all required output directories if they do not already exist
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Select GPU if available, otherwise fall back to CPU
    print(f"Device: {device}\n")  # Print the selected compute device

    # ---- Step 1: Generate change masks ----
    print("=" * 60)  # Print a separator line for Step 1
    print("  STEP 1 / 3 : Generating binary change masks (NDBI)")  # Print the step title
    print("=" * 60)  # Print a closing separator line
    summary = generate_change_masks(  # Call the mask generation function with configured parameters
        PATCHES_DIR, MASKS_DIR, YEAR_PAIRS,  # Pass the patches directory, output masks directory, and year pairs
        min_valid=MIN_VALID_FRACTION,  # Pass the minimum valid pixel fraction threshold
    )
    for pair, s in summary.items():  # Iterate over the summary statistics for each year pair
        print(f"  {pair}: {s['total']} masks ({s['with_change']} with change)")  # Print the count of generated masks and those containing change

    # ---- Step 2: Train ----
    print("\n" + "=" * 60)  # Print a separator line for Step 2
    print("  STEP 2 / 3 : Training Siamese U-Net")  # Print the step title
    print("=" * 60)  # Print a closing separator line

    train_loader, val_loader, test_loader = create_dataloaders(  # Create the three DataLoaders from the patches and masks
        PATCHES_DIR, MASKS_DIR, YEAR_PAIRS,  # Pass the directory paths and year pairs
        batch_size=BATCH_SIZE,  # Pass the configured batch size
        train_ratio=TRAIN_RATIO,  # Pass the training set ratio
        val_ratio=VAL_RATIO,  # Pass the validation set ratio
        num_workers=NUM_WORKERS,  # Pass the number of data loading workers
        seed=SEED,  # Pass the random seed for reproducible splits
    )

    model = SiameseUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)  # Instantiate the Siamese U-Net model with configured channel counts
    history = train_model(  # Run the full training loop and capture the training history
        model, train_loader, val_loader,  # Pass the model and the train/val DataLoaders
        device=device,  # Pass the compute device (GPU or CPU)
        num_epochs=NUM_EPOCHS,  # Pass the maximum number of training epochs
        lr=LEARNING_RATE,  # Pass the initial learning rate
        weight_decay=WEIGHT_DECAY,  # Pass the L2 regularization weight decay
        patience=EARLY_STOP_PATIENCE,  # Pass the early stopping patience
        checkpoint_dir=CHECKPOINTS_DIR,  # Pass the directory for saving model checkpoints
    )

    plot_training_curves(  # Plot and save the training/validation loss and IoU curves
        history,  # Pass the training history dictionary
        save_path=os.path.join(VIS_DIR, "training_curves.png"),  # Save the plot to the visualizations directory
    )

    # ---- Step 3: Evaluate ----
    print("\n" + "=" * 60)  # Print a separator line for Step 3
    print("  STEP 3 / 3 : Evaluating on test set")  # Print the step title
    print("=" * 60)  # Print a closing separator line

    metrics = compute_metrics(model, test_loader, device=device)  # Compute all evaluation metrics on the held-out test set
    print_metrics(metrics, title="Test Set Results")  # Print the formatted evaluation metrics to the console

    plot_confusion_matrix(  # Plot and save the confusion matrix heatmap
        metrics,  # Pass the computed metrics containing TP, FP, FN, TN
        save_path=os.path.join(VIS_DIR, "confusion_matrix.png"),  # Save the plot to the visualizations directory
    )

    plot_predictions(  # Plot and save side-by-side prediction comparisons
        model, test_loader, device=device, num_samples=6,  # Pass the model, test loader, device, and number of samples to display
        save_path=os.path.join(VIS_DIR, "sample_predictions.png"),  # Save the plot to the visualizations directory
    )

    print("\nPipeline complete. Results saved to:", VIS_DIR)  # Print the completion message with the output directory path


if __name__ == "__main__":  # Check if this script is being run directly (not imported as a module)
    main()  # Execute the main pipeline function
