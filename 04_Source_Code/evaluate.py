"""
Evaluation metrics and visualization for change detection.
"""

import os  # Import OS module for file path construction
import numpy as np  # Import NumPy for numerical operations on confusion matrix arrays
import torch  # Import PyTorch for model inference and tensor operations
import matplotlib.pyplot as plt  # Import matplotlib for creating plots and figures
from matplotlib.colors import ListedColormap  # Import ListedColormap to create custom discrete color maps


# ──────────────────── Metrics ────────────────────

def compute_metrics(model, loader, device="cuda", threshold=0.5):  # Compute pixel-level evaluation metrics over an entire DataLoader
    """
    Compute pixel-level metrics on a DataLoader.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, iou, confusion (TP/FP/FN/TN)
    """
    model.eval()  # Set the model to evaluation mode (disables dropout and uses running BatchNorm statistics)
    tp = fp = fn = tn = 0  # Initialize the four confusion matrix counters to zero

    with torch.no_grad():  # Disable gradient computation to save memory and speed up inference
        for t1, t2, mask in loader:  # Iterate over each batch from the DataLoader
            t1 = t1.to(device)  # Transfer the NDBI t1 tensor to the target device (GPU or CPU)
            t2 = t2.to(device)  # Transfer the NDBI t2 tensor to the target device
            mask = mask.to(device)  # Transfer the ground truth change mask to the target device

            pred = (model(t1, t2) > threshold).float()  # Forward pass and binarize predictions at the given threshold

            tp += (pred * mask).sum().item()  # Count true positives: pixels predicted as change that are actually change
            fp += (pred * (1 - mask)).sum().item()  # Count false positives: pixels predicted as change that are actually no-change
            fn += ((1 - pred) * mask).sum().item()  # Count false negatives: pixels predicted as no-change that are actually change
            tn += ((1 - pred) * (1 - mask)).sum().item()  # Count true negatives: pixels predicted as no-change that are actually no-change

    total = tp + fp + fn + tn  # Compute the total number of pixels evaluated
    precision = tp / (tp + fp + 1e-7)  # Compute precision: fraction of predicted change pixels that are correct; epsilon prevents division by zero
    recall = tp / (tp + fn + 1e-7)  # Compute recall: fraction of actual change pixels that were detected
    f1 = 2 * precision * recall / (precision + recall + 1e-7)  # Compute F1-score: harmonic mean of precision and recall
    iou = tp / (tp + fp + fn + 1e-7)  # Compute IoU (Intersection over Union) for the change class
    accuracy = (tp + tn) / (total + 1e-7)  # Compute overall pixel accuracy: fraction of all pixels correctly classified

    return {  # Return all metrics and confusion matrix counts as a dictionary
        "accuracy": accuracy,  # Overall pixel classification accuracy
        "precision": precision,  # Precision for the change class
        "recall": recall,  # Recall for the change class
        "f1": f1,  # F1-score for the change class
        "iou": iou,  # Intersection over Union for the change class
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),  # Raw confusion matrix counts as integers
    }


def print_metrics(metrics, title="Evaluation"):  # Print a formatted summary of evaluation metrics to the console
    print(f"\n{'=' * 50}")  # Print a top separator line
    print(f"  {title}")  # Print the title of the evaluation section
    print(f"{'=' * 50}")  # Print a middle separator line
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")  # Print the overall accuracy to 4 decimal places
    print(f"  Precision : {metrics['precision']:.4f}")  # Print the precision to 4 decimal places
    print(f"  Recall    : {metrics['recall']:.4f}")  # Print the recall to 4 decimal places
    print(f"  F1-Score  : {metrics['f1']:.4f}")  # Print the F1-score to 4 decimal places
    print(f"  IoU       : {metrics['iou']:.4f}")  # Print the IoU to 4 decimal places
    print(f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  "  # Print the true positive and false positive counts with commas
          f"FN={metrics['fn']:,}  TN={metrics['tn']:,}")  # Print the false negative and true negative counts with commas
    print(f"{'=' * 50}\n")  # Print a bottom separator line


# ─────────────────── Visualization ───────────────

def plot_training_curves(history, save_path=None):  # Plot training and validation loss/IoU curves from the training history
    """Plot loss and IoU curves from training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Create a figure with two side-by-side subplots

    epochs = range(1, len(history["train_loss"]) + 1)  # Generate a sequence of epoch numbers starting from 1

    ax1.plot(epochs, history["train_loss"], "b-", label="Train")  # Plot the training loss curve in blue
    ax1.plot(epochs, history["val_loss"], "r-", label="Val")  # Plot the validation loss curve in red
    ax1.set_xlabel("Epoch")  # Label the x-axis as "Epoch"
    ax1.set_ylabel("Loss (BCE + Dice)")  # Label the y-axis with the loss type
    ax1.set_title("Training & Validation Loss")  # Set the subplot title
    ax1.legend()  # Display the legend showing train/val labels
    ax1.grid(True, alpha=0.3)  # Add a semi-transparent grid for readability

    ax2.plot(epochs, history["train_iou"], "b-", label="Train")  # Plot the training IoU curve in blue
    ax2.plot(epochs, history["val_iou"], "r-", label="Val")  # Plot the validation IoU curve in red
    ax2.set_xlabel("Epoch")  # Label the x-axis as "Epoch"
    ax2.set_ylabel("IoU")  # Label the y-axis as "IoU"
    ax2.set_title("Training & Validation IoU")  # Set the subplot title
    ax2.legend()  # Display the legend
    ax2.grid(True, alpha=0.3)  # Add a semi-transparent grid

    plt.tight_layout()  # Adjust subplot spacing to prevent label overlap
    if save_path:  # If a save path was provided
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Save the figure as a PNG at 150 DPI with tight bounding box
    plt.show()  # Display the figure in the notebook or GUI


def plot_predictions(model, loader, device="cuda", num_samples=6,  # Visualize model predictions alongside inputs and ground truth
                     save_path=None):
    """Show side-by-side: NDBI_t1 | NDBI_t2 | Ground Truth | Prediction."""
    model.eval()  # Set the model to evaluation mode
    shown = 0  # Initialize a counter for the number of samples displayed so far
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))  # Create a grid of subplots: num_samples rows x 4 columns
    if num_samples == 1:  # If only one sample is requested
        axes = axes[np.newaxis]  # Add an extra dimension so indexing remains consistent

    cmap_change = ListedColormap(["black", "red"])  # Create a colormap: 0 (no change) = black, 1 (change) = red

    with torch.no_grad():  # Disable gradient computation during inference
        for t1, t2, mask in loader:  # Iterate over each batch from the DataLoader
            t1_d = t1.to(device)  # Transfer the NDBI t1 batch to the target device
            t2_d = t2.to(device)  # Transfer the NDBI t2 batch to the target device
            pred = model(t1_d, t2_d).cpu()  # Run the forward pass and move predictions back to CPU for plotting

            for i in range(t1.size(0)):  # Iterate over each sample in the batch
                if shown >= num_samples:  # If we have already displayed enough samples
                    break  # Stop processing more samples

                axes[shown, 0].imshow(t1[i, 0], cmap="RdYlGn", vmin=-1, vmax=1)  # Display the NDBI t1 image with a diverging colormap
                axes[shown, 0].set_title("NDBI t1")  # Set the column title for the t1 input
                axes[shown, 0].axis("off")  # Hide the axis ticks and borders

                axes[shown, 1].imshow(t2[i, 0], cmap="RdYlGn", vmin=-1, vmax=1)  # Display the NDBI t2 image
                axes[shown, 1].set_title("NDBI t2")  # Set the column title for the t2 input
                axes[shown, 1].axis("off")  # Hide the axis ticks and borders

                axes[shown, 2].imshow(mask[i, 0], cmap=cmap_change, vmin=0, vmax=1)  # Display the ground truth change mask in black/red
                axes[shown, 2].set_title("Ground Truth")  # Set the column title for the ground truth
                axes[shown, 2].axis("off")  # Hide the axis ticks and borders

                axes[shown, 3].imshow((pred[i, 0] > 0.5).float(), cmap=cmap_change,  # Display the binarized model prediction in black/red
                                      vmin=0, vmax=1)
                axes[shown, 3].set_title("Prediction")  # Set the column title for the prediction
                axes[shown, 3].axis("off")  # Hide the axis ticks and borders

                shown += 1  # Increment the displayed sample counter
            if shown >= num_samples:  # If enough samples have been displayed
                break  # Exit the batch loop

    plt.tight_layout()  # Adjust spacing between subplots
    if save_path:  # If a save path was provided
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Save the figure to disk
    plt.show()  # Display the figure


def plot_confusion_matrix(metrics, save_path=None):  # Plot a 2x2 confusion matrix heatmap from the computed metrics
    """Plot a 2x2 confusion matrix."""
    cm = np.array([[metrics["tn"], metrics["fp"]],  # Build the confusion matrix: row 0 = actual no-change (TN, FP)
                    [metrics["fn"], metrics["tp"]]])  # Row 1 = actual change (FN, TP)

    fig, ax = plt.subplots(figsize=(6, 5))  # Create a single figure and axis for the confusion matrix
    im = ax.imshow(cm, cmap="Blues")  # Display the confusion matrix as a blue heatmap image

    labels = ["No Change", "Change"]  # Define the class labels for the axes
    ax.set_xticks([0, 1])  # Set x-axis tick positions for two classes
    ax.set_yticks([0, 1])  # Set y-axis tick positions for two classes
    ax.set_xticklabels(labels)  # Label the x-axis ticks with class names
    ax.set_yticklabels(labels)  # Label the y-axis ticks with class names
    ax.set_xlabel("Predicted")  # Label the x-axis as "Predicted"
    ax.set_ylabel("Actual")  # Label the y-axis as "Actual"
    ax.set_title("Confusion Matrix")  # Set the plot title

    for i in range(2):  # Iterate over each row of the confusion matrix
        for j in range(2):  # Iterate over each column of the confusion matrix
            ax.text(j, i, f"{cm[i, j]:,.0f}", ha="center", va="center",  # Annotate each cell with its count value, centered
                    color="white" if cm[i, j] > cm.max() / 2 else "black",  # Use white text on dark cells, black text on light cells
                    fontsize=12)  # Set the annotation font size

    plt.colorbar(im, ax=ax)  # Add a color bar showing the value-to-color mapping
    plt.tight_layout()  # Adjust the layout to prevent clipping
    if save_path:  # If a save path was provided
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Save the confusion matrix plot to disk
    plt.show()  # Display the confusion matrix plot
