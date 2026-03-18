"""
Evaluation metrics and visualization for change detection.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ──────────────────── Metrics ────────────────────

def compute_metrics(model, loader, device="cuda", threshold=0.5):
    """
    Compute pixel-level metrics on a DataLoader.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, iou, confusion (TP/FP/FN/TN)
    """
    model.eval()
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for t1, t2, mask in loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device)

            pred = (model(t1, t2) > threshold).float()

            tp += (pred * mask).sum().item()
            fp += (pred * (1 - mask)).sum().item()
            fn += ((1 - pred) * mask).sum().item()
            tn += ((1 - pred) * (1 - mask)).sum().item()

    total = tp + fp + fn + tn
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    accuracy = (tp + tn) / (total + 1e-7)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def print_metrics(metrics, title="Evaluation"):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  "
          f"FN={metrics['fn']:,}  TN={metrics['tn']:,}")
    print(f"{'=' * 50}\n")


# ─────────────────── Visualization ───────────────

def plot_training_curves(history, save_path=None):
    """Plot loss and IoU curves from training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE + Dice)")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_iou"], "b-", label="Train")
    ax2.plot(epochs, history["val_iou"], "r-", label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("Training & Validation IoU")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_predictions(model, loader, device="cuda", num_samples=6,
                     save_path=None):
    """Show side-by-side: NDBI_t1 | NDBI_t2 | Ground Truth | Prediction."""
    model.eval()
    shown = 0
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis]

    cmap_change = ListedColormap(["black", "red"])

    with torch.no_grad():
        for t1, t2, mask in loader:
            t1_d = t1.to(device)
            t2_d = t2.to(device)
            pred = model(t1_d, t2_d).cpu()

            for i in range(t1.size(0)):
                if shown >= num_samples:
                    break

                axes[shown, 0].imshow(t1[i, 0], cmap="RdYlGn", vmin=-1, vmax=1)
                axes[shown, 0].set_title("NDBI t1")
                axes[shown, 0].axis("off")

                axes[shown, 1].imshow(t2[i, 0], cmap="RdYlGn", vmin=-1, vmax=1)
                axes[shown, 1].set_title("NDBI t2")
                axes[shown, 1].axis("off")

                axes[shown, 2].imshow(mask[i, 0], cmap=cmap_change, vmin=0, vmax=1)
                axes[shown, 2].set_title("Ground Truth")
                axes[shown, 2].axis("off")

                axes[shown, 3].imshow((pred[i, 0] > 0.5).float(), cmap=cmap_change,
                                      vmin=0, vmax=1)
                axes[shown, 3].set_title("Prediction")
                axes[shown, 3].axis("off")

                shown += 1
            if shown >= num_samples:
                break

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(metrics, save_path=None):
    """Plot a 2×2 confusion matrix."""
    cm = np.array([[metrics["tn"], metrics["fp"]],
                    [metrics["fn"], metrics["tp"]]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["No Change", "Change"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
