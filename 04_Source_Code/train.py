"""
Training loop for the Siamese U-Net change detector.
Includes combined BCE + Dice loss, early stopping, and checkpointing.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn


# ───────────────────── Loss ─────────────────────

class DiceBCELoss(nn.Module):
    """Weighted sum of Binary Cross-Entropy and Dice loss."""

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice


# ──────────────── Training helpers ──────────────

def _run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for t1, t2, mask in loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device)

            pred = model(t1, t2)
            loss = criterion(pred, mask)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * t1.size(0)

            # batch IoU
            p_bin = (pred > 0.5).float()
            inter = (p_bin * mask).sum(dim=(1, 2, 3))
            union = p_bin.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) - inter
            iou = (inter + 1e-7) / (union + 1e-7)
            total_iou += iou.sum().item()
            n += t1.size(0)

    return total_loss / n, total_iou / n


def train_model(model, train_loader, val_loader, *,
                device="cuda", num_epochs=50, lr=1e-4, weight_decay=1e-5,
                patience=10, checkpoint_dir="checkpoints"):
    """
    Full training loop.

    Returns
    -------
    history : dict  with keys 'train_loss', 'val_loss', 'train_iou', 'val_iou'
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_val_loss = float("inf")
    wait = 0
    best_path = os.path.join(checkpoint_dir, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_iou = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_iou = _run_epoch(model, val_loader, criterion, None, device, False)

        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(vl_iou)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{num_epochs} │ "
            f"train loss {tr_loss:.4f}  IoU {tr_iou:.4f} │ "
            f"val loss {vl_loss:.4f}  IoU {vl_iou:.4f} │ "
            f"lr {lr_now:.1e} │ {elapsed:.0f}s"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            wait = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ saved best model (val_loss={vl_loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return history
