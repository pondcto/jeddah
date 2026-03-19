"""
Training loop for the Siamese U-Net change detector.
Includes combined BCE + Dice loss, early stopping, and checkpointing.
"""

import os  # Import OS module for file path operations and directory creation
import time  # Import time module to measure epoch duration
import numpy as np  # Import NumPy (available for potential numerical operations)
import torch  # Import the core PyTorch library for tensor operations and GPU support
import torch.nn as nn  # Import the neural network module containing loss functions and layer definitions


# ───────────────────── Loss ─────────────────────

class DiceBCELoss(nn.Module):  # Define a custom loss class combining Binary Cross-Entropy and Dice loss
    """Weighted sum of Binary Cross-Entropy and Dice loss."""

    def __init__(self, bce_weight=0.5, smooth=1.0):  # Constructor with configurable BCE blend ratio and Dice smoothing factor
        super().__init__()  # Initialize the parent nn.Module class
        self.bce = nn.BCELoss()  # Instantiate the standard Binary Cross-Entropy loss function
        self.bce_weight = bce_weight  # Store the weighting factor for BCE in the combined loss (0.5 = equal blend)
        self.smooth = smooth  # Store the Dice smoothing constant to prevent division by zero

    def forward(self, pred, target):  # Compute the combined loss given predictions and ground truth
        bce_loss = self.bce(pred, target)  # Compute the standard binary cross-entropy loss between predictions and targets

        pred_flat = pred.view(-1)  # Flatten the prediction tensor to a 1-D vector for Dice computation
        target_flat = target.view(-1)  # Flatten the target tensor to a 1-D vector
        intersection = (pred_flat * target_flat).sum()  # Compute the element-wise product and sum to get the intersection
        dice = 1 - (2.0 * intersection + self.smooth) / (  # Compute Dice loss as 1 minus the Dice coefficient
            pred_flat.sum() + target_flat.sum() + self.smooth  # Denominator: sum of all predicted + all target values + smoothing
        )
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice  # Return the weighted combination of BCE and Dice losses


# ──────────────── Training helpers ──────────────

def _run_epoch(model, loader, criterion, optimizer, device, is_train):  # Execute one complete epoch (training or validation)
    model.train() if is_train else model.eval()  # Set the model to training mode (enables dropout/BN updates) or evaluation mode
    total_loss = 0.0  # Initialize the cumulative loss accumulator for this epoch
    total_iou = 0.0  # Initialize the cumulative IoU accumulator for this epoch
    n = 0  # Initialize the sample counter

    ctx = torch.enable_grad() if is_train else torch.no_grad()  # Enable gradient computation for training; disable for validation
    with ctx:  # Enter the gradient context (either enabled or disabled)
        for t1, t2, mask in loader:  # Iterate over each batch from the DataLoader
            t1 = t1.to(device)  # Transfer the NDBI t1 tensor to the target device (GPU or CPU)
            t2 = t2.to(device)  # Transfer the NDBI t2 tensor to the target device
            mask = mask.to(device)  # Transfer the ground truth change mask to the target device

            pred = model(t1, t2)  # Forward pass: feed both temporal images through the Siamese U-Net to get predictions
            loss = criterion(pred, mask)  # Compute the combined BCE + Dice loss between predictions and ground truth

            if is_train:  # If in training mode, perform backpropagation and weight update
                optimizer.zero_grad()  # Reset all parameter gradients to zero before the backward pass
                loss.backward()  # Compute gradients of the loss with respect to all model parameters
                optimizer.step()  # Update model weights using the computed gradients and the Adam optimizer

            total_loss += loss.item() * t1.size(0)  # Accumulate the batch loss weighted by the number of samples in the batch

            p_bin = (pred > 0.5).float()  # Binarize predictions at threshold 0.5 to get hard change/no-change decisions
            inter = (p_bin * mask).sum(dim=(1, 2, 3))  # Compute per-sample intersection: pixels predicted as change AND actually changed
            union = p_bin.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) - inter  # Compute per-sample union using inclusion-exclusion
            iou = (inter + 1e-7) / (union + 1e-7)  # Compute per-sample IoU with a small epsilon to avoid division by zero
            total_iou += iou.sum().item()  # Accumulate the sum of per-sample IoU values
            n += t1.size(0)  # Add the batch size to the total sample count

    return total_loss / n, total_iou / n  # Return the average loss and average IoU for the entire epoch


def train_model(model, train_loader, val_loader, *,  # Main training function with keyword-only arguments after the asterisk
                device="cuda", num_epochs=50, lr=1e-4, weight_decay=1e-5,
                patience=10, checkpoint_dir="checkpoints"):
    """
    Full training loop.

    Returns
    -------
    history : dict  with keys 'train_loss', 'val_loss', 'train_iou', 'val_iou'
    """
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create the checkpoint directory if it does not already exist
    model = model.to(device)  # Move the model's parameters and buffers to the target device (GPU or CPU)
    criterion = DiceBCELoss()  # Instantiate the combined BCE + Dice loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Create an Adam optimizer with the specified learning rate and L2 regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Create a learning rate scheduler that reduces LR when validation loss plateaus
        optimizer, mode="min", factor=0.5, patience=5  # Monitor minimum val loss; halve the LR if no improvement for 5 epochs
    )

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}  # Initialize empty lists to record training metrics per epoch
    best_val_loss = float("inf")  # Initialize the best validation loss to infinity (any real loss will be better)
    wait = 0  # Initialize the early stopping counter to zero
    best_path = os.path.join(checkpoint_dir, "best_model.pth")  # Define the file path for saving the best model checkpoint

    for epoch in range(1, num_epochs + 1):  # Loop over each epoch from 1 to num_epochs
        t0 = time.time()  # Record the start time of this epoch

        tr_loss, tr_iou = _run_epoch(model, train_loader, criterion, optimizer, device, True)  # Run one training epoch and get the average loss and IoU
        vl_loss, vl_iou = _run_epoch(model, val_loader, criterion, None, device, False)  # Run one validation epoch (no optimizer needed) and get the average loss and IoU

        scheduler.step(vl_loss)  # Update the learning rate scheduler based on the current validation loss

        history["train_loss"].append(tr_loss)  # Record the training loss for this epoch
        history["val_loss"].append(vl_loss)  # Record the validation loss for this epoch
        history["train_iou"].append(tr_iou)  # Record the training IoU for this epoch
        history["val_iou"].append(vl_iou)  # Record the validation IoU for this epoch

        elapsed = time.time() - t0  # Calculate the elapsed time for this epoch in seconds
        lr_now = optimizer.param_groups[0]["lr"]  # Retrieve the current learning rate from the optimizer
        print(  # Print the epoch summary with all key metrics
            f"Epoch {epoch:3d}/{num_epochs} | "  # Print the current epoch number and total
            f"train loss {tr_loss:.4f}  IoU {tr_iou:.4f} | "  # Print the training loss and IoU
            f"val loss {vl_loss:.4f}  IoU {vl_iou:.4f} | "  # Print the validation loss and IoU
            f"lr {lr_now:.1e} | {elapsed:.0f}s"  # Print the current learning rate and epoch duration
        )

        if vl_loss < best_val_loss:  # Check if the current validation loss is the best seen so far
            best_val_loss = vl_loss  # Update the best validation loss record
            wait = 0  # Reset the early stopping patience counter
            torch.save(model.state_dict(), best_path)  # Save the current model weights as the best checkpoint
            print(f"  -> saved best model (val_loss={vl_loss:.4f})")  # Print confirmation that the best model was saved
        else:  # If the validation loss did not improve
            wait += 1  # Increment the early stopping patience counter
            if wait >= patience:  # If the patience limit has been reached
                print(f"\n  Early stopping at epoch {epoch} (patience={patience})")  # Print the early stopping message
                break  # Exit the training loop early

    model.load_state_dict(torch.load(best_path, map_location=device))  # Load the best model weights back into the model
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")  # Print the final training completion message with the best loss
    return history  # Return the training history dictionary containing per-epoch metrics
