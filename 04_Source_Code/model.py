"""
Siamese U-Net for binary change detection.

Architecture
────────────
Two input images (t1, t2) pass through a shared encoder.
Skip features from both branches are concatenated and fed to
a single decoder that produces a binary change-probability map.

Input  : two tensors of shape (B, C, 256, 256)
Output : one  tensor of shape (B, 1, 256, 256)  ∈ [0, 1]
"""

import torch  # Import the core PyTorch library for tensor operations
import torch.nn as nn  # Import the neural network module containing layers, losses, and model base classes


class _ConvBlock(nn.Module):  # Define a reusable double-convolution building block inheriting from nn.Module
    """Conv 3×3 → BN → ReLU  ×2"""

    def __init__(self, in_ch, out_ch):  # Constructor takes input channel count and output channel count
        super().__init__()  # Initialize the parent nn.Module class
        self.block = nn.Sequential(  # Create a sequential container that chains layers in order
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # First 3x3 convolution with same-padding; bias disabled because BatchNorm follows
            nn.BatchNorm2d(out_ch),  # Batch normalization to stabilize and accelerate training
            nn.ReLU(inplace=True),  # ReLU activation function applied in-place to save memory
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),  # Second 3x3 convolution maintaining the same channel count
            nn.BatchNorm2d(out_ch),  # Second batch normalization layer
            nn.ReLU(inplace=True),  # Second ReLU activation applied in-place
        )

    def forward(self, x):  # Define the forward pass for the convolutional block
        return self.block(x)  # Pass the input tensor through the sequential block and return the result


class SiameseUNet(nn.Module):  # Define the Siamese U-Net model inheriting from nn.Module

    def __init__(self, in_channels=1, out_channels=1):  # Constructor with default 1 input channel (NDBI) and 1 output channel (binary mask)
        super().__init__()  # Initialize the parent nn.Module class
        self.pool = nn.MaxPool2d(2)  # 2x2 max pooling layer to downsample spatial dimensions by half

        # ---------- shared encoder ----------
        self.enc1 = _ConvBlock(in_channels, 64)  # Encoder level 1: input channels → 64 feature maps (256x256 → 256x256)
        self.enc2 = _ConvBlock(64, 128)  # Encoder level 2: 64 → 128 feature maps (128x128 → 128x128)
        self.enc3 = _ConvBlock(128, 256)  # Encoder level 3: 128 → 256 feature maps (64x64 → 64x64)
        self.enc4 = _ConvBlock(256, 512)  # Encoder level 4: 256 → 512 feature maps (32x32 → 32x32)

        # ---------- bottleneck (fused) ----------
        self.bottleneck = _ConvBlock(512 * 2, 1024)  # Bottleneck: concatenated features from both branches (1024 input) → 1024 output (16x16)

        # ---------- decoder ----------
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # Upsample bottleneck output from 16x16 to 32x32 with 512 channels
        self.dec4 = _ConvBlock(512 + 512 * 2, 512)  # Decoder level 4: 512 (upsampled) + 512 (skip_t1) + 512 (skip_t2) = 1536 → 512

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Upsample from 32x32 to 64x64 with 256 channels
        self.dec3 = _ConvBlock(256 + 256 * 2, 256)  # Decoder level 3: 256 + 256 + 256 = 768 → 256

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Upsample from 64x64 to 128x128 with 128 channels
        self.dec2 = _ConvBlock(128 + 128 * 2, 128)  # Decoder level 2: 128 + 128 + 128 = 384 → 128

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsample from 128x128 to 256x256 with 64 channels
        self.dec1 = _ConvBlock(64 + 64 * 2, 64)  # Decoder level 1: 64 + 64 + 64 = 192 → 64

        self.head = nn.Conv2d(64, out_channels, kernel_size=1)  # 1x1 convolution to map 64 features to the final output channel count
        self._init_weights()  # Call custom weight initialization after all layers are defined

    def _init_weights(self):  # Custom method to initialize all model weights
        for m in self.modules():  # Iterate over every layer/module in the network
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):  # Check if the module is a convolutional or transposed-convolutional layer
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # Apply He initialization suited for ReLU activations
            elif isinstance(m, nn.BatchNorm2d):  # Check if the module is a batch normalization layer
                nn.init.constant_(m.weight, 1)  # Set BatchNorm scale (gamma) to 1
                nn.init.constant_(m.bias, 0)  # Set BatchNorm shift (beta) to 0

    def forward(self, x1, x2):  # Define the forward pass taking two temporal input images
        # ---- encoder branch 1 ----
        e1_1 = self.enc1(x1);  p1_1 = self.pool(e1_1)  # Encode x1 at level 1 (skip features e1_1), then downsample to 128x128
        e2_1 = self.enc2(p1_1); p2_1 = self.pool(e2_1)  # Encode at level 2 (skip features e2_1), then downsample to 64x64
        e3_1 = self.enc3(p2_1); p3_1 = self.pool(e3_1)  # Encode at level 3 (skip features e3_1), then downsample to 32x32
        e4_1 = self.enc4(p3_1); p4_1 = self.pool(e4_1)  # Encode at level 4 (skip features e4_1), then downsample to 16x16

        # ---- encoder branch 2 (shared weights) ----
        e1_2 = self.enc1(x2);  p1_2 = self.pool(e1_2)  # Encode x2 at level 1 using the same encoder weights (weight-tied)
        e2_2 = self.enc2(p1_2); p2_2 = self.pool(e2_2)  # Encode x2 at level 2 with shared weights
        e3_2 = self.enc3(p2_2); p3_2 = self.pool(e3_2)  # Encode x2 at level 3 with shared weights
        e4_2 = self.enc4(p3_2); p4_2 = self.pool(e4_2)  # Encode x2 at level 4 with shared weights

        # ---- bottleneck ----
        b = self.bottleneck(torch.cat([p4_1, p4_2], dim=1))  # Concatenate both branch outputs along channel axis and process through the bottleneck

        # ---- decoder ----
        d = self.dec4(torch.cat([self.up4(b),  e4_1, e4_2], dim=1))  # Upsample bottleneck, concatenate with level-4 skip features from both branches, decode
        d = self.dec3(torch.cat([self.up3(d),  e3_1, e3_2], dim=1))  # Upsample, concatenate with level-3 skip features from both branches, decode
        d = self.dec2(torch.cat([self.up2(d),  e2_1, e2_2], dim=1))  # Upsample, concatenate with level-2 skip features from both branches, decode
        d = self.dec1(torch.cat([self.up1(d),  e1_1, e1_2], dim=1))  # Upsample, concatenate with level-1 skip features from both branches, decode

        return torch.sigmoid(self.head(d))  # Apply 1x1 conv then sigmoid to produce pixel-wise change probabilities in [0, 1]
