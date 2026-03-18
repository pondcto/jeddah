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

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv 3×3 → BN → ReLU  ×2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SiameseUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # ---------- shared encoder ----------
        self.enc1 = _ConvBlock(in_channels, 64)
        self.enc2 = _ConvBlock(64, 128)
        self.enc3 = _ConvBlock(128, 256)
        self.enc4 = _ConvBlock(256, 512)

        # ---------- bottleneck (fused) ----------
        self.bottleneck = _ConvBlock(512 * 2, 1024)

        # ---------- decoder ----------
        # Each level: upsample + cat(skip_t1, skip_t2) → conv block
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _ConvBlock(512 + 512 * 2, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(256 + 256 * 2, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(128 + 128 * 2, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(64 + 64 * 2, 64)

        self.head = nn.Conv2d(64, out_channels, kernel_size=1)
        self._init_weights()

    # -- weight init -------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # -- forward ------------------------------------------------------------
    def forward(self, x1, x2):
        # ---- encoder branch 1 ----
        e1_1 = self.enc1(x1);  p1_1 = self.pool(e1_1)
        e2_1 = self.enc2(p1_1); p2_1 = self.pool(e2_1)
        e3_1 = self.enc3(p2_1); p3_1 = self.pool(e3_1)
        e4_1 = self.enc4(p3_1); p4_1 = self.pool(e4_1)

        # ---- encoder branch 2 (shared weights) ----
        e1_2 = self.enc1(x2);  p1_2 = self.pool(e1_2)
        e2_2 = self.enc2(p1_2); p2_2 = self.pool(e2_2)
        e3_2 = self.enc3(p2_2); p3_2 = self.pool(e3_2)
        e4_2 = self.enc4(p3_2); p4_2 = self.pool(e4_2)

        # ---- bottleneck ----
        b = self.bottleneck(torch.cat([p4_1, p4_2], dim=1))

        # ---- decoder ----
        d = self.dec4(torch.cat([self.up4(b),  e4_1, e4_2], dim=1))
        d = self.dec3(torch.cat([self.up3(d),  e3_1, e3_2], dim=1))
        d = self.dec2(torch.cat([self.up2(d),  e2_1, e2_2], dim=1))
        d = self.dec1(torch.cat([self.up1(d),  e1_1, e1_2], dim=1))

        return torch.sigmoid(self.head(d))
