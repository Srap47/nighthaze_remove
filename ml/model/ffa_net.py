"""FFA-Net: Feature Fusion Attention Network for Single Image Dehazing.

Paper: Qin, X., Wang, Z., Bai, Y., Xie, X., & Jia, H. (2020). AAAI 2020.
Official implementation: https://github.com/zhilin007/FFA-Net
Pretrained weights: its_train_ffa_3_19.pkl
  - gps=3 (3 groups/gates)
  - blocks=19 (19 blocks per group)
  - Trained on RESIDE ITS dataset (synthetic indoor haze)

Architecture highlights:
- Multi-group structure (3 parallel feature extraction paths)
- Gate-based feature fusion (learns to weight each group)
- Attention at multiple levels (pixel and channel)
- Residual connection from input to output (skip dehazing if not needed)

CRITICAL: Architecture must match official repo exactly for load_state_dict() to work
with pretrained weights. Any changes break checkpoint compatibility.
"""

import torch
import torch.nn as nn

from .blocks import default_conv, PALayer, CALayer, Group


class FFA(nn.Module):
    """Feature Fusion Attention Network for image dehazing.

    Architecture overview:
    1. Pre-process: RGB → feature (3 → 64 channels)
    2. Three parallel groups (g1, g2, g3) extract features independently
    3. Gate (ca): learns to weight/fuse outputs from all 3 groups
    4. Pixel attention (palayer): refines the fused features
    5. Post-process: features → RGB + residual skip from input

    Key design: multiple extraction paths with learned fusion prevents
    mode collapse and allows the network to capture diverse features.
    """
    def __init__(self, gps, blocks, conv=default_conv):
        """Initialize FFA-Net architecture.

        Args:
            gps: Number of gate/group branches (typically 3). Must be 3 for pretrained weights.
            blocks: Number of Block units per Group (typically 19). Affects model capacity.
            conv: Convolution function to use (default: same-padding Conv2d)

        TWEAK NOTES:
        - gps must be 3 to match pretrained weight structure
        - blocks controls model depth; higher = more capacity but slower inference
        - dim=64 (internal feature channels) is fixed in pretrained weights
        """
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64  # Internal feature channels (fixed for pretrained weights)
        kernel_size = 3

        # Pre-processing: Map RGB input (3 channels) to feature space (64 channels)
        pre_process = [conv(3, self.dim, kernel_size)]

        # Sanity check: pretrained weights assume gps=3
        assert self.gps == 3, f"Pretrained weights expect gps=3, got {self.gps}"

        # Three parallel feature extraction paths (each stack of N blocks + convolutions)
        # Each group independently processes features through multiple residual blocks
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)

        # Gate fusion: learns per-group weights to combine outputs from g1, g2, g3
        # Input: concatenated features (64*3=192 channels)
        # Output: gps weights (one per group, normalized by sigmoid)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),  # Bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()  # Normalize weights to [0,1]
        )

        # Pixel attention on the fused features (refines spatial importance)
        self.palayer = PALayer(self.dim)

        # Post-processing: Map features back to RGB (64 → 64 → 3 channels)
        post_process = [
            conv(self.dim, self.dim, kernel_size),  # Refine features
            conv(self.dim, 3, kernel_size)          # Map to RGB output
        ]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        """Process image through FFA-Net.

        Args:
            x1: Input image tensor (batch, 3, H, W) with values in [0,1]

        Returns:
            Dehazed image tensor (batch, 3, H, W) with same range as input
            (includes residual skip from input, so output ≈ input if no dehaze needed)
        """
        # Pre-process: RGB → feature space
        x = self.pre(x1)  # (batch, 64, H, W)

        # Extract features through three parallel groups
        res1 = self.g1(x)  # Group 1 output: (batch, 64, H, W)
        res2 = self.g2(res1)  # Group 2 output: (batch, 64, H, W)
        res3 = self.g3(res2)  # Group 3 output: (batch, 64, H, W)

        # Gate-based fusion: learn weights for each group's contribution
        # Concatenate all group outputs
        w = self.ca(torch.cat([res1, res2, res3], dim=1))  # (batch, 192, 1, 1)
        # Reshape weights: (batch, 3, 64, 1, 1) → broadcast to each group
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]

        # Weighted sum of group outputs: w[0]*res1 + w[1]*res2 + w[2]*res3
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3

        # Apply pixel attention to refined features
        out = self.palayer(out)

        # Post-process: feature space → RGB
        x = self.post(out)  # (batch, 3, H, W)

        # Residual connection: output = dehazed + input
        # If dehaze is not needed, network can learn to output near-zero correction
        return x + x1
