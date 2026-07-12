"""Building blocks for the FFA-Net architecture.

Defines reusable components:
- default_conv: Standard 2D convolution with automatic padding
- PALayer: Pixel Attention (per-pixel importance weighting)
- CALayer: Channel Attention (per-channel importance weighting)
- Block: FFA basic unit (conv → attention → residual)
- Group: Stack of blocks with long skip connection
"""

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """Create a Conv2d layer with same-padding (preserves spatial dimensions).

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel (e.g., 3)
        bias: Whether to include bias term

    Returns:
        nn.Conv2d with padding=(kernel_size // 2) for same-padding
        E.g., kernel_size=3 → padding=1 preserves spatial dimensions
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias
    )


class PALayer(nn.Module):
    """Pixel Attention Layer — learns per-pixel importance weights.

    Produces a spatial attention map (same spatial dimensions as input, 1 channel).
    Uses a small bottleneck MLP: channel → channel//8 → 1.
    Output: weights [0,1] (via Sigmoid) per spatial location.
    The attention map reweights each pixel independently (emphasizing relevant regions).
    """
    def __init__(self, channel):
        super(PALayer, self).__init__()
        # Bottleneck: reduce channel dimension to channel//8 for efficiency
        # Output: single attention map (1 channel) at same spatial resolution
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()  # Normalize attention weights to [0, 1]
        )

    def forward(self, x):
        """Apply pixel attention: multiply input by spatial attention map.

        Args:
            x: Tensor of shape (batch, channel, height, width)

        Returns:
            Element-wise product of x and spatial attention map (same shape as x)
        """
        y = self.pa(x)  # Shape: (batch, 1, height, width)
        return x * y    # Broadcast multiply; each pixel reweighted


class CALayer(nn.Module):
    """Channel Attention Layer — learns per-channel importance weights.

    Produces a channel attention vector (1 weight per channel).
    Uses global average pooling followed by bottleneck MLP.
    Output: weights [0,1] (via Sigmoid) per channel.
    The attention vector reweights each feature channel (emphasizing informative channels).
    """
    def __init__(self, channel):
        super(CALayer, self).__init__()
        # Global average pooling: (batch, channel, H, W) → (batch, channel, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Bottleneck MLP: channel → channel//8 → channel
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()  # Normalize attention weights to [0, 1]
        )

    def forward(self, x):
        """Apply channel attention: multiply input by per-channel weights.

        Args:
            x: Tensor of shape (batch, channel, height, width)

        Returns:
            Element-wise product of x and per-channel attention vector (same shape as x)
        """
        y = self.avg_pool(x)  # Global avg pool: (batch, channel, 1, 1)
        y = self.ca(y)        # Channel attention weights: (batch, channel, 1, 1)
        return x * y          # Broadcast multiply; each channel reweighted


class Block(nn.Module):
    """FFA basic block: residual conv path + attention gates.

    Architecture:
    1. Conv → ReLU (first conv + activation)
    2. Add residual (shortcut connection from input)
    3. Conv (second conv)
    4. Channel Attention (CA layer)
    5. Pixel Attention (PA layer)
    6. Add residual again (output skipped from input)

    The dual-branch design (main path + attention) allows the network
    to learn what to preserve and what to enhance at each stage.
    """
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        # Main path: two convolutions with activation and residual
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        # Attention gates: channel-wise and pixel-wise
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        """Apply block transformation with attention.

        Flow:
        res = conv1(x) → relu → + x (residual)
        res = conv2(res) → CA(res) → PA(res) → + x (residual)
        """
        # First conv path: conv → activation → add skip
        res = self.act1(self.conv1(x))
        res = res + x
        # Second conv path: conv → attention gates → add skip
        res = self.conv2(res)
        res = self.calayer(res)  # Channel attention
        res = self.palayer(res)  # Pixel attention
        res += x  # Final skip connection (input bypasses this block)
        return res


class Group(nn.Module):
    """Group of N Blocks followed by a final conv, with long skip connection.

    A hierarchical building block: stacks multiple Blocks and adds
    a final convolution. The entire group is surrounded by a skip connection,
    allowing features to bypass the group (enables very deep networks).

    Architecture:
    - N × Block (multi-block stack)
    - 1 × Conv (final convolution)
    - Add long skip (output + input)
    """
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        # Build sequence: N blocks + 1 final conv
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))  # Final conv
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        """Apply group transformation with long skip connection.

        Flow:
        res = gp(x) (N blocks + final conv)
        res = res + x (long skip)
        """
        res = self.gp(x)  # Process through all blocks + final conv
        res += x          # Long skip: bypass entire group
        return res
