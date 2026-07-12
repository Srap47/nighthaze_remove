"""Standalone image-quality metrics library.

Deliberately free of any application (FastAPI/app.*) imports so these can be
used directly from notebooks, scripts, or evaluation harnesses:

    from ml.evaluation.metrics import compute_psnr, compute_ssim

All functions take uint8 BGR images (OpenCV convention) unless noted.
Provides both full-reference metrics (comparing to reference image) and
no-reference metrics (measuring properties independently).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB. Higher is better. (Full-reference metric)

    Measures pixel-level accuracy: penalizes any deviation from the reference.
    Very sensitive to small shifts; doesn't always correlate with perceptual quality.
    Typical range: 15-40 dB; >25 dB is good, >30 dB is excellent.

    Args:
        original: uint8 BGR reference image (ground truth / expected output)
        processed: uint8 BGR image to score (actual output)

    Returns:
        PSNR score in decibels (dB); higher is better
    """
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return float(
        peak_signal_noise_ratio(original_rgb, processed_rgb, data_range=255)
    )


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Structural Similarity Index. Range [-1, 1]. Higher is better. (Full-reference metric)

    Measures perceptual similarity by comparing luminance, contrast, and structure.
    More closely matches human perception than PSNR.
    Typical range: 0.5-1.0 for natural images; >0.9 is excellent.

    Args:
        original: uint8 BGR reference image (ground truth / expected output)
        processed: uint8 BGR image to score (actual output)

    Returns:
        SSIM score in [-1, 1]; higher is better (1.0 = identical)

    Note:
        Uses channel_axis parameter (scikit-image >= 0.19); older versions used
        multichannel (now removed). Automatically handles multi-channel images.
    """
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return float(
        structural_similarity(
            original_rgb, processed_rgb, channel_axis=2, data_range=255
        )
    )


def compute_colorfulness(image: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness metric. Higher = more colorful. (No-reference metric)

    Quantifies color saturation using opponent color spaces (rg and yb channels).
    Useful for detecting desaturation caused by dehazing (a common artifact).
    Works independently; doesn't require a reference image.

    The metric combines:
    - Saturation variance (how spread out colors are)
    - Mean magnitude (how strong the colors are)

    Args:
        image: uint8 BGR image

    Returns:
        Colorfulness score (higher = more saturated/vibrant colors)
        Typical range: 0-100+ depending on image content
    """
    blue, green, red = cv2.split(image.astype(np.float32))
    # Opponent color channels: red-green axis, yellow-blue axis (human perception)
    rg = red - green
    yb = 0.5 * (red + green) - blue

    # Calculate saturation statistics in opponent space
    # Variance term (0.3 weight on mean) captures both saturation and color bias
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def compute_niqe_simplified(image: np.ndarray, patch_size: int = 32) -> float:
    """Simplified NIQE (Natural Image Quality Evaluator) naturalness score. Lower = more natural. (No-reference metric)

    Measures deviation from natural image statistics by analyzing patch-wise
    MSCN (Mean-Subtracted Contrast Normalized) coefficients. Natural images
    have MSCN distributions close to Gaussian (kurtosis ≈ 3.0); deviations
    indicate artifacts, distortion, or unnatural content.

    Useful for detecting:
    - Over-sharpening (high kurtosis)
    - Blocking/compression artifacts (high kurtosis)
    - Over-smoothing (high kurtosis in texture regions)
    - Realistic content (kurtosis near 3.0)

    Args:
        image: uint8 BGR image
        patch_size: Side length of non-overlapping square patches (32 = efficient default)

    Returns:
        NIQE score (average |kurtosis - 3| across patches)
        Lower is better (closer to 0 = more natural)
        Typical range: 0-2 for natural images; >3 suggests heavy artifacts
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    scores: list[float] = []
    # Tile image with non-overlapping patches
    for y in range(0, gray.shape[0] - patch_size, patch_size):
        for x in range(0, gray.shape[1] - patch_size, patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            # Normalize patch: subtract mean, divide by standard deviation
            mu = patch.mean()
            sigma = patch.std() + 1e-6  # Small epsilon prevents division by zero
            mscn = (patch - mu) / sigma  # MSCN-normalized coefficients
            # Kurtosis = E[x^4] where x is MSCN (Gaussian has kurtosis = 3.0)
            kurtosis = float(np.mean(mscn**4))
            # Track deviation from natural kurtosis baseline
            scores.append(abs(kurtosis - 3.0))

    return float(np.mean(scores)) if scores else 0.0
