"""
Standalone image-quality metrics.

Deliberately free of any application (FastAPI/app.*) imports so these can be
used directly from notebooks, scripts, or evaluation harnesses:

    from ml.evaluation.metrics import compute_psnr, compute_ssim

All functions take uint8 BGR images (OpenCV convention) unless noted.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB. Higher is better.

    Args:
        original: uint8 BGR reference image.
        processed: uint8 BGR image to score.
    """
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return float(
        peak_signal_noise_ratio(original_rgb, processed_rgb, data_range=255)
    )


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Structural Similarity Index in [-1, 1]. Higher is better.

    Uses ``channel_axis`` (scikit-image >= 0.19); the old ``multichannel``
    keyword was removed.
    """
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return float(
        structural_similarity(
            original_rgb, processed_rgb, channel_axis=2, data_range=255
        )
    )


def compute_colorfulness(image: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness metric. Higher is more colorful.

    Args:
        image: uint8 BGR image.
    """
    blue, green, red = cv2.split(image.astype(np.float32))
    rg = red - green
    yb = 0.5 * (red + green) - blue

    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def compute_niqe_simplified(image: np.ndarray, patch_size: int = 32) -> float:
    """Simplified NIQE-style naturalness score. Lower is more natural.

    Measures how far the MSCN (Mean-Subtracted Contrast-Normalized) coefficients
    of each patch deviate from a Gaussian: natural images have MSCN kurtosis
    near 3.0, so the score is the mean |kurtosis - 3|.

    Args:
        image: uint8 BGR image.
        patch_size: Side length of the square analysis patches.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    scores: list[float] = []
    for y in range(0, gray.shape[0] - patch_size, patch_size):
        for x in range(0, gray.shape[1] - patch_size, patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            mu = patch.mean()
            sigma = patch.std() + 1e-6
            mscn = (patch - mu) / sigma
            kurtosis = float(np.mean(mscn**4))
            scores.append(abs(kurtosis - 3.0))

    return float(np.mean(scores)) if scores else 0.0
