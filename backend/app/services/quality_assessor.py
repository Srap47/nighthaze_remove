"""
Quality assessor service.

Computes image-quality metrics — both full-reference (PSNR, SSIM) and
no-reference (BRISQUE, a simplified NIQE) — plus haze-visibility and
colorfulness measures.

Module contract: INPUT = two uint8 BGR images (original, dehazed).
OUTPUT = :class:`~app.models.schemas.DehazeMetrics`.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from app.config import Settings
from app.models.schemas import DehazeMetrics

logger = logging.getLogger(__name__)


class QualityAssessor:
    """Scores a dehazed image against its hazy original."""

    def __init__(self, settings: Settings | None = None) -> None:
        # Metrics use fixed formulas; settings is accepted for a uniform
        # service-construction interface but is optional.
        self.settings = settings

    def score(self, original: np.ndarray, dehazed: np.ndarray) -> DehazeMetrics:
        """Compute quality metrics comparing ``original`` (hazy) to ``dehazed``.

        Args:
            original: uint8 BGR hazy input image.
            dehazed: uint8 BGR dehazed output image.

        Returns:
            A populated :class:`DehazeMetrics` (``processing_time_ms`` left at
            0.0 for the pipeline to fill in).
        """
        # Convert BGR to RGB for scikit-image metrics (they expect RGB channel order)
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        dehazed_rgb = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)

        # 1. PSNR (Peak Signal-to-Noise Ratio) — full-reference metric
        # Compares hazy original to dehazed output; higher is better
        # Range typically 15-40 dB; >25 dB = good quality, >30 dB = excellent
        psnr = peak_signal_noise_ratio(orig_rgb, dehazed_rgb, data_range=255)

        # 2. SSIM (Structural Similarity Index) — full-reference metric
        # Measures structural/perceptual similarity; higher is better (range 0-1)
        # Incorporates luminance, contrast, and structure comparisons
        # scikit-image 0.22: use channel_axis=2 (the old multichannel= was removed).
        ssim = structural_similarity(
            orig_rgb, dehazed_rgb, channel_axis=2, data_range=255
        )

        # 3. BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) — no-reference metric
        # Lower is better; evaluates distortion without reference image
        # Gracefully handles failures (returns -1.0) to prevent pipeline crashes
        try:
            # Compatibility shim: brisque 0.0.15 references scipy.ndarray, an
            # alias removed in modern SciPy. Restore it before importing brisque.
            import scipy
            import numpy as _np

            if not hasattr(scipy, "ndarray"):
                scipy.ndarray = _np.ndarray

            from brisque import BRISQUE

            brisque_obj = BRISQUE(url=False)
            brisque_score = brisque_obj.score(dehazed_rgb)
        except Exception as exc:  # noqa: BLE001 — brisque can fail on odd inputs
            logger.warning("BRISQUE scoring failed (%s); returning -1.0.", exc)
            brisque_score = -1.0

        # 4. NIQE (Natural Image Quality Evaluator) — simplified, computed on dehazed image
        # Lower is better (closer to 0 = more natural); measures deviation from natural image statistics
        # Useful for detecting artifacts and unnatural distortions
        niqe_score = self._compute_niqe(dehazed)

        # 5. Visibility score — estimates atmospheric haze removal effectiveness
        # Uses dark channel prior: higher score = more haze removed (range 0-1)
        # Approximates transmission map (darker values = thicker haze)
        orig_float = original.astype(np.float32) / 255.0
        dark_ch = orig_float.min(axis=2)  # Minimum across RGB channels (dark channel prior)
        trans_approx = 1.0 - 0.95 * dark_ch  # 0.95 = omega parameter for transmission
        visibility_score = float(1.0 - trans_approx.mean())  # Mean transmission = haze level

        # 6. Colorfulness (Hasler & Susstrunk, 2003) — measures color vibrancy
        # Higher score = more saturated/vibrant colors
        # Computes statistics on rg (Red-Green) and yb (Yellow-Blue) opponent color channels
        cf_before = self._colorfulness(original)  # Hazy image colorfulness
        cf_after = self._colorfulness(dehazed)    # Dehazed image colorfulness
        # Calculate percentage improvement in color vibrancy
        improvement_pct = ((cf_after - cf_before) / (cf_before + 1e-6)) * 100

        return DehazeMetrics(
            psnr=round(float(psnr), 3),
            ssim=round(float(ssim), 4),
            niqe=round(float(niqe_score), 3),
            brisque=round(float(brisque_score), 3),
            visibility_score=round(float(visibility_score), 4),
            colorfulness_before=round(float(cf_before), 3),
            colorfulness_after=round(float(cf_after), 3),
            colorfulness_improvement_pct=round(float(improvement_pct), 2),
            processing_time_ms=0.0,  # filled in by the pipeline
        )

    def _colorfulness(self, image: np.ndarray) -> float:
        """Hasler & Susstrunk colorfulness metric for a uint8 BGR image.

        Measures color saturation using opponent color channels (rg and yb).
        Higher values = more saturated/vibrant colors; lower = more desaturated/grayscale.
        """
        B, G, R = cv2.split(image.astype(np.float32))
        # Opponent color channels: rg (red-green axis), yb (yellow-blue axis)
        rg = R - G  # Red-green differential
        yb = 0.5 * (R + G) - B  # Yellow-blue differential (perceptually weighted)
        # Extract mean and standard deviation for each opponent channel
        std_rg, mean_rg = rg.std(), rg.mean()
        std_yb, mean_yb = yb.std(), yb.mean()
        # Colorfulness = saturation variance + 0.3 * mean magnitude
        # (Heavier weight on variance for perceptual sensitivity)
        return float(
            np.sqrt(std_rg ** 2 + std_yb ** 2)
            + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        )

    def _compute_niqe(self, image: np.ndarray) -> float:
        """Simplified NIQE (Natural Image Quality Evaluator) via MSCN-patch kurtosis.

        Measures deviation from natural image statistics. Lower is better (more natural).
        Natural images have MSCN (Mean-Subtracted Contrast Normalized) kurtosis near 3.0
        (Gaussian distribution); artifacts cause deviation from this baseline.
        Averaged over non-overlapping 32x32 patches for computational efficiency.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        # TWEAK NOTE: patch_size (32) controls the local window for quality assessment
        # Larger patches = smoother estimate, less sensitive to fine details
        # Smaller patches = more granular, detects local artifacts better
        patch_size = 32
        scores: list[float] = []
        # Tile image with non-overlapping patches
        for y in range(0, gray.shape[0] - patch_size, patch_size):
            for x in range(0, gray.shape[1] - patch_size, patch_size):
                patch = gray[y:y + patch_size, x:x + patch_size]
                # Normalize patch: subtract mean, divide by standard deviation
                mu = patch.mean()
                sigma = patch.std() + 1e-6  # Small epsilon prevents division by zero
                mscn = (patch - mu) / sigma  # Mean-Subtracted Contrast Normalized
                # Kurtosis = E[x^4], where x is MSCN-normalized pixel values
                # Gaussian has kurtosis = 3.0; deviations indicate artifacts/distortion
                kurtosis = float(np.mean(mscn ** 4))
                # Track absolute deviation from 3.0 (target Gaussian kurtosis)
                scores.append(abs(kurtosis - 3.0))
        # Return mean deviation; lower = more natural image
        return float(np.mean(scores)) if scores else 0.0
