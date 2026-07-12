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
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        dehazed_rgb = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)

        # 1. PSNR (Peak Signal-to-Noise Ratio).
        psnr = peak_signal_noise_ratio(orig_rgb, dehazed_rgb, data_range=255)

        # 2. SSIM (Structural Similarity Index).
        # scikit-image 0.22: use channel_axis=2 (the old multichannel= was removed).
        ssim = structural_similarity(
            orig_rgb, dehazed_rgb, channel_axis=2, data_range=255
        )

        # 3. BRISQUE (no-reference; lower = better). Never let it crash the pipeline.
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

        # 4. NIQE (simplified, computed on the dehazed image; lower = better).
        niqe_score = self._compute_niqe(dehazed)

        # 5. Visibility score — how much haze was removed (higher = more).
        orig_float = original.astype(np.float32) / 255.0
        dark_ch = orig_float.min(axis=2)
        trans_approx = 1.0 - 0.95 * dark_ch
        visibility_score = float(1.0 - trans_approx.mean())

        # 6. Colorfulness (Hasler & Susstrunk, 2003) + improvement.
        cf_before = self._colorfulness(original)
        cf_after = self._colorfulness(dehazed)
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
        """Hasler & Susstrunk colorfulness metric for a uint8 BGR image."""
        B, G, R = cv2.split(image.astype(np.float32))
        rg = R - G
        yb = 0.5 * (R + G) - B
        std_rg, mean_rg = rg.std(), rg.mean()
        std_yb, mean_yb = yb.std(), yb.mean()
        return float(
            np.sqrt(std_rg ** 2 + std_yb ** 2)
            + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        )

    def _compute_niqe(self, image: np.ndarray) -> float:
        """Simplified NIQE via MSCN-patch kurtosis deviation from 3.0.

        Lower is more natural. Natural images have MSCN kurtosis near 3.0
        (Gaussian); deviation is averaged over 32x32 patches.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        patch_size = 32
        scores: list[float] = []
        for y in range(0, gray.shape[0] - patch_size, patch_size):
            for x in range(0, gray.shape[1] - patch_size, patch_size):
                patch = gray[y:y + patch_size, x:x + patch_size]
                mu = patch.mean()
                sigma = patch.std() + 1e-6
                mscn = (patch - mu) / sigma  # Mean-Subtracted Contrast Normalized
                kurtosis = float(np.mean(mscn ** 4))
                scores.append(abs(kurtosis - 3.0))
        return float(np.mean(scores)) if scores else 0.0
