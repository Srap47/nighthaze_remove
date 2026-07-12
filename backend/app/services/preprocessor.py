"""
Preprocessor service — first pipeline stage.

Module contract: INPUT = raw uint8 BGR image. OUTPUT = float32 [0,1] BGR plus
metadata (:class:`PreprocessResult`).

Recall the global contract: pipeline-internal images are float32 BGR in [0,1];
uint8 is used only at I/O boundaries. This stage *crosses* that boundary — it
receives uint8, does its work in uint8, and emits float32 [0,1] for the rest of
the pipeline. It validates the image, normalises channel layout, tames sensor
noise (adaptive bilateral filtering), and lifts very dark regions (selective
gamma) before normalising.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings
from app.services import image_utils


@dataclass
class PreprocessResult:
    """Output of :meth:`Preprocessor.prepare`."""

    image: np.ndarray               # float32 [0,1] BGR, ready for the pipeline
    original_size: tuple[int, int]  # (H, W) before any resizing
    scale_factor: float             # applied resize factor (1.0 if unchanged)
    noise_level: str                # 'low' | 'medium' | 'high'
    noise_value: float              # raw Laplacian variance


class Preprocessor:
    """Validates and conditions a raw image for the dehazing pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare(self, image: np.ndarray) -> PreprocessResult:
        """Validate, denoise, gamma-correct and normalise ``image``.

        Args:
            image: Raw uint8 image (grayscale, BGR, or BGRA).

        Returns:
            A :class:`PreprocessResult` whose ``image`` is float32 [0,1] BGR.
        """
        image_utils.validate_image(image, self.settings)
        image = image_utils.ensure_bgr(image)
        original_size = (int(image.shape[0]), int(image.shape[1]))

        image, scale_factor = image_utils.resize_maintain_aspect(
            image, self.settings.max_image_dimension
        )

        noise_value = self._estimate_noise(image)
        # Lower Laplacian variance ⇒ blurrier/noisier ⇒ stronger filtering.
        # TWEAK NOTE: Thresholds (50, 200) control noise classification; adjust if image quality changes
        if noise_value < 50:
            noise_level = "high"
        elif noise_value < 200:
            noise_level = "medium"
        else:
            noise_level = "low"

        # TWEAK NOTE: Bilateral filter diameter (d) and sigma values control denoising strength
        # Larger d and sigma = more aggressive smoothing (slower but less noise)
        # High noise: d=9, sigmaColor=75, sigmaSpace=75 (aggressive)
        # Medium noise: d=5, sigmaColor=50, sigmaSpace=50 (moderate)
        if noise_level == "high":
            image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        elif noise_level == "medium":
            image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

        image = self._apply_gamma_correction(image)
        image = image_utils.normalize(image)  # → float32 [0,1]

        return PreprocessResult(
            image=image,
            original_size=original_size,
            scale_factor=scale_factor,
            noise_level=noise_level,
            noise_value=noise_value,
        )

    def _estimate_noise(self, image: np.ndarray) -> float:
        """Return the Laplacian variance of ``image`` as a noise/blur proxy."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Selectively brighten very dark regions (luminance < 30) with gamma 0.7."""
        img_float = image.astype(np.float32) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # TWEAK NOTE: dark_mask threshold (30) isolates very dark pixels for brightening
        # Lower threshold = target darker regions; higher = broader application
        dark_mask = (gray < 30).astype(np.float32)
        # out = in^(1/gamma), gamma = 0.7 → brightens shadows.
        # TWEAK NOTE: gamma value (0.7) controls shadow brightening strength
        # Lower gamma (e.g., 0.6) brightens more; higher (e.g., 0.8) brightens less
        gamma_corrected = np.power(np.clip(img_float, 1e-6, 1.0), 1.0 / 0.7)
        # Blend: apply gamma correction only to dark regions, preserve bright areas
        result = (
            img_float * (1 - dark_mask[:, :, None])
            + gamma_corrected * dark_mask[:, :, None]
        )
        return np.clip(result * 255, 0, 255).astype(np.uint8)
