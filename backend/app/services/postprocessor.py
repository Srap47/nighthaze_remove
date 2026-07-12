"""
Postprocessor service — final enhancement stage.

Polishes the recovered image: local contrast (CLAHE), light denoising, unsharp
masking, and a saturation boost to counter the desaturation typical of nighttime
dehazing. All tunable parameters come from :class:`~app.config.Settings`.

Module contract: INPUT = uint8 BGR. OUTPUT = uint8 BGR.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings


class Postprocessor:
    """Applies the final cosmetic enhancement chain to a dehazed image."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance ``image`` (uint8 BGR) and return uint8 BGR.

        Order: CLAHE (LAB L-channel) → colour denoising → unsharp mask →
        HSV saturation boost.
        """
        settings = self.settings

        # Step 1: CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel
        # Enhances local contrast in LAB colorspace (L = brightness, independent of color)
        # Prevents color shifts and halo artifacts that global histogram equalization causes
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # TWEAK NOTE: clahe_clip_limit controls contrast amplification strength (from settings)
        # Higher value (e.g., 3.0) = more contrast boost; lower (e.g., 1.0) = subtle enhancement
        # TWEAK NOTE: clahe_tile_size divides the image into tiles for local processing
        # Larger tiles = smoother result; smaller tiles = more local detail enhancement
        clahe = cv2.createCLAHE(
            clipLimit=settings.clahe_clip_limit,
            tileGridSize=(settings.clahe_tile_size, settings.clahe_tile_size),
        )
        l_enhanced = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

        # Step 2: Light colour denoising (Non-Local Means Denoising)
        # Reduces noise by averaging similar patches across the image (preserves edges better than Gaussian blur)
        # h parameter controls filtering strength; higher h = more aggressive denoising (softer image)
        image = cv2.fastNlMeansDenoisingColored(
            image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
        )

        # Step 3: Unsharp masking (high-pass sharpening)
        # Technique: subtract blurred version from original to enhance edges and fine details
        # TWEAK NOTE: sharpen_alpha (from settings) controls contribution of original image
        # Higher alpha = more sharpening; lower = more subtle
        # TWEAK NOTE: sharpen_beta (from settings) controls blurred version contribution (usually negative)
        # More negative beta = stronger sharpening effect
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        image = cv2.addWeighted(
            image, settings.sharpen_alpha,
            blurred, settings.sharpen_beta,
            0,
        )
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Step 4: Saturation boost (nighttime dehazing tends to wash out colour).
        # Enhance color vibrancy by amplifying the S (Saturation) channel in HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        # TWEAK NOTE: saturation_boost multiplier (from settings) scales saturation intensity
        # Values > 1.0 increase color vibrancy; < 1.0 desaturate (mute colors)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * settings.saturation_boost, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return image
