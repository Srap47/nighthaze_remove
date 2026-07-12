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

        # Step 1: CLAHE on the L channel for local contrast without over-brightening.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=settings.clahe_clip_limit,
            tileGridSize=(settings.clahe_tile_size, settings.clahe_tile_size),
        )
        l_enhanced = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

        # Step 2: Light colour denoising.
        image = cv2.fastNlMeansDenoisingColored(
            image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
        )

        # Step 3: Unsharp masking (sharpen).
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        image = cv2.addWeighted(
            image, settings.sharpen_alpha,
            blurred, settings.sharpen_beta,
            0,
        )
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Step 4: Saturation boost (nighttime dehazing tends to wash out colour).
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * settings.saturation_boost, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return image
