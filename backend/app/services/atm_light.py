"""
Atmospheric light estimator service.

Estimates atmospheric light *per region* rather than globally — critical for
nighttime scenes where light sources are non-uniform and localised.

Module contract: INPUT = float32 [0,1] BGR + glow mask/regions.
OUTPUT = :class:`AtmLightResult` (float32 maps; ``transmission_vis`` is uint8
BGR for display only).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings
from app.services import image_utils
from app.services.glow_detector import GlowRegion


@dataclass
class AtmLightResult:
    """Output of :meth:`AtmosphericLightEstimator.estimate`."""

    A_global: np.ndarray               # shape (3,) float32 — global atmospheric light
    atmospheric_light_map: np.ndarray  # shape H x W x 3 float32 — per-pixel atm light
    transmission_vis: np.ndarray       # uint8 BGR — grayscale transmission visualization


class AtmosphericLightEstimator:
    """Computes a per-pixel atmospheric light map for nighttime dehazing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def estimate(
        self,
        image: np.ndarray,
        glow_mask: np.ndarray,
        glow_regions: list[GlowRegion],
    ) -> AtmLightResult:
        """Estimate global and per-region atmospheric light.

        Args:
            image: float32 [0,1] BGR image.
            glow_mask: float32 [0,1] H x W mask of glow regions.
            glow_regions: Detected :class:`GlowRegion` objects (may be empty).

        Returns:
            An :class:`AtmLightResult`. If ``glow_regions`` is empty the map is a
            uniform ``A_global`` everywhere.
        """
        H, W = image.shape[:2]

        # --- Global atmospheric light (for non-glow regions) ---
        dark_channel = self._compute_dark_channel(image, glow_mask, patch_size=15)

        num_pixels = H * W
        top_n = max(1, int(num_pixels * 0.001))  # top 0.1% brightest in dark channel
        flat_dark = dark_channel.flatten()
        top_indices = np.argsort(flat_dark)[-top_n:]
        rows = top_indices // W
        cols = top_indices % W
        candidates = image[rows, cols]                      # (top_n, 3)
        brightest_idx = int(np.argmax(candidates.max(axis=1)))
        A_global = candidates[brightest_idx].astype(np.float32)  # (3,)

        # --- Per-pixel atmospheric light map ---
        atm_map = np.ones((H, W, 3), dtype=np.float32) * A_global

        for region in glow_regions:
            x, y, w, h = region.bbox
            cx, cy = region.center
            radius = max(w, h) / 2.0
            if radius <= 0:
                continue

            y_coords, x_coords = np.mgrid[y:y + h, x:x + w]
            distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            # 1.0 at the source centre, fading to 0.0 at ``radius``.
            blend_weight = np.clip(1.0 - distances / radius, 0.0, 1.0).astype(np.float32)

            region_slice = atm_map[y:y + h, x:x + w]  # view into atm_map
            local_light = region.local_atm_light
            for c in range(3):
                region_slice[:, :, c] = (
                    local_light[c] * blend_weight
                    + A_global[c] * (1.0 - blend_weight)
                )

        # --- Transmission visualization (estimated from the dark channel) ---
        transmission = 1.0 - self.settings.omega * dark_channel
        transmission = np.clip(transmission, self.settings.transmission_min_clip, 1.0)
        trans_vis = image_utils.denormalize(
            np.stack([transmission, transmission, transmission], axis=2)
        )

        return AtmLightResult(A_global, atm_map, trans_vis)

    def _compute_dark_channel(
        self, image: np.ndarray, glow_mask: np.ndarray, patch_size: int = 15
    ) -> np.ndarray:
        """Dark channel (per-patch min over BGR), with glow regions excluded."""
        min_channel = image.min(axis=2)               # per-pixel min across BGR
        min_channel = min_channel * (1.0 - glow_mask)  # zero out glow regions
        kernel = np.ones((patch_size, patch_size), np.uint8)
        dark = cv2.erode(min_channel.astype(np.float32), kernel)
        return dark
