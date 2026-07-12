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
        # Compute dark channel prior (He et al. single image dehazing) for regions without glow
        # TWEAK NOTE: patch_size (15) controls the window for dark channel computation
        # Larger patch = smoother atmospheric estimation; smaller patch = more detail-sensitive
        dark_channel = self._compute_dark_channel(image, glow_mask, patch_size=15)

        # Select top 0.1% brightest pixels in dark channel as candidates for atmospheric light
        num_pixels = H * W
        # TWEAK NOTE: Percentile (0.001 = top 0.1%) affects atmospheric light estimate quality
        # Higher percentile (e.g., 0.01) = more conservative, smoother; lower = purer peak value
        top_n = max(1, int(num_pixels * 0.001))  # top 0.1% brightest in dark channel
        flat_dark = dark_channel.flatten()
        top_indices = np.argsort(flat_dark)[-top_n:]
        rows = top_indices // W
        cols = top_indices % W
        candidates = image[rows, cols]                      # (top_n, 3)
        # Pick the candidate with highest max channel value (most saturated/brightest)
        brightest_idx = int(np.argmax(candidates.max(axis=1)))
        A_global = candidates[brightest_idx].astype(np.float32)  # (3,)

        # --- Per-pixel atmospheric light map ---
        # Initialize with global atmospheric light; will be blended with local light source colors
        atm_map = np.ones((H, W, 3), dtype=np.float32) * A_global

        # For each detected glow region, blend its local light color into the map
        for region in glow_regions:
            x, y, w, h = region.bbox
            cx, cy = region.center
            # Radius defines the blend falloff distance (distance from source center)
            radius = max(w, h) / 2.0
            if radius <= 0:
                continue

            # Compute Euclidean distance from each pixel to the light source center
            y_coords, x_coords = np.mgrid[y:y + h, x:x + w]
            distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            # Radial blend weight: 1.0 at the source centre, fading to 0.0 at ``radius``.
            # TWEAK NOTE: Radius-to-distance falloff controls how local light spreads
            # Steeper falloff (direct distance) preserves localization; smoother blending softens edges
            blend_weight = np.clip(1.0 - distances / radius, 0.0, 1.0).astype(np.float32)

            # Update atmospheric map: interpolate between local glow light and global light
            region_slice = atm_map[y:y + h, x:x + w]  # view into atm_map
            local_light = region.local_atm_light
            for c in range(3):
                region_slice[:, :, c] = (
                    local_light[c] * blend_weight
                    + A_global[c] * (1.0 - blend_weight)
                )

        # --- Transmission visualization (estimated from the dark channel) ---
        # Transmission t(x) estimates the fraction of light reaching the camera
        # Higher transmission = clearer, less haze; lower = more haze/fog
        # TWEAK NOTE: omega (from settings) weights the dark channel's influence on transmission
        # Higher omega (default 0.95) = stronger dark channel influence; lower = more conservative
        transmission = 1.0 - self.settings.omega * dark_channel
        # TWEAK NOTE: transmission_min_clip (from settings) sets minimum transmission floor
        # Prevents overestimating clarity in severely hazy regions
        transmission = np.clip(transmission, self.settings.transmission_min_clip, 1.0)
        # Convert to uint8 visualization (grayscale: brighter = clearer, darker = hazier)
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
