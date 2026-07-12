"""
Glow detector service.

Detects artificial light sources (street lamps, neon signs, headlights) and
their glow halos using classical computer vision — no deep learning here.

Module contract: INPUT = float32 [0,1] BGR. OUTPUT = :class:`GlowDetectionResult`.
Internally the image is briefly denormalised to uint8 for OpenCV colour/analysis
ops, but every value returned to the pipeline respects the float32 [0,1] contract
(the glow mask is float32 [0,1]; only ``glow_mask_vis`` is uint8 for display).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings
from app.services import image_utils


@dataclass
class GlowRegion:
    """A single detected light source and its estimated local atmospheric light."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    center: tuple[int, int]          # (cx, cy)
    local_atm_light: np.ndarray      # shape (3,) float32 [0,1], BGR
    intensity: float                 # blurred-brightness of the core [0,1]


@dataclass
class GlowDetectionResult:
    """Output of :meth:`GlowDetector.detect`."""

    glow_mask: np.ndarray       # float32 [0,1], H x W — 1.0 inside glow regions
    glow_mask_vis: np.ndarray   # uint8 BGR — green overlay visualization
    glow_regions: list[GlowRegion]
    num_sources: int


class GlowDetector:
    """Locates bright light sources and their halos in a nighttime image."""

    def __init__(self, settings: Settings) -> None:
        # Detection constants are algorithm-specific (not user-tunable); settings
        # is retained for a uniform service-construction interface.
        self.settings = settings

    def detect(self, image: np.ndarray) -> GlowDetectionResult:
        """Detect light sources and glow halos.

        Args:
            image: float32 [0,1] BGR image.

        Returns:
            A :class:`GlowDetectionResult`. For a pitch-dark image with no light
            sources, returns an empty region list and an all-zero mask.
        """
        H, W = image.shape[:2]
        img_uint8 = image_utils.denormalize(image)

        # 1. Brightness (V) channel in [0,1].
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32) / 255.0

        # 2. Blur to approximate glow spread.
        v_blurred = cv2.GaussianBlur(v_channel, (21, 21), 0)

        # Edge case: pitch-dark frame → no meaningful light sources.
        max_brightness = float(v_blurred.max())
        if max_brightness <= 1e-6:
            return GlowDetectionResult(
                glow_mask=np.zeros((H, W), dtype=np.float32),
                glow_mask_vis=img_uint8.copy(),
                glow_regions=[],
                num_sources=0,
            )

        # 3. Threshold at 85% of peak brightness.
        threshold = 0.85 * max_brightness
        light_mask = (v_blurred > threshold).astype(np.uint8) * 255

        # 4. Dilate with an elliptical kernel to capture the halo.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        glow_mask_uint8 = cv2.dilate(light_mask, kernel)

        # 5. Connected components = individual light sources.
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            light_mask, connectivity=8
        )

        glow_regions: list[GlowRegion] = []
        for i in range(1, num_labels):  # label 0 is background
            x, y, w, h, area = stats[i]
            if area < 50:  # ignore tiny noise blobs
                continue

            cx = min(max(int(centroids[i][0]), 0), W - 1)
            cy = min(max(int(centroids[i][1]), 0), H - 1)

            # Expanded ROI (2x bounding box, clamped to the image).
            ex = max(0, x - w // 2)
            ey = max(0, y - h // 2)
            ew = min(W, x + w + w // 2) - ex
            eh = min(H, y + h + h // 2) - ey
            roi = image[ey:ey + eh, ex:ex + ew]
            if roi.size == 0:
                continue

            # Local atmospheric light = mean of the top 0.1% brightest ROI pixels.
            roi_gray = cv2.cvtColor(image_utils.denormalize(roi), cv2.COLOR_BGR2GRAY)
            flat_brightness = roi_gray.flatten()
            top_pct_count = max(1, int(len(flat_brightness) * 0.001))
            top_indices = np.argsort(flat_brightness)[-top_pct_count:]
            roi_flat = roi.reshape(-1, 3)
            local_atm_light = roi_flat[top_indices].mean(axis=0).astype(np.float32)

            glow_regions.append(
                GlowRegion(
                    bbox=(int(x), int(y), int(w), int(h)),
                    center=(cx, cy),
                    local_atm_light=local_atm_light,
                    intensity=float(v_blurred[cy, cx]),
                )
            )

        # 6. Float32 [0,1] glow mask.
        glow_mask = (glow_mask_uint8 / 255.0).astype(np.float32)

        # 7. Green-overlay visualization on the original.
        vis = img_uint8.copy()
        vis[glow_mask_uint8 > 0] = (0, 200, 0)
        glow_mask_vis = cv2.addWeighted(img_uint8, 0.7, vis, 0.3, 0)

        return GlowDetectionResult(
            glow_mask=glow_mask,
            glow_mask_vis=glow_mask_vis,
            glow_regions=glow_regions,
            num_sources=len(glow_regions),
        )
