"""
Radiance recovery service.

Applies the atmospheric scattering model inversion, refining the FFA-Net output
with the per-region atmospheric light map. Physics recovery can over-brighten
near light sources, so results are blended back toward the FFA-Net output inside
glow regions.

Module contract: INPUT = float32 [0,1] BGR (FFA output + atm map + glow mask).
OUTPUT = uint8 BGR (I/O boundary).
"""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.services import image_utils
from app.services.atm_light import AtmLightResult


class RadianceRecovery:
    """Physics-based refinement of the FFA-Net dehazed image."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def recover(
        self,
        ffa_dehazed: np.ndarray,
        atm_result: AtmLightResult,
        glow_mask: np.ndarray,
    ) -> np.ndarray:
        """Invert the scattering model and blend with the FFA output.

        Args:
            ffa_dehazed: float32 [0,1] BGR — FFA-Net output.
            atm_result: Atmospheric light estimation for this image.
            glow_mask: float32 [0,1] H x W glow-region mask.

        Returns:
            uint8 BGR recovered image.
        """
        t_min = self.settings.transmission_min_clip
        omega = self.settings.omega

        A = atm_result.atmospheric_light_map              # H x W x 3
        # Transmission approximation: brighter atm light ⇒ lower transmission.
        t_map = np.clip(1.0 - omega * A.mean(axis=2), t_min, 1.0)  # H x W
        t_clamped = np.maximum(t_map[:, :, None], t_min)          # H x W x 1

        # Scattering-model inversion: J = (I - A) / max(t, t_min) + A
        recovered = (ffa_dehazed - A) / t_clamped + A
        recovered = np.clip(recovered, 0.0, 1.0)

        # Glow-aware blending: trust FFA-Net more inside glow regions (70%).
        glow_3ch = glow_mask[:, :, None]
        final = recovered * (1.0 - glow_3ch * 0.7) + ffa_dehazed * (glow_3ch * 0.7)
        final = np.clip(final, 0.0, 1.0)

        return image_utils.denormalize(final)  # uint8 BGR
