"""Unit tests for the GlowDetector service.

Tests verify:
- Light source detection (finds multiple bright spots in dark scenes)
- Local atmospheric light estimation (per-region color/intensity)
- Output format (glow mask dtype, range, visualization)
- Edge cases (pitch-black images, grayscale handling)
"""

import cv2
import numpy as np

from app.services import image_utils
from app.services.glow_detector import GlowDetector


def _synthetic_night_scene() -> np.ndarray:
    """Create a synthetic nighttime scene with three known light sources.

    Generates a dark image (mostly black) with three bright white circles,
    blurred to simulate glow halos. Used to test glow detection in a controlled,
    deterministic environment. Returns float32 [0,1] BGR image.
    """
    # Start with dark background (mostly black, value ~20/255)
    image = np.full((400, 600, 3), 20, dtype=np.uint8)
    # Paint three white light sources at known positions
    for center in [(150, 100), (300, 200), (450, 300)]:
        cv2.circle(image, center, 30, (250, 250, 250), thickness=-1)
    # Blur to simulate glow spread and soften edges
    image = cv2.GaussianBlur(image, (31, 31), 0)
    # Convert to float32 [0,1] for pipeline processing
    return image_utils.normalize(image)


def test_detects_multiple_light_sources(settings):
    """Verify glow detector finds multiple light sources in the synthetic scene.

    The synthetic scene has 3 blurred light sources. The detector should
    identify at least 2 (allowing for edge cases where small circles merge).
    Checks that num_sources matches the glow_regions list length.
    """
    result = GlowDetector(settings).detect(_synthetic_night_scene())
    assert result.num_sources >= 2
    assert len(result.glow_regions) == result.num_sources


def test_glow_regions_carry_local_atm_light(settings):
    """Verify each detected glow region has an estimated local atmospheric light.

    Each GlowRegion carries local_atm_light (float32, shape (3,) for BGR).
    This is used later by radiance recovery to blend the glow light color
    into the atmospheric light map. Checks dtype and shape.
    """
    result = GlowDetector(settings).detect(_synthetic_night_scene())
    assert result.glow_regions, "expected at least one glow region"
    for region in result.glow_regions:
        # Shape (3,) = BGR channels; float32 = normalized [0,1] range
        assert region.local_atm_light.shape == (3,)
        assert region.local_atm_light.dtype == np.float32


def test_pitch_black_image_has_no_sources(settings):
    """Verify all-black image (no light sources) returns empty detection.

    Edge case: completely dark image should trigger early exit
    and return empty glow_regions list, all-zero mask, and 0 sources.
    Tests graceful handling of degenerate inputs.
    """
    black = np.zeros((200, 300, 3), dtype=np.float32)
    result = GlowDetector(settings).detect(black)

    assert result.num_sources == 0
    assert result.glow_regions == []
    assert result.glow_mask.shape == (200, 300)
    assert float(result.glow_mask.max()) == 0.0


def test_glow_mask_dtype_and_range(settings, sample_image_float):
    """Verify glow mask has correct dtype and value range.

    The glow_mask is float32 [0,1] where 1.0 indicates glow regions
    and 0.0 indicates non-glow. Shape matches image (H, W).
    Tests the pipeline contract for this output.
    """
    result = GlowDetector(settings).detect(sample_image_float)

    assert result.glow_mask.dtype == np.float32
    assert result.glow_mask.min() >= 0.0
    assert result.glow_mask.max() <= 1.0
    assert result.glow_mask.shape == sample_image_float.shape[:2]


def test_glow_mask_vis_is_uint8_bgr(settings, sample_image_float):
    """Verify visualization mask has correct type for display.

    The glow_mask_vis is uint8 BGR (suitable for OpenCV display and JPEG encoding).
    Shape matches the input image (H, W, 3). This is for frontend visualization only.
    """
    result = GlowDetector(settings).detect(sample_image_float)
    assert result.glow_mask_vis.dtype == np.uint8
    assert result.glow_mask_vis.shape == sample_image_float.shape
