"""Unit tests for the GlowDetector service."""

import cv2
import numpy as np

from app.services import image_utils
from app.services.glow_detector import GlowDetector


def _synthetic_night_scene() -> np.ndarray:
    """Dark scene with three blurred light sources, as float32 [0,1] BGR."""
    image = np.full((400, 600, 3), 20, dtype=np.uint8)
    for center in [(150, 100), (300, 200), (450, 300)]:
        cv2.circle(image, center, 30, (250, 250, 250), thickness=-1)
    image = cv2.GaussianBlur(image, (31, 31), 0)
    return image_utils.normalize(image)


def test_detects_multiple_light_sources(settings):
    result = GlowDetector(settings).detect(_synthetic_night_scene())
    assert result.num_sources >= 2
    assert len(result.glow_regions) == result.num_sources


def test_glow_regions_carry_local_atm_light(settings):
    result = GlowDetector(settings).detect(_synthetic_night_scene())
    assert result.glow_regions, "expected at least one glow region"
    for region in result.glow_regions:
        assert region.local_atm_light.shape == (3,)
        assert region.local_atm_light.dtype == np.float32


def test_pitch_black_image_has_no_sources(settings):
    black = np.zeros((200, 300, 3), dtype=np.float32)
    result = GlowDetector(settings).detect(black)

    assert result.num_sources == 0
    assert result.glow_regions == []
    assert result.glow_mask.shape == (200, 300)
    assert float(result.glow_mask.max()) == 0.0


def test_glow_mask_dtype_and_range(settings, sample_image_float):
    result = GlowDetector(settings).detect(sample_image_float)

    assert result.glow_mask.dtype == np.float32
    assert result.glow_mask.min() >= 0.0
    assert result.glow_mask.max() <= 1.0
    assert result.glow_mask.shape == sample_image_float.shape[:2]


def test_glow_mask_vis_is_uint8_bgr(settings, sample_image_float):
    result = GlowDetector(settings).detect(sample_image_float)
    assert result.glow_mask_vis.dtype == np.uint8
    assert result.glow_mask_vis.shape == sample_image_float.shape
