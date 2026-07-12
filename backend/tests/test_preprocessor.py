"""Unit tests for the Preprocessor service."""

import numpy as np
import pytest

from app.core.exceptions import InvalidImageError
from app.services.preprocessor import Preprocessor


def test_output_is_float32_unit_range(settings, sample_image):
    result = Preprocessor(settings).prepare(sample_image)
    assert result.image.dtype == np.float32
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_output_is_three_channel(settings, sample_image):
    result = Preprocessor(settings).prepare(sample_image)
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3


def test_noise_level_is_valid(settings, sample_image):
    result = Preprocessor(settings).prepare(sample_image)
    assert result.noise_level in {"low", "medium", "high"}
    assert isinstance(result.noise_value, float)


def test_grayscale_is_promoted_to_bgr(settings):
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (200, 300), dtype=np.uint8)
    result = Preprocessor(settings).prepare(gray)
    assert result.image.shape == (200, 300, 3)


def test_oversized_image_is_resized(settings):
    # Random content keeps the Laplacian variance high => 'low' noise => the
    # (slow) bilateral filter is skipped, so this test stays fast.
    rng = np.random.default_rng(1)
    oversized = rng.integers(0, 256, (2100, 2800, 3), dtype=np.uint8)

    result = Preprocessor(settings).prepare(oversized)

    assert result.original_size == (2100, 2800)
    assert max(result.image.shape[:2]) == settings.max_image_dimension
    assert result.scale_factor < 1.0


def test_tiny_image_raises_invalid_image_error(settings):
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        Preprocessor(settings).prepare(tiny)
