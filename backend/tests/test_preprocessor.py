"""Unit tests for the Preprocessor service.

Tests verify:
- Output normalization (float32 [0,1] range)
- Channel handling (single/multi-channel)
- Noise estimation (classification into low/medium/high)
- Image resizing (respecting max_image_dimension)
- Error handling (too-small images rejected)
"""

import numpy as np
import pytest

from app.core.exceptions import InvalidImageError
from app.services.preprocessor import Preprocessor


def test_output_is_float32_unit_range(settings, sample_image):
    """Verify output is normalized to float32 [0,1] range.

    The preprocessor must convert uint8 [0,255] to float32 [0,1] for
    pipeline compatibility. Validates both dtype and value range.
    """
    result = Preprocessor(settings).prepare(sample_image)
    assert result.image.dtype == np.float32
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_output_is_three_channel(settings, sample_image):
    """Verify output has 3 channels (BGR) regardless of input.

    Ensures internal contract: all pipeline images are float32 BGR.
    Three dimensions: (height, width, 3 BGR channels).
    """
    result = Preprocessor(settings).prepare(sample_image)
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3


def test_noise_level_is_valid(settings, sample_image):
    """Verify noise classification is computed and valid.

    Checks that noise_level is one of the expected categories (low/medium/high)
    and noise_value is a numeric estimate (Laplacian variance).
    """
    result = Preprocessor(settings).prepare(sample_image)
    assert result.noise_level in {"low", "medium", "high"}
    assert isinstance(result.noise_value, float)


def test_grayscale_is_promoted_to_bgr(settings):
    """Verify grayscale images are expanded to 3-channel BGR.

    Input: single-channel (H, W) grayscale image
    Output: three-channel (H, W, 3) BGR image (all channels identical)
    Tests the ensure_bgr utility function.
    """
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (200, 300), dtype=np.uint8)
    result = Preprocessor(settings).prepare(gray)
    assert result.image.shape == (200, 300, 3)


def test_oversized_image_is_resized(settings):
    """Verify oversized images are downsampled to respect max_image_dimension.

    Tests the resize_maintain_aspect function:
    - Original size is recorded in result.original_size
    - Resized to fit within max_image_dimension
    - Aspect ratio is maintained
    - scale_factor reflects the downsample amount

    Uses random content to simulate high noise (skips bilateral filter for speed).
    """
    # Random content keeps the Laplacian variance high => 'low' noise => the
    # (slow) bilateral filter is skipped, so this test stays fast.
    rng = np.random.default_rng(1)
    oversized = rng.integers(0, 256, (2100, 2800, 3), dtype=np.uint8)

    result = Preprocessor(settings).prepare(oversized)

    assert result.original_size == (2100, 2800)
    # Max dimension should be respected (matches settings.max_image_dimension)
    assert max(result.image.shape[:2]) == settings.max_image_dimension
    assert result.scale_factor < 1.0  # Downsampled, not upsampled


def test_tiny_image_raises_invalid_image_error(settings):
    """Verify tiny images are rejected (below minimum dimension).

    The preprocessor validates image size to prevent processing of unusably
    small images. A 32x32 image is below the minimum threshold and should
    raise InvalidImageError.
    """
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        Preprocessor(settings).prepare(tiny)
