"""
Shared image I/O and conversion utilities.

Color/dtype contract:
    All pipeline-internal images are float32 BGR in [0,1]. uint8 only at I/O
    boundaries.

Every function here is pure (no side effects, no shared state). uint8 BGR is
used only where images cross the process boundary — decoding uploads, encoding
responses — while the pipeline itself operates on float32 BGR in [0,1].
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from app.config import Settings
from app.core.exceptions import InvalidImageError

_DATA_URI_PREFIX = "data:image/png;base64,"


def to_base64(image: np.ndarray) -> str:
    """Encode a BGR uint8 image as a base64 PNG data URI.

    Args:
        image: BGR ``uint8`` array, shape ``(H, W, 3)`` (or 2-D grayscale).

    Returns:
        A ``"data:image/png;base64,<...>"`` string suitable for an ``<img src>``.

    Raises:
        InvalidImageError: If the image cannot be PNG-encoded.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise InvalidImageError("Failed to PNG-encode image for base64 output.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"{_DATA_URI_PREFIX}{encoded}"


def from_base64(data_uri: str) -> np.ndarray:
    """Decode a base64 PNG data URI back into a BGR uint8 image.

    The inverse of :func:`to_base64`. Accepts either a full data URI or a bare
    base64 string.

    Args:
        data_uri: ``"data:image/png;base64,<...>"`` or raw base64 text.

    Returns:
        BGR ``uint8`` array, shape ``(H, W, 3)``.

    Raises:
        InvalidImageError: If the payload cannot be decoded into an image.
    """
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    try:
        raw = base64.b64decode(payload)
    except (base64.binascii.Error, ValueError) as exc:  # type: ignore[attr-defined]
        raise InvalidImageError("Malformed base64 image payload.") from exc
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("Base64 payload did not decode to a valid image.")
    return image


def file_to_numpy(file_bytes: bytes) -> np.ndarray:
    """Decode raw uploaded file bytes into a BGR uint8 image.

    Args:
        file_bytes: Raw bytes of an uploaded image file (PNG/JPEG/etc.).

    Returns:
        BGR ``uint8`` array, shape ``(H, W, 3)``.

    Raises:
        InvalidImageError: If the bytes are not a decodable image.
    """
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("Uploaded file is not a valid/decodable image.")
    return image


def normalize(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 [0,255] image to float32 [0,1]."""
    return image.astype(np.float32) / 255.0


def denormalize(image: np.ndarray) -> np.ndarray:
    """Convert a float32 [0,1] image to uint8 [0,255], clipping out-of-range values."""
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def resize_maintain_aspect(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is at most ``max_dim``, preserving aspect ratio.

    Images already within ``max_dim`` are returned unchanged with a scale of 1.0.

    Args:
        image: Input image (any dtype cv2 accepts).
        max_dim: Maximum allowed length of the longest side, in pixels.

    Returns:
        ``(resized_image, scale_factor)`` where ``scale_factor = new_size /
        original_size`` (1.0 if no resize occurred). The factor lets callers
        rescale downstream results back to the original resolution.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0

    scale = max_dim / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized, scale


def validate_image(image: np.ndarray, settings: Settings) -> None:
    """Validate that ``image`` is a usable array of workable size.

    Note on large images: ``max_image_dimension`` is a *processing cap*, not a
    rejection threshold. Oversized images (e.g. a 4032x3024 phone photo) are
    downscaled by :func:`resize_maintain_aspect` in the preprocessor rather than
    refused, so no maximum-dimension check happens here. Upload size is still
    bounded separately by ``max_image_size_mb`` at the API layer.

    Args:
        image: Candidate image array.
        settings: Active :class:`~app.config.Settings` providing the dimension bounds.

    Raises:
        InvalidImageError: If the image is ``None``/not an array, has an
            unsupported number of dimensions, or is too small to process.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise InvalidImageError("No image provided (input is None or not a numpy array).")

    if image.ndim not in (2, 3):
        raise InvalidImageError(
            f"Unsupported image shape: expected 2-D or 3-D array, got {image.ndim}-D."
        )

    height, width = image.shape[:2]
    smallest = min(height, width)

    if smallest < settings.min_image_dimension:
        raise InvalidImageError(
            f"Image too small: {width}x{height}px, minimum side is "
            f"{settings.min_image_dimension}px."
        )


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce an image to 3-channel BGR.

    - Grayscale (``ndim == 2``) is expanded to BGR.
    - BGRA (4 channels) has its alpha channel dropped.
    - 3-channel BGR is returned unchanged.

    Args:
        image: Grayscale, BGR, or BGRA image.

    Returns:
        A 3-channel BGR image of the same dtype.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
