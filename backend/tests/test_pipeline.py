"""Integration tests for the full dehazing pipeline.

Tests verify end-to-end pipeline behavior:
- Successful completion and output generation
- Correct output shape and format (base64-encoded images)
- Presence and validity of quality metrics
- Processing stage ordering and timing
- Edge case handling (grayscale images, degenerate inputs)

All tests marked @pytest.mark.slow because they drive FFA-Net inference (~10-20s each).
Tests reuse the cached pipeline_result fixture (session-scoped) to avoid redundant runs.
"""

import cv2
import numpy as np
import pytest

from app.services import image_utils

pytestmark = pytest.mark.slow  # every test here drives FFA-Net on CPU


def test_pipeline_succeeds(pipeline_result):
    """Verify end-to-end pipeline execution completed successfully.

    Checks:
    - success flag is True (no errors during processing)
    - job_id is present (unique identifier for this run)
    """
    assert pipeline_result.success is True
    assert pipeline_result.job_id


def test_pipeline_output_shape(pipeline_result, sample_image):
    """Verify dehazed output shape matches input (no unexpected cropping/resizing).

    Decodes the base64-encoded output image and compares shape to original.
    Ensures the pipeline doesn't alter image dimensions.
    """
    dehazed = image_utils.from_base64(pipeline_result.dehazed_image_b64)
    assert dehazed.shape == sample_image.shape


def test_pipeline_metrics_present(pipeline_result):
    """Verify all expected quality metrics are computed and present.

    Checks that DehazeMetrics object has the required fields:
    psnr, ssim, niqe, brisque, visibility_score (and others).
    All values should be float (or -1.0 if computation failed).
    """
    for key in ("psnr", "ssim", "niqe", "brisque", "visibility_score"):
        assert hasattr(pipeline_result.metrics, key)
        assert isinstance(getattr(pipeline_result.metrics, key), float)


def test_pipeline_output_differs_from_input(pipeline_result, sample_image):
    """Verify the pipeline actually changes the image (not a no-op).

    Decodes output and computes mean absolute pixel difference.
    A successful dehaze should change pixels substantially (> 1.0).
    """
    dehazed = image_utils.from_base64(pipeline_result.dehazed_image_b64)
    pixel_diff = np.abs(sample_image.astype(float) - dehazed.astype(float)).mean()
    assert pixel_diff > 1.0, "dehazed output is essentially identical to input"


def test_pipeline_processing_time_reasonable(pipeline_result):
    """Verify total processing time is within expected bounds.

    Generous headroom: FFA-Net inference alone ~10-20s, plus other stages.
    30s limit allows for variation across hardware and system load.
    """
    # Generous CPU headroom; FFA-Net inference alone is ~10s.
    assert pipeline_result.metrics.processing_time_ms < 30000


def test_pipeline_has_six_stages(pipeline_result):
    """Verify pipeline runs all six stages in the correct order.

    Expected stages:
    1. preprocessing: validate, denoise, normalize
    2. glow_detection: find light sources
    3. ffa_net_inference: deep learning dehaze
    4. radiance_recovery: physics-based refinement
    5. postprocessing: CLAHE, sharpen, desaturate
    6. quality_assessment: compute metrics
    """
    stages = [stage.stage for stage in pipeline_result.pipeline_stages]
    assert stages == [
        "preprocessing",
        "glow_detection",
        "ffa_net_inference",
        "radiance_recovery",
        "postprocessing",
        "quality_assessment",
    ]


def test_pipeline_returns_all_four_images(pipeline_result):
    """Verify pipeline returns all four expected output images (base64-encoded).

    Expected images:
    - original_image_b64: input (for comparison)
    - dehazed_image_b64: main output
    - transmission_map_b64: haze density visualization
    - glow_mask_b64: detected light source regions
    All should be PNG-encoded data URIs (data:image/png;base64,...)
    """
    for field in (
        pipeline_result.original_image_b64,
        pipeline_result.dehazed_image_b64,
        pipeline_result.transmission_map_b64,
        pipeline_result.glow_mask_b64,
    ):
        assert field.startswith("data:image/png;base64,")


def test_pipeline_handles_grayscale(pipeline, sample_image):
    """Verify pipeline handles single-channel grayscale input gracefully.

    Converts sample to grayscale, feeds to pipeline.
    Pipeline's preprocessor should expand to 3-channel BGR internally.
    """
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    assert gray.ndim == 2
    result = pipeline.process(gray)
    assert result.success is True
