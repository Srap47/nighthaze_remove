"""Integration tests for the full dehazing pipeline."""

import cv2
import numpy as np
import pytest

from app.services import image_utils

pytestmark = pytest.mark.slow  # every test here drives FFA-Net on CPU


def test_pipeline_succeeds(pipeline_result):
    assert pipeline_result.success is True
    assert pipeline_result.job_id


def test_pipeline_output_shape(pipeline_result, sample_image):
    dehazed = image_utils.from_base64(pipeline_result.dehazed_image_b64)
    assert dehazed.shape == sample_image.shape


def test_pipeline_metrics_present(pipeline_result):
    for key in ("psnr", "ssim", "niqe", "brisque", "visibility_score"):
        assert hasattr(pipeline_result.metrics, key)
        assert isinstance(getattr(pipeline_result.metrics, key), float)


def test_pipeline_output_differs_from_input(pipeline_result, sample_image):
    dehazed = image_utils.from_base64(pipeline_result.dehazed_image_b64)
    pixel_diff = np.abs(sample_image.astype(float) - dehazed.astype(float)).mean()
    assert pixel_diff > 1.0, "dehazed output is essentially identical to input"


def test_pipeline_processing_time_reasonable(pipeline_result):
    # Generous CPU headroom; FFA-Net inference alone is ~10s.
    assert pipeline_result.metrics.processing_time_ms < 30000


def test_pipeline_has_six_stages(pipeline_result):
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
    for field in (
        pipeline_result.original_image_b64,
        pipeline_result.dehazed_image_b64,
        pipeline_result.transmission_map_b64,
        pipeline_result.glow_mask_b64,
    ):
        assert field.startswith("data:image/png;base64,")


def test_pipeline_handles_grayscale(pipeline, sample_image):
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    assert gray.ndim == 2
    result = pipeline.process(gray)
    assert result.success is True
