"""Shared pytest fixtures.

The expensive objects (FFA-Net model, and one full pipeline run) are
session-scoped so the ~10s CPU inference happens as few times as possible.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Make the `app` package importable no matter where pytest is invoked from.
# This allows running tests from any working directory.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.core.pipeline import DehazingPipeline  # noqa: E402
from app.main import app  # noqa: E402
from app.models.schemas import DehazeResponse  # noqa: E402
from app.services import image_utils  # noqa: E402

# Path to bundled sample nighttime image used across tests
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_hazy_night.jpg"


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    """Absolute path to the bundled sample nighttime image.

    Scope: session (shared across all tests in the run)
    Used by tests that need to load the sample image from disk.
    """
    return FIXTURE_PATH


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Application settings loaded from environment/defaults.

    Scope: session (created once, reused for all tests)
    The default ``model_weights_path`` ("../ml/weights/...") is resolved by
    FFANetService relative to backend/, so it already works from any CWD — no
    test override is needed. Returns the same instance for all tests.
    """
    return Settings()


@pytest.fixture
def sample_image() -> np.ndarray:
    """The sample hazy night image as uint8 BGR.

    Scope: function (created fresh for each test that uses it)
    Skips test if fixture file is missing or corrupted.
    Allows individual tests to load and manipulate the image without side effects.
    """
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Sample fixture missing: {FIXTURE_PATH}")
    image = cv2.imread(str(FIXTURE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        pytest.skip(f"Sample fixture is not a decodable image: {FIXTURE_PATH}")
    return image


@pytest.fixture
def sample_image_float(sample_image: np.ndarray) -> np.ndarray:
    """The sample image normalized to float32 [0,1] BGR.

    Scope: function (depends on sample_image fixture, so created per test)
    Converts uint8 [0,255] BGR to float32 [0,1] BGR for pipeline processing.
    Derived from sample_image fixture, so test isolation is automatic.
    """
    return image_utils.normalize(sample_image)


@pytest.fixture(scope="session")
def pipeline(settings: Settings) -> DehazingPipeline:
    """A fully constructed dehazing pipeline.

    Scope: session (expensive: built once, reused across all tests)
    Initializes all services:
    - Preprocessor, Glow Detector, Atmospheric Light Estimator,
    - Radiance Recovery, FFA-Net inference, Postprocessor, Quality Assessor
    - FFA-Net model weights load once (takes ~5-10s), avoiding reload overhead
    Skips tests if model weights fail to load (graceful degradation for CI without GPU).
    """
    built = DehazingPipeline(settings)
    if not built.ffa_service.model_loaded:
        pytest.skip("FFA-Net weights unavailable — skipping pipeline tests.")
    return built


@pytest.fixture(scope="session")
def pipeline_result(pipeline: DehazingPipeline) -> DehazeResponse:
    """One full end-to-end pipeline run (cached for all tests).

    Scope: session (very expensive: FFA-Net inference takes ~10-20s per image)
    Runs the complete pipeline once on the sample image, then reuses the
    cached DehazeResponse across all tests that need it. Dramatically speeds
    up test suite (avoid re-running inference for every test).
    Tests that need a different image or modified pipeline should create their own.
    """
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Sample fixture missing: {FIXTURE_PATH}")
    image = cv2.imread(str(FIXTURE_PATH), cv2.IMREAD_COLOR)
    return pipeline.process(image)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """TestClient for making HTTP requests to the FastAPI app.

    Scope: session (created once, reused across all tests)
    The TestClient context manager automatically runs the FastAPI lifespan
    (app startup), which initializes app.state.pipeline. Tests can make
    GET/POST requests to endpoints and receive responses.

    Example usage in a test:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    """
    with TestClient(app) as test_client:
        yield test_client
