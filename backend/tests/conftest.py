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
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.core.pipeline import DehazingPipeline  # noqa: E402
from app.main import app  # noqa: E402
from app.models.schemas import DehazeResponse  # noqa: E402
from app.services import image_utils  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_hazy_night.jpg"


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    """Absolute path to the bundled sample nighttime image."""
    return FIXTURE_PATH


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Application settings.

    The default ``model_weights_path`` ("../ml/weights/...") is resolved by
    FFANetService relative to backend/, so it already works from any CWD — no
    test override is needed.
    """
    return Settings()


@pytest.fixture
def sample_image() -> np.ndarray:
    """The sample hazy night image as uint8 BGR."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Sample fixture missing: {FIXTURE_PATH}")
    image = cv2.imread(str(FIXTURE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        pytest.skip(f"Sample fixture is not a decodable image: {FIXTURE_PATH}")
    return image


@pytest.fixture
def sample_image_float(sample_image: np.ndarray) -> np.ndarray:
    """The sample image normalized to float32 [0,1] BGR."""
    return image_utils.normalize(sample_image)


@pytest.fixture(scope="session")
def pipeline(settings: Settings) -> DehazingPipeline:
    """A fully constructed pipeline. Model weights load once per session."""
    built = DehazingPipeline(settings)
    if not built.ffa_service.model_loaded:
        pytest.skip("FFA-Net weights unavailable — skipping pipeline tests.")
    return built


@pytest.fixture(scope="session")
def pipeline_result(pipeline: DehazingPipeline) -> DehazeResponse:
    """One full pipeline run, shared across tests (FFA-Net inference is slow)."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Sample fixture missing: {FIXTURE_PATH}")
    image = cv2.imread(str(FIXTURE_PATH), cv2.IMREAD_COLOR)
    return pipeline.process(image)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """TestClient with lifespan run (so app.state.pipeline exists)."""
    with TestClient(app) as test_client:
        yield test_client
