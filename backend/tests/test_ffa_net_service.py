"""Unit tests for the FFA-Net inference service."""

import numpy as np
import pytest

from app.core.exceptions import ModelNotLoadedError
from app.services.ffa_net_service import FFANetService

BAD_WEIGHTS = "../ml/weights/does_not_exist.pkl"


def test_model_loads(settings):
    service = FFANetService(settings)
    assert service.model_loaded is True
    assert service.model is not None


@pytest.mark.slow
def test_dehaze_output_contract(settings, sample_image_float):
    service = FFANetService(settings)
    output = service.dehaze(sample_image_float)

    assert output.dtype == np.float32
    assert output.shape == sample_image_float.shape
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_missing_weights_degrade_gracefully(settings):
    """A bad weights path must NOT crash construction — /health reports it."""
    broken = settings.model_copy(update={"model_weights_path": BAD_WEIGHTS})

    service = FFANetService(broken)

    assert service.model_loaded is False
    assert service.model is None


def test_dehaze_raises_when_model_not_loaded(settings):
    broken = settings.model_copy(update={"model_weights_path": BAD_WEIGHTS})
    service = FFANetService(broken)

    with pytest.raises(ModelNotLoadedError):
        service.dehaze(np.zeros((64, 64, 3), dtype=np.float32))
