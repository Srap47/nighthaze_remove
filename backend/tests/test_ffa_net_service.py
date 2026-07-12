"""Unit tests for the FFA-Net inference service.

Tests verify:
- Model loading (weights successfully loaded)
- Inference contract (output dtype, shape, range)
- Graceful degradation (missing weights don't crash startup)
- Error handling (dehaze() raises when model unavailable)
"""

import numpy as np
import pytest

from app.core.exceptions import ModelNotLoadedError
from app.services.ffa_net_service import FFANetService

# Invalid weights path used to test graceful degradation
BAD_WEIGHTS = "../ml/weights/does_not_exist.pkl"


def test_model_loads(settings):
    """Verify FFA-Net model weights load successfully at startup.

    Checks both the model_loaded flag and the model object itself.
    If weights are missing, this test is skipped by conftest.py fixture logic.
    """
    service = FFANetService(settings)
    assert service.model_loaded is True
    assert service.model is not None


@pytest.mark.slow
def test_dehaze_output_contract(settings, sample_image_float):
    """Verify inference output respects the pipeline contract.

    Checks:
    - dtype: float32 (internal pipeline format)
    - shape: matches input (H, W, 3) — no resizing side effect
    - range: [0, 1] (normalized)

    Marked @pytest.mark.slow because FFA-Net inference is CPU-intensive (~10-20s).
    """
    service = FFANetService(settings)
    output = service.dehaze(sample_image_float)

    assert output.dtype == np.float32
    assert output.shape == sample_image_float.shape
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_missing_weights_degrade_gracefully(settings):
    """Verify missing weights don't crash app startup (graceful degradation).

    FFANetService.__init__ catches exceptions from _load_model() and logs
    warnings instead of raising. This allows the app to start in degraded mode
    (model_loaded=False) and lets /health endpoint report the issue.
    Frontend can then guide users to download weights.
    """
    broken = settings.model_copy(update={"model_weights_path": BAD_WEIGHTS})

    service = FFANetService(broken)

    # Model failed to load, but service still created
    assert service.model_loaded is False
    assert service.model is None


def test_dehaze_raises_when_model_not_loaded(settings):
    """Verify dehaze() raises ModelNotLoadedError when weights unavailable.

    If the model failed to load during __init__, any call to dehaze()
    should raise ModelNotLoadedError (raising early rather than silently failing).
    """
    broken = settings.model_copy(update={"model_weights_path": BAD_WEIGHTS})
    service = FFANetService(broken)

    with pytest.raises(ModelNotLoadedError):
        service.dehaze(np.zeros((64, 64, 3), dtype=np.float32))
