"""
FFA-Net inference service — the deep-learning core of the pipeline.

Wraps the pretrained FFA-Net model (Qin et al., AAAI 2020) for single-image
dehazing.

Module contract: INPUT = float32 [0,1] BGR. OUTPUT = float32 [0,1] BGR.

Fault tolerance: if the weights file is missing or fails to load, the service
does NOT crash the application. It logs a clear warning (with download
instructions) and leaves ``model_loaded = False`` so ``/health`` can report the
degraded state; ``dehaze`` then raises :class:`ModelNotLoadedError` if called.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from app.config import Settings
from app.core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)

_DOWNLOAD_HINT = (
    "Download 'its_train_ffa_3_19.pkl' from "
    "https://github.com/zhilin007/FFA-Net and place it at the configured "
    "MODEL_WEIGHTS_PATH."
)


class FFANetService:
    """Loads and runs the pretrained FFA-Net model for dehazing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: torch.nn.Module | None = None
        self.model_loaded: bool = False
        try:
            self._load_model()
        except Exception as exc:  # noqa: BLE001 — never let startup crash here
            # Degraded mode: /health reports model_loaded=False; dehaze() will 503.
            logger.warning(
                "FFA-Net model could not be loaded (%s). Running in degraded "
                "mode without dehazing. %s",
                exc,
                _DOWNLOAD_HINT,
            )
            self.model = None
            self.model_loaded = False

    def _load_model(self) -> None:
        """Load FFA-Net weights into ``self.model``. Raises on any failure."""
        # The ``ml`` package lives at the project root (one level above backend/).
        # services → app → backend → project_root
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from ml.model.ffa_net import FFA

        # Resolve the weights path relative to backend/ so it works regardless of
        # the current working directory. backend/ = parents[2].
        backend_dir = Path(__file__).resolve().parents[2]
        weights_path = (backend_dir / self.settings.model_weights_path).resolve()
        if not weights_path.exists():
            raise ModelNotLoadedError(
                f"FFA-Net weights not found at {weights_path}. {_DOWNLOAD_HINT}"
            )

        model = FFA(gps=self.settings.model_gps, blocks=self.settings.model_blocks)

        checkpoint = torch.load(str(weights_path), map_location=self.settings.device)
        # The checkpoint is a training snapshot: keys are
        # ['step', 'max_psnr', 'max_ssim', 'ssims', 'psnrs', 'losses', 'model'].
        # Weights live under 'model' and carry a 'module.' prefix (DataParallel).
        raw_state = checkpoint["model"]
        state_dict = {k.replace("module.", ""): v for k, v in raw_state.items()}

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self.settings.device)

        self.model = model
        self.model_loaded = True
        logger.info(
            "FFA-Net loaded (gps=%d, blocks=%d) on %s from %s",
            self.settings.model_gps,
            self.settings.model_blocks,
            self.settings.device,
            weights_path,
        )

    def dehaze(self, image: np.ndarray) -> np.ndarray:
        """Dehaze a float32 [0,1] BGR image, returning float32 [0,1] BGR.

        Args:
            image: float32 [0,1] BGR image, shape ``(H, W, 3)``.

        Returns:
            Dehazed float32 [0,1] BGR image at the original ``(H, W)``.

        Raises:
            ModelNotLoadedError: If the model failed to load at startup.
        """
        if not self.model_loaded or self.model is None:
            raise ModelNotLoadedError(
                f"FFA-Net model is not loaded. {_DOWNLOAD_HINT}"
            )

        device = self.settings.device
        H, W = image.shape[:2]
        input_size = self.settings.ffa_input_size

        # 1. Resize to the network's square input size.
        resized = cv2.resize(
            image, (input_size, input_size), interpolation=cv2.INTER_LINEAR
        )

        # 2. BGR → RGB, HWC → CHW, add batch dim.
        rgb = resized[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).float()

        # 3. Inference.
        with torch.no_grad():
            output = self.model(tensor)

        # 4. CHW → HWC, RGB → BGR, clip.
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_bgr = output_np[:, :, ::-1].copy()
        output_bgr = np.clip(output_bgr, 0.0, 1.0)

        # 5. Resize back to the original dimensions.
        result = cv2.resize(output_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        return result.astype(np.float32)
