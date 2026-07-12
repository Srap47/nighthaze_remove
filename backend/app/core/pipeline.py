"""
Dehazing pipeline orchestrator.

:class:`DehazingPipeline` wires the seven services together and runs a raw image
through all six stages, producing a :class:`DehazeResponse`. It is instantiated
once at application startup and reused for every request.

Dimension contract: the ``original`` image used for metrics and returned as
``original_image_b64`` is resized with the *same* ``resize_maintain_aspect`` step
the preprocessor applies, so ``original`` and ``dehazed`` always share identical
dimensions (required for PSNR/SSIM).
"""

from __future__ import annotations

import logging
import time
import uuid

import numpy as np

from app.config import Settings
from app.core.exceptions import NightHazeError, PipelineError
from app.models.schemas import DehazeResponse, PipelineStage
from app.services import image_utils
from app.services.atm_light import AtmosphericLightEstimator
from app.services.ffa_net_service import FFANetService
from app.services.glow_detector import GlowDetector
from app.services.postprocessor import Postprocessor
from app.services.preprocessor import Preprocessor
from app.services.quality_assessor import QualityAssessor
from app.services.radiance_recovery import RadianceRecovery

logger = logging.getLogger(__name__)


class DehazingPipeline:
    """Orchestrates preprocessing → glow → FFA-Net → recovery → post → metrics."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.preprocessor = Preprocessor(settings)
        self.glow_detector = GlowDetector(settings)
        self.ffa_service = FFANetService(settings)  # exposed for the health route
        self.atm_light = AtmosphericLightEstimator(settings)
        self.radiance_recovery = RadianceRecovery(settings)
        self.postprocessor = Postprocessor(settings)
        self.quality_assessor = QualityAssessor(settings)

    def process(self, image: np.ndarray) -> DehazeResponse:
        """Run the full dehazing pipeline on a raw uint8 BGR image.

        Args:
            image: Raw decoded image (uint8; grayscale/BGR/BGRA accepted).

        Returns:
            A populated :class:`DehazeResponse`.

        Raises:
            ModelNotLoadedError: If FFA-Net weights are unavailable.
            InvalidImageError: If the input fails validation.
            PipelineError: For any other unexpected stage failure (with context).
        """
        job_id = str(uuid.uuid4())
        stages: list[PipelineStage] = []

        # Reference image at pipeline resolution: mirror the preprocessor's resize
        # so original_resized matches every downstream (dehazed) frame exactly.
        bgr = image_utils.ensure_bgr(image)
        original_resized, _scale = image_utils.resize_maintain_aspect(
            bgr, self.settings.max_image_dimension
        )
        original_resized = np.ascontiguousarray(original_resized)  # uint8 BGR

        current_stage = "initialization"
        try:
            # Stage 1: Preprocessing.
            current_stage = "preprocessing"
            t0 = time.time()
            prep_result = self.preprocessor.prepare(image)
            stages.append(PipelineStage(stage="preprocessing", time_ms=(time.time() - t0) * 1000))

            # Stage 2: Glow detection.
            current_stage = "glow_detection"
            t0 = time.time()
            glow_result = self.glow_detector.detect(prep_result.image)
            stages.append(PipelineStage(stage="glow_detection", time_ms=(time.time() - t0) * 1000))

            # Stage 3: FFA-Net inference.
            current_stage = "ffa_net_inference"
            t0 = time.time()
            ffa_dehazed = self.ffa_service.dehaze(prep_result.image)
            stages.append(PipelineStage(stage="ffa_net_inference", time_ms=(time.time() - t0) * 1000))

            # Stage 4: Atmospheric light estimation + glow-aware radiance recovery.
            current_stage = "radiance_recovery"
            t0 = time.time()
            atm_result = self.atm_light.estimate(
                prep_result.image, glow_result.glow_mask, glow_result.glow_regions
            )
            recovered = self.radiance_recovery.recover(
                ffa_dehazed, atm_result, glow_result.glow_mask
            )
            stages.append(PipelineStage(stage="radiance_recovery", time_ms=(time.time() - t0) * 1000))

            # Stage 5: Post-processing.
            current_stage = "postprocessing"
            t0 = time.time()
            final_image = self.postprocessor.enhance(recovered)
            stages.append(PipelineStage(stage="postprocessing", time_ms=(time.time() - t0) * 1000))

            # Stage 6: Quality assessment (original vs final, identical dimensions).
            current_stage = "quality_assessment"
            t0 = time.time()
            metrics = self.quality_assessor.score(original_resized, final_image)
            stages.append(PipelineStage(stage="quality_assessment", time_ms=(time.time() - t0) * 1000))
        except NightHazeError:
            # Domain errors (ModelNotLoadedError, InvalidImageError, ...) carry
            # their own HTTP mapping — let them propagate unchanged.
            raise
        except Exception as exc:  # noqa: BLE001 — annotate with stage, then wrap
            raise PipelineError(
                f"Pipeline failed during stage '{current_stage}': {exc}"
            ) from exc

        total_ms = sum(s.time_ms for s in stages)
        metrics.processing_time_ms = total_ms
        logger.info("Pipeline %s complete in %.1f ms", job_id, total_ms)

        return DehazeResponse(
            success=True,
            job_id=job_id,
            original_image_b64=image_utils.to_base64(original_resized),
            dehazed_image_b64=image_utils.to_base64(final_image),
            transmission_map_b64=image_utils.to_base64(atm_result.transmission_vis),
            glow_mask_b64=image_utils.to_base64(glow_result.glow_mask_vis),
            metrics=metrics,
            pipeline_stages=stages,
        )
