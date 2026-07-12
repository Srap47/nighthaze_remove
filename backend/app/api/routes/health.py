"""Health / readiness route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(request: Request) -> HealthResponse:
    """Report service health status and deployment metadata.

    Used by monitoring systems (Kubernetes, load balancers) and the frontend to check
    if the backend is ready for requests. Critical status: model_loaded flag indicates
    if the FFA-Net model weights were successfully loaded at startup.

    Returns:
        HealthResponse with:
        - status: "ok" (always succeeds if this endpoint runs)
        - model_loaded: bool indicating if FFA-Net weights are available
        - model_name: identifier of the loaded model architecture
        - device: where inference runs (cuda or cpu)
        - version: app version from settings

    Note:
        If model_loaded=false, POST /dehaze endpoints will return 503 (service unavailable).
        This allows graceful degradation if model weights fail to download during deployment.
    """
    pipeline = request.app.state.pipeline
    return HealthResponse(
        status="ok",  # Always returns successfully if endpoint is reachable
        model_loaded=pipeline.ffa_service.model_loaded,  # False if weights missing/failed to load
        model_name="FFA-Net (its_train_ffa_3_19)",  # Model architecture identifier
        device=settings.device,  # cuda or cpu
        version=settings.app_version,  # Semantic version for deployment tracking
    )
