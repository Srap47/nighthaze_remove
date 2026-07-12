"""Health / readiness route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(request: Request) -> HealthResponse:
    """Report service status and whether the FFA-Net model is loaded."""
    pipeline = request.app.state.pipeline
    return HealthResponse(
        status="ok",
        model_loaded=pipeline.ffa_service.model_loaded,
        model_name="FFA-Net (its_train_ffa_3_19)",
        device=settings.device,
        version=settings.app_version,
    )
