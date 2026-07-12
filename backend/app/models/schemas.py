"""
Pydantic request/response schemas for the NightHaze API.

These models define the JSON contract exposed to the frontend. Image payloads
are transported as base64 PNG data URIs (``data:image/png;base64,...``) so the
response is self-contained and directly assignable to an ``<img src>``.
"""

from pydantic import BaseModel, ConfigDict


class PipelineStage(BaseModel):
    """Timing record for a single stage of the dehazing pipeline."""

    stage: str
    time_ms: float


class DehazeMetrics(BaseModel):
    """Quantitative quality metrics comparing input and dehazed output."""

    psnr: float
    ssim: float
    niqe: float
    brisque: float
    visibility_score: float
    colorfulness_before: float
    colorfulness_after: float
    colorfulness_improvement_pct: float
    processing_time_ms: float


class DehazeResponse(BaseModel):
    """Successful dehazing result returned to the client."""

    success: bool
    job_id: str
    original_image_b64: str      # "data:image/png;base64,..."
    dehazed_image_b64: str       # "data:image/png;base64,..."
    transmission_map_b64: str    # "data:image/png;base64,..." (grayscale visualization)
    glow_mask_b64: str           # "data:image/png;base64,..." (binary mask visualization)
    metrics: DehazeMetrics
    pipeline_stages: list[PipelineStage]


class HealthResponse(BaseModel):
    """Service health / readiness payload."""

    # ``model_loaded`` / ``model_name`` collide with Pydantic's protected
    # "model_" namespace; opt out so the field names can stay as specified.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str
    device: str
    version: str


class ErrorResponse(BaseModel):
    """Uniform error envelope for all failure responses."""

    success: bool = False
    error: str
    detail: str | None = None
