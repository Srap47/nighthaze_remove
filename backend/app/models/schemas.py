"""
Pydantic request/response schemas for the NightHaze API.

These models define the JSON contract exposed to the frontend. Image payloads
are transported as base64 PNG data URIs (``data:image/png;base64,...``) so the
response is self-contained and directly assignable to an ``<img src>``.
"""

from pydantic import BaseModel, ConfigDict


class PipelineStage(BaseModel):
    """Timing record for a single stage of the dehazing pipeline.

    Returned in the pipeline_stages array so the frontend can show which
    stage took the longest and attribute slowdowns correctly.
    """

    stage: str       # "preprocessing", "glow_detection", etc.
    time_ms: float   # Wall time in milliseconds


class DehazeMetrics(BaseModel):
    """Quantitative quality metrics comparing input and dehazed output.

    Includes both full-reference metrics (PSNR, SSIM — require a reference)
    and no-reference metrics (NIQE, BRISQUE — estimate quality from the image alone).
    Also includes perceptual metrics like visibility (haze removal estimate)
    and colorfulness (vibrancy measurement).
    """

    # Full-reference metrics (compare original vs dehazed)
    psnr: float           # Peak Signal-to-Noise Ratio; higher is better (dB)
    ssim: float           # Structural Similarity Index; higher is better (0-1)

    # No-reference quality metrics
    niqe: float           # Naturalness Image Quality Evaluator; lower is more natural
    brisque: float        # Blind/Referenceless Image Spatial Quality Evaluator; lower is better (0-100)
                          # Special value: -1 means unavailable (e.g. on systems without scikit-image)

    # Perceptual metrics
    visibility_score: float              # Estimated haze removal (0-1); higher = more haze removed
    colorfulness_before: float           # Hasler & Susstrunk colorfulness in input
    colorfulness_after: float            # Hasler & Susstrunk colorfulness in output
    colorfulness_improvement_pct: float  # Percentage change in colorfulness

    # Timing
    processing_time_ms: float  # Sum of all pipeline stage times


class DehazeResponse(BaseModel):
    """Successful dehazing result returned to the client.

    Contains 4 base64-encoded PNG images (original, dehazed, transmission map,
    glow mask) suitable for direct assignment to <img src>, plus metrics and
    timing information. All images returned as data: URIs so no separate fetch
    is needed.
    """

    success: bool               # Always true for this schema
    job_id: str                 # UUID for this processing run (useful for logging)
    # All images are base64-encoded PNG data URIs: data:image/png;base64,...
    original_image_b64: str      # Input image (after processing-cap resize)
    dehazed_image_b64: str       # Final result (the dehazed image)
    transmission_map_b64: str    # Transmission estimate (grayscale: 0=opaque, 1=clear)
    glow_mask_b64: str           # Detected light sources (green overlay visualization)
    metrics: DehazeMetrics       # Quality scores and timings
    pipeline_stages: list[PipelineStage]  # Per-stage wall times (always 6 entries)


class HealthResponse(BaseModel):
    """Service health / readiness payload.

    Returned by the GET /api/v1/health endpoint. Allows the frontend to detect
    whether the model loaded successfully and which device (CPU/CUDA) is in use.
    The service starts even if weights are missing; this endpoint reports
    model_loaded=False in that case rather than crashing.
    """

    # ``model_loaded`` / ``model_name`` collide with Pydantic's protected
    # "model_" namespace; opt out so the field names can stay as specified.
    model_config = ConfigDict(protected_namespaces=())

    status: str              # "ok" = service is running normally
    model_loaded: bool       # True if FFA-Net weights loaded successfully
    model_name: str          # "FFA-Net (its_train_ffa_3_19)" or similar
    device: str              # "cpu" or "cuda"
    version: str             # App version (e.g. "1.0.0")


class ErrorResponse(BaseModel):
    """Uniform error envelope for all failure responses.

    All HTTP error responses (400, 404, 413, 422, 500, 503) use this schema
    so the frontend has a single error handling pattern. The error field is
    a machine-readable code; detail is the human-readable message.
    """

    success: bool = False   # Always False for errors
    error: str              # Machine-readable error code: "not_found", "payload_too_large", etc.
    detail: str | None = None  # Human-readable error message (may be None for generic errors)
