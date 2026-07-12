"""
Dehazing routes.

- ``POST /dehaze/upload`` — dehaze a user-uploaded image.
- ``GET  /dehaze/demo``   — dehaze a bundled sample so the frontend can demo
  without an upload.

Both routes are ``async`` and offload the CPU-bound pipeline to the default
thread-pool executor so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import settings
from app.models.schemas import DehazeResponse
from app.services import image_utils

router = APIRouter()

# Whitelist of accepted image MIME types (security: prevents non-image uploads)
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Path to bundled demo image used for /dehaze/demo endpoint
# Resolved relative to this file to be independent of working directory
# (routes → api → app → backend) so the current working directory is irrelevant.
_DEMO_FIXTURE = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sample_hazy_night.jpg"
)


async def _run_pipeline(request: Request, image) -> DehazeResponse:
    """Execute the CPU-bound dehazing pipeline in a thread-pool executor.

    The dehazing pipeline (FFA-Net inference, CLAHE, etc.) is synchronous and
    computationally expensive. Running it on the event loop's main thread would
    block other requests. This helper offloads it to the default thread-pool
    executor so the async event loop remains responsive to other requests.

    Args:
        request: FastAPI request (provides access to app.state.pipeline)
        image: numpy array (uint8 BGR) to dehaze

    Returns:
        DehazeResponse with dehazed image (base64), metrics, and timing info
    """
    pipeline = request.app.state.pipeline
    loop = asyncio.get_event_loop()
    # partial() binds 'image' arg; executor calls pipeline.process(image) in a thread
    return await loop.run_in_executor(None, partial(pipeline.process, image))


@router.post("/upload", response_model=DehazeResponse, tags=["dehaze"])
async def dehaze_upload(request: Request, image: UploadFile = File(...)) -> DehazeResponse:
    """Dehaze a user-uploaded image.

    Accepts JPEG, PNG, or WebP images (up to configured size limit).
    Offloads CPU-intensive pipeline to thread-pool executor.

    Args:
        image: Multipart form-file upload (Content-Type validated against whitelist)

    Returns:
        DehazeResponse with:
        - dehazed_image_base64: output image (PNG-encoded)
        - metrics: quality scores (PSNR, SSIM, BRISQUE, colorfulness, etc.)
        - processing_time_ms: end-to-end timing
        - processing_stages: stage-by-stage timing breakdown

    Raises:
        HTTPException 400: unsupported MIME type
        HTTPException 413: file too large
        (Domain exceptions from pipeline converted to HTTP by exception handlers)
    """
    # Validate MIME type against whitelist (security: prevent non-image uploads)
    if image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{image.content_type}'. "
                "Allowed: image/jpeg, image/png, image/webp."
            ),
        )

    # Read file and validate size before processing (fail-fast to avoid wasted CPU)
    file_bytes = await image.read()
    # TWEAK NOTE: max_image_size_mb (from settings) sets the upload size limit
    # Lower limit = faster validation but restricts max image resolution
    # Higher limit = supports larger images but uses more memory/bandwidth
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {settings.max_image_size_mb} MB size limit.",
        )

    # Decode file bytes to numpy array (uint8 BGR format)
    img = image_utils.file_to_numpy(file_bytes)
    # Execute pipeline in thread pool (see _run_pipeline for async details)
    return await _run_pipeline(request, img)


@router.get("/demo", response_model=DehazeResponse, tags=["dehaze"])
async def dehaze_demo(request: Request) -> DehazeResponse:
    """Dehaze the bundled sample nighttime image (no upload required).

    This endpoint allows the frontend to demonstrate the dehazing capability
    without requiring the user to upload an image first. Uses a pre-bundled
    sample nighttime photograph (sample_hazy_night.jpg).

    Returns:
        DehazeResponse: same format as /upload, with demo image and metrics

    Raises:
        HTTPException 404: fixture file missing (broken build/deployment)
    """
    # Verify the demo fixture exists (sanity check for deployment)
    if not _DEMO_FIXTURE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Demo fixture not found at {_DEMO_FIXTURE}.",
        )

    # Load demo fixture and run pipeline (same as /upload)
    img = image_utils.file_to_numpy(_DEMO_FIXTURE.read_bytes())
    return await _run_pipeline(request, img)
