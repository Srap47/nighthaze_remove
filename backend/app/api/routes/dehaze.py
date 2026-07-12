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

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# backend/tests/fixtures/sample_hazy_night.jpg, resolved relative to this file
# (routes → api → app → backend) so the current working directory is irrelevant.
_DEMO_FIXTURE = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sample_hazy_night.jpg"
)


async def _run_pipeline(request: Request, image) -> DehazeResponse:
    """Execute the CPU-bound pipeline in the default thread-pool executor."""
    pipeline = request.app.state.pipeline
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(pipeline.process, image))


@router.post("/upload", response_model=DehazeResponse, tags=["dehaze"])
async def dehaze_upload(request: Request, image: UploadFile = File(...)) -> DehazeResponse:
    """Dehaze an uploaded image (JPEG/PNG/WebP, up to the configured size limit)."""
    if image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{image.content_type}'. "
                "Allowed: image/jpeg, image/png, image/webp."
            ),
        )

    file_bytes = await image.read()
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {settings.max_image_size_mb} MB size limit.",
        )

    img = image_utils.file_to_numpy(file_bytes)
    return await _run_pipeline(request, img)


@router.get("/demo", response_model=DehazeResponse, tags=["dehaze"])
async def dehaze_demo(request: Request) -> DehazeResponse:
    """Dehaze the bundled sample nighttime image."""
    if not _DEMO_FIXTURE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Demo fixture not found at {_DEMO_FIXTURE}.",
        )

    img = image_utils.file_to_numpy(_DEMO_FIXTURE.read_bytes())
    return await _run_pipeline(request, img)
