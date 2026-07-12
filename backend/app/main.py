"""
NightHaze FastAPI application.

Wires together the lifespan (which builds the dehazing pipeline once at startup),
CORS, domain-exception handlers, and the versioned API router.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.config import settings
from app.core.exceptions import (
    ImageTooLargeError,
    InvalidImageError,
    ModelNotLoadedError,
    PipelineError,
)
from app.core.logging import setup_logging
from app.core.pipeline import DehazingPipeline
from app.models.schemas import ErrorResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the dehazing pipeline at startup; log lifecycle events."""
    setup_logging(settings.debug)
    t0 = time.time()
    logger.info("Starting %s v%s — initializing pipeline...", settings.app_name, settings.app_version)

    pipeline = DehazingPipeline(settings)
    app.state.pipeline = pipeline

    logger.info(
        "Pipeline ready in %.2fs | device=%s | model_loaded=%s",
        time.time() - t0,
        settings.device,
        pipeline.ffa_service.model_loaded,
    )
    yield
    logger.info("Shutting down %s.", settings.app_name)


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_json(status_code: int, error: str, detail: str | None = None) -> JSONResponse:
    """Build a JSONResponse wrapping an :class:`ErrorResponse`."""
    payload = ErrorResponse(error=error, detail=detail).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(InvalidImageError)
async def _handle_invalid_image(request: Request, exc: InvalidImageError) -> JSONResponse:
    return _error_json(400, "invalid_image", exc.message)


@app.exception_handler(ImageTooLargeError)
async def _handle_image_too_large(request: Request, exc: ImageTooLargeError) -> JSONResponse:
    return _error_json(413, "image_too_large", exc.message)


@app.exception_handler(ModelNotLoadedError)
async def _handle_model_not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    return _error_json(503, "model_not_loaded", exc.message)


@app.exception_handler(PipelineError)
async def _handle_pipeline_error(request: Request, exc: PipelineError) -> JSONResponse:
    logger.exception("Pipeline error: %s", exc.message)
    return _error_json(500, "pipeline_error", exc.message)


# Maps HTTP status codes to the ``error`` label in our ErrorResponse envelope.
_HTTP_ERROR_LABELS = {
    400: "bad_request",
    404: "not_found",
    413: "payload_too_large",
}


@app.exception_handler(StarletteHTTPException)
async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Reformat any HTTPException (including framework 404/405) into ErrorResponse.

    Registered against Starlette's base HTTPException so it also captures
    framework-generated errors (e.g. unknown-route 404s), not just the
    ``fastapi.HTTPException`` instances raised by our routes.
    """
    error = _HTTP_ERROR_LABELS.get(exc.status_code, "http_error")
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    payload = ErrorResponse(error=error, detail=detail).model_dump()
    return JSONResponse(
        status_code=exc.status_code,
        content=payload,
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Reformat request-validation (422) errors into a readable ErrorResponse."""
    messages = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "invalid")
        messages.append(f"{loc}: {msg}" if loc else msg)
    detail = "; ".join(messages) if messages else "Request validation failed."
    return _error_json(422, "validation_error", detail)


@app.exception_handler(Exception)
async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled errors. Logs the traceback; never leaks it."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return _error_json(500, "internal_error", "An unexpected error occurred.")


app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    """Welcome endpoint."""
    return {"message": "NightHaze API", "docs": "/docs"}
