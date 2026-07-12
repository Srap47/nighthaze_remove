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
    """Build the dehazing pipeline at startup; log lifecycle events.

    This context manager is called once when FastAPI starts up and once when shutting down.
    It initializes the shared DehazingPipeline (services, models) and makes it available
    to all request handlers via app.state.pipeline.
    """
    # Configure logging based on debug flag (verbose in debug mode)
    setup_logging(settings.debug)
    t0 = time.time()
    logger.info("Starting %s v%s — initializing pipeline...", settings.app_name, settings.app_version)

    # Instantiate the complete dehazing pipeline (all services: preprocessor, glow detector, etc.)
    # Pipeline is built once and reused for all requests (computationally expensive to recreate)
    pipeline = DehazingPipeline(settings)
    app.state.pipeline = pipeline

    # Report readiness: how long startup took, device (GPU/CPU), and model load status
    logger.info(
        "Pipeline ready in %.2fs | device=%s | model_loaded=%s",
        time.time() - t0,
        settings.device,
        pipeline.ffa_service.model_loaded,
    )
    yield  # Application runs while this yields
    logger.info("Shutting down %s.", settings.app_name)  # Cleanup on shutdown


# Initialize FastAPI app with custom lifespan handler (pipeline initialization/cleanup)
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

# Configure CORS (Cross-Origin Resource Sharing) to allow frontend requests
# TWEAK NOTE: allowed_origins (from settings) controls which domains can access the API
# Should be restricted to trusted frontend URLs in production (not "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,  # Allow cookies/auth tokens across origins
    allow_methods=["*"],     # All HTTP methods (GET, POST, etc.)
    allow_headers=["*"],     # All headers (Authorization, etc.)
)


def _error_json(status_code: int, error: str, detail: str | None = None) -> JSONResponse:
    """Build a JSONResponse wrapping an ErrorResponse schema.

    Ensures consistent error response format: { error, detail, timestamp }
    """
    payload = ErrorResponse(error=error, detail=detail).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


# ============== Exception Handlers (Domain Exceptions) ==============
# These catch and convert domain exceptions (raised by services/pipeline) into HTTP responses.
# Order matters: more specific exceptions handled first, then general fallbacks.

@app.exception_handler(InvalidImageError)
async def _handle_invalid_image(request: Request, exc: InvalidImageError) -> JSONResponse:
    """Handle invalid image errors (corrupt file, unsupported format, wrong dimensions)."""
    return _error_json(400, "invalid_image", exc.message)


@app.exception_handler(ImageTooLargeError)
async def _handle_image_too_large(request: Request, exc: ImageTooLargeError) -> JSONResponse:
    """Handle image size violations (exceeds max dimension)."""
    return _error_json(413, "image_too_large", exc.message)


@app.exception_handler(ModelNotLoadedError)
async def _handle_model_not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    """Handle model loading failures (weights file missing, hardware incompatibility).

    Returns 503 (Service Unavailable) — server is degraded but may recover after weights download.
    """
    return _error_json(503, "model_not_loaded", exc.message)


@app.exception_handler(PipelineError)
async def _handle_pipeline_error(request: Request, exc: PipelineError) -> JSONResponse:
    """Handle general pipeline failures (internal processing errors).

    Logs full exception traceback for debugging; returns generic message to client.
    """
    logger.exception("Pipeline error: %s", exc.message)
    return _error_json(500, "pipeline_error", exc.message)


# ============== HTTP Framework Exception Handler ==============
# Maps HTTP status codes to human-readable error labels for ErrorResponse envelope
_HTTP_ERROR_LABELS = {
    400: "bad_request",      # Malformed request
    404: "not_found",        # Route not found
    413: "payload_too_large", # Request body too large
}


@app.exception_handler(StarletteHTTPException)
async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Reformat any HTTPException (including framework 404/405) into ErrorResponse.

    Registered against Starlette's base HTTPException so it also captures
    framework-generated errors (e.g. unknown-route 404s), not just the
    fastapi.HTTPException instances raised by our routes. Ensures all HTTP errors
    use the same ErrorResponse format.
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
    """Reformat Pydantic request-validation errors (422) into a readable ErrorResponse.

    Extracts validation errors from each field and combines them into a single
    detail message (e.g., "image_data: field required; file_type: invalid enum value").
    """
    messages = []
    for err in exc.errors():
        # Build field path (e.g., "request.body.image_data")
        loc = ".".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "invalid")
        messages.append(f"{loc}: {msg}" if loc else msg)
    detail = "; ".join(messages) if messages else "Request validation failed."
    return _error_json(422, "validation_error", detail)


@app.exception_handler(Exception)
async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any unhandled exceptions in route handlers or middleware.

    Logs the full traceback for debugging; returns generic message to client
    without exposing internal details (security best practice).
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return _error_json(500, "internal_error", "An unexpected error occurred.")


app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    """Welcome endpoint."""
    return {"message": "NightHaze API", "docs": "/docs"}
