"""Top-level API router aggregating all route modules.

This module composes the health and dehaze route modules into a single router.
The api_router is included in main.py with prefix="/api/v1", creating endpoints like:
  - GET  /api/v1/health
  - POST /api/v1/dehaze/upload
  - GET  /api/v1/dehaze/demo
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import dehaze, health

# Main API router (versioned with /api/v1 prefix in main.py)
api_router = APIRouter()

# Include health check endpoint at root of /api/v1
api_router.include_router(health.router)

# Include dehazing endpoints under /dehaze prefix
# Routes within dehaze.router (e.g., /upload) become /api/v1/dehaze/upload
api_router.include_router(dehaze.router, prefix="/dehaze")
