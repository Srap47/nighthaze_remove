"""Top-level API router aggregating all route modules."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import dehaze, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(dehaze.router, prefix="/dehaze")
