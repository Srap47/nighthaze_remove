"""
Logging configuration.

A single :func:`setup_logging` entry point, called once at application startup
(from the FastAPI lifespan), configures the root logger with a consistent,
structured line format.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(debug: bool) -> None:
    """Configure the root logger.

    Args:
        debug: If True, log at DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    # force=True so we win even if a host (e.g. uvicorn) already touched root.
    logging.basicConfig(level=level, format=_LOG_FORMAT, force=True)
