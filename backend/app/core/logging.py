"""
Logging configuration.

A single :func:`setup_logging` entry point, called once at application startup
(from the FastAPI lifespan), configures the root logger with a consistent,
structured line format.
"""

from __future__ import annotations

import logging

# TWEAK NOTE: Log format
# Change _LOG_FORMAT to adjust what information appears in logs.
# Current format: timestamp | level (padded) | logger name | message
# Useful additions: %(filename)s (file), %(funcName)s (function), %(lineno)d (line number)
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(debug: bool) -> None:
    """Configure the root logger with a consistent format.

    Called once at app startup from the FastAPI lifespan context manager.
    Sets the root logger level and format so all loggers inherit consistent
    formatting (timestamp, level, module name, message).

    Args:
        debug: If True, log at DEBUG level; otherwise INFO.
             Controlled by the DEBUG environment variable.
    """
    level = logging.DEBUG if debug else logging.INFO
    # force=True ensures this config wins even if uvicorn or other frameworks
    # have already configured the root logger.
    logging.basicConfig(level=level, format=_LOG_FORMAT, force=True)
