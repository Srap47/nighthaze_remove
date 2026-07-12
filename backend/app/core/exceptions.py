"""
Application-specific exceptions.

Every exception derives from :class:`NightHazeError` and carries a human-readable
``message`` attribute. FastAPI exception handlers (see ``app/main.py``) map each
type to an HTTP status code and serialise ``message`` into an ``ErrorResponse``.
"""


class NightHazeError(Exception):
    """Base exception for all application errors.

    Guarantees every subclass instance exposes a ``message`` string. If no
    message is supplied, the subclass docstring is used as a sensible fallback.
    """

    def __init__(self, message: str | None = None) -> None:
        self.message: str = message or (self.__doc__ or self.__class__.__name__).strip()
        super().__init__(self.message)


class InvalidImageError(NightHazeError):
    """Raised when uploaded file is not a valid image or has invalid dimensions.

    Mapped to HTTP 400 by the FastAPI exception handler in app/main.py.
    """


class ImageTooLargeError(NightHazeError):
    """Raised when image exceeds size limits.

    Mapped to HTTP 413 (Payload Too Large) by the exception handler.
    Raised by image_utils.validate_image() for both file size and
    image dimension violations.
    """


class ModelNotLoadedError(NightHazeError):
    """Raised when FFA-Net weights file is missing or corrupted.

    Mapped to HTTP 503 (Service Unavailable). The app still starts
    gracefully if weights are missing—this exception is only raised when
    a dehazing endpoint is called. Allows the health check to report
    model_loaded=False without crashing.
    """


class PipelineError(NightHazeError):
    """Raised when any pipeline stage fails unexpectedly.

    Wraps exceptions from services (preprocessor, glow_detector, etc.)
    and adds context about which stage failed. Mapped to HTTP 500 by
    the exception handler.
    """
