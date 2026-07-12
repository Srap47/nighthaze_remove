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
    """Raised when uploaded file is not a valid image or has invalid dimensions."""


class ImageTooLargeError(NightHazeError):
    """Raised when image exceeds size limits."""


class ModelNotLoadedError(NightHazeError):
    """Raised when FFA-Net weights file is missing or corrupted."""


class PipelineError(NightHazeError):
    """Raised when any pipeline stage fails unexpectedly."""
