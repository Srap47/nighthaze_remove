"""
Application configuration.

Central Pydantic-settings object for the NightHaze backend. All tunable
parameters — model, image constraints, physics, post-processing, and CORS —
live here so they can be overridden via environment variables or a ``.env``
file without touching code.

The default ``model_weights_path`` is expressed *relative to the ``backend/``
working directory* (``../ml/weights/...``), because the API process is started
from ``backend/`` (see the Makefile). This keeps the checkout portable: no
absolute machine-specific paths are baked in.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings, populated from env / ``.env``."""

    # App metadata
    app_name: str = "NightHaze API"
    app_version: str = "1.0.0"
    debug: bool = False

    # TWEAK NOTE: Model configuration
    # These settings control FFA-Net architecture. Do not change unless using
    # different pretrained weights that were trained with different parameters.
    model_weights_path: str = "../ml/weights/its_train_ffa_3_19.pkl"
    model_gps: int = 3       # FFA-Net groups (must match checkpoint)
    model_blocks: int = 19   # FFA-Net blocks per group (must match checkpoint)
    device: str = "cpu"      # 'cuda' if GPU available; override via DEVICE env var

    # TWEAK NOTE: Image processing constraints
    # max_image_size_mb: limits upload size (HTTP 413 if exceeded)
    # max_image_dimension: images taller/wider than this are downscaled in preprocessing
    # min_image_dimension: images smaller than this are rejected (need meaningful detail)
    # ffa_input_size: FFA-Net always runs at this resolution for consistent inference time
    max_image_size_mb: int = 10
    max_image_dimension: int = 2048
    min_image_dimension: int = 64
    ffa_input_size: int = 512

    # TWEAK NOTE: Physics parameters (atmospheric scattering model)
    # transmission_min_clip: prevents division by zero in radiance recovery J=(I-A)/max(t,t0)
    # omega: controls dehazing strength (0.95 = keep some haze for natural look)
    transmission_min_clip: float = 0.1
    omega: float = 0.95

    # TWEAK NOTE: Post-processing tweaks
    # These affect the final visual quality. Adjust to tune sharpness vs. noise:
    # - clahe_clip_limit: higher = more local contrast (0-4 typical range)
    # - clahe_tile_size: smaller = more localized enhancement
    # - saturation_boost: > 1.0 adds vibrancy, < 1.0 desaturates
    # - sharpen_alpha/beta: unsharp mask coefficients (alpha-beta = kernel)
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    saturation_boost: float = 1.2
    sharpen_alpha: float = 1.5
    sharpen_beta: float = -0.5

    # TWEAK NOTE: CORS allowed origins
    # Frontend app origins permitted to make requests from a browser.
    # Add production domain here when deploying.
    allowed_origins: list[str] = [
        "http://localhost:5173",  # Vite dev server default port
        "http://localhost:3000",  # Alternative dev port
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Several fields are named ``model_*`` (model_weights_path, model_gps,
        # model_blocks); opt out of Pydantic's protected "model_" namespace.
        protected_namespaces=(),
    )


settings = Settings()
