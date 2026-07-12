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

    app_name: str = "NightHaze API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model — FFA-Net pretrained weights.
    # Relative to backend/ (the process CWD), so it resolves to <repo>/ml/weights/...
    model_weights_path: str = "../ml/weights/its_train_ffa_3_19.pkl"
    model_gps: int = 3       # FFA-Net groups
    model_blocks: int = 19   # FFA-Net blocks per group
    device: str = "cpu"      # 'cuda' if a GPU is available; overridable via DEVICE

    # Image constraints
    max_image_size_mb: int = 10
    max_image_dimension: int = 2048
    min_image_dimension: int = 64
    ffa_input_size: int = 512   # FFA-Net inference resize target

    # Physics parameters
    transmission_min_clip: float = 0.1
    omega: float = 0.95          # dehazing strength

    # Post-processing
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    saturation_boost: float = 1.2
    sharpen_alpha: float = 1.5
    sharpen_beta: float = -0.5

    # API — origins permitted by the CORS middleware
    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
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
