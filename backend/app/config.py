from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or backend/.env."""

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "LipiSetu"
    env: str = "development"

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'lipisetu.db').as_posix()}"
    storage_dir: Path = BACKEND_DIR / "data" / "storage"
    frontend_dist: Path = BACKEND_DIR.parent / "frontend" / "dist"

    jwt_secret: str = "dev-only-change-me-in-production-0f3c9a"
    jwt_access_minutes: int = 60 * 8
    jwt_refresh_days: int = 7

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    max_upload_mb: int = 25
    max_image_pixels: int = 60_000_000
    processing_max_side: int = 2400

    # Job queue
    embedded_worker: bool = True
    worker_poll_seconds: float = 1.0
    job_lease_seconds: int = 300
    job_max_attempts: int = 3

    # Rate limits (requests per minute per client)
    rate_limit_per_minute: int = 240
    rate_limit_multiplier: float = 1.0  # scales every per-endpoint bucket (auth, upload, OCR, search...)
    ocr_jobs_per_minute: int = 30
    # The CRNN engine is only offered when its checkpoint reports a validation CER at or below this.
    crnn_max_val_cer: float = 0.30
    # "auto" reads with the ensemble (per-line choice between engines) whenever two or more local engines exist.
    auto_use_ensemble: bool = True
    # A page is reported PARTIAL (not COMPLETE) when most text boxes were rejected or the read is weak.
    ocr_max_dropped_fraction: float = 0.5
    ocr_min_mean_confidence: float = 0.5

    # Optional providers
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    tesseract_cmd: str = ""
    osrm_url: str = "https://router.project-osrm.org"
    http_timeout_seconds: float = 8.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    if settings.is_sqlite:
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return settings
