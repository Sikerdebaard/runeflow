# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cache_dir() -> Path:
    """
    Return the OS-appropriate cache directory for runeflow.

    Respects ``XDG_CACHE_HOME`` on Linux/BSD, falls back to ``~/.cache``.
    On macOS this resolves to ``~/Library/Caches`` when XDG is not set,
    but ``~/.cache`` is also acceptable there.
    """
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "runeflow"


class AppConfig(BaseSettings):
    """
    Top-level application settings.

    All fields can be overridden by environment variables.
    Call ``AppConfig.from_env()`` to create an instance from a custom
    env dict (useful in tests).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Zone ──────────────────────────────────────────────────────────────────
    zone: str = "NL"

    # ── Storage ───────────────────────────────────────────────────────────────
    # Default: ~/.cache/runeflow  (respects XDG_CACHE_HOME on Linux)
    # Override with CACHE_DIR env var or in .env
    cache_dir: Path = Field(default_factory=_default_cache_dir)

    # ── ENTSO-E ───────────────────────────────────────────────────────────────
    entsoe_api_key: str = ""

    # ── NED (NL supplemental data) ────────────────────────────────────────────
    ned_api_key: str = ""

    # ── Open-Meteo API endpoints ──────────────────────────────────────────────
    openmeteo_historical_api: str = "https://archive-api.open-meteo.com/v1/archive"
    openmeteo_forecast_api: str = "https://api.open-meteo.com/v1/forecast"
    openmeteo_ensemble_api: str = "https://ensemble-api.open-meteo.com/v1/ensemble"

    # ── Inference ─────────────────────────────────────────────────────────────
    inference_warmup_days: int = 16
    forecast_horizon_days: int = 9

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _expand_cache_dir(cls, v: object) -> Path:
        return Path(str(v)).expanduser().resolve()

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def prices_cache_dir(self) -> Path:
        return self.cache_dir / "prices"

    @property
    def weather_cache_dir(self) -> Path:
        return self.cache_dir / "weather"

    @property
    def generation_cache_dir(self) -> Path:
        return self.cache_dir / "generation"

    @property
    def models_cache_dir(self) -> Path:
        return self.cache_dir / "models"

    @property
    def forecasts_cache_dir(self) -> Path:
        return self.cache_dir / "forecasts"

    def ensure_dirs(self) -> None:
        """Create all cache directories if they do not exist."""
        for d in [
            self.cache_dir,
            self.prices_cache_dir,
            self.weather_cache_dir,
            self.generation_cache_dir,
            self.models_cache_dir,
            self.forecasts_cache_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        zone: str | None = None,
    ) -> "AppConfig":
        """
        Create an AppConfig from a custom env dict.

        Useful in tests to inject specific values without touching
        the process environment.
        """
        env = env or dict(os.environ)
        overrides: dict[str, object] = {}
        if zone is not None:
            overrides["zone"] = zone
        # Map canonical env var names → config field names
        mapping = {
            "ENTSOE": "entsoe_api_key",
            "NED": "ned_api_key",
            "CACHE_DIR": "cache_dir",
            "LOG_LEVEL": "log_level",
        }
        for env_key, field_name in mapping.items():
            if env_key in env:
                overrides[field_name] = env[env_key]

        return cls(**overrides)