# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""OpenMeteo weather adapter — historical, forecast and ensemble members."""

from __future__ import annotations

import datetime
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from loguru import logger

from runeflow.domain.weather import WeatherLocation, WeatherSeries
from runeflow.exceptions import DataUnavailableError, DownloadError, RateLimitError
from runeflow.ports.weather import WeatherPort

# ── API URLs ──────────────────────────────────────────────────────────────────

_DEFAULT_HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"
_DEFAULT_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
_DEFAULT_ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"

# ── Rate-limit back-off ───────────────────────────────────────────────────────

_BACKOFF_SECONDS = [30, 60, 120, 240]

# ── Variables ─────────────────────────────────────────────────────────────────

DEFAULT_HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "relative_humidity_2m",
    "precipitation",
    "is_day",
]

ENSEMBLE_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "precipitation",
]

_MAX_RETRIES = 4


class OpenMeteoAdapter(WeatherPort):
    """Download weather data from Open-Meteo."""

    def __init__(
        self,
        timezone: str = "UTC",
        historical_api: str = _DEFAULT_HISTORICAL_API,
        forecast_api: str = _DEFAULT_FORECAST_API,
        ensemble_api: str = _DEFAULT_ENSEMBLE_API,
        hourly_vars: list[str] | None = None,
    ) -> None:
        self._tz = timezone
        self._historical_api = historical_api
        self._forecast_api = forecast_api
        self._ensemble_api = ensemble_api
        self._hourly_vars = hourly_vars or DEFAULT_HOURLY_VARS

    # ── Public API ────────────────────────────────────────────────────────────

    def download_historical(
        self,
        locations: list[WeatherLocation],
        start: datetime.date,
        end: datetime.date,
    ) -> WeatherSeries:
        frames: dict[str, pd.DataFrame] = {}
        for loc in locations:
            # Download per year to respect API limits
            year_frames = []
            for year in range(start.year, end.year + 1):
                year_start = max(start, datetime.date(year, 1, 1))
                year_end = min(end, datetime.date(year, 12, 31))
                df = self._fetch_historical(loc, year_start, year_end)
                if df is not None and not df.empty:
                    year_frames.append(df)
                time.sleep(0.3)
            if year_frames:
                frames[loc.name] = pd.concat(year_frames, ignore_index=False)
        if not frames:
            raise DataUnavailableError(
                f"Open-Meteo returned no historical weather for {start}→{end}."
            )
        return WeatherSeries.from_location_frames(frames, source="open-meteo-historical")

    def download_forecast(
        self,
        locations: list[WeatherLocation],
        horizon_days: int = 9,
    ) -> WeatherSeries:
        frames: dict[str, pd.DataFrame] = {}
        for loc in locations:
            df = self._fetch_forecast(loc, forecast_days=horizon_days)
            if df is not None and not df.empty:
                frames[loc.name] = df
            time.sleep(0.3)
        if not frames:
            raise DataUnavailableError("Open-Meteo returned no forecast weather data.")
        return WeatherSeries.from_location_frames(frames, source="open-meteo-forecast")

    def download_ensemble_forecast(
        self,
        locations: list[WeatherLocation],
        horizon_days: int = 9,
    ) -> list[WeatherSeries]:
        """Return list of WeatherSeries, one per ensemble member."""
        # Collect per-member frames for each location
        member_location_frames: dict[int, dict[str, pd.DataFrame]] = {}

        for loc in locations:
            members_dict = self._fetch_ensemble_members(loc, forecast_days=horizon_days)
            if members_dict is None:
                logger.warning(f"[OpenMeteo] No ensemble members for {loc.name}")
                continue
            for m_idx, df in members_dict.items():
                member_location_frames.setdefault(m_idx, {})[loc.name] = df
            time.sleep(0.5)

        if not member_location_frames:
            raise DataUnavailableError("Open-Meteo returned no ensemble member data.")

        return [
            WeatherSeries.from_location_frames(
                loc_frames, source=f"open-meteo-ensemble-member-{m_idx}"
            )
            for m_idx, loc_frames in sorted(member_location_frames.items())
        ]

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get_with_retry(
        self,
        url: str,
        params: dict[str, Any],
        max_retries: int = _MAX_RETRIES,
    ) -> requests.Response | None:
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=60)
                if resp.status_code == 429:
                    if attempt < max_retries - 1:
                        wait = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
                        logger.warning(f"[OpenMeteo] Rate limited — waiting {wait}s…")
                        time.sleep(wait)
                        continue
                    raise RateLimitError("[OpenMeteo] Rate limit exceeded.")
                resp.raise_for_status()
                return resp
            except RateLimitError:
                raise
            except requests.RequestException as exc:
                logger.warning(
                    f"[OpenMeteo] Request failed (attempt {attempt + 1}/{max_retries}): {exc}"
                )
                if attempt == max_retries - 1:
                    raise DownloadError(f"Open-Meteo request failed: {exc}") from exc
                time.sleep(0.5)
        return None

    # ── Fetch implementations ─────────────────────────────────────────────────

    def _fetch_historical(
        self,
        loc: WeatherLocation,
        start: datetime.date,
        end: datetime.date,
    ) -> pd.DataFrame | None:
        if end > datetime.date.today():
            end = datetime.date.today()

        params: dict[str, Any] = {
            "latitude": loc.lat,
            "longitude": loc.lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": self._hourly_vars,
            "timezone": self._tz,
        }
        try:
            resp = self._get_with_retry(self._historical_api, params)
            if resp is None:
                return None
            data = resp.json()
            return self._parse_hourly(data)
        except Exception as exc:
            logger.error(f"[OpenMeteo] Historical fetch failed for {loc.name}: {exc}")
            raise

    def _fetch_forecast(self, loc: WeatherLocation, forecast_days: int = 9) -> pd.DataFrame | None:
        params: dict[str, Any] = {
            "latitude": loc.lat,
            "longitude": loc.lon,
            "hourly": self._hourly_vars,
            "forecast_days": forecast_days,
            "timezone": self._tz,
        }
        try:
            resp = self._get_with_retry(self._forecast_api, params)
            if resp is None:
                return None
            data = resp.json()
            return self._parse_hourly(data)
        except Exception as exc:
            logger.error(f"[OpenMeteo] Forecast fetch failed for {loc.name}: {exc}")
            return None

    def _fetch_ensemble_members(
        self,
        loc: WeatherLocation,
        model: str = "ecmwf_ifs025",
        forecast_days: int = 9,
    ) -> dict[int, pd.DataFrame] | None:
        params: dict[str, Any] = {
            "latitude": loc.lat,
            "longitude": loc.lon,
            "hourly": ENSEMBLE_VARS,
            "models": model,
            "forecast_days": forecast_days,
            "timezone": self._tz,
        }
        try:
            resp = self._get_with_retry(self._ensemble_api, params)
            if resp is None:
                return None
            data = resp.json()
        except Exception as exc:
            logger.error(f"[OpenMeteo] Ensemble fetch failed for {loc.name}: {exc}")
            return None

        hourly = data.get("hourly", {})
        if not hourly:
            return None

        times = pd.to_datetime(hourly.get("time", [])).tz_localize(
            self._tz, ambiguous="NaT", nonexistent="shift_forward"
        )

        # Discover number of members
        n_members = 1
        for var in ENSEMBLE_VARS:
            member_keys = [k for k in hourly if k.startswith(f"{var}_member")]
            if member_keys:
                n_members = len(member_keys) + 1  # +1 for control run
                break

        members: dict[int, pd.DataFrame] = {}
        for m in range(n_members):
            df = pd.DataFrame({"date": times})
            df = df.set_index("date")
            for var in ENSEMBLE_VARS:
                if m == 0:
                    df[var] = hourly.get(var, np.nan)
                else:
                    key = f"{var}_member{m:02d}"
                    if key in hourly:
                        df[var] = hourly[key]
                    elif not any(k.startswith(f"{var}_member") for k in hourly):
                        df[var] = hourly.get(var, np.nan)
                    else:
                        df[var] = np.nan
            members[m] = df

        logger.info(
            f"[OpenMeteo] Downloaded {n_members} ensemble members "
            f"for {loc.name}, {len(times)} timesteps."
        )
        return members

    @staticmethod
    def _parse_hourly(data: dict[str, Any]) -> pd.DataFrame | None:
        """Parse Open-Meteo JSON response into a DataFrame with DatetimeIndex."""
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            return None

        tz = data.get("timezone", "UTC")
        times = pd.to_datetime(hourly["time"]).tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        df = pd.DataFrame(index=times)
        df.index.name = "date"
        for key, values in hourly.items():
            if key != "time":
                df[key] = values

        return df
