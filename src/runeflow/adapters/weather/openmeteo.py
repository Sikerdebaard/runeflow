# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""OpenMeteo weather adapter — historical, forecast and ensemble members."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import tempfile
import time
from typing import Any, cast
from urllib.parse import urlparse

import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from loguru import logger
from retry_requests import retry

from runeflow.domain.weather import WeatherLocation, WeatherSeries
from runeflow.exceptions import DailyRateLimitError, DataUnavailableError, DownloadError
from runeflow.ports.weather import WeatherPort

# ── API URLs ──────────────────────────────────────────────────────────────────

_DEFAULT_HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"
_DEFAULT_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
_DEFAULT_ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"

# ── HTTP-level cache TTL (seconds) ────────────────────────────────────────────

# Historical archive data is immutable, so cache it forever via urls_expire_after.
# Forecast / ensemble data is cached for one model update cycle (6 h).
# Both ICON-EU-EPS and ECMWF IFS 0.25° ensemble have update_interval_seconds=21600,
# so a 6 h TTL avoids redundant API calls within the same model run.
_HTTP_CACHE_TTL = 21600  # 6 hours — matches ensemble model update interval

# The current (still-growing) quarter uses a shorter TTL so that new
# observations are picked up periodically without causing a daily URL cache miss.
_CURRENT_QUARTER_CACHE_TTL = 14 * 24 * 3600  # 14 days

# Archive requests are chunked into sub-yearly slices to avoid triggering the
# Open-Meteo minutely rate limit (a full year of 11 variables in one request is
# enough to exhaust it).  Completed quarters are cached forever; the current
# quarter uses _CURRENT_QUARTER_CACHE_TTL above.
_ARCHIVE_CHUNK_MONTHS = 3  # quarterly

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

# ── Mixed ensemble configuration ─────────────────────────────────────────────
# ICON-EU: 13 km European regional model, hourly, 40 members, ~5-day horizon.
# ECMWF IFS 0.25°: 25 km global model, 3-hourly (interpolated → hourly),
#   51 members, ~15-day horizon.
# Combined: 91 members total; near-term (0–5 d) has both models, far-term
#   (5–9 d) has ECMWF only.  The percentile-based aggregation in
#   InferenceService._build_result handles the variable member count.
_ICON_EU_MODEL = "icon_eu"
_ICON_EU_HORIZON_DAYS = 5
_ECMWF_MODEL = "ecmwf_ifs025"
N_MIXED_ENSEMBLE_MEMBERS = 91  # 40 ICON-EU + 51 ECMWF

# Metadata endpoints for dynamic cache invalidation.  Maps the model name used
# in API calls to the Open-Meteo meta.json URL (fetched outside requests_cache).
_MODEL_META_URLS: dict[str, str] = {
    _ICON_EU_MODEL: "https://ensemble-api.open-meteo.com/data/dwd_icon_eu_eps/static/meta.json",
    _ECMWF_MODEL: "https://ensemble-api.open-meteo.com/data/ecmwf_ifs025_ensemble/static/meta.json",
}


def _iter_date_chunks(
    start: datetime.date,
    end: datetime.date,
    chunk_months: int,
) -> list[tuple[datetime.date, datetime.date]]:
    """Yield (chunk_start, chunk_end) pairs covering [start, end] inclusive.

    Each chunk spans at most *chunk_months* calendar months.  The last chunk
    is clipped to *end* so the caller never receives out-of-range dates.
    """
    chunks: list[tuple[datetime.date, datetime.date]] = []
    cursor = start
    while cursor <= end:
        # First day of the next chunk = cursor + chunk_months months
        m = cursor.month - 1 + chunk_months
        next_cursor = datetime.date(cursor.year + m // 12, m % 12 + 1, 1)
        chunk_end = min(next_cursor - datetime.timedelta(days=1), end)
        chunks.append((cursor, chunk_end))
        cursor = next_cursor
    return chunks


def _biweekly_anchor(d: datetime.date) -> datetime.date:
    """Return the most recent fortnightly anchor date on or before *d*.

    Rounds *d* down to the nearest 14-day boundary derived from the proleptic
    Gregorian ordinal.  Used by :meth:`OpenMeteoAdapter.download_historical`
    to produce a stable ``end_date`` query parameter for the current quarter,
    ensuring the HTTP cache entry lives for the full
    :data:`_CURRENT_QUARTER_CACHE_TTL` instead of being invalidated every day
    as the date advances.
    """
    return datetime.date.fromordinal((d.toordinal() // 14) * 14)


class OpenMeteoAdapter(WeatherPort):
    """Download weather data from Open-Meteo.

    Uses openmeteo-requests (protobuf SDK) with HTTP-level caching via
    requests-cache and automatic retries via retry-requests.  Forecast and
    ensemble requests are batched: all locations for a zone are fetched in a
    single API call instead of one call per location.
    """

    def __init__(
        self,
        timezone: str = "UTC",
        historical_api: str = _DEFAULT_HISTORICAL_API,
        forecast_api: str = _DEFAULT_FORECAST_API,
        ensemble_api: str = _DEFAULT_ENSEMBLE_API,
        hourly_vars: list[str] | None = None,
        http_cache_dir: str | None = None,
    ) -> None:
        self._tz = timezone
        self._historical_api = historical_api
        self._forecast_api = forecast_api
        self._ensemble_api = ensemble_api
        self._hourly_vars = hourly_vars or DEFAULT_HOURLY_VARS

        cache_path = os.path.join(http_cache_dir or tempfile.gettempdir(), "openmeteo_http_cache")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # Historical archive responses never change — cache them forever so the free-tier
        # hourly/daily quota is not exhausted across the 3 daily cron runs.
        # Forecast/ensemble responses fall back to a 6 h TTL; the dynamic
        # _maybe_invalidate_ensemble_cache() mechanism proactively clears them
        # as soon as a new model run is detected via each model's meta.json.
        cache_session = requests_cache.CachedSession(
            cache_path,
            expire_after=_HTTP_CACHE_TTL,
            urls_expire_after={
                "https://archive-api.open-meteo.com/*": requests_cache.NEVER_EXPIRE,
            },
        )
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self._client = openmeteo_requests.Client(session=retry_session)
        # Keep a direct reference to the cache backend for targeted invalidation.
        self._cache_session = cache_session
        # Sidecar JSON file: {model_name: last_run_availability_time} (Unix ts)
        self._model_state_path = pathlib.Path(cache_path).parent / "openmeteo_model_state.json"

    # ── Public API ────────────────────────────────────────────────────────────

    def download_historical(
        self,
        locations: list[WeatherLocation],
        start: datetime.date,
        end: datetime.date,
    ) -> WeatherSeries:
        today = datetime.date.today()
        frames: dict[str, pd.DataFrame] = {}
        for loc in locations:
            # Download in quarterly chunks to avoid triggering the minutely rate
            # limit.  Completed (past) quarters are cached forever; the current
            # quarter is pinned to a stable fortnightly boundary and cached for
            # 14 days so recent observations are refreshed periodically.
            chunk_frames = []
            for chunk_start, chunk_end in _iter_date_chunks(start, end, _ARCHIVE_CHUNK_MONTHS):
                is_current = chunk_end >= today
                if is_current:
                    # Round today down to the nearest fortnightly boundary so
                    # the URL is stable for 14 days.  max() guards against the
                    # anchor falling before the start of a brand-new quarter.
                    effective_end = max(_biweekly_anchor(today), chunk_start)
                    original_ue = self._cache_session.settings.urls_expire_after
                    self._cache_session.settings.urls_expire_after = {
                        "https://archive-api.open-meteo.com/*": _CURRENT_QUARTER_CACHE_TTL,
                    }
                else:
                    effective_end = chunk_end
                try:
                    df = self._fetch_historical(loc, chunk_start, effective_end)
                finally:
                    if is_current:
                        self._cache_session.settings.urls_expire_after = original_ue
                if df is not None and not df.empty:
                    chunk_frames.append(df)
            if chunk_frames:
                frames[loc.name] = pd.concat(chunk_frames, ignore_index=False)
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
        raw = self._fetch_batch(
            self._forecast_api,
            locations,
            hourly=self._hourly_vars,
            forecast_days=horizon_days,
        )
        frames: dict[str, pd.DataFrame] = {}
        for loc, response in zip(locations, raw, strict=False):
            if response is None:
                continue
            df = self._parse_hourly_sdk(response, self._hourly_vars)
            if df is not None and not df.empty:
                frames[loc.name] = df
        if not frames:
            raise DataUnavailableError(
                "Open-Meteo returned no forecast weather data "
                f"(all {len(locations)} location(s) failed or returned empty responses)."
            )
        return WeatherSeries.from_location_frames(frames, source="open-meteo-forecast")

    def download_ensemble_forecast(
        self,
        locations: list[WeatherLocation],
        horizon_days: int = 9,
    ) -> list[WeatherSeries]:
        """Download mixed ensemble: ICON-EU (hourly, ≤5 d) + ECMWF IFS (3 h→1 h, ≤9 d).

        Returns a flat list of WeatherSeries: ICON-EU members first
        (indices 0–39), then ECMWF members (indices 40–90).  Near-term
        timestamps have 91 members, far-term (>5 d) has 51 ECMWF-only.
        """
        all_member_frames: dict[int, dict[str, pd.DataFrame]] = {}

        # ── ICON-EU: hourly, ~5 days, 40 members ─────────────────────────
        icon_frames = self._fetch_ensemble_model(
            locations,
            _ICON_EU_MODEL,
            min(horizon_days, _ICON_EU_HORIZON_DAYS),
            interpolate_to_hourly=False,
        )
        for m_idx, loc_frames in sorted(icon_frames.items()):
            all_member_frames[m_idx] = loc_frames
        icon_count = len(icon_frames)

        # ── ECMWF IFS: 3-hourly → interpolated to hourly, ~9 days, 51 members
        ecmwf_frames = self._fetch_ensemble_model(
            locations,
            _ECMWF_MODEL,
            horizon_days,
            interpolate_to_hourly=True,
        )
        for m_idx, loc_frames in sorted(ecmwf_frames.items()):
            all_member_frames[icon_count + m_idx] = loc_frames

        if not all_member_frames:
            raise DataUnavailableError("Open-Meteo returned no ensemble member data.")

        logger.info(
            "[OpenMeteo] Mixed ensemble: {} ICON-EU + {} ECMWF = {} total members",
            icon_count,
            len(ecmwf_frames),
            len(all_member_frames),
        )

        return [
            WeatherSeries.from_location_frames(
                loc_frames, source=f"open-meteo-ensemble-member-{m_idx}"
            )
            for m_idx, loc_frames in sorted(all_member_frames.items())
        ]

    # ── Dynamic ensemble cache invalidation ────────────────────────────────

    def _get_model_run_availability(self, model: str) -> int | None:
        """Return last_run_availability_time (Unix ts) from the model's meta.json.

        Uses a plain (non-cached) HTTP GET so we always see the latest value.
        Returns None on any failure so the caller can skip invalidation safely.
        """
        meta_url = _MODEL_META_URLS.get(model)
        if not meta_url:
            return None
        try:
            resp = requests.get(meta_url, timeout=5)
            resp.raise_for_status()
            return int(resp.json()["last_run_availability_time"])
        except Exception as exc:
            logger.warning("[OpenMeteo] Could not read meta for {}: {}", model, exc)
            return None

    def _load_model_state(self) -> dict[str, int]:
        """Load persisted {model: last_seen_availability_time} from the sidecar file."""
        try:
            return json.loads(self._model_state_path.read_text())  # type: ignore[no-any-return]
        except Exception:
            return {}

    def _save_model_state(self, state: dict[str, int]) -> None:
        """Persist {model: last_seen_availability_time} to the sidecar file."""
        try:
            self._model_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._model_state_path.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            logger.warning("[OpenMeteo] Could not save model state: {}", exc)

    def _maybe_invalidate_ensemble_cache(self, model: str) -> None:
        """Proactively invalidate cached ensemble responses when the model has a new run.

        Fetches the model's meta.json, compares last_run_availability_time with
        the value stored in the sidecar state file.  If a newer run is available,
        all cached responses for the ensemble API host are deleted so the next
        _fetch_batch call retrieves fresh data.  The 6 h TTL acts as a fallback
        when this check cannot reach the metadata endpoint.
        """
        availability_time = self._get_model_run_availability(model)
        if availability_time is None:
            return

        state = self._load_model_state()
        if availability_time <= state.get(model, 0):
            logger.debug(
                "[OpenMeteo] {} ensemble cache is current (run={})",
                model,
                datetime.datetime.fromtimestamp(availability_time, tz=datetime.UTC).isoformat(),
            )
            return

        run_dt = datetime.datetime.fromtimestamp(availability_time, tz=datetime.UTC)
        logger.info(
            "[OpenMeteo] {} has a new run ({}), invalidating ensemble cache",
            model,
            run_dt.isoformat(),
        )
        try:
            parsed = urlparse(self._ensemble_api)
            self._cache_session.cache.delete(urls=[f"{parsed.scheme}://{parsed.netloc}/*"])
        except Exception as exc:
            logger.warning("[OpenMeteo] Cache invalidation failed for {}: {}", model, exc)
            return  # Don't update state — we'll retry on the next call

        state[model] = availability_time
        self._save_model_state(state)

    # ── Ensemble model helpers ───────────────────────────────────────────────

    def _fetch_ensemble_model(
        self,
        locations: list[WeatherLocation],
        model: str,
        forecast_days: int,
        *,
        interpolate_to_hourly: bool = False,
    ) -> dict[int, dict[str, pd.DataFrame]]:
        """Fetch ensemble data for a single NWP model.

        Returns ``{member_idx: {location_name: DataFrame}}``.  When
        *interpolate_to_hourly* is set, sub-hourly or multi-hourly data
        (e.g. ECMWF's 3-hourly) is resampled + linearly interpolated to 1 h.
        """
        self._maybe_invalidate_ensemble_cache(model)
        raw = self._fetch_batch(
            self._ensemble_api,
            locations,
            hourly=ENSEMBLE_VARS,
            models=model,
            forecast_days=forecast_days,
        )
        member_frames: dict[int, dict[str, pd.DataFrame]] = {}
        for loc, response in zip(locations, raw, strict=False):
            if response is None:
                logger.warning("[OpenMeteo] No {} ensemble data for {}", model, loc.name)
                continue
            members = self._parse_ensemble_sdk(response, ENSEMBLE_VARS)
            if members is None:
                continue
            for m_idx, df in members.items():
                if interpolate_to_hourly and not df.empty:
                    freq = pd.infer_freq(cast(pd.DatetimeIndex, df.index))
                    if freq is not None and freq != "h":
                        df = df.resample("1h").interpolate(method="linear")
                member_frames.setdefault(m_idx, {})[loc.name] = df

        if member_frames:
            logger.info(
                "[OpenMeteo] {} — parsed {} members for {} locations",
                model,
                len(member_frames),
                len(locations),
            )
        return member_frames

    # ── Batch fetch ───────────────────────────────────────────────────────────

    def _fetch_batch(
        self,
        url: str,
        locations: list[WeatherLocation],
        **params: Any,
    ) -> list[Any]:
        """Fetch all locations in a single API call.

        Returns a list of raw SDK response objects aligned with *locations*.
        On failure returns ``[None, ...]`` so callers degrade gracefully.
        """
        full_params: dict[str, Any] = {
            "latitude": [loc.lat for loc in locations],
            "longitude": [loc.lon for loc in locations],
            "timezone": self._tz,
            **params,
        }
        try:
            return self._client.weather_api(url, params=full_params)
        except Exception as exc:
            exc_lower = str(exc).lower()
            is_daily = "daily" in exc_lower and "tomorrow" in exc_lower
            is_hourly = "hourly" in exc_lower and "rate" in exc_lower
            is_minutely = "minutely" in exc_lower
            is_rate_limited = (
                is_daily or is_hourly or is_minutely or "request limit exceeded" in exc_lower
            )

            if is_daily:
                logger.warning("[OpenMeteo] Daily rate limit hit on {} — aborting.", url)
                raise DailyRateLimitError(
                    f"Open-Meteo daily rate limit exceeded on {url}; quota resets tomorrow."
                ) from exc
            if is_rate_limited:
                logger.warning("[OpenMeteo] Rate limit hit on {}: {}", url, exc)
                raise DownloadError(f"Open-Meteo rate limit on {url}: {exc}") from exc

            logger.error("[OpenMeteo] Batch request to {} failed: {}", url, exc)
            return [None] * len(locations)

    # ── Historical fetch (per-location, year-chunked) ─────────────────────────

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
        _max_retries = 5
        for attempt in range(_max_retries):
            try:
                responses = self._client.weather_api(self._historical_api, params=params)
                return self._parse_hourly_sdk(responses[0], self._hourly_vars)
            except Exception as exc:
                exc_lower = str(exc).lower()
                is_minutely = "minutely" in exc_lower
                is_hourly = "hourly" in exc_lower
                is_rate_limited = is_minutely or is_hourly or "request limit exceeded" in exc_lower

                if is_rate_limited:
                    is_daily = "daily" in exc_lower and "tomorrow" in exc_lower
                    if is_daily:
                        # Daily quota exhausted — retrying today won't help at all.
                        # Raise DailyRateLimitError so the caller can abort all
                        # remaining zones and sleep until tomorrow.
                        logger.warning(
                            "[OpenMeteo] Daily rate limit hit for {} — aborting, "
                            "quota resets tomorrow.",
                            loc.name,
                        )
                        raise DailyRateLimitError(
                            f"Open-Meteo daily rate limit exceeded for {loc.name}; "
                            "quota resets tomorrow."
                        ) from exc
                    if is_hourly:
                        # Hourly quota exhausted — retrying within the same hour won't help.
                        # Raise immediately; the cron will retry at the next scheduled run.
                        logger.warning(
                            "[OpenMeteo] Hourly rate limit hit for {} — aborting, "
                            "will retry on next scheduled run.",
                            loc.name,
                        )
                        raise DownloadError(
                            f"Open-Meteo hourly rate limit exceeded for {loc.name}; "
                            "will retry on next scheduled run."
                        ) from exc
                    if attempt < _max_retries - 1:
                        wait = 65
                        logger.warning(
                            "[OpenMeteo] Minutely rate limit hit for {} ({}→{}) "
                            "— sleeping {}s before retry {}/{}",
                            loc.name,
                            start,
                            end,
                            wait,
                            attempt + 1,
                            _max_retries - 1,
                        )
                        time.sleep(wait)
                        continue
                logger.error(
                    "[OpenMeteo] Historical fetch failed for {} ({}→{}): {}",
                    loc.name,
                    start,
                    end,
                    exc,
                )
                raise DownloadError(f"Open-Meteo historical fetch failed: {exc}") from exc
        # unreachable, but satisfies type checkers
        raise DownloadError(f"Open-Meteo historical fetch failed after {_max_retries} attempts")

    # ── SDK parsers ───────────────────────────────────────────────────────────

    def _make_time_index(self, hourly: Any) -> pd.DatetimeIndex:
        return pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_convert(self._tz)

    def _parse_hourly_sdk(self, response: Any, var_names: list[str]) -> pd.DataFrame | None:
        """Parse a deterministic SDK response into a DataFrame.

        Variables are returned by the SDK in the same order they were
        requested, so we map them positionally to *var_names*.
        """
        hourly = response.Hourly()
        if hourly is None or hourly.VariablesLength() == 0:
            return None
        times = self._make_time_index(hourly)
        df = pd.DataFrame(index=times)
        df.index.name = "date"
        for i, var_name in enumerate(var_names):
            if i < hourly.VariablesLength():
                df[var_name] = hourly.Variables(i).ValuesAsNumpy()
        return df

    def _parse_ensemble_sdk(
        self,
        response: Any,
        var_names: list[str],
    ) -> dict[int, pd.DataFrame] | None:
        """Parse an ensemble SDK response into per-member DataFrames.

        The SDK returns one Variable object per (variable, member) combination.
        We group by ``EnsembleMember()`` and map variables positionally within
        each group (preserving request order).
        """
        hourly = response.Hourly()
        if hourly is None or hourly.VariablesLength() == 0:
            return None
        times = self._make_time_index(hourly)

        # Group variable objects by ensemble member index
        member_vars: dict[int, list[Any]] = {}
        for i in range(hourly.VariablesLength()):
            var_obj = hourly.Variables(i)
            member_vars.setdefault(var_obj.EnsembleMember(), []).append(var_obj)

        if not member_vars:
            return None

        members: dict[int, pd.DataFrame] = {}
        for m, var_objs in sorted(member_vars.items()):
            df = pd.DataFrame(index=times)
            df.index.name = "date"
            for var_name, var_obj in zip(var_names, var_objs, strict=False):
                df[var_name] = var_obj.ValuesAsNumpy()
            members[m] = df

        logger.info(
            "[OpenMeteo] Parsed %d ensemble members, %d timesteps.", len(members), len(times)
        )
        return members
