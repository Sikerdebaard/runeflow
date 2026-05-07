# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""EiaCommodityAdapter — energy commodity prices via the EIA Open Data API v2.

Fetches three global energy commodity benchmarks:

* **Crude oil (WTI)**   — monthly spot price, USD/barrel
  Route: ``/v2/petroleum/pri/spt/data/``
* **Natural gas (Henry Hub)** — monthly spot price, USD/MMBtu
  Route: ``/v2/natural-gas/pri/sum/data/``
* **Coal (Central Appalachian)** — quarterly spot price, USD/short ton
  Route: ``/v2/coal/price/data/``

All series are resampled to an hourly DatetimeIndex via forward-fill so
that the result can be joined directly to the hourly electricity price
DataFrame without frequency mismatch.

Disk caching
------------
Previous (complete) calendar years are persisted as Parquet files under
``{cache_dir}/commodity/{year}.parquet``.  A completed year is never
re-downloaded, which avoids large redundant API calls on repeated runs.
The current year is always refreshed once per process (in-memory TTL).
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from runeflow.exceptions import AuthenticationError, DownloadError
from runeflow.ports.commodity import CommodityPricePort

_BASE_URL = "https://api.eia.gov/v2"

# --- EIA v2 route configuration -------------------------------------------
# Each entry: (route, frequency, facet_key, facet_value, value_column,
#              output_column, unit_label)
_COMMODITY_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "petroleum/pri/spt/data",
        "monthly",
        "product",
        "EPC0",
        "commodity_oil_usd_bbl",
        "$/barrel",
    ),
    (
        "natural-gas/pri/sum/data",
        "monthly",
        "series",
        "RNGWHHD",
        "commodity_gas_usd_mmbtu",
        "$/MMBtu",
    ),
    (
        "coal/price/data",
        "quarterly",
        "series",
        "COAL2",
        "commodity_coal_usd_t",
        "$/short ton",
    ),
]

# Second-priority coal series if COAL2 returns empty (it was retired 2016).
_COAL_FALLBACK_SERIES = "COAL3"

# TTL for the current-year in-memory cache (seconds).
_CURRENT_YEAR_TTL = 86_400  # 24 hours


class EiaCommodityAdapter(CommodityPricePort):
    """Fetch oil, natural gas and coal prices from EIA Open Data API v2.

    Parameters
    ----------
    api_key:
        EIA API key (free registration at https://www.eia.gov/opendata/).
    cache_dir:
        Directory where completed-year Parquet files will be stored.
        Typically ``AppConfig.commodity_cache_dir``.
    """

    def __init__(self, api_key: str, cache_dir: Path) -> None:
        if not api_key:
            raise AuthenticationError(
                "EIA API key is required.  "
                "Register for free at https://www.eia.gov/opendata/ "
                "and set the EIA_KEY environment variable."
            )
        self._api_key = api_key
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory TTL for current-year data
        self._mem_cache: dict[int, tuple[pd.DataFrame, float]] = {}

    # ── CommodityPricePort interface ──────────────────────────────────────────

    def download(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> pd.DataFrame | None:
        """Return hourly forward-filled commodity prices for *start*–*end*."""
        years = list(range(start.year, end.year + 1))
        frames: list[pd.DataFrame] = []

        for year in years:
            df_year = self._get_year(year)
            if df_year is not None and not df_year.empty:
                frames.append(df_year)

        if not frames:
            return None

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Crop to requested range (hourly boundaries)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
        df = df.loc[start_ts:end_ts]
        return df if not df.empty else None

    def download_latest(self) -> pd.DataFrame | None:
        """Return rolling 12-month commodity prices ending today."""
        today = datetime.date.today()
        start = today.replace(year=today.year - 1)
        return self.download(start, today)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _year_cache_path(self, year: int) -> Path:
        return self._cache_dir / f"{year}.parquet"

    def _get_year(self, year: int) -> pd.DataFrame | None:
        """Return hourly commodity DataFrame for *year*, using disk/mem cache."""
        today = datetime.date.today()
        is_complete_year = year < today.year

        # --- Disk cache for completed years ----------------------------------
        path = self._year_cache_path(year)
        if is_complete_year and path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Commodity disk-cache read failed for %d: %s", year, exc)

        # --- In-memory TTL for current year ----------------------------------
        if not is_complete_year and year in self._mem_cache:
            df_cached, expires = self._mem_cache[year]
            if time.monotonic() < expires:
                return df_cached

        # --- Fetch from EIA --------------------------------------------------
        df = self._fetch_year(year)
        if df is None or df.empty:
            return None

        if is_complete_year:
            try:
                df.to_parquet(path, compression="snappy")
                logger.debug("Commodity disk-cache saved for year %d", year)
            except Exception as exc:
                logger.warning("Commodity disk-cache write failed for %d: %s", year, exc)
        else:
            self._mem_cache[year] = (df, time.monotonic() + _CURRENT_YEAR_TTL)

        return df

    def _fetch_year(self, year: int) -> pd.DataFrame | None:
        """Download all three commodities for *year* and return hourly frame."""
        start_str = f"{year}-01-01"
        end_str = f"{year}-12-31"

        series_frames: dict[str, pd.Series] = {}
        for route, frequency, facet_key, facet_value, out_col, _unit in _COMMODITY_SPECS:
            raw = self._fetch_series(route, frequency, facet_key, facet_value, start_str, end_str)

            # Coal fallback: COAL2 was retired in 2016
            if raw is None and out_col == "commodity_coal_usd_t":
                raw = self._fetch_series(
                    route, frequency, facet_key, _COAL_FALLBACK_SERIES, start_str, end_str
                )

            if raw is not None and not raw.empty:
                series_frames[out_col] = raw
            else:
                logger.debug("No EIA data for %s in %d — column will be NaN", out_col, year)

        if not series_frames:
            return None

        df = pd.DataFrame(series_frames)

        # Resample to hourly UTC and forward-fill within the year
        hourly_idx = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31 23:00:00",
            freq="h",
            tz="UTC",
        )
        df = df.reindex(hourly_idx).ffill()
        return df

    def _fetch_series(
        self,
        route: str,
        frequency: str,
        facet_key: str,
        facet_value: str,
        start: str,
        end: str,
    ) -> pd.Series | None:
        """Call EIA v2 API and return a UTC-indexed Series of float values."""
        url = f"{_BASE_URL}/{route}/"
        params: dict[str, str] = {
            "frequency": frequency,
            "data[0]": "value",
            f"facets[{facet_key}][]": facet_value,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": "0",
            "length": "5000",
            "start": start,
            "end": end,
            "api_key": self._api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 403:
                raise AuthenticationError("EIA API key rejected (HTTP 403).")
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise DownloadError(f"EIA request failed ({route}): {exc}") from exc

        records = payload.get("response", {}).get("data", [])
        if not records:
            return None

        rows: list[tuple[pd.Timestamp, float]] = []
        for rec in records:
            period = rec.get("period", "")
            value = rec.get("value")
            if value is None:
                continue
            try:
                ts = _parse_eia_period(period, frequency)
                rows.append((ts, float(value)))
            except (ValueError, KeyError):
                continue

        if not rows:
            return None

        idx, vals = zip(*rows, strict=False)
        return pd.Series(vals, index=pd.DatetimeIndex(idx, tz="UTC"), name=facet_value, dtype=float)


def _parse_eia_period(period: str, frequency: str) -> pd.Timestamp:
    """Convert an EIA period string to a UTC Timestamp at period start.

    EIA formats:
    * monthly  → ``"2023-01"``
    * quarterly → ``"2023-Q1"``
    * annual   → ``"2023"``
    * daily    → ``"2023-01-15"``
    """
    if frequency == "monthly":
        return pd.Timestamp(f"{period}-01", tz="UTC")
    if frequency == "quarterly":
        # e.g. "2023-Q1" → 2023-01-01, "2023-Q2" → 2023-04-01
        year, q = period.split("-Q")
        month = (int(q) - 1) * 3 + 1
        return pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC")
    if frequency == "annual":
        return pd.Timestamp(f"{period}-01-01", tz="UTC")
    # daily or unknown: attempt ISO parse
    return pd.Timestamp(period, tz="UTC")
