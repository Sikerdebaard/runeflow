# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""CommodityAdapter — European energy commodity prices, no API key required.

Three free data sources, each chosen for European energy market relevance:

* **European gas** — `Bundesnetzagentur <https://www.bundesnetzagentur.de>`_
  daily German spot price (€/MWh).  The German gas benchmark closely tracks
  TTF and serves as the proxy used by similar tools.
  Column: ``commodity_gas_eu_eur_mwh``.

* **Brent crude oil** — `Yahoo Finance <https://finance.yahoo.com>`_ ticker
  ``BZ=F`` (ICE Brent Last Day futures), daily closing price in USD/bbl.
  Falls back to the FRED ``DCOILBRENTEU`` daily series if Yahoo is
  unavailable.  Column: ``commodity_brent_usd_bbl``.

* **Thermal coal** — `FRED <https://fred.stlouisfed.org>`_ series
  ``PCOALAUUSDM`` (Australian Newcastle benchmark, USD/mt), monthly
  frequency, forward-filled to hourly.  No API key required.
  Column: ``commodity_coal_usd_t``.

Disk caching
------------
Completed calendar years are persisted as Parquet files under
``{cache_dir}/{year}.parquet``.  A finished year is never re-downloaded.
The current (in-progress) year is refreshed once per day via an in-memory
TTL.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf  # type: ignore[import-untyped]
from loguru import logger

from runeflow.ports.commodity import CommodityPricePort

# ── Source URLs ───────────────────────────────────────────────────────────────

# Bundesnetzagentur JSON chart endpoint (German daily gas spot price, €/MWh)
_BUND_URL = "https://www.bundesnetzagentur.de/_tools/SVG/js2/_functions/json.html"
_BUND_GAS_ID = "870302"

# Yahoo Finance ticker for ICE Brent Last Day futures (USD/bbl, daily)
_BRENT_TICKER = "BZ=F"

# FRED no-key CSV endpoint — fallback for Brent and primary source for coal
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRED_BRENT_SERIES = "DCOILBRENTEU"  # USD/bbl, daily
_FRED_COAL_SERIES = "PCOALAUUSDM"  # USD/metric ton, monthly

# TTL for the current-year in-memory cache (seconds).
_CURRENT_YEAR_TTL = 86_400  # 24 hours


class CommodityAdapter(CommodityPricePort):
    """Fetch European energy commodity prices from three free public sources.

    * Gas  → Bundesnetzagentur daily German spot price (€/MWh)
    * Brent → Yahoo Finance ``BZ=F`` daily (USD/bbl) with FRED fallback
    * Coal  → FRED ``PCOALAUUSDM`` monthly (USD/mt) — no API key required

    Parameters
    ----------
    cache_dir:
        Directory where completed-year Parquet files will be stored.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[int, tuple[pd.DataFrame, float]] = {}

    # ── CommodityPricePort interface ──────────────────────────────────────────

    def download(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> pd.DataFrame | None:
        """Return hourly forward-filled commodity prices for *start*–*end*."""
        years = list(range(start.year, end.year + 1))
        frames = [f for y in years if (f := self._get_year(y)) is not None and not f.empty]
        if not frames:
            return None

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]

        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
        df = df.loc[start_ts:end_ts]
        return df if not df.empty else None

    def download_latest(self) -> pd.DataFrame | None:
        """Return rolling 12-month commodity prices ending today."""
        today = datetime.date.today()
        return self.download(today.replace(year=today.year - 1), today)

    # ── Year-level caching ────────────────────────────────────────────────────

    def _year_cache_path(self, year: int) -> Path:
        return self._cache_dir / f"{year}.parquet"

    def _get_year(self, year: int) -> pd.DataFrame | None:
        today = datetime.date.today()
        is_complete = year < today.year

        path = self._year_cache_path(year)
        if is_complete and path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Commodity disk-cache read failed for %d: %s", year, exc)

        if not is_complete and year in self._mem_cache:
            df_cached, expires = self._mem_cache[year]
            if time.monotonic() < expires:
                return df_cached

        df = self._fetch_year(year)
        if df is None or df.empty:
            return None

        if is_complete:
            try:
                df.to_parquet(path, compression="snappy")
                logger.debug("Commodity disk-cache saved for year %d", year)
            except Exception as exc:
                logger.warning("Commodity disk-cache write failed for %d: %s", year, exc)
        else:
            self._mem_cache[year] = (df, time.monotonic() + _CURRENT_YEAR_TTL)

        return df

    def _fetch_year(self, year: int) -> pd.DataFrame | None:
        today = datetime.date.today()
        y_start = datetime.date(year, 1, 1)
        y_end = min(datetime.date(year, 12, 31), today)

        series: dict[str, pd.Series] = {}

        gas = self._fetch_gas(y_start, y_end)
        if gas is not None and not gas.empty:
            series["commodity_gas_eu_eur_mwh"] = gas
        else:
            logger.debug("No Bundesnetzagentur gas data for %d", year)

        brent = self._fetch_brent_yahoo(y_start, y_end)
        if brent is None or brent.empty:
            logger.debug("Yahoo Brent empty for %d — trying FRED fallback", year)
            brent = self._fetch_brent_fred(y_start, y_end)
        if brent is not None and not brent.empty:
            series["commodity_brent_usd_bbl"] = brent
        else:
            logger.debug("No Brent data for %d", year)

        coal = self._fetch_coal_fred(y_start, y_end)
        if coal is not None and not coal.empty:
            series["commodity_coal_usd_t"] = coal
        else:
            logger.debug("No FRED coal data for %d", year)

        if not series:
            return None

        df = pd.DataFrame(series)
        hourly_idx = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31 23:00:00",
            freq="h",
            tz="UTC",
        )
        df = df.reindex(hourly_idx).ffill().bfill()
        return df

    # ── Per-source fetch helpers ──────────────────────────────────────────────

    def _fetch_gas(self, start: datetime.date, end: datetime.date) -> pd.Series | None:
        """Fetch daily German gas spot price from Bundesnetzagentur (€/MWh)."""
        # Add a small buffer so the last day is always included
        qend = end + datetime.timedelta(days=5)
        params = {
            "view": "json",
            "id": _BUND_GAS_ID,
            "xMin": start.strftime("%d.%m.%Y"),
            "xMax": qend.strftime("%d.%m.%Y"),
            "singleType": "1",
        }
        try:
            resp = requests.get(
                _BUND_URL,
                params=params,
                timeout=30,
                headers={"accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Bundesnetzagentur gas request failed: %s", exc)
            return None

        try:
            timestamps = data["labels"]
            prices = data["datasets"][1]["data"]
        except (KeyError, IndexError) as exc:
            logger.warning("Bundesnetzagentur response parse error: %s", exc)
            return None

        rows: dict[pd.Timestamp, float] = {}
        for t, p in zip(timestamps, prices, strict=False):
            if not isinstance(p, int | float):
                continue
            try:
                dt = datetime.datetime.strptime(t, "%d.%m.%Y")
                rows[pd.Timestamp(dt, tz="UTC")] = float(p)
            except ValueError:
                continue

        if not rows:
            return None

        return pd.Series(rows, dtype=float, name="commodity_gas_eu_eur_mwh").sort_index()

    def _fetch_brent_yahoo(self, start: datetime.date, end: datetime.date) -> pd.Series | None:
        """Fetch daily Brent closing price from Yahoo Finance (BZ=F, USD/bbl)."""
        try:
            raw = yf.download(
                _BRENT_TICKER,
                start=start.isoformat(),
                # yfinance end is exclusive
                end=(end + datetime.timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.warning("Yahoo Finance Brent request failed: %s", exc)
            return None

        if raw is None or raw.empty:
            return None

        # yfinance >= 0.2 may return MultiIndex columns when downloading one ticker
        close = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]

        close = close.dropna()
        if close.empty:
            return None

        if close.index.tz is None:
            close.index = close.index.tz_localize("UTC")
        else:
            close.index = close.index.tz_convert("UTC")

        return pd.Series(close, dtype=float, name="commodity_brent_usd_bbl")

    def _fetch_brent_fred(self, start: datetime.date, end: datetime.date) -> pd.Series | None:
        """Fetch daily Brent crude from FRED DCOILBRENTEU (USD/bbl, no API key)."""
        return self._fetch_fred_csv(_FRED_BRENT_SERIES, start, end, "commodity_brent_usd_bbl")

    def _fetch_coal_fred(self, start: datetime.date, end: datetime.date) -> pd.Series | None:
        """Fetch monthly Australian thermal coal from FRED PCOALAUUSDM (USD/mt).

        Monthly data is fetched with one extra month of lead-in so the
        hourly forward-fill covers the full year from the first hour.
        """
        # Pull from one month before the year start so ffill has a seed value
        lead_start = (datetime.date(start.year, 1, 1) - datetime.timedelta(days=32)).replace(day=1)
        return self._fetch_fred_csv(_FRED_COAL_SERIES, lead_start, end, "commodity_coal_usd_t")

    def _fetch_fred_csv(
        self,
        series_id: str,
        start: datetime.date,
        end: datetime.date,
        col_name: str,
    ) -> pd.Series | None:
        """Download a FRED series via the no-key public CSV endpoint."""
        try:
            resp = requests.get(_FRED_CSV_URL, params={"id": series_id}, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("FRED %s request failed: %s", series_id, exc)
            return None

        rows: dict[pd.Timestamp, float] = {}
        for line in resp.text.strip().splitlines()[1:]:  # skip header
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                dt = datetime.date.fromisoformat(parts[0])
                val = float(parts[1])
            except ValueError:
                continue
            if start <= dt <= end:
                rows[pd.Timestamp(dt, tz="UTC")] = val

        if not rows:
            return None

        return pd.Series(rows, dtype=float, name=col_name).sort_index()
