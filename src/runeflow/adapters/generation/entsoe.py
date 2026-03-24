# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""ENTSO-E generation, load and cross-border flow adapter."""

from __future__ import annotations

import datetime

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from runeflow.domain.generation import GenerationSeries
from runeflow.exceptions import AuthenticationError
from runeflow.ports.generation import GenerationPort

_NL_NEIGHBORS = ["DE_LU", "BE"]


class EntsoeGenerationAdapter(GenerationPort):
    """Download generation mix, load forecasts and cross-border flows from ENTSO-E."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise AuthenticationError("ENTSO-E API key is required.")
        self._client = EntsoePandasClient(api_key=api_key)

    def supports_zone(self, zone: str) -> bool:
        # ENTSO-E covers all European bidding zones
        return bool(zone)

    def download_generation(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> GenerationSeries | None:
        ts_start = pd.Timestamp(start)
        ts_start = (
            ts_start.tz_localize("UTC") if ts_start.tzinfo is None else ts_start.tz_convert("UTC")
        )
        ts_end = pd.Timestamp(end)
        ts_end = ts_end.tz_localize("UTC") if ts_end.tzinfo is None else ts_end.tz_convert("UTC")
        ts_end += pd.Timedelta("23h")

        dfs: list[pd.DataFrame] = []

        # Load forecast
        df_load = self._fetch_load_forecast(zone, ts_start, ts_end)
        if df_load is not None and not df_load.empty:
            dfs.append(df_load)

        # Wind & solar generation forecast
        df_ws = self._fetch_wind_solar_forecast(zone, ts_start, ts_end)
        if df_ws is not None and not df_ws.empty:
            dfs.append(df_ws)

        if not dfs:
            logger.warning(f"[ENTSO-E Generation] No data for {zone} ({start}→{end})")
            return None

        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.join(df, how="outer")

        return GenerationSeries(
            zone=zone,
            df=combined,
            source="entsoe-generation",
            fetched_at=pd.Timestamp.now("UTC"),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_load_forecast(
        self, zone: str, ts_start: pd.Timestamp, ts_end: pd.Timestamp
    ) -> pd.DataFrame | None:
        try:
            series = self._client.query_load_forecast(zone, start=ts_start, end=ts_end)
            if series is None or series.empty:
                return None
            return series.rename("load_forecast_mw").rename_axis("date").to_frame()  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning(f"[ENTSO-E Generation] Load forecast failed for {zone}: {exc}")
            return None

    def _fetch_wind_solar_forecast(
        self, zone: str, ts_start: pd.Timestamp, ts_end: pd.Timestamp
    ) -> pd.DataFrame | None:
        try:
            df = self._client.query_wind_and_solar_forecast(zone, start=ts_start, end=ts_end)
            if df is None or df.empty:
                return None
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(c).strip() for c in df.columns]
            df.columns = [
                "forecast_" + c.lower().replace(" ", "_").replace("-", "_") for c in df.columns
            ]
            df.index.name = "date"
            return df  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning(f"[ENTSO-E Generation] Wind/solar forecast failed for {zone}: {exc}")
            return None
