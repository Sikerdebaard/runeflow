# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""ENTSO-E generation forecast feature group."""

from __future__ import annotations

import logging
import os

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup

logger = logging.getLogger(__name__)


def _flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# Columns present in historical generation data but NOT available during inference.
# These must never be used as model features.
GENERATION_COLUMNS_HISTORICAL_ONLY = [
    "gen_",  # Actual generation by fuel type
    "load_mw",  # Actual load (only load_forecast_mw is available for the future)
    "flow_",  # Cross-border flows (actual, not forecast)
]


class GenerationForecastFeatures(FeatureGroup):
    """
    TSO day-ahead forecasts (load, wind, solar) joined from generation data.

    Only uses inference-safe columns (no actual generation/load/flows).
    """

    name = "generation"
    _warned_no_gen_cols: bool = False

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        # Find inference-available generation columns already present in df
        # (generation data should have been joined upstream)
        gen_cols = [
            c
            for c in df.columns
            if not any(p in c.lower() for p in GENERATION_COLUMNS_HISTORICAL_ONLY)
            and any(k in c.lower() for k in ["forecast", "load_forecast"])
        ]

        if not gen_cols:
            if not GenerationForecastFeatures._warned_no_gen_cols:
                logger.debug("No inference-available generation columns found in df (logged once)")
                GenerationForecastFeatures._warned_no_gen_cols = True
            return df

        forecast_renewable_cols = [
            c
            for c in gen_cols
            if "forecast" in c.lower() and any(r in c.lower() for r in ["wind", "solar"])
        ]

        if forecast_renewable_cols:
            df["forecast_total_renewable_mw"] = df[forecast_renewable_cols].sum(axis=1)
            df["forecast_renewable_same_hour_1d"] = df["forecast_total_renewable_mw"].shift(24)
            df["forecast_renewable_change_24h"] = df["forecast_total_renewable_mw"] - df[
                "forecast_total_renewable_mw"
            ].shift(24)

        if "load_forecast_mw" in df.columns:
            df["load_forecast_same_hour_1d"] = df["load_forecast_mw"].shift(24)
            df["load_forecast_change_24h"] = df["load_forecast_mw"] - df["load_forecast_mw"].shift(
                24
            )
            if "forecast_total_renewable_mw" in df.columns:
                df["forecast_residual_load_mw"] = (
                    df["load_forecast_mw"] - df["forecast_total_renewable_mw"]
                )

            if _flag_enabled("RUNEFLOW_ENABLE_DATACENTER_PROXY", default=True):
                # Baseload and non-weather load proxy signals (AI/datacenter pressure).
                overnight_mask = df.index.hour.isin([0, 1, 2, 3, 4, 5]).astype(float)  # type: ignore[attr-defined]
                overnight_load = df["load_forecast_mw"].where(overnight_mask > 0)
                df["baseload_min_rolling"] = overnight_load.rolling(
                    7 * 24,
                    min_periods=48,
                ).quantile(0.10)
                df["baseload_level_7d"] = overnight_load.rolling(
                    7 * 24,
                    min_periods=48,
                ).mean()
                df["baseload_level_30d"] = overnight_load.rolling(
                    30 * 24,
                    min_periods=7 * 24,
                ).mean()
                df["baseload_trend_30d"] = df["baseload_level_7d"] - df["baseload_level_30d"]
                df["baseload_trend_90d"] = (
                    df["load_forecast_mw"].rolling(30 * 24, min_periods=7 * 24).mean()
                    - df["load_forecast_mw"].rolling(90 * 24, min_periods=14 * 24).mean()
                )

                # Weather-insensitive residual load proxy using available weather columns.
                weather_cols = [
                    c
                    for c in df.columns
                    if any(k in c.lower() for k in ["temperature", "wind_speed"])
                ]
                if weather_cols:
                    weather_proxy = df[weather_cols].mean(axis=1)
                    baseline = weather_proxy.shift(1).rolling(7 * 24, min_periods=48).mean()
                    weather_anom = weather_proxy - baseline
                    df["non_weather_load_residual"] = df["load_forecast_mw"] - (
                        df["load_forecast_mw"].rolling(30 * 24, min_periods=7 * 24).mean()
                        + weather_anom * 100.0
                    )

                if "baseload_trend_30d" in df.columns:
                    # Normalized pressure proxy likely to track persistent demand uplift.
                    trend = df["baseload_trend_30d"]
                    scale = trend.shift(1).rolling(30 * 24, min_periods=7 * 24).std() + 1e-8
                    df["datacenter_pressure_proxy"] = (trend / scale).clip(-5, 5)

        return df
