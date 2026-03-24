# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""ENTSO-E generation forecast feature group."""
from __future__ import annotations

import logging

import pandas as pd

from runeflow.zones.config import ZoneConfig
from .base import FeatureGroup

logger = logging.getLogger(__name__)

# Columns present in historical generation data but NOT available during inference.
# These must never be used as model features.
GENERATION_COLUMNS_HISTORICAL_ONLY = [
    "gen_",      # Actual generation by fuel type
    "load_mw",   # Actual load (only load_forecast_mw is available for the future)
    "flow_",     # Cross-border flows (actual, not forecast)
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
            c for c in df.columns
            if not any(p in c.lower() for p in GENERATION_COLUMNS_HISTORICAL_ONLY)
            and any(k in c.lower() for k in ["forecast", "load_forecast"])
        ]

        if not gen_cols:
            if not GenerationForecastFeatures._warned_no_gen_cols:
                logger.debug("No inference-available generation columns found in df (logged once)")
                GenerationForecastFeatures._warned_no_gen_cols = True
            return df

        forecast_renewable_cols = [
            c for c in gen_cols
            if "forecast" in c.lower()
            and any(r in c.lower() for r in ["wind", "solar"])
        ]

        if forecast_renewable_cols:
            df["forecast_total_renewable_mw"] = df[forecast_renewable_cols].sum(axis=1)
            df["forecast_renewable_same_hour_1d"] = df["forecast_total_renewable_mw"].shift(24)
            df["forecast_renewable_change_24h"] = (
                df["forecast_total_renewable_mw"]
                - df["forecast_total_renewable_mw"].shift(24)
            )

        if "load_forecast_mw" in df.columns:
            df["load_forecast_same_hour_1d"] = df["load_forecast_mw"].shift(24)
            df["load_forecast_change_24h"] = (
                df["load_forecast_mw"] - df["load_forecast_mw"].shift(24)
            )
            if "forecast_total_renewable_mw" in df.columns:
                df["forecast_residual_load_mw"] = (
                    df["load_forecast_mw"] - df["forecast_total_renewable_mw"]
                )

        return df