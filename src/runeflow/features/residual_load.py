# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Residual load proxy feature group."""
from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig
from .base import FeatureGroup


class ResidualLoadFeatures(FeatureGroup):
    """
    Net-load after subtracting estimated renewable generation.

    Uses zone_cfg.installed_solar_capacity_mw and installed_wind_capacity_mw.
    Falls back gracefully when columns are absent.
    """

    name = "residual_load"

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        has_ned = "ned_utilization_kwh" in df.columns
        has_solar = "solar_power_output" in df.columns
        if not (has_ned and has_solar):
            return df

        solar_cap = float(zone_cfg.installed_solar_capacity_mw or 9000.0)
        wind_cap = float(zone_cfg.installed_wind_capacity_mw or 8000.0)

        solar_gen_mw = df["solar_power_output"] * solar_cap

        if "wind_power_potential" in df.columns:
            wind_max = (
                df["wind_power_potential"].rolling(168, min_periods=72).quantile(0.99) + 1e-8
            )
            wind_cf = (df["wind_power_potential"] / wind_max).clip(0, 1)
            wind_gen_mw = wind_cf * wind_cap
        else:
            wind_gen_mw = 0.0

        load_mw = df["ned_utilization_kwh"] / 1000.0  # kWh per h → MW
        df["residual_load_mw"] = load_mw - solar_gen_mw - wind_gen_mw

        rl_mean = df["residual_load_mw"].rolling(168, min_periods=72).mean()
        rl_std = df["residual_load_mw"].rolling(168, min_periods=72).std() + 1e-8
        df["residual_load_zscore"] = (df["residual_load_mw"] - rl_mean) / rl_std

        # NOTE: low_residual_load / high_residual_load removed to match
        # production feature set (215 features does not include these bins).

        return df