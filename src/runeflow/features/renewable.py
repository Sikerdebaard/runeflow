# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Combined renewable pressure feature group."""
from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig
from .base import FeatureGroup


class RenewablePressureFeatures(FeatureGroup):
    """
    Composite wind+solar renewable pressure on prices.

    Requires solar_power_output + wind_power_potential (calculated upstream).
    """

    name = "renewable_pressure"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("solar_power_output", "wind_power_potential")

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        has_solar = "solar_power_output" in df.columns
        has_wind = "wind_power_potential" in df.columns

        if not has_solar and not has_wind:
            return df

        if has_wind:
            wind_p95 = (
                df["wind_power_potential"].rolling(168, min_periods=72).quantile(0.95) + 1e-8
            )
            wind_norm = (df["wind_power_potential"] / wind_p95).clip(0, 2)
        else:
            wind_norm = pd.Series(0.0, index=df.index)

        solar = df["solar_power_output"] if has_solar else pd.Series(0.0, index=df.index)

        df["renewable_pressure"] = solar + wind_norm
        df["renewable_pressure_24h"] = df["renewable_pressure"].rolling(24, min_periods=12).mean()

        rp_p90 = (
            df["renewable_pressure"].shift(1).rolling(168, min_periods=72).quantile(0.9)
        )
        df["high_renewable_risk"] = (df["renewable_pressure"] > rp_p90).fillna(0).astype(int)

        # Solar-peak-hour interactions
        if has_solar and "is_peak_hour" in df.columns:
            df["peak_hour_solar_scarcity"] = df["is_peak_hour"] * df.get(
                "solar_scarcity", 1.0 / (solar + 0.01)
            )
            is_evening_peak = df.index.hour.isin([17, 18, 19, 20]).astype(int)
            solar_ramp_down = df.get("solar_ramp_down", (-solar.diff(1)).clip(lower=0))
            df["evening_solar_rampdown"] = is_evening_peak * solar_ramp_down

        return df