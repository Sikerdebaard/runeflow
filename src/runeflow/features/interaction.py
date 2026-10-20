# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Peak interaction feature group."""

from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class PeakInteractionFeatures(FeatureGroup):
    """
    Peak-hour interactions with temperature and spike indicators.
    Requires temperature, hdd/cdd, and spike features to be computed first.
    """

    name = "interaction"

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        if "is_peak_hour" not in df.columns:
            return df

        # Temperature × peak
        if "hdd_24h" in df.columns and "temp_extremeness" in df.columns:
            df["peak_hour_cold_interaction"] = df["is_peak_hour"] * df["hdd_24h"]
            df["peak_hour_cold_extreme"] = (
                df["is_peak_hour"] * df["temp_extremeness"] * df["hdd_24h"]
            )

        if "cdd_24h" in df.columns:
            df["peak_hour_heat_interaction"] = df["is_peak_hour"] * df["cdd_24h"]

        if "diesel_crack_spread_level" in df.columns:
            df["diesel_crack_x_peak_hour"] = df["diesel_crack_spread_level"] * df["is_peak_hour"]

            if "wind_scarcity" in df.columns:
                low_wind = (df["wind_scarcity"] > df["wind_scarcity"].median()).astype(int)
                df["diesel_crack_x_low_wind"] = df["diesel_crack_spread_level"] * low_wind

            if "market_stress_index" in df.columns:
                df["diesel_crack_x_market_stress"] = (
                    df["diesel_crack_spread_level"] * df["market_stress_index"]
                )

        if "datacenter_pressure_proxy" in df.columns:
            df["datacenter_pressure_x_peak"] = df["datacenter_pressure_proxy"] * df["is_peak_hour"]

        return df
