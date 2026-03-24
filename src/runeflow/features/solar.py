# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Solar position and power feature groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class SolarPositionFeatures(FeatureGroup):
    """Solar zenith/azimuth/elevation + clear-sky GHI from pvlib."""

    name = "solar_position"

    @property
    def produces(self) -> tuple[str, ...]:
        return ("solar_zenith", "solar_azimuth", "solar_elevation", "clear_sky_ghi")

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        loc = zone_cfg.weather_locations[0]  # primary location
        for wl in zone_cfg.weather_locations:
            if wl.name == zone_cfg.primary_weather_location.name:
                loc = wl
                break

        solar_pos = pvlib.solarposition.get_solarposition(
            time=df.index,
            latitude=loc.lat,
            longitude=loc.lon,
            altitude=0,
            pressure=None,
            method="nrel_numpy",
        )
        df["solar_zenith"] = solar_pos["zenith"]
        df["solar_azimuth"] = solar_pos["azimuth"]
        df["solar_elevation"] = solar_pos["elevation"].clip(lower=0)

        cos_zenith = np.cos(np.radians(solar_pos["zenith"])).clip(lower=0)
        df["clear_sky_ghi"] = 1000.0 * cos_zenith

        return df


class SolarPowerFeatures(FeatureGroup):
    """Solar power output estimate + rolling aggregations + ramp features."""

    name = "solar_power"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("clear_sky_ghi",)

    @property
    def produces(self) -> tuple[str, ...]:
        return (
            "solar_power_output",
            "solar_power_6h",
            "solar_power_24h",
            "solar_ramp_rate",
            "solar_ramp_down",
            "solar_midday_surplus",
            "solar_scarcity",
            "clear_sky_index",
            "direct_diffuse_ratio",
            "solar_output_12h",
            "is_sunny_period",
        )

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        radiation_cols = [c for c in df.columns if "shortwave_radiation" in c.lower()]
        cloud_cols = [
            c
            for c in df.columns
            if "cloudcover" in c.lower()
            and "low" not in c.lower()
            and "mid" not in c.lower()
            and "high" not in c.lower()
        ]
        direct_cols = [c for c in df.columns if "direct_radiation" in c.lower()]
        diffuse_cols = [c for c in df.columns if "diffuse_radiation" in c.lower()]

        # Solar power output (normalised ~0-1.5)
        if radiation_cols:
            df["solar_power_output"] = (df[radiation_cols[0]] / 1000.0).clip(0, 1.5)
        elif cloud_cols:
            df["solar_power_output"] = (
                df["clear_sky_ghi"] * (1 - df[cloud_cols[0]] / 100.0) / 1000.0
            ).clip(0, 1.5)
        else:
            df["solar_power_output"] = (df["clear_sky_ghi"] / 1000.0).clip(0, 1.5)

        # Rolling aggregations
        spo = df["solar_power_output"]
        df["solar_power_6h"] = spo.rolling(6, min_periods=3).mean()
        df["solar_power_24h"] = spo.rolling(24, min_periods=12).mean()

        # Ramp
        df["solar_ramp_rate"] = spo.diff(1)
        df["solar_ramp_down"] = (-df["solar_ramp_rate"]).clip(lower=0)

        # Midday surplus
        is_midday = df.index.hour.isin([10, 11, 12, 13, 14, 15]).astype(int)  # type: ignore[attr-defined]
        df["solar_midday_surplus"] = is_midday * spo

        # Scarcity
        df["solar_scarcity"] = (1.0 / (spo + 0.01)).clip(upper=100.0)

        # Clear-sky index
        if radiation_cols:
            df["clear_sky_index"] = (df[radiation_cols[0]] / (df["clear_sky_ghi"] + 1.0)).clip(
                0, 1.5
            )
        elif cloud_cols:
            df["clear_sky_index"] = (1 - df[cloud_cols[0]] / 100.0).clip(0, 1.0)

        # Direct/diffuse ratio
        if direct_cols and diffuse_cols:
            df["direct_diffuse_ratio"] = (df[direct_cols[0]] / (df[diffuse_cols[0]] + 1.0)).clip(
                0, 10.0
            )

        # Accumulated output + sunny-period indicator
        df["solar_output_12h"] = spo.rolling(12, min_periods=6).sum()
        df["is_sunny_period"] = (
            df["solar_output_12h"]
            > df["solar_output_12h"].shift(1).rolling(168, min_periods=72).quantile(0.75)
        ).astype(int)

        return df
