# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Duck-curve severity feature group."""

from __future__ import annotations

import numpy as np
import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class DuckCurveFeatures(FeatureGroup):
    """
    Duck-curve morning and evening demand pressure features.

    Evening: solar_ramp_down × heating_demand × wind_scarcity.
    Morning: pre-solar demand build in the hours around sunrise.
    """

    name = "duck_curve"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("solar_ramp_down", "wind_scarcity", "hdd")

    @property
    def produces(self) -> tuple[str, ...]:
        return (
            "evening_ramp_severity",
            "evening_ramp_severity_peak",
            "morning_load_pressure",
            "pre_solar_demand_peak",
            "duck_curve_asymmetry",
        )

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        has_solar_ramp = "solar_ramp_down" in df.columns
        has_wind_scarcity = "wind_scarcity" in df.columns
        has_hdd = "hdd" in df.columns

        if not (has_solar_ramp and has_wind_scarcity and has_hdd):
            return df

        wind_factor = 1.0 + df["wind_scarcity"].clip(0, 10.0)
        hdd_demand = np.maximum(df["hdd"], 1.0)

        # ── Evening ramp ──────────────────────────────────────────────────────
        df["evening_ramp_severity"] = df["solar_ramp_down"] * hdd_demand * wind_factor

        if "is_solar_cliff" in df.columns:
            df["evening_ramp_severity_peak"] = df["is_solar_cliff"] * df["evening_ramp_severity"]

        # ── Morning demand pressure ───────────────────────────────────────────
        # Pre-sunrise window (up to 4 h before sunrise): demand rising, no solar.
        # hours_before_sunrise > 0 means it's dark and sunrise is approaching.
        if "hours_before_sunrise" in df.columns:
            pre_solar_mask = (
                df["hours_before_sunrise"].gt(0) & df["hours_before_sunrise"].le(4.0)
            ).astype(float)
            df["pre_solar_demand_peak"] = pre_solar_mask * hdd_demand * wind_factor
        else:
            # Fallback: winter mornings 5–8 h are the pre-solar demand window
            morning_mask = df.index.hour.isin([5, 6, 7, 8]).astype(float)  # type: ignore[attr-defined]
            short_day = (
                df["solar_day_length"].lt(10.0).astype(float)
                if "solar_day_length" in df.columns
                else pd.Series(1.0, index=df.index)
            )
            df["pre_solar_demand_peak"] = morning_mask * short_day * hdd_demand

        # Early-morning ramp onset (0–2.5 h after sunrise): solar just starting,
        # demand still near its overnight peak — prices remain elevated.
        if "hours_since_sunrise" in df.columns:
            onset_mask = df["hours_since_sunrise"].between(0.0, 2.5).astype(float)
            df["morning_load_pressure"] = onset_mask * hdd_demand * wind_factor
        else:
            # Fallback: fixed morning hour window
            morning_onset = df.index.hour.isin([6, 7, 8, 9]).astype(float)  # type: ignore[attr-defined]
            df["morning_load_pressure"] = morning_onset * hdd_demand

        # ── Asymmetry between morning and evening pressure ────────────────────
        # Positive when evening dominates (typical summer solar-heavy day),
        # negative when morning dominates (winter heating-led morning peak).
        if "morning_load_pressure" in df.columns and "evening_ramp_severity" in df.columns:
            eve_24h = df["evening_ramp_severity"].rolling(24, min_periods=12).mean()
            morn_24h = df["morning_load_pressure"].rolling(24, min_periods=12).mean()
            df["duck_curve_asymmetry"] = eve_24h - morn_24h

        return df
