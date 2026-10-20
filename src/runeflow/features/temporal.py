# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Temporal / calendar feature group."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


def _flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class TemporalFeatures(FeatureGroup):
    """Hour-of-day, day-of-week, month, season, time-block indicators."""

    name = "temporal"

    @property
    def produces(self) -> tuple[str, ...]:
        return (
            "hour_of_day",
            "day_of_week",
            "day_of_year",
            "month",
            "is_weekend",
            "season",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
            "is_morning_ramp",
            "is_solar_midday",
            "is_evening_peak",
            "is_night_valley",
            "is_solar_cliff",
            "is_peak_hour",
            "is_overnight",
        )

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)
        idx = df.index

        df["hour_of_day"] = idx.hour  # type: ignore[attr-defined]
        df["day_of_week"] = idx.dayofweek  # type: ignore[attr-defined]
        df["day_of_year"] = idx.dayofyear  # type: ignore[attr-defined]
        df["month"] = idx.month  # type: ignore[attr-defined]
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)  # type: ignore[attr-defined]
        # 1=Winter, 2=Spring, 3=Summer, 4=Fall
        df["season"] = (idx.month % 12 + 3) // 3  # type: ignore[attr-defined]

        if _flag_enabled("RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL", default=True):
            # Cyclical calendar encoding avoids discontinuities at boundaries.
            hour = df["hour_of_day"].astype(float)
            dow = df["day_of_week"].astype(float)
            doy = df["day_of_year"].astype(float)
            df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
            df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
            df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
            df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
            df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
            df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

        # Coarse blocks (backward-compat)
        df["is_peak_hour"] = idx.hour.isin([7, 8, 9, 17, 18, 19, 20]).astype(int)  # type: ignore[attr-defined]
        df["is_overnight"] = idx.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)  # type: ignore[attr-defined]

        # Granular blocks
        df["is_morning_ramp"] = idx.hour.isin([6, 7, 8, 9]).astype(int)  # type: ignore[attr-defined]
        df["is_solar_midday"] = idx.hour.isin([10, 11, 12, 13, 14, 15]).astype(int)  # type: ignore[attr-defined]
        df["is_evening_peak"] = idx.hour.isin([17, 18, 19, 20]).astype(int)  # type: ignore[attr-defined]
        df["is_night_valley"] = idx.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)  # type: ignore[attr-defined]
        # Solar cliff: solar drops, but demand hasn't peaked yet
        df["is_solar_cliff"] = idx.hour.isin([15, 16, 17]).astype(int)  # type: ignore[attr-defined]

        return df
