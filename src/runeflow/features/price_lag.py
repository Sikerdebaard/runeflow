# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Price lag feature group."""
from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig
from .base import FeatureGroup

_LAGS = (1, 2, 3, 6, 12, 24, 48, 72)
_DAILY_LAGS = (1, 2, 3, 7)


class PriceLagFeatures(FeatureGroup):
    """Absolute lags, same-hour-of-day lags, rolling statistics."""

    name = "price_lag"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("Price_EUR_MWh",)

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)
        col = "Price_EUR_MWh"
        if col not in df.columns:
            return df

        # Absolute lags
        for lag in _LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Daily mean lags (lag-N-day rolling daily mean)
        valid_prices = df[col].dropna()
        if not valid_prices.empty:
            daily_mean = valid_prices.resample("D").mean()
            for lag in _DAILY_LAGS:
                shifted_dates = df.index.normalize() - pd.Timedelta(days=lag)
                df[f"{col}_mean_lagdays_{lag}"] = daily_mean.reindex(shifted_dates).to_numpy()

        # Rolling statistics (shifted by 1 to prevent leakage)
        shifted = df[col].shift(1)
        df[f"{col}_rolling_24h_mean"] = shifted.rolling(24, min_periods=12).mean()
        df[f"{col}_rolling_24h_std"] = shifted.rolling(24, min_periods=12).std()
        df[f"{col}_rolling_168h_mean"] = shifted.rolling(168, min_periods=72).mean()

        # Same-hour-of-day lags
        df[f"{col}_same_hour_1d"] = df[col].shift(24)
        df[f"{col}_same_hour_2d"] = df[col].shift(48)
        df[f"{col}_same_hour_7d"] = df[col].shift(168)

        df[f"{col}_same_hour_7d_mean"] = (
            df.groupby(df.index.hour)[col]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
        )
        df[f"{col}_same_hour_deviation"] = (
            df[f"{col}_same_hour_1d"] - df[f"{col}_same_hour_7d_mean"]
        )

        return df