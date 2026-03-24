# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Price regime / volatility feature group."""

from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class PriceRegimeFeatures(FeatureGroup):
    """30-day price regime, volatility, regime-change ratio."""

    name = "price_regime"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("Price_EUR_MWh",)

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)
        col = "Price_EUR_MWh"
        if col not in df.columns:
            return df

        shifted = df[col].shift(1)

        # 30-day rolling median = "crisis level"
        df[f"{col}_regime_30d"] = shifted.rolling(720, min_periods=168).median()

        rolling_168h_mean = shifted.rolling(168, min_periods=72).mean()
        df[f"{col}_rolling_168h_mean"] = rolling_168h_mean  # may already exist, harmless

        df[f"{col}_regime_change_7d"] = rolling_168h_mean / (df[f"{col}_regime_30d"] + 1e-8)

        # Volatility
        df[f"{col}_volatility_24h"] = shifted.rolling(24, min_periods=12).std()
        df[f"{col}_volatility_168h"] = shifted.rolling(168, min_periods=72).std()

        # Range-based volatility
        r24 = shifted.rolling(24, min_periods=12)
        df[f"{col}_range_24h"] = r24.max() - r24.min()
        r168 = shifted.rolling(168, min_periods=72)
        df[f"{col}_range_168h"] = r168.max() - r168.min()

        # Momentum (price diff)
        df[f"{col}_diff_1h"] = shifted.diff(1)
        df[f"{col}_diff_24h"] = shifted.diff(24)
        price_changes = shifted.diff(1)
        df[f"{col}_momentum_24h"] = price_changes.clip(lower=0).rolling(24, min_periods=12).sum()

        # Peak interactions with volatility
        if "is_peak_hour" in df.columns:
            df[f"{col}_peak_volatility"] = df["is_peak_hour"] * df[f"{col}_volatility_24h"]
            df[f"{col}_momentum_peak"] = df["is_peak_hour"] * df[f"{col}_momentum_24h"]
            df[f"{col}_volatility_peak"] = df["is_peak_hour"] * df[f"{col}_volatility_24h"]

        # Price acceleration
        if f"{col}_diff_1h" in df.columns:
            df[f"{col}_acceleration_1h"] = df[f"{col}_diff_1h"].diff(1)
            df[f"{col}_acceleration_24h"] = df[f"{col}_diff_24h"].diff(24)

        return df
