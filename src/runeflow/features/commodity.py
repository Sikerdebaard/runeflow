# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Energy commodity price feature group — oil, natural gas, coal.

Expects columns produced by
:class:`~runeflow.adapters.supplemental.commodity.CommodityAdapter`
(joined upstream in the training/warmup assembly):

* ``commodity_brent_usd_bbl``   — Brent crude in USD/barrel (Yahoo Finance BZ=F)
* ``commodity_gas_eu_eur_mwh``  — German/European gas spot price in €/MWh (Bundesnetzagentur)
* ``commodity_coal_usd_t``      — thermal coal (FRED PCOALAUUSDM) in USD/mt

All three columns are forward-filled to fill any hourly gaps left after
frequency up-sampling.  The group then derives:

* 24 h and 168 h (1-week) lag columns for each commodity
* 7-day and 30-day trailing means (rolling on hourly index)
* 24 h percentage rate-of-change
* gas-to-oil and coal-to-oil price ratios

If none of the commodity columns are present the group returns the
DataFrame unchanged — it is safe to include even when no commodity adapter
is configured.
"""

from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup

_COMMODITY_COLS = (
    "commodity_brent_usd_bbl",
    "commodity_gas_eu_eur_mwh",
    "commodity_coal_usd_t",
)


class CommodityPriceFeatures(FeatureGroup):
    """Energy commodity price context features (oil, gas, coal)."""

    name = "commodity"

    @property
    def requires(self) -> tuple[str, ...]:
        # All columns are optional — guard inside transform()
        return ()

    @property
    def produces(self) -> tuple[str, ...]:
        produced = []
        for col in _COMMODITY_COLS:
            produced += [
                col,
                f"{col}_lag_24h",
                f"{col}_lag_168h",
                f"{col}_ma7d",
                f"{col}_ma30d",
                f"{col}_pct_change_24h",
            ]
        produced += [
            "commodity_gas_oil_ratio",
            "commodity_coal_oil_ratio",
        ]
        return tuple(produced)

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        present = [c for c in _COMMODITY_COLS if c in df.columns]
        if not present:
            return df

        df = self._copy(df)

        # --- Forward-fill: carry the most recent known price into every hour ---
        # Commodity data arrives at monthly/quarterly native frequency; after
        # joining on an hourly electricity index most rows will be NaN.
        for col in present:
            df[col] = df[col].ffill()

        # --- Per-commodity lag and rolling features ---------------------------
        for col in present:
            series = df[col]

            # Same-hour lags (hourly shift)
            df[f"{col}_lag_24h"] = series.shift(24)
            df[f"{col}_lag_168h"] = series.shift(168)

            # Rolling trailing means (min_periods keeps NaN at the very start)
            df[f"{col}_ma7d"] = series.rolling(7 * 24, min_periods=48).mean()
            df[f"{col}_ma30d"] = series.rolling(30 * 24, min_periods=168).mean()

            # 24 h percentage change (avoid division by zero)
            lagged = series.shift(24)
            df[f"{col}_pct_change_24h"] = (series - lagged) / (lagged.abs() + 1e-8)

        # --- Cross-commodity ratios -------------------------------------------
        oil = df.get("commodity_brent_usd_bbl")
        gas = df.get("commodity_gas_eu_eur_mwh")
        coal = df.get("commodity_coal_usd_t")

        if oil is not None and gas is not None:
            # gas (€/MWh) / oil (USD/bbl) — different units, but the relative
            # spread still captures fuel-switching pressure as a model signal
            df["commodity_gas_oil_ratio"] = gas / (oil + 1e-8)

        if oil is not None and coal is not None:
            df["commodity_coal_oil_ratio"] = coal / (oil + 1e-8)

        return df
