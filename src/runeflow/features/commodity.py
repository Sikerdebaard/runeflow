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

import os

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup

_COMMODITY_COLS = (
    "commodity_brent_usd_bbl",
    "commodity_gas_eu_eur_mwh",
    "commodity_coal_usd_t",
    "commodity_diesel_usd_gal",
)


def _flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


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
            "commodity_diesel_oil_ratio",
            "gas_power_cost_proxy",
            "gas_shock_24h",
            "gas_shock_7d",
            "fuel_cost_proxy",
            "power_crack_proxy",
            "diesel_crack_spread_level",
            "diesel_crack_spread_z_30d",
            "diesel_crack_spread_momentum_24h",
            "diesel_crack_spread_momentum_7d",
            "diesel_crack_stress_regime",
            "diesel_data_quality_flag",
            "market_stress_index",
            "stress_regime",
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

        diesel = df.get("commodity_diesel_usd_gal")
        if gas is not None and diesel is not None and oil is not None:
            df["commodity_diesel_oil_ratio"] = diesel / (oil + 1e-8)

        # --- Fuel cost and crack-style power spread proxies ------------------
        if _flag_enabled("RUNEFLOW_ENABLE_SPREAD_FEATURES", default=True) and gas is not None:
            # Proxy marginal fuel pressure in EUR/MWh space.
            df["gas_power_cost_proxy"] = gas
            df["gas_shock_24h"] = gas - gas.shift(24)
            df["gas_shock_7d"] = gas - gas.shift(168)

        gas_series = gas
        coal_series = coal

        if (
            _flag_enabled("RUNEFLOW_ENABLE_SPREAD_FEATURES", default=True)
            and gas_series is not None
            and coal_series is not None
        ):
            df["fuel_cost_proxy"] = 0.7 * gas_series + 0.3 * coal_series
        elif (
            _flag_enabled("RUNEFLOW_ENABLE_SPREAD_FEATURES", default=True)
            and gas_series is not None
        ):
            df["fuel_cost_proxy"] = gas_series
        elif (
            _flag_enabled("RUNEFLOW_ENABLE_SPREAD_FEATURES", default=True)
            and coal_series is not None
        ):
            df["fuel_cost_proxy"] = coal_series

        if "fuel_cost_proxy" in df.columns and "Price_EUR_MWh" in df.columns:
            # Inference-safe: use lagged price level only.
            df["power_crack_proxy"] = df["Price_EUR_MWh"].shift(1) - df["fuel_cost_proxy"]

        # --- Diesel crack spread (world-economy proxy) -----------------------
        # Prefer explicit diesel series when present; otherwise derive a stable
        # proxy from Brent so downstream features can still be computed.
        if _flag_enabled("RUNEFLOW_ENABLE_DIESEL_CRACK", default=True) and diesel is not None:
            diesel_refined = diesel
            df["diesel_data_quality_flag"] = 1
        elif _flag_enabled("RUNEFLOW_ENABLE_DIESEL_CRACK", default=True) and oil is not None:
            # Approximation: refinery uplift over crude when no diesel benchmark
            # is available in this zone/time window.
            diesel_refined = oil * 1.35
            df["diesel_data_quality_flag"] = 0
        else:
            diesel_refined = None

        if diesel_refined is not None and oil is not None:
            crack = diesel_refined - oil
            df["diesel_crack_spread_level"] = crack
            crack_mean = crack.shift(1).rolling(30 * 24, min_periods=7 * 24).mean()
            crack_std = crack.shift(1).rolling(30 * 24, min_periods=7 * 24).std()
            df["diesel_crack_spread_z_30d"] = (crack - crack_mean) / (crack_std + 1e-8)
            df["diesel_crack_spread_momentum_24h"] = crack - crack.shift(24)
            df["diesel_crack_spread_momentum_7d"] = crack - crack.shift(168)

            # Regime bins: 0=normal, 1=elevated, 2=stressed
            q70 = crack.shift(1).rolling(60 * 24, min_periods=14 * 24).quantile(0.70)
            q90 = crack.shift(1).rolling(60 * 24, min_periods=14 * 24).quantile(0.90)
            elevated = pd.Series(crack > q70, index=crack.index, dtype="bool").fillna(False)
            stressed = pd.Series(crack > q90, index=crack.index, dtype="bool").fillna(False)
            df["diesel_crack_stress_regime"] = elevated.astype(int) + stressed.astype(int)

        # --- Endogenous market stress index ----------------------------------
        stress_parts: list[pd.Series] = []

        if "gas_shock_24h" in df.columns:
            gs = df["gas_shock_24h"].abs()
            gs_n = gs / (gs.shift(1).rolling(30 * 24, min_periods=7 * 24).quantile(0.95) + 1e-8)
            stress_parts.append(gs_n.clip(0, 3))

        if "Price_EUR_MWh" in df.columns:
            pstd = df["Price_EUR_MWh"].shift(1).rolling(24, min_periods=12).std() / (
                df["Price_EUR_MWh"].shift(1).rolling(30 * 24, min_periods=7 * 24).std() + 1e-8
            )
            stress_parts.append(pstd.clip(0, 3))

        if "diesel_crack_spread_z_30d" in df.columns:
            stress_parts.append(df["diesel_crack_spread_z_30d"].abs().clip(0, 3))

        if _flag_enabled("RUNEFLOW_ENABLE_MARKET_STRESS", default=True) and stress_parts:
            msi = pd.concat(stress_parts, axis=1).mean(axis=1)
            df["market_stress_index"] = msi
            # 0=normal, 1=elevated, 2=crisis
            df["stress_regime"] = (msi > 1.0).fillna(False).astype(int) + (msi > 1.6).fillna(
                False
            ).astype(int)

        return df
