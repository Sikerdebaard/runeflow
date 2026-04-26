# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Bulgaria (BG) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

BG = ZoneConfig(
    zone="BG",
    name="Bulgaria",
    timezone="Europe/Sofia",
    workalendar_country="BG",
    primary_weather_location=WeatherLocation("bulgaria_central", 42.7339, 25.4858, "primary"),
    weather_locations=(
        WeatherLocation("bulgaria_central", 42.7339, 25.4858, "primary"),
        # Nuclear (Kozloduy — 2 × 1000 MW VVER units)
        WeatherLocation("kozloduy", 43.7968, 23.7836, "nuclear"),
        # Hydro (Rhodope mountains)
        WeatherLocation("rhodope", 41.8333, 24.5833, "hydro"),
        # Wind (Black Sea coast / Dobroudja)
        WeatherLocation("dobroudja_wind", 43.5660, 27.8270, "wind"),
        # Solar (Thrace plain)
        WeatherLocation("thrace_solar", 42.0000, 25.5000, "solar"),
    ),
    installed_solar_capacity_mw=2000.0,  # ~2 GW
    installed_wind_capacity_mw=700.0,  # ~0.7 GW
    typical_load_mw=5500.0,
    neighbors=(
        NeighborZone(
            zone="RO",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("romania_central", 45.7489, 26.1026, "primary"),),
        ),
        NeighborZone(
            zone="GR",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("greece_central", 39.0742, 21.8243, "primary"),),
        ),
    ),
    has_energyzero=False,
    has_ned=False,
    tariff_formulas={"wholesale": WHOLESALE_FORMULA},
    feature_groups=(
        "temporal",
        "solar_position",
        "solar_power",
        "holiday",
        "price_lag",
        "price_regime",
        "spike_momentum",
        "temperature",
        "wind",
        "renewable_pressure",
        "duck_curve",
        "spike_risk",
        "market_structure",
        "generation",
    ),
    models=("xgboost_quantile", "extreme_high", "extreme_low"),
    ensemble_strategy="condition_gated",
    historical_years=tuple(range(2020, 2027)),
    min_training_years=2,
)

ZoneRegistry.register(BG)
