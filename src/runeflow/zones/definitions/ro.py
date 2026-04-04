# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Romania (RO) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

RO = ZoneConfig(
    zone="RO",
    name="Romania",
    timezone="Europe/Bucharest",
    workalendar_country="RO",
    primary_weather_location=WeatherLocation("romania_central", 45.7489, 26.1026, "primary"),
    weather_locations=(
        WeatherLocation("romania_central", 45.7489, 26.1026, "primary"),
        # Hydro (Carpathian arc, Iron Gates)
        WeatherLocation("carpathian_arc", 45.5000, 25.0000, "hydro"),
        WeatherLocation("iron_gates", 44.6500, 22.5700, "hydro"),
        # Wind (Dobrogea — one of Europe's best wind resources)
        WeatherLocation("dobrogea_wind", 44.1698, 28.6348, "wind"),
        # Nuclear (Cernavoda)
        WeatherLocation("cernavoda", 44.3333, 28.0500, "nuclear"),
        # Solar (south plains)
        WeatherLocation("muntenia", 44.4268, 26.1025, "solar"),
    ),
    installed_solar_capacity_mw=1400.0,  # ~1.4 GW (growing)
    installed_wind_capacity_mw=3000.0,  # ~3 GW
    typical_load_mw=8000.0,
    neighbors=(
        NeighborZone(
            zone="HU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("hungary_central", 47.1625, 19.5033, "primary"),),
        ),
        NeighborZone(
            zone="BG",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("bulgaria_central", 42.7339, 25.4858, "primary"),),
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
        "precipitation",
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

ZoneRegistry.register(RO)
