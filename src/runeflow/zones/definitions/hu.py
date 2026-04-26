# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Hungary (HU) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

HU = ZoneConfig(
    zone="HU",
    name="Hungary",
    timezone="Europe/Budapest",
    workalendar_country="HU",
    primary_weather_location=WeatherLocation("hungary_central", 47.1625, 19.5033, "primary"),
    weather_locations=(
        WeatherLocation("hungary_central", 47.1625, 19.5033, "primary"),
        # Nuclear (Paks — ~50 % of generation)
        WeatherLocation("paks", 46.5832, 18.8531, "nuclear"),
        # Solar (Great Plain)
        WeatherLocation("great_plain", 47.0000, 20.0000, "solar"),
        WeatherLocation("transdanubia", 47.2000, 17.5000, "solar"),
    ),
    installed_solar_capacity_mw=4000.0,  # ~4 GW rapid build-out
    installed_wind_capacity_mw=400.0,  # limited wind
    typical_load_mw=5500.0,
    neighbors=(
        NeighborZone(
            zone="SK",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("slovakia_central", 48.6690, 19.6990, "primary"),),
        ),
        NeighborZone(
            zone="AT",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("austria_central", 47.5162, 14.5501, "primary"),),
        ),
        NeighborZone(
            zone="RO",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("romania_central", 45.7489, 26.1026, "primary"),),
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

ZoneRegistry.register(HU)
