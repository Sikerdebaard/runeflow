# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Germany-Luxembourg (DE_LU) zone definition."""
from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.de import DE_TARIFF_FORMULAS

DE_LU = ZoneConfig(
    zone="DE_LU",
    name="Germany-Luxembourg",
    timezone="Europe/Berlin",
    workalendar_country="DE",
    primary_weather_location=WeatherLocation(
        "germany_central", 51.1657, 10.4515, "primary"
    ),
    weather_locations=(
        WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),
        WeatherLocation("lower_saxony", 52.6367, 9.8508, "wind"),
        WeatherLocation("schleswig_holstein", 54.2194, 9.6961, "wind"),
        WeatherLocation("bavaria", 48.7904, 11.4979, "solar"),
        # French nuclear sites (for import estimation)
        WeatherLocation("normandy", 49.1829, -0.3707, "nuclear"),
        WeatherLocation("grand_est", 48.6833, 6.1833, "nuclear"),
    ),
    installed_solar_capacity_mw=65000.0,  # ~65 GW
    installed_wind_capacity_mw=60000.0,   # ~60 GW onshore + offshore
    typical_load_mw=55000.0,
    neighbors=(
        NeighborZone(
            zone="FR",
            purpose="nuclear_import",
            weather_locations=(
                WeatherLocation("normandy", 49.1829, -0.3707, "nuclear"),
                WeatherLocation("grand_est", 48.6833, 6.1833, "nuclear"),
            ),
        ),
        NeighborZone(
            zone="NL",
            purpose="demand_coupling",
            weather_locations=(
                WeatherLocation("de_bilt", 52.1009, 5.1762, "primary"),
            ),
        ),
    ),
    has_energyzero=False,
    has_ned=False,
    tariff_formulas=DE_TARIFF_FORMULAS,
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
    historical_years=tuple(range(2020, 2026)),
    min_training_years=2,
)

ZoneRegistry.register(DE_LU)