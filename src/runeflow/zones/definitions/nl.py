# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Netherlands (NL) zone definition."""
from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.nl import NL_TARIFF_FORMULAS

NL = ZoneConfig(
    zone="NL",
    name="Netherlands",
    timezone="Europe/Amsterdam",
    workalendar_country="NL",
    primary_weather_location=WeatherLocation("de_bilt", 52.1009, 5.1762, "primary"),
    weather_locations=(
        WeatherLocation("de_bilt", 52.1009, 5.1762, "primary"),
        # German wind farms (for cross-border import estimation)
        WeatherLocation("lower_saxony", 52.6367, 9.8508, "wind"),
        WeatherLocation("brandenburg", 52.4125, 12.5316, "wind"),
        WeatherLocation("schleswig_holstein", 54.2194, 9.6961, "wind"),
        # French nuclear sites (for import/cooling risk)
        WeatherLocation("normandy", 49.1829, -0.3707, "nuclear"),
        WeatherLocation("rhone_alpes", 45.4472, 4.3881, "nuclear"),
        WeatherLocation("grand_est", 48.6833, 6.1833, "nuclear"),
    ),
    installed_solar_capacity_mw=9000.0,
    installed_wind_capacity_mw=8000.0,
    typical_load_mw=12000.0,
    neighbors=(
        NeighborZone(
            zone="DE_LU",
            purpose="wind_import",
            weather_locations=(
                WeatherLocation("lower_saxony", 52.6367, 9.8508, "wind"),
                WeatherLocation("brandenburg", 52.4125, 12.5316, "wind"),
                WeatherLocation("schleswig_holstein", 54.2194, 9.6961, "wind"),
            ),
        ),
        NeighborZone(
            zone="FR",
            purpose="nuclear_import",
            weather_locations=(
                WeatherLocation("normandy", 49.1829, -0.3707, "nuclear"),
                WeatherLocation("rhone_alpes", 45.4472, 4.3881, "nuclear"),
                WeatherLocation("grand_est", 48.6833, 6.1833, "nuclear"),
            ),
        ),
    ),
    has_energyzero=True,
    has_ned=True,
    tariff_formulas=NL_TARIFF_FORMULAS,
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
        "cloud",
        "renewable_pressure",
        "residual_load",
        "cross_border",
        "duck_curve",
        "spike_risk",
        "market_structure",
        "generation",
        "interaction",
    ),
    models=("xgboost_quantile", "extreme_high", "extreme_low"),
    ensemble_strategy="condition_gated",
    historical_years=tuple(range(2020, 2027)),
    min_training_years=2,
)

ZoneRegistry.register(NL)