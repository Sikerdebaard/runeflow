# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""France (FR) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

FR = ZoneConfig(
    zone="FR",
    name="France",
    timezone="Europe/Paris",
    workalendar_country="FR",
    primary_weather_location=WeatherLocation("france_central", 46.2276, 2.2137, "primary"),
    weather_locations=(
        WeatherLocation("france_central", 46.2276, 2.2137, "primary"),
        # Nuclear sites (France is ~70 % nuclear)
        WeatherLocation("normandy", 49.1829, -0.3707, "nuclear"),
        WeatherLocation("rhone_alpes", 45.4472, 4.3881, "nuclear"),
        WeatherLocation("grand_est", 48.6833, 6.1833, "nuclear"),
        WeatherLocation("loire_valley", 47.2506, 0.6333, "nuclear"),
        # Wind regions (north-west Atlantic coast)
        WeatherLocation("brittany", 48.2020, -2.9326, "wind"),
        WeatherLocation("hauts_de_france", 50.4801, 2.7937, "wind"),
        # Solar south
        WeatherLocation("occitanie", 43.8927, 3.2828, "solar"),
    ),
    installed_solar_capacity_mw=20000.0,  # ~20 GW
    installed_wind_capacity_mw=23000.0,  # ~23 GW onshore + offshore
    typical_load_mw=75000.0,
    neighbors=(
        NeighborZone(
            zone="DE_LU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),),
        ),
        NeighborZone(
            zone="ES",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("spain_central", 40.4168, -3.7038, "primary"),),
        ),
        NeighborZone(
            zone="BE",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("belgium_central", 50.5039, 4.4699, "primary"),),
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

ZoneRegistry.register(FR)
