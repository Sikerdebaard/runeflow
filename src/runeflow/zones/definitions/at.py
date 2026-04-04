# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Austria (AT) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

AT = ZoneConfig(
    zone="AT",
    name="Austria",
    timezone="Europe/Vienna",
    workalendar_country="AT",
    primary_weather_location=WeatherLocation("austria_central", 47.5162, 14.5501, "primary"),
    weather_locations=(
        WeatherLocation("austria_central", 47.5162, 14.5501, "primary"),
        # Alpine hydro reservoirs (north/south alps)
        WeatherLocation("tyrol", 47.2537, 11.6010, "hydro"),
        WeatherLocation("salzburg", 47.7981, 13.0460, "hydro"),
        WeatherLocation("carinthia", 46.7224, 14.1806, "hydro"),
        # Wind corridor (Burgenland / Lower Austria)
        WeatherLocation("burgenland", 47.5667, 16.4227, "wind"),
        WeatherLocation("lower_austria", 48.1817, 15.9753, "wind"),
    ),
    installed_solar_capacity_mw=3500.0,  # ~3.5 GW
    installed_wind_capacity_mw=3800.0,  # ~3.8 GW
    typical_load_mw=8000.0,
    neighbors=(
        NeighborZone(
            zone="DE_LU",
            purpose="demand_coupling",
            weather_locations=(
                WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),
                WeatherLocation("bavaria", 48.7904, 11.4979, "solar"),
            ),
        ),
        NeighborZone(
            zone="CH",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("switzerland_central", 46.8182, 8.2275, "primary"),),
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
        "residual_load",
        "cross_border",
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

ZoneRegistry.register(AT)
