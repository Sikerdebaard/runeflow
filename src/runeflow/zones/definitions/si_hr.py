# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Slovenia (SI) and Croatia (HR) zone definitions."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

SI = ZoneConfig(
    zone="SI",
    name="Slovenia",
    timezone="Europe/Ljubljana",
    workalendar_country="SI",
    primary_weather_location=WeatherLocation("slovenia_central", 46.1512, 14.9955, "primary"),
    weather_locations=(
        WeatherLocation("slovenia_central", 46.1512, 14.9955, "primary"),
        # Alpine hydro (Sava, Drava rivers)
        WeatherLocation("sava_valley", 46.0569, 14.5058, "hydro"),
        WeatherLocation("drava_valley", 46.5547, 15.6467, "hydro"),
        # Nuclear (Krško)
        WeatherLocation("krsko", 45.8208, 15.5700, "nuclear"),
        # Solar (south/coastal)
        WeatherLocation("primorska", 45.7000, 13.7000, "solar"),
    ),
    installed_solar_capacity_mw=700.0,  # ~0.7 GW
    installed_wind_capacity_mw=3.0,  # negligible
    typical_load_mw=1700.0,
    neighbors=(
        NeighborZone(
            zone="AT",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("austria_central", 47.5162, 14.5501, "primary"),),
        ),
        NeighborZone(
            zone="IT_NORD",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("milan", 45.4654, 9.1859, "primary"),),
        ),
        NeighborZone(
            zone="HR",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("croatia_central", 45.1000, 15.2000, "primary"),),
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

HR = ZoneConfig(
    zone="HR",
    name="Croatia",
    timezone="Europe/Zagreb",
    workalendar_country="HR",
    primary_weather_location=WeatherLocation("croatia_central", 45.1000, 15.2000, "primary"),
    weather_locations=(
        WeatherLocation("croatia_central", 45.1000, 15.2000, "primary"),
        # Hydro (Drava, Sava rivers)
        WeatherLocation("drava_hr", 45.8000, 16.5000, "hydro"),
        # Wind (Dalmatian karst corridor — bora/jugo)
        WeatherLocation("dalmatia_wind", 43.5000, 16.5000, "wind"),
        WeatherLocation("zagora_wind", 43.7000, 16.0000, "wind"),
        # Solar (Adriatic coast)
        WeatherLocation("split_solar", 43.5081, 16.4402, "solar"),
    ),
    installed_solar_capacity_mw=600.0,  # ~0.6 GW (growing fast)
    installed_wind_capacity_mw=1200.0,  # ~1.2 GW
    typical_load_mw=2400.0,
    neighbors=(
        NeighborZone(
            zone="SI",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("slovenia_central", 46.1512, 14.9955, "primary"),),
        ),
        NeighborZone(
            zone="HU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("hungary_central", 47.1625, 19.5033, "primary"),),
        ),
        NeighborZone(
            zone="RS",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("serbia_central", 44.0165, 21.0059, "primary"),),
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

ZoneRegistry.register(SI)
ZoneRegistry.register(HR)
