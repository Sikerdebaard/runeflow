# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Czechia (CZ) and Slovakia (SK) zone definitions."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

CZ = ZoneConfig(
    zone="CZ",
    name="Czechia",
    timezone="Europe/Prague",
    workalendar_country="CZ",
    primary_weather_location=WeatherLocation("czech_central", 49.8175, 15.4730, "primary"),
    weather_locations=(
        WeatherLocation("czech_central", 49.8175, 15.4730, "primary"),
        # Nuclear (Dukovany, Temelin)
        WeatherLocation("dukovany", 49.0867, 16.1469, "nuclear"),
        WeatherLocation("temelin", 49.1817, 14.3751, "nuclear"),
        # Solar south Bohemia
        WeatherLocation("south_bohemia", 49.0992, 14.5787, "solar"),
    ),
    installed_solar_capacity_mw=3000.0,  # ~3 GW
    installed_wind_capacity_mw=350.0,  # minimal wind
    typical_load_mw=9000.0,
    neighbors=(
        NeighborZone(
            zone="DE_LU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),),
        ),
        NeighborZone(
            zone="SK",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("slovakia_central", 48.6690, 19.6990, "primary"),),
        ),
        NeighborZone(
            zone="PL",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("poland_central", 52.0693, 19.4803, "primary"),),
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
    ),
    models=("xgboost_quantile", "extreme_high", "extreme_low"),
    ensemble_strategy="condition_gated",
    historical_years=tuple(range(2020, 2027)),
    min_training_years=2,
)

SK = ZoneConfig(
    zone="SK",
    name="Slovakia",
    timezone="Europe/Bratislava",
    workalendar_country="SK",
    primary_weather_location=WeatherLocation("slovakia_central", 48.6690, 19.6990, "primary"),
    weather_locations=(
        WeatherLocation("slovakia_central", 48.6690, 19.6990, "primary"),
        # Nuclear (Mochovce, Bohunice)
        WeatherLocation("mochovce", 48.3667, 18.5333, "nuclear"),
        WeatherLocation("bohunice", 48.4931, 17.6803, "nuclear"),
        # Solar (Danubian plain)
        WeatherLocation("danube_plain", 47.9756, 17.9783, "solar"),
    ),
    installed_solar_capacity_mw=800.0,
    installed_wind_capacity_mw=3.0,  # negligible wind
    typical_load_mw=4000.0,
    neighbors=(
        NeighborZone(
            zone="CZ",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("czech_central", 49.8175, 15.4730, "primary"),),
        ),
        NeighborZone(
            zone="HU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("hungary_central", 47.1625, 19.5033, "primary"),),
        ),
        NeighborZone(
            zone="AT",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("austria_central", 47.5162, 14.5501, "primary"),),
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

ZoneRegistry.register(CZ)
ZoneRegistry.register(SK)
