# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Greece (GR) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

GR = ZoneConfig(
    zone="GR",
    name="Greece",
    timezone="Europe/Athens",
    workalendar_country="GR",
    primary_weather_location=WeatherLocation("greece_central", 39.0742, 21.8243, "primary"),
    weather_locations=(
        WeatherLocation("greece_central", 39.0742, 21.8243, "primary"),
        # Wind (Aegean islands, Thrace)
        WeatherLocation("thrace_wind", 41.0000, 25.5000, "wind"),
        WeatherLocation("cyclades_wind", 37.1000, 25.4000, "wind"),
        WeatherLocation("peloponnese_wind", 37.5900, 22.0800, "wind"),
        # Solar (very high irradiation)
        WeatherLocation("attica_solar", 38.0068, 23.7457, "solar"),
        WeatherLocation("thessaly_solar", 39.4500, 22.3000, "solar"),
        WeatherLocation("crete_solar", 35.2401, 24.8093, "solar"),
        # Hydro
        WeatherLocation("epirus_hydro", 39.5622, 20.7669, "hydro"),
    ),
    installed_solar_capacity_mw=7000.0,  # ~7 GW
    installed_wind_capacity_mw=5000.0,  # ~5 GW
    typical_load_mw=6000.0,
    neighbors=(
        NeighborZone(
            zone="BG",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("bulgaria_central", 42.7339, 25.4858, "primary"),),
        ),
        NeighborZone(
            zone="IT_CNOR",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("florence", 43.7696, 11.2558, "primary"),),
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
        "cloud",
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

ZoneRegistry.register(GR)
