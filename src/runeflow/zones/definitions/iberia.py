# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Iberian Peninsula zone definitions: Spain (ES) and Portugal (PT)."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

ES = ZoneConfig(
    zone="ES",
    name="Spain",
    timezone="Europe/Madrid",
    workalendar_country="ES",
    primary_weather_location=WeatherLocation("spain_central", 40.4168, -3.7038, "primary"),
    weather_locations=(
        WeatherLocation("spain_central", 40.4168, -3.7038, "primary"),
        # Wind corridors (Castile, Aragon, Cadiz)
        WeatherLocation("castile_leon", 41.8528, -4.4161, "wind"),
        WeatherLocation("aragon", 41.5976, -0.9057, "wind"),
        WeatherLocation("cadiz", 36.5271, -6.2886, "wind"),
        # Solar south (La Mancha, Andalusia, Murcia)
        WeatherLocation("la_mancha", 38.9967, -3.9271, "solar"),
        WeatherLocation("andalusia", 37.3891, -5.9845, "solar"),
        WeatherLocation("murcia", 37.9922, -1.1307, "solar"),
        # Nuclear sites
        WeatherLocation("trillo", 40.6823, -2.5930, "nuclear"),
        WeatherLocation("cofrentes", 39.2441, -1.0635, "nuclear"),
    ),
    installed_solar_capacity_mw=23000.0,  # ~23 GW PV
    installed_wind_capacity_mw=30000.0,  # ~30 GW
    typical_load_mw=28000.0,
    neighbors=(
        NeighborZone(
            zone="FR",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("france_central", 46.2276, 2.2137, "primary"),),
        ),
        NeighborZone(
            zone="PT",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("portugal_central", 39.3999, -8.2245, "primary"),),
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

PT = ZoneConfig(
    zone="PT",
    name="Portugal",
    timezone="Europe/Lisbon",
    workalendar_country="PT",
    primary_weather_location=WeatherLocation("portugal_central", 39.3999, -8.2245, "primary"),
    weather_locations=(
        WeatherLocation("portugal_central", 39.3999, -8.2245, "primary"),
        # Atlantic wind (north coast)
        WeatherLocation("minho", 41.6944, -8.4294, "wind"),
        WeatherLocation("beiras", 40.2033, -8.4103, "wind"),
        # Solar south (Alentejo)
        WeatherLocation("alentejo", 38.5675, -8.0076, "solar"),
        WeatherLocation("algarve", 37.0194, -7.9304, "solar"),
    ),
    installed_solar_capacity_mw=2000.0,  # ~2 GW
    installed_wind_capacity_mw=8500.0,  # ~8.5 GW
    typical_load_mw=7000.0,
    neighbors=(
        NeighborZone(
            zone="ES",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("spain_central", 40.4168, -3.7038, "primary"),),
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

ZoneRegistry.register(ES)
ZoneRegistry.register(PT)
