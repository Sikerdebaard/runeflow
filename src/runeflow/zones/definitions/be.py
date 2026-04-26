# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Belgium (BE) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

BE = ZoneConfig(
    zone="BE",
    name="Belgium",
    timezone="Europe/Brussels",
    workalendar_country="BE",
    primary_weather_location=WeatherLocation("belgium_central", 50.5039, 4.4699, "primary"),
    weather_locations=(
        WeatherLocation("belgium_central", 50.5039, 4.4699, "primary"),
        # Nuclear sites (Doel, Tihange)
        WeatherLocation("antwerp", 51.2194, 4.4025, "nuclear"),
        WeatherLocation("liege", 50.6326, 5.5797, "nuclear"),
        # Offshore wind (North Sea)
        WeatherLocation("north_sea_be", 51.5500, 2.7000, "wind"),
        # French nuclear for import
        WeatherLocation("normandy", 49.1829, -0.3707, "nuclear"),
    ),
    installed_solar_capacity_mw=8600.0,  # ~8.6 GW
    installed_wind_capacity_mw=4700.0,  # ~4.7 GW onshore + offshore
    typical_load_mw=13000.0,
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
            zone="DE_LU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),),
        ),
        NeighborZone(
            zone="NL",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("de_bilt", 52.1009, 5.1762, "primary"),),
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

ZoneRegistry.register(BE)
