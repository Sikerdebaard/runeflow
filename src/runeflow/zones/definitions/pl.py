# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Poland (PL) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

PL = ZoneConfig(
    zone="PL",
    name="Poland",
    timezone="Europe/Warsaw",
    workalendar_country="PL",
    primary_weather_location=WeatherLocation("poland_central", 52.0693, 19.4803, "primary"),
    weather_locations=(
        WeatherLocation("poland_central", 52.0693, 19.4803, "primary"),
        # Baltic coast wind
        WeatherLocation("pomerania", 54.3520, 18.6466, "wind"),
        WeatherLocation("warmia", 53.8753, 20.6287, "wind"),
        # Solar south
        WeatherLocation("silesia", 50.2945, 18.6714, "solar"),
        WeatherLocation("lesser_poland", 50.0647, 19.9450, "solar"),
    ),
    installed_solar_capacity_mw=18000.0,  # ~18 GW (rapid growth)
    installed_wind_capacity_mw=12000.0,  # ~12 GW onshore
    typical_load_mw=22000.0,
    neighbors=(
        NeighborZone(
            zone="DE_LU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),),
        ),
        NeighborZone(
            zone="CZ",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("czech_central", 49.8175, 15.4730, "primary"),),
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

ZoneRegistry.register(PL)
