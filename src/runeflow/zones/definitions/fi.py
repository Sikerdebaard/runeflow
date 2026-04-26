# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Finland (FI) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

FI = ZoneConfig(
    zone="FI",
    name="Finland",
    timezone="Europe/Helsinki",
    workalendar_country="FI",
    primary_weather_location=WeatherLocation("finland_south", 61.9241, 25.7482, "primary"),
    weather_locations=(
        WeatherLocation("finland_south", 61.9241, 25.7482, "primary"),
        # Wind (western coast, Ostrobothnia — best onshore wind in Nordic)
        WeatherLocation("ostrobothnia_wind", 63.0000, 22.0000, "wind"),
        WeatherLocation("satakunta_wind", 61.5000, 22.0000, "wind"),
        WeatherLocation("lapland_wind", 67.7000, 26.3000, "wind"),
        # Nuclear (Olkiluoto 1–3, Loviisa 1–2)
        WeatherLocation("olkiluoto", 61.2347, 21.4444, "nuclear"),
        WeatherLocation("loviisa", 60.3931, 26.3625, "nuclear"),
        # Hydro (mostly northern)
        WeatherLocation("oulu_hydro", 65.0124, 25.4682, "hydro"),
    ),
    installed_solar_capacity_mw=1200.0,  # ~1.2 GW (limited by latitude)
    installed_wind_capacity_mw=8000.0,  # ~8 GW (rapid growth)
    typical_load_mw=9000.0,
    neighbors=(
        NeighborZone(
            zone="SE_1",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("sweden_north", 66.0000, 20.0000, "primary"),),
        ),
        NeighborZone(
            zone="SE_3",
            purpose="demand_coupling",
            weather_locations=(
                WeatherLocation("sweden_south_central", 59.0000, 15.0000, "primary"),
            ),
        ),
        NeighborZone(
            zone="EE",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("estonia_central", 58.5953, 25.0136, "primary"),),
        ),
    ),
    has_energyzero=False,
    has_ned=False,
    has_nordpool=True,
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

ZoneRegistry.register(FI)
