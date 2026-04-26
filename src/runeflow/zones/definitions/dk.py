# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Denmark zone definitions: DK_1 (West/Jutland) and DK_2 (East/Zealand+Bornholm)."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

# ---------------------------------------------------------------------------
# DK_1 — Western Denmark (Jutland + Funen, synchronous with Continental Europe)
# ---------------------------------------------------------------------------

DK_1 = ZoneConfig(
    zone="DK_1",
    name="Denmark West",
    timezone="Europe/Copenhagen",
    workalendar_country="DK",
    primary_weather_location=WeatherLocation("jutland_central", 56.2639, 9.5018, "primary"),
    weather_locations=(
        WeatherLocation("jutland_central", 56.2639, 9.5018, "primary"),
        # Onshore wind (Jutland ridge)
        WeatherLocation("jutland_west_wind", 55.8604, 8.4012, "wind"),
        WeatherLocation("jutland_north_wind", 57.2367, 9.9022, "wind"),
        # Offshore wind (North Sea / Horns Rev)
        WeatherLocation("horns_rev", 55.5000, 7.9000, "wind"),
        WeatherLocation("north_sea_dk", 56.5000, 7.0000, "wind"),
        # Solar
        WeatherLocation("jutland_solar", 55.9000, 9.5000, "solar"),
    ),
    installed_solar_capacity_mw=2500.0,  # ~2.5 GW
    installed_wind_capacity_mw=6000.0,  # ~6 GW onshore + offshore
    typical_load_mw=3500.0,
    neighbors=(
        NeighborZone(
            zone="DE_LU",
            purpose="wind_import",
            weather_locations=(WeatherLocation("schleswig_holstein", 54.2194, 9.6961, "wind"),),
        ),
        NeighborZone(
            zone="NO_2",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("norway_sw", 58.9700, 5.7331, "hydro"),),
        ),
        NeighborZone(
            zone="SE_3",
            purpose="demand_coupling",
            weather_locations=(
                WeatherLocation("sweden_south_central", 59.0000, 15.0000, "primary"),
            ),
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

# ---------------------------------------------------------------------------
# DK_2 — Eastern Denmark (Zealand, Lolland-Falster, Bornholm; synchronous with Nordic)
# ---------------------------------------------------------------------------

DK_2 = ZoneConfig(
    zone="DK_2",
    name="Denmark East",
    timezone="Europe/Copenhagen",
    workalendar_country="DK",
    primary_weather_location=WeatherLocation("zealand_central", 55.6761, 12.5683, "primary"),
    weather_locations=(
        WeatherLocation("zealand_central", 55.6761, 12.5683, "primary"),
        # Offshore wind (Ørsted / Kriegers Flak, Baltic)
        WeatherLocation("kriegers_flak", 55.0000, 13.2000, "wind"),
        WeatherLocation("oresund_wind", 55.8000, 12.8000, "wind"),
        # Solar
        WeatherLocation("zealand_solar", 55.4000, 12.0000, "solar"),
    ),
    installed_solar_capacity_mw=1000.0,  # ~1 GW
    installed_wind_capacity_mw=2000.0,  # ~2 GW (including offshore)
    typical_load_mw=2000.0,
    neighbors=(
        NeighborZone(
            zone="SE_4",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("sweden_south", 55.8000, 13.3000, "primary"),),
        ),
        NeighborZone(
            zone="DE_LU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),),
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

ZoneRegistry.register(DK_1)
ZoneRegistry.register(DK_2)
