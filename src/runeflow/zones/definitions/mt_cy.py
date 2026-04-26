# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Malta (MT) and Cyprus (CY) zone definitions.

Both are island systems isolated from the continental European grid:
  - MT is connected to Sicily via the Malta–Sicily HVDC interconnector (200 MW)
  - CY has no AC interconnection; imports via submarine cable when available

ENTSO-E Day-Ahead price data availability may be limited for these zones.
The system will fall back gracefully via DataUnavailableError when data is absent.
All zones start with wholesale-only tariff.
"""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

MT = ZoneConfig(
    zone="MT",
    name="Malta",
    timezone="Europe/Malta",
    workalendar_country="MT",
    primary_weather_location=WeatherLocation("malta", 35.9375, 14.3754, "primary"),
    weather_locations=(
        WeatherLocation("malta", 35.9375, 14.3754, "primary"),
        # Solar (excellent Mediterranean irradiation)
        WeatherLocation("malta_solar", 35.8500, 14.5000, "solar"),
    ),
    installed_solar_capacity_mw=250.0,  # ~250 MW and growing rapidly
    installed_wind_capacity_mw=0.0,  # no utility-scale wind
    typical_load_mw=500.0,
    neighbors=(
        NeighborZone(
            zone="IT_SICI",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("palermo", 38.1157, 13.3615, "primary"),),
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
    disabled_reason=(
        "Malta is an island market connected only via HVDC to Sicily. No day-ahead prices "
        "are published on the ENTSO-E Transparency Platform (NoMatchingDataError confirmed). "
        "A dedicated Maltese market adapter would be required to re-enable this zone."
    ),
)

CY = ZoneConfig(
    zone="CY",
    name="Cyprus",
    timezone="Asia/Nicosia",
    workalendar_country="CY",
    primary_weather_location=WeatherLocation("cyprus", 35.1264, 33.4299, "primary"),
    weather_locations=(
        WeatherLocation("cyprus", 35.1264, 33.4299, "primary"),
        # Solar (very high DNI)
        WeatherLocation("nicosia_solar", 35.1856, 33.3823, "solar"),
        WeatherLocation("limassol_solar", 34.6851, 33.0392, "solar"),
        # Wind (Troodos range)
        WeatherLocation("troodos_wind", 34.9175, 32.8792, "wind"),
    ),
    installed_solar_capacity_mw=300.0,  # 300 MW+ and rapid growth
    installed_wind_capacity_mw=160.0,  # ~160 MW onshore
    typical_load_mw=900.0,
    neighbors=(),  # Isolated island system; no reliable AC interconnection
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
    disabled_reason=(
        "Cyprus is an isolated island grid with no reliable AC interconnection to continental "
        "Europe. No day-ahead prices are published on ENTSO-E (NoMatchingDataError confirmed). "
        "A dedicated Cypriot market adapter would be required to re-enable this zone."
    ),
)

ZoneRegistry.register(MT)
ZoneRegistry.register(CY)
