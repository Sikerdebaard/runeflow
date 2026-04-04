# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Italy bidding zone definitions.

ENTSO-E divides Italy into regional bidding zones:
  IT_NORD  — Northern Italy (largest zone, ~70 % of load)
  IT_CNOR  — Central-North Italy
  IT_CSUD  — Central-South Italy
  IT_SUD   — Southern Italy
  IT_SICI  — Sicily
  IT_SARD  — Sardinia

Workalendar does not have zone-specific Italian calendars; 'IT' is used for all.
"""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

# ---------------------------------------------------------------------------
# IT_NORD — Northern Italy
# ---------------------------------------------------------------------------

IT_NORD = ZoneConfig(
    zone="IT_NORD",
    name="Italy North",
    timezone="Europe/Rome",
    workalendar_country="IT",
    primary_weather_location=WeatherLocation("milan", 45.4654, 9.1859, "primary"),
    weather_locations=(
        WeatherLocation("milan", 45.4654, 9.1859, "primary"),
        # Alpine hydro
        WeatherLocation("aosta_valley", 45.7376, 7.3206, "hydro"),
        WeatherLocation("trentino", 46.0604, 11.1242, "hydro"),
        # Northern solar (Po Valley)
        WeatherLocation("po_valley", 45.0703, 10.6411, "solar"),
        # French nuclear for import
        WeatherLocation("rhone_alpes", 45.4472, 4.3881, "nuclear"),
        # Swiss hydro
        WeatherLocation("switzerland_central", 46.8182, 8.2275, "hydro"),
    ),
    installed_solar_capacity_mw=18000.0,  # ~18 GW (north-Italy share)
    installed_wind_capacity_mw=1500.0,  # limited in north
    typical_load_mw=30000.0,
    neighbors=(
        NeighborZone(
            zone="FR",
            purpose="nuclear_import",
            weather_locations=(WeatherLocation("rhone_alpes", 45.4472, 4.3881, "nuclear"),),
        ),
        NeighborZone(
            zone="CH",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("switzerland_central", 46.8182, 8.2275, "primary"),),
        ),
        NeighborZone(
            zone="AT",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("tyrol", 47.2537, 11.6010, "hydro"),),
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
    ),
    models=("xgboost_quantile", "extreme_high", "extreme_low"),
    ensemble_strategy="condition_gated",
    historical_years=tuple(range(2020, 2027)),
    min_training_years=2,
)

# ---------------------------------------------------------------------------
# IT_CNOR — Central-North Italy
# ---------------------------------------------------------------------------

IT_CNOR = ZoneConfig(
    zone="IT_CNOR",
    name="Italy Central North",
    timezone="Europe/Rome",
    workalendar_country="IT",
    primary_weather_location=WeatherLocation("florence", 43.7696, 11.2558, "primary"),
    weather_locations=(
        WeatherLocation("florence", 43.7696, 11.2558, "primary"),
        WeatherLocation("tuscany_solar", 43.4677, 11.1630, "solar"),
        WeatherLocation("adriatic_wind", 43.9161, 12.8985, "wind"),
    ),
    installed_solar_capacity_mw=3000.0,
    installed_wind_capacity_mw=1200.0,
    typical_load_mw=8000.0,
    neighbors=(
        NeighborZone(
            zone="IT_NORD",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("milan", 45.4654, 9.1859, "primary"),),
        ),
        NeighborZone(
            zone="IT_CSUD",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("rome", 41.9028, 12.4964, "primary"),),
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

# ---------------------------------------------------------------------------
# IT_CSUD — Central-South Italy
# ---------------------------------------------------------------------------

IT_CSUD = ZoneConfig(
    zone="IT_CSUD",
    name="Italy Central South",
    timezone="Europe/Rome",
    workalendar_country="IT",
    primary_weather_location=WeatherLocation("rome", 41.9028, 12.4964, "primary"),
    weather_locations=(
        WeatherLocation("rome", 41.9028, 12.4964, "primary"),
        WeatherLocation("lazio_solar", 41.6552, 12.9899, "solar"),
        WeatherLocation("campania_wind", 41.2833, 14.7500, "wind"),
    ),
    installed_solar_capacity_mw=3500.0,
    installed_wind_capacity_mw=2000.0,
    typical_load_mw=9000.0,
    neighbors=(
        NeighborZone(
            zone="IT_CNOR",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("florence", 43.7696, 11.2558, "primary"),),
        ),
        NeighborZone(
            zone="IT_SUD",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("naples", 40.8518, 14.2681, "primary"),),
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

# ---------------------------------------------------------------------------
# IT_SUD — Southern Italy
# ---------------------------------------------------------------------------

IT_SUD = ZoneConfig(
    zone="IT_SUD",
    name="Italy South",
    timezone="Europe/Rome",
    workalendar_country="IT",
    primary_weather_location=WeatherLocation("naples", 40.8518, 14.2681, "primary"),
    weather_locations=(
        WeatherLocation("naples", 40.8518, 14.2681, "primary"),
        # Puglia wind (one of Europe's best wind resources)
        WeatherLocation("puglia_wind", 41.1253, 16.8619, "wind"),
        WeatherLocation("basilicata_wind", 40.6399, 15.8055, "wind"),
        WeatherLocation("calabria_solar", 38.9057, 16.5948, "solar"),
    ),
    installed_solar_capacity_mw=5000.0,
    installed_wind_capacity_mw=5500.0,
    typical_load_mw=10000.0,
    neighbors=(
        NeighborZone(
            zone="IT_CSUD",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("rome", 41.9028, 12.4964, "primary"),),
        ),
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
)

# ---------------------------------------------------------------------------
# IT_SICI — Sicily
# ---------------------------------------------------------------------------

IT_SICI = ZoneConfig(
    zone="IT_SICI",
    name="Italy Sicily",
    timezone="Europe/Rome",
    workalendar_country="IT",
    primary_weather_location=WeatherLocation("palermo", 38.1157, 13.3615, "primary"),
    weather_locations=(
        WeatherLocation("palermo", 38.1157, 13.3615, "primary"),
        WeatherLocation("agrigento_solar", 37.3110, 13.5765, "solar"),
        WeatherLocation("trapani_wind", 37.9851, 12.5297, "wind"),
    ),
    installed_solar_capacity_mw=1800.0,
    installed_wind_capacity_mw=1800.0,
    typical_load_mw=3000.0,
    neighbors=(
        NeighborZone(
            zone="IT_SUD",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("naples", 40.8518, 14.2681, "primary"),),
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

# ---------------------------------------------------------------------------
# IT_SARD — Sardinia
# ---------------------------------------------------------------------------

IT_SARD = ZoneConfig(
    zone="IT_SARD",
    name="Italy Sardinia",
    timezone="Europe/Rome",
    workalendar_country="IT",
    primary_weather_location=WeatherLocation("cagliari", 39.2238, 9.1217, "primary"),
    weather_locations=(
        WeatherLocation("cagliari", 39.2238, 9.1217, "primary"),
        WeatherLocation("sardinia_wind", 40.1209, 9.0129, "wind"),
        WeatherLocation("sardinia_solar", 39.6666, 8.5557, "solar"),
    ),
    installed_solar_capacity_mw=600.0,
    installed_wind_capacity_mw=1100.0,
    typical_load_mw=1800.0,
    neighbors=(
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

ZoneRegistry.register(IT_NORD)
ZoneRegistry.register(IT_CNOR)
ZoneRegistry.register(IT_CSUD)
ZoneRegistry.register(IT_SUD)
ZoneRegistry.register(IT_SICI)
ZoneRegistry.register(IT_SARD)
