# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Baltic state zone definitions: Estonia (EE), Latvia (LV), Lithuania (LT)."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

# ---------------------------------------------------------------------------
# EE — Estonia
# ---------------------------------------------------------------------------

EE = ZoneConfig(
    zone="EE",
    name="Estonia",
    timezone="Europe/Tallinn",
    workalendar_country="EE",
    primary_weather_location=WeatherLocation("estonia_central", 58.5953, 25.0136, "primary"),
    weather_locations=(
        WeatherLocation("estonia_central", 58.5953, 25.0136, "primary"),
        # Wind (western islands and coast)
        WeatherLocation("hiiumaa_wind", 58.9161, 22.5833, "wind"),
        WeatherLocation("saaremaa_wind", 58.4747, 22.5668, "wind"),
        WeatherLocation("parnumaa_wind", 58.3858, 24.4998, "wind"),
        # Solar
        WeatherLocation("tallinn_solar", 59.4370, 24.7536, "solar"),
    ),
    installed_solar_capacity_mw=500.0,  # ~0.5 GW growing
    installed_wind_capacity_mw=700.0,  # ~0.7 GW
    typical_load_mw=1100.0,
    neighbors=(
        NeighborZone(
            zone="FI",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("finland_south", 61.9241, 25.7482, "primary"),),
        ),
        NeighborZone(
            zone="LV",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("latvia_central", 56.8796, 24.6032, "primary"),),
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
# LV — Latvia
# ---------------------------------------------------------------------------

LV = ZoneConfig(
    zone="LV",
    name="Latvia",
    timezone="Europe/Riga",
    workalendar_country="LV",
    primary_weather_location=WeatherLocation("latvia_central", 56.8796, 24.6032, "primary"),
    weather_locations=(
        WeatherLocation("latvia_central", 56.8796, 24.6032, "primary"),
        # Hydro (Daugava river cascades — main generation source)
        WeatherLocation("daugava_hydro", 56.5000, 24.5000, "hydro"),
        # Wind (western coast)
        WeatherLocation("vidzeme_wind", 57.2000, 25.3000, "wind"),
        WeatherLocation("kurzeme_wind", 56.8000, 21.5000, "wind"),
    ),
    installed_solar_capacity_mw=200.0,
    installed_wind_capacity_mw=150.0,
    typical_load_mw=900.0,
    neighbors=(
        NeighborZone(
            zone="EE",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("estonia_central", 58.5953, 25.0136, "primary"),),
        ),
        NeighborZone(
            zone="LT",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("lithuania_central", 55.1694, 23.8813, "primary"),),
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
# LT — Lithuania
# ---------------------------------------------------------------------------

LT = ZoneConfig(
    zone="LT",
    name="Lithuania",
    timezone="Europe/Vilnius",
    workalendar_country="LT",
    primary_weather_location=WeatherLocation("lithuania_central", 55.1694, 23.8813, "primary"),
    weather_locations=(
        WeatherLocation("lithuania_central", 55.1694, 23.8813, "primary"),
        # Wind (western coast — Klaipeda, Baltic Sea)
        WeatherLocation("klaipeda_wind", 55.7034, 21.1443, "wind"),
        WeatherLocation("samogitia_wind", 55.9000, 22.3000, "wind"),
        # Solar (south)
        WeatherLocation("suvalkai_solar", 54.1000, 23.0000, "solar"),
    ),
    installed_solar_capacity_mw=800.0,  # growing fast
    installed_wind_capacity_mw=900.0,  # ~0.9 GW
    typical_load_mw=1500.0,
    neighbors=(
        NeighborZone(
            zone="LV",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("latvia_central", 56.8796, 24.6032, "primary"),),
        ),
        NeighborZone(
            zone="PL",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("poland_central", 52.0693, 19.4803, "primary"),),
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

ZoneRegistry.register(EE)
ZoneRegistry.register(LV)
ZoneRegistry.register(LT)
