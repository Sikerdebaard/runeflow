# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Balkan zone definitions: Serbia (RS), Bosnia-Herzegovina (BA),
Montenegro (ME), North Macedonia (MK), and Kosovo (XK).

Workalendar notes:
  - RS (Serbia): workalendar_country="RS"
  - BA (Bosnia): workalendar has no BA entry; "RS" used as closest fallback
  - ME (Montenegro): no workalendar entry; "RS" used as closest fallback
  - MK (North Macedonia): no workalendar entry; "RS" used as closest fallback
  - XK (Kosovo): no workalendar entry; "RS" used as closest fallback

All zones start with wholesale-only tariff and 12 core feature groups.
Data availability from ENTSO-E may be limited for some sub-zones.
"""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

# ---------------------------------------------------------------------------
# RS — Serbia
# ---------------------------------------------------------------------------

RS = ZoneConfig(
    zone="RS",
    name="Serbia",
    timezone="Europe/Belgrade",
    workalendar_country="RS",
    primary_weather_location=WeatherLocation("serbia_central", 44.0165, 21.0059, "primary"),
    weather_locations=(
        WeatherLocation("serbia_central", 44.0165, 21.0059, "primary"),
        # Wind (south Banat / Vojvodina plain)
        WeatherLocation("vojvodina_wind", 45.2671, 19.8335, "wind"),
        # Solar (Sumadija)
        WeatherLocation("sumadija_solar", 44.0000, 20.5000, "solar"),
        # Hydro (Iron Gates, Djerdap)
        WeatherLocation("djerdap_hydro", 44.6500, 22.5700, "hydro"),
    ),
    installed_solar_capacity_mw=500.0,  # growing
    installed_wind_capacity_mw=700.0,  # ~0.7 GW
    typical_load_mw=5000.0,
    neighbors=(
        NeighborZone(
            zone="RO",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("romania_central", 45.7489, 26.1026, "primary"),),
        ),
        NeighborZone(
            zone="HU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("hungary_central", 47.1625, 19.5033, "primary"),),
        ),
        NeighborZone(
            zone="BG",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("bulgaria_central", 42.7339, 25.4858, "primary"),),
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

# ---------------------------------------------------------------------------
# BA — Bosnia-Herzegovina  (disabled: no ENTSO-E day-ahead prices published)
# ---------------------------------------------------------------------------

BA = ZoneConfig(
    zone="BA",
    name="Bosnia-Herzegovina",
    timezone="Europe/Sarajevo",
    workalendar_country="RS",  # workalendar has no BA; using RS as regional fallback
    primary_weather_location=WeatherLocation("sarajevo", 43.8563, 18.4131, "primary"),
    weather_locations=(
        WeatherLocation("sarajevo", 43.8563, 18.4131, "primary"),
        # Hydro (Neretva, Vrbas, Una rivers)
        WeatherLocation("neretva_hydro", 43.3365, 17.8153, "hydro"),
        WeatherLocation("mostar", 43.3438, 17.8078, "solar"),
        # Wind (Dinara)
        WeatherLocation("dinara_wind", 44.0000, 16.5000, "wind"),
    ),
    installed_solar_capacity_mw=200.0,
    installed_wind_capacity_mw=400.0,
    typical_load_mw=2000.0,
    neighbors=(
        NeighborZone(
            zone="HR",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("croatia_central", 45.1000, 15.2000, "primary"),),
        ),
        NeighborZone(
            zone="RS",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("serbia_central", 44.0165, 21.0059, "primary"),),
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
    disabled_reason=(
        "Bosnia-Herzegovina does not publish day-ahead prices on the ENTSO-E Transparency "
        "Platform (NoMatchingDataError confirmed). No alternative price data source available."
    ),
)

# ---------------------------------------------------------------------------
# ME — Montenegro
# ---------------------------------------------------------------------------

ME = ZoneConfig(
    zone="ME",
    name="Montenegro",
    timezone="Europe/Podgorica",
    workalendar_country="RS",  # workalendar has no ME; using RS as regional fallback
    primary_weather_location=WeatherLocation("podgorica", 42.4304, 19.2594, "primary"),
    weather_locations=(
        WeatherLocation("podgorica", 42.4304, 19.2594, "primary"),
        # Hydro (Piva river)
        WeatherLocation("piva_hydro", 43.0000, 18.8000, "hydro"),
        # Wind (Krnovo plateau)
        WeatherLocation("krnovo_wind", 42.8667, 19.3333, "wind"),
    ),
    installed_solar_capacity_mw=100.0,
    installed_wind_capacity_mw=120.0,
    typical_load_mw=500.0,
    neighbors=(
        NeighborZone(
            zone="RS",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("serbia_central", 44.0165, 21.0059, "primary"),),
        ),
        NeighborZone(
            zone="BA",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("sarajevo", 43.8563, 18.4131, "primary"),),
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
# MK — North Macedonia
# ---------------------------------------------------------------------------

MK = ZoneConfig(
    zone="MK",
    name="North Macedonia",
    timezone="Europe/Skopje",
    workalendar_country="RS",  # workalendar has no MK; using RS as regional fallback
    primary_weather_location=WeatherLocation("skopje", 41.9973, 21.4280, "primary"),
    weather_locations=(
        WeatherLocation("skopje", 41.9973, 21.4280, "primary"),
        # Hydro (Crn Drim, Treska)
        WeatherLocation("crn_drim_hydro", 41.5500, 20.6800, "hydro"),
        # Wind / solar (Vardar valley)
        WeatherLocation("vardar_solar", 41.7000, 22.0000, "solar"),
    ),
    installed_solar_capacity_mw=200.0,
    installed_wind_capacity_mw=50.0,
    typical_load_mw=1200.0,
    neighbors=(
        NeighborZone(
            zone="RS",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("serbia_central", 44.0165, 21.0059, "primary"),),
        ),
        NeighborZone(
            zone="GR",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("greece_central", 39.0742, 21.8243, "primary"),),
        ),
        NeighborZone(
            zone="BG",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("bulgaria_central", 42.7339, 25.4858, "primary"),),
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

# ---------------------------------------------------------------------------
# XK — Kosovo  (disabled: no ENTSO-E day-ahead prices published)
# ---------------------------------------------------------------------------

XK = ZoneConfig(
    zone="XK",
    name="Kosovo",
    timezone="Europe/Belgrade",
    workalendar_country="RS",  # workalendar has no XK; using RS as regional fallback
    primary_weather_location=WeatherLocation("pristina", 42.6629, 21.1655, "primary"),
    weather_locations=(
        WeatherLocation("pristina", 42.6629, 21.1655, "primary"),
        # Wind (Drenica plateau)
        WeatherLocation("drenica_wind", 42.7000, 20.9000, "wind"),
    ),
    installed_solar_capacity_mw=50.0,
    installed_wind_capacity_mw=130.0,
    typical_load_mw=1100.0,
    neighbors=(
        NeighborZone(
            zone="RS",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("serbia_central", 44.0165, 21.0059, "primary"),),
        ),
        NeighborZone(
            zone="MK",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("skopje", 41.9973, 21.4280, "primary"),),
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
    disabled_reason=(
        "Kosovo does not publish day-ahead prices on the ENTSO-E Transparency Platform "
        "(NoMatchingDataError confirmed). No alternative price data source available."
    ),
)

ZoneRegistry.register(RS)
ZoneRegistry.register(BA)
ZoneRegistry.register(ME)
ZoneRegistry.register(MK)
ZoneRegistry.register(XK)
