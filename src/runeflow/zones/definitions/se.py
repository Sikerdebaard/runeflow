# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Sweden bidding zone definitions.

Sweden is divided into four price areas from north to south:
  SE_1 — Luleå (north, vast hydro + wind, exports heavy)
  SE_2 — Sundsvall (north-central, hydro + growing wind)
  SE_3 — Stockholm (south-central, largest demand zone, nuclear nearby)
  SE_4 — Malmö (south, wind-rich, strong coupling to DE/DK/PL)

All zones have high hydro component (precipitation-sensitive) and
SE_3 / SE_4 have significant nuclear input (Forsmark, Ringhals, Oskarshamn).
"""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

# ---------------------------------------------------------------------------
# SE_1 — Northern Sweden (Luleå/Norrbotten)
# ---------------------------------------------------------------------------

SE_1 = ZoneConfig(
    zone="SE_1",
    name="Sweden North",
    timezone="Europe/Stockholm",
    workalendar_country="SE",
    primary_weather_location=WeatherLocation("sweden_north", 66.0000, 20.0000, "primary"),
    weather_locations=(
        WeatherLocation("sweden_north", 66.0000, 20.0000, "primary"),
        # Major hydro rivers (Lule älv, Pite älv, Ume älv upper)
        WeatherLocation("lule_alv_hydro", 66.5000, 19.5000, "hydro"),
        WeatherLocation("pite_alv_hydro", 65.5000, 18.5000, "hydro"),
        # Wind (Norrland plateau)
        WeatherLocation("vasterbotten_wind", 65.0000, 17.5000, "wind"),
        WeatherLocation("norrbotten_wind", 66.5000, 22.0000, "wind"),
    ),
    installed_solar_capacity_mw=100.0,  # minimal at this latitude
    installed_wind_capacity_mw=3000.0,  # large onshore buildout
    typical_load_mw=3000.0,  # low demand, large exporter
    neighbors=(
        NeighborZone(
            zone="SE_2",
            purpose="hydro_coupling",
            weather_locations=(
                WeatherLocation("sweden_north_central", 63.0000, 17.0000, "primary"),
            ),
        ),
        NeighborZone(
            zone="NO_4",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("norway_north", 68.0000, 15.0000, "primary"),),
        ),
        NeighborZone(
            zone="FI",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("finland_south", 61.9241, 25.7482, "primary"),),
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

# ---------------------------------------------------------------------------
# SE_2 — North-Central Sweden (Sundsvall / Härnösand region)
# ---------------------------------------------------------------------------

SE_2 = ZoneConfig(
    zone="SE_2",
    name="Sweden North-Central",
    timezone="Europe/Stockholm",
    workalendar_country="SE",
    primary_weather_location=WeatherLocation("sweden_north_central", 63.0000, 17.0000, "primary"),
    weather_locations=(
        WeatherLocation("sweden_north_central", 63.0000, 17.0000, "primary"),
        # Hydro (Ångermanälven, Indalsälven)
        WeatherLocation("angermanalven_hydro", 63.5000, 17.5000, "hydro"),
        WeatherLocation("indalsalven_hydro", 63.0000, 14.5000, "hydro"),
        # Wind
        WeatherLocation("jamtland_wind", 63.1792, 14.6357, "wind"),
        WeatherLocation("vasternorrland_wind", 62.8000, 17.5000, "wind"),
    ),
    installed_solar_capacity_mw=200.0,
    installed_wind_capacity_mw=2500.0,
    typical_load_mw=4000.0,
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
            zone="NO_3",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("trondheim", 63.4305, 10.3951, "primary"),),
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

# ---------------------------------------------------------------------------
# SE_3 — South-Central Sweden (Stockholm / Mälardalen)
# ---------------------------------------------------------------------------

SE_3 = ZoneConfig(
    zone="SE_3",
    name="Sweden South-Central",
    timezone="Europe/Stockholm",
    workalendar_country="SE",
    primary_weather_location=WeatherLocation("sweden_south_central", 59.0000, 15.0000, "primary"),
    weather_locations=(
        WeatherLocation("sweden_south_central", 59.0000, 15.0000, "primary"),
        # Nuclear (Forsmark, Oskarshamn)
        WeatherLocation("forsmark", 60.4097, 18.1706, "nuclear"),
        WeatherLocation("oskarshamn", 57.4173, 16.3976, "nuclear"),
        # Hydro (Dalälven)
        WeatherLocation("dalalven_hydro", 60.5000, 15.0000, "hydro"),
        # Wind (Svealand)
        WeatherLocation("svealand_wind", 59.5000, 16.0000, "wind"),
        # Solar (growing in south)
        WeatherLocation("stockholm_solar", 59.3293, 18.0686, "solar"),
    ),
    installed_solar_capacity_mw=1200.0,
    installed_wind_capacity_mw=4000.0,
    typical_load_mw=12000.0,  # largest demand zone in Sweden
    neighbors=(
        NeighborZone(
            zone="SE_2",
            purpose="hydro_coupling",
            weather_locations=(
                WeatherLocation("sweden_north_central", 63.0000, 17.0000, "primary"),
            ),
        ),
        NeighborZone(
            zone="SE_4",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("sweden_south", 55.8000, 13.3000, "primary"),),
        ),
        NeighborZone(
            zone="FI",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("finland_south", 61.9241, 25.7482, "primary"),),
        ),
        NeighborZone(
            zone="NO_1",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("oslo", 59.9139, 10.7522, "primary"),),
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
# SE_4 — Southern Sweden (Malmö / Skåne)
# ---------------------------------------------------------------------------

SE_4 = ZoneConfig(
    zone="SE_4",
    name="Sweden South",
    timezone="Europe/Stockholm",
    workalendar_country="SE",
    primary_weather_location=WeatherLocation("sweden_south", 55.8000, 13.3000, "primary"),
    weather_locations=(
        WeatherLocation("sweden_south", 55.8000, 13.3000, "primary"),
        # Nuclear (Ringhals)
        WeatherLocation("ringhals", 57.2583, 11.9849, "nuclear"),
        # Wind (Skåne / Baltic Sea / Öresund)
        WeatherLocation("skane_wind", 55.5000, 13.5000, "wind"),
        WeatherLocation("baltic_se_wind", 56.5000, 17.0000, "wind"),
        # Solar (southernmost Sweden)
        WeatherLocation("skane_solar", 55.7000, 13.2000, "solar"),
    ),
    installed_solar_capacity_mw=1500.0,
    installed_wind_capacity_mw=3000.0,
    typical_load_mw=6000.0,
    neighbors=(
        NeighborZone(
            zone="SE_3",
            purpose="demand_coupling",
            weather_locations=(
                WeatherLocation("sweden_south_central", 59.0000, 15.0000, "primary"),
            ),
        ),
        NeighborZone(
            zone="DK_2",
            purpose="wind_import",
            weather_locations=(WeatherLocation("zealand_central", 55.6761, 12.5683, "primary"),),
        ),
        NeighborZone(
            zone="DE_LU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),),
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
        "precipitation",
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

ZoneRegistry.register(SE_1)
ZoneRegistry.register(SE_2)
ZoneRegistry.register(SE_3)
ZoneRegistry.register(SE_4)
