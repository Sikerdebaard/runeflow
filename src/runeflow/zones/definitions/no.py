# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Norway bidding zone definitions.

Norway is divided into five price areas (NO_1 – NO_5) based on grid topology:
  NO_1 — South-East (Oslo region, Statnett NE) — largest demand zone
  NO_2 — South-West (Stavanger / Bergen area) — heavy offshore wind, wind-rich
  NO_3 — Middle / Trondheim region — large hydro, also some wind
  NO_4 — North (Tromsø / Finnmark) — hydro heavy, low demand
  NO_5 — West (Hordaland / Bergen) — hydro-dominant, interconnected with NO_2

All Norwegian zones are almost entirely hydro-powered (~88 % of generation).
Precipitation drives prices more than any other input.
"""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

# ---------------------------------------------------------------------------
# NO_1 — South-East Norway (Østlandet / Oslo region)
# ---------------------------------------------------------------------------

NO_1 = ZoneConfig(
    zone="NO_1",
    name="Norway South-East",
    timezone="Europe/Oslo",
    workalendar_country="NO",
    primary_weather_location=WeatherLocation("oslo", 59.9139, 10.7522, "primary"),
    weather_locations=(
        WeatherLocation("oslo", 59.9139, 10.7522, "primary"),
        # Hydro reservoirs (Glomma/Lågen basin)
        WeatherLocation("glomma_hydro", 61.0000, 11.0000, "hydro"),
        WeatherLocation("mjosa_hydro", 60.7400, 10.8100, "hydro"),
        # Wind (inland ridges)
        WeatherLocation("innlandet_wind", 61.5000, 10.5000, "wind"),
    ),
    installed_solar_capacity_mw=400.0,  # low latitude benefit
    installed_wind_capacity_mw=800.0,
    typical_load_mw=9000.0,  # largest demand zone
    neighbors=(
        NeighborZone(
            zone="SE_3",
            purpose="hydro_coupling",
            weather_locations=(
                WeatherLocation("sweden_south_central", 59.0000, 15.0000, "primary"),
            ),
        ),
        NeighborZone(
            zone="SE_4",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("sweden_south", 55.8000, 13.3000, "primary"),),
        ),
        NeighborZone(
            zone="DK_1",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("jutland_central", 56.2639, 9.5018, "primary"),),
        ),
        NeighborZone(
            zone="NO_2",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("norway_sw", 58.9700, 5.7331, "primary"),),
        ),
        NeighborZone(
            zone="NO_5",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("hordaland", 60.4720, 6.3279, "hydro"),),
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
# NO_2 — South-West Norway (Rogaland / Stavanger region)
# ---------------------------------------------------------------------------

NO_2 = ZoneConfig(
    zone="NO_2",
    name="Norway South-West",
    timezone="Europe/Oslo",
    workalendar_country="NO",
    primary_weather_location=WeatherLocation("norway_sw", 58.9700, 5.7331, "primary"),
    weather_locations=(
        WeatherLocation("norway_sw", 58.9700, 5.7331, "primary"),
        # Hydro reservoirs (Ulla-Førre, Sira-Kvina)
        WeatherLocation("ulla_forre_hydro", 59.5000, 6.5000, "hydro"),
        WeatherLocation("sira_kvina_hydro", 58.5000, 6.5000, "hydro"),
        # Offshore wind (North Sea — Hywind, Equinor projects)
        WeatherLocation("north_sea_no", 57.5000, 3.0000, "wind"),
        WeatherLocation("utsira_wind", 59.3000, 4.8833, "wind"),
    ),
    installed_solar_capacity_mw=100.0,
    installed_wind_capacity_mw=1200.0,  # highest wind in Norway
    typical_load_mw=4000.0,
    neighbors=(
        NeighborZone(
            zone="NO_1",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("oslo", 59.9139, 10.7522, "primary"),),
        ),
        NeighborZone(
            zone="NO_5",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("hordaland", 60.4720, 6.3279, "hydro"),),
        ),
        NeighborZone(
            zone="DK_1",
            purpose="wind_import",
            weather_locations=(WeatherLocation("jutland_central", 56.2639, 9.5018, "primary"),),
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
# NO_3 — Middle Norway (Trøndelag / Trondheim region)
# ---------------------------------------------------------------------------

NO_3 = ZoneConfig(
    zone="NO_3",
    name="Norway Middle",
    timezone="Europe/Oslo",
    workalendar_country="NO",
    primary_weather_location=WeatherLocation("trondheim", 63.4305, 10.3951, "primary"),
    weather_locations=(
        WeatherLocation("trondheim", 63.4305, 10.3951, "primary"),
        # Hydro (Nea-Nidelva, Orkla)
        WeatherLocation("nea_nidelva_hydro", 63.0000, 11.5000, "hydro"),
        # Wind (Fosen peninsula, Snillfjord — large wind parks)
        WeatherLocation("fosen_wind", 63.8000, 9.5000, "wind"),
        WeatherLocation("snillfjord_wind", 63.4000, 9.0000, "wind"),
    ),
    installed_solar_capacity_mw=100.0,  # minimal at this latitude
    installed_wind_capacity_mw=2000.0,  # Fosen is one of Europe's largest onshore parks
    typical_load_mw=3500.0,
    neighbors=(
        NeighborZone(
            zone="NO_1",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("oslo", 59.9139, 10.7522, "primary"),),
        ),
        NeighborZone(
            zone="NO_4",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("norway_north", 68.0000, 15.0000, "primary"),),
        ),
        NeighborZone(
            zone="SE_2",
            purpose="hydro_coupling",
            weather_locations=(
                WeatherLocation("sweden_north_central", 63.0000, 17.0000, "primary"),
            ),
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
# NO_4 — Northern Norway (Nordland, Troms, Finnmark)
# ---------------------------------------------------------------------------

NO_4 = ZoneConfig(
    zone="NO_4",
    name="Norway North",
    timezone="Europe/Oslo",
    workalendar_country="NO",
    primary_weather_location=WeatherLocation("norway_north", 68.0000, 15.0000, "primary"),
    weather_locations=(
        WeatherLocation("norway_north", 68.0000, 15.0000, "primary"),
        # Hydro (Svartisen, Rana, Beiarn)
        WeatherLocation("svartisen_hydro", 66.9000, 14.0000, "hydro"),
        WeatherLocation("rana_hydro", 66.3000, 14.2000, "hydro"),
        # Wind
        WeatherLocation("nordland_wind", 67.0000, 15.5000, "wind"),
    ),
    installed_solar_capacity_mw=30.0,  # Arctic — very limited
    installed_wind_capacity_mw=700.0,
    typical_load_mw=2500.0,
    neighbors=(
        NeighborZone(
            zone="NO_3",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("trondheim", 63.4305, 10.3951, "primary"),),
        ),
        NeighborZone(
            zone="SE_1",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("sweden_north", 66.0000, 20.0000, "primary"),),
        ),
        NeighborZone(
            zone="FI",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("finland_south", 61.9241, 25.7482, "primary"),),
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
# NO_5 — West Norway (Hordaland / Bergen region)
# ---------------------------------------------------------------------------

NO_5 = ZoneConfig(
    zone="NO_5",
    name="Norway West",
    timezone="Europe/Oslo",
    workalendar_country="NO",
    primary_weather_location=WeatherLocation("hordaland", 60.4720, 6.3279, "primary"),
    weather_locations=(
        WeatherLocation("hordaland", 60.4720, 6.3279, "primary"),
        # Hydro (Bergen region; heavily maritime rainfall)
        WeatherLocation("hardanger_hydro", 60.3300, 6.8800, "hydro"),
        WeatherLocation("west_fjords_hydro", 61.1000, 6.0000, "hydro"),
        # Wind
        WeatherLocation("stord_wind", 59.8000, 5.5000, "wind"),
    ),
    installed_solar_capacity_mw=80.0,
    installed_wind_capacity_mw=400.0,
    typical_load_mw=3500.0,
    neighbors=(
        NeighborZone(
            zone="NO_1",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("oslo", 59.9139, 10.7522, "primary"),),
        ),
        NeighborZone(
            zone="NO_2",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("norway_sw", 58.9700, 5.7331, "primary"),),
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

ZoneRegistry.register(NO_1)
ZoneRegistry.register(NO_2)
ZoneRegistry.register(NO_3)
ZoneRegistry.register(NO_4)
ZoneRegistry.register(NO_5)
