# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Great Britain (GB) and Ireland (IE) zone definitions."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

# ---------------------------------------------------------------------------
# GB — Great Britain (England, Scotland, Wales)
# ---------------------------------------------------------------------------

GB = ZoneConfig(
    zone="GB",
    name="Great Britain",
    timezone="Europe/London",
    workalendar_country="GB",
    primary_weather_location=WeatherLocation("gb_central", 52.3555, -1.1743, "primary"),
    weather_locations=(
        WeatherLocation("gb_central", 52.3555, -1.1743, "primary"),
        # Offshore wind (North Sea — Hornsea, Dogger Bank, Beatrice)
        WeatherLocation("north_sea_gb", 54.5000, 1.0000, "wind"),
        WeatherLocation("dogger_bank", 54.7500, 2.5000, "wind"),
        # Scottish onshore wind
        WeatherLocation("scotland_wind", 57.0000, -4.0000, "wind"),
        WeatherLocation("scotland_north_wind", 58.5000, -3.5000, "wind"),
        # Solar (south England)
        WeatherLocation("south_england_solar", 51.0000, -1.5000, "solar"),
        WeatherLocation("east_england_solar", 52.5000, 0.5000, "solar"),
        # Nuclear (Hinkley, Sizewell, Heysham)
        WeatherLocation("somerset_nuclear", 51.2081, -3.1318, "nuclear"),
        WeatherLocation("suffolk_nuclear", 52.2079, 1.6197, "nuclear"),
    ),
    installed_solar_capacity_mw=14000.0,  # ~14 GW
    installed_wind_capacity_mw=29000.0,  # ~15 GW offshore + 14 GW onshore
    typical_load_mw=40000.0,
    neighbors=(
        NeighborZone(
            zone="FR",
            purpose="nuclear_import",
            weather_locations=(WeatherLocation("normandy", 49.1829, -0.3707, "nuclear"),),
        ),
        NeighborZone(
            zone="BE",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("belgium_central", 50.5039, 4.4699, "primary"),),
        ),
        NeighborZone(
            zone="NL",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("de_bilt", 52.1009, 5.1762, "primary"),),
        ),
        NeighborZone(
            zone="IE",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("ireland_central", 53.1424, -7.6921, "primary"),),
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
        "interaction",
    ),
    models=("xgboost_quantile", "extreme_high", "extreme_low"),
    ensemble_strategy="condition_gated",
    historical_years=tuple(range(2020, 2027)),
    min_training_years=2,
    disabled_reason=(
        "Great Britain left ENTSO-E Day-Ahead market coupling after Brexit (2021). "
        "Prices are no longer published on the Transparency Platform. "
        "A dedicated BMRS/Elexon adapter would be required to re-enable this zone."
    ),
)

# ---------------------------------------------------------------------------
# IE — Ireland (Single Electricity Market, shared with Northern Ireland)
# ---------------------------------------------------------------------------
# IE — Ireland  (disabled: no ENTSO-E historical day-ahead prices available)
# ---------------------------------------------------------------------------

IE = ZoneConfig(
    zone="IE",
    name="Ireland",
    timezone="Europe/Dublin",
    workalendar_country="IE",
    primary_weather_location=WeatherLocation("ireland_central", 53.1424, -7.6921, "primary"),
    weather_locations=(
        WeatherLocation("ireland_central", 53.1424, -7.6921, "primary"),
        # Atlantic onshore wind (west coast — among best in Europe)
        WeatherLocation("connacht_wind", 53.7000, -9.0000, "wind"),
        WeatherLocation("munster_wind", 52.3000, -8.5000, "wind"),
        WeatherLocation("donegal_wind", 54.9000, -8.2000, "wind"),
        # Solar (south)
        WeatherLocation("leinster_solar", 52.8000, -7.0000, "solar"),
    ),
    installed_solar_capacity_mw=1000.0,  # ~1 GW growing
    installed_wind_capacity_mw=6000.0,  # ~6 GW onshore
    typical_load_mw=4000.0,
    neighbors=(
        NeighborZone(
            zone="GB",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("gb_central", 52.3555, -1.1743, "primary"),),
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
        "Ireland's Single Electricity Market (SEM/IE_SEM) does not publish historical "
        "day-ahead prices on the ENTSO-E Transparency Platform (NoMatchingDataError for all "
        "years 2020-2026 confirmed). A dedicated SEMO/SEMOPX adapter would be required to "
        "re-enable this zone."
    ),
)

ZoneRegistry.register(GB)
ZoneRegistry.register(IE)
