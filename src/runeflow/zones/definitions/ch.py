# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Switzerland (CH) zone definition."""

from __future__ import annotations

from runeflow.domain.weather import WeatherLocation
from runeflow.zones.config import NeighborZone, ZoneConfig
from runeflow.zones.registry import ZoneRegistry
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

CH = ZoneConfig(
    zone="CH",
    name="Switzerland",
    timezone="Europe/Zurich",
    workalendar_country="CH",
    primary_weather_location=WeatherLocation("switzerland_central", 46.8182, 8.2275, "primary"),
    weather_locations=(
        WeatherLocation("switzerland_central", 46.8182, 8.2275, "primary"),
        # Alpine hydro reservoirs
        WeatherLocation("valais", 46.2044, 7.3599, "hydro"),
        WeatherLocation("graubunden", 46.6569, 9.5737, "hydro"),
        WeatherLocation("ticino", 46.3317, 8.8004, "hydro"),
        # Nuclear sites (Leibstadt, Gösgen, Beznau, Mühleberg decommissioned)
        WeatherLocation("leibstadt", 47.6000, 8.1833, "nuclear"),
        WeatherLocation("gosgen", 47.3644, 7.9669, "nuclear"),
        # Solar (Swiss plateau)
        WeatherLocation("zurich_plateau", 47.3769, 8.5417, "solar"),
    ),
    installed_solar_capacity_mw=5000.0,  # ~5 GW
    installed_wind_capacity_mw=100.0,  # minimal onshore wind
    typical_load_mw=8500.0,
    neighbors=(
        NeighborZone(
            zone="DE_LU",
            purpose="demand_coupling",
            weather_locations=(WeatherLocation("germany_central", 51.1657, 10.4515, "primary"),),
        ),
        NeighborZone(
            zone="FR",
            purpose="nuclear_import",
            weather_locations=(WeatherLocation("rhone_alpes", 45.4472, 4.3881, "nuclear"),),
        ),
        NeighborZone(
            zone="AT",
            purpose="hydro_coupling",
            weather_locations=(WeatherLocation("austria_central", 47.5162, 14.5501, "primary"),),
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

ZoneRegistry.register(CH)
