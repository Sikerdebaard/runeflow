# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Dependency injection configuration.

Call ``configure_injector(zone)`` **once** from the CLI entry point,
never at import time.  Tests may call it with ``allow_override=True``.
"""

from __future__ import annotations

import os

import inject
from loguru import logger as _logger

from runeflow.config import AppConfig
from runeflow.ports.generation import GenerationPort
from runeflow.ports.price import PricePort
from runeflow.ports.store import DataStore
from runeflow.ports.supplemental import SupplementalDataPort
from runeflow.ports.validator import DataValidator
from runeflow.ports.weather import WeatherPort
from runeflow.zones.config import ZoneConfig
from runeflow.zones.registry import ZoneRegistry


def configure_injector(
    zone: str,
    env: dict[str, str] | None = None,
    allow_override: bool = True,
) -> None:
    """
    Configure inject bindings for *zone*.

    Args:
        zone: ENTSO-E zone code (e.g. ``"NL"``).
        env: Environment variable dict.  Defaults to :data:`os.environ`.
        allow_override: Allow rebinding (needed for tests).
    """
    env = env or dict(os.environ)
    zone_cfg = ZoneRegistry.get(zone)

    def _binder(binder: inject.Binder) -> None:
        config = AppConfig.from_env(env, zone=zone)
        config.ensure_dirs()

        binder.bind(AppConfig, config)
        binder.bind(ZoneConfig, zone_cfg)  # class-based lookup (autoparams)
        binder.bind("zone_config", zone_cfg)  # string-based lookup (inject.attr)
        binder.bind("logger", _logger)

        # ── Storage ───────────────────────────────────────────────────────────
        from runeflow.adapters.store.parquet import ParquetStore

        store = ParquetStore(cache_dir=config.cache_dir)
        binder.bind(DataStore, store)

        # ── Price adapters ────────────────────────────────────────────────────
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        price_adapters: list[PricePort] = []

        if entsoe_key := env.get("ENTSOE", config.entsoe_api_key):
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            price_adapters.append(EntsoePriceAdapter(api_key=entsoe_key))

        if zone_cfg.has_energyzero:
            from runeflow.adapters.price.energyzero import EnergyZeroPriceAdapter

            price_adapters.append(EnergyZeroPriceAdapter())

        binder.bind(PricePort, FallbackPriceAdapter(price_adapters))

        # ── Weather adapter ───────────────────────────────────────────────────
        import datetime

        from runeflow.adapters.weather.caching import CachingWeatherAdapter
        from runeflow.adapters.weather.openmeteo import OpenMeteoAdapter
        from runeflow.adapters.weather.strategies import TTLCachingStrategy

        _inner_weather = OpenMeteoAdapter(
            timezone=zone_cfg.timezone,
            historical_api=config.openmeteo_historical_api,
            forecast_api=config.openmeteo_forecast_api,
            ensemble_api=config.openmeteo_ensemble_api,
        )
        binder.bind(
            WeatherPort,
            CachingWeatherAdapter(
                _inner_weather,
                store,
                zone,
                strategy=TTLCachingStrategy(ttl=datetime.timedelta(hours=3)),
            ),
        )

        # ── Generation adapter (optional ENTSO-E) ─────────────────────────────
        if entsoe_key := env.get("ENTSOE", config.entsoe_api_key):
            from runeflow.adapters.generation.entsoe import EntsoeGenerationAdapter

            binder.bind(GenerationPort, EntsoeGenerationAdapter(api_key=entsoe_key))

        # ── Supplemental adapter (NED for NL) ─────────────────────────────────
        if zone_cfg.has_ned:
            ned_key = env.get("NED", config.ned_api_key)
            if ned_key:
                from runeflow.adapters.supplemental.ned import NedAdapter

                binder.bind(SupplementalDataPort, NedAdapter(api_key=ned_key))

        # ── Validator ─────────────────────────────────────────────────────────
        from runeflow.validators.composite import default_validator

        binder.bind(DataValidator, default_validator())

    inject.configure(_binder, allow_override=allow_override)
