# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
WarmupService — builds and caches the feature warmup window needed for
autoregressive inference.
"""
from __future__ import annotations

import logging

import inject
import pandas as pd

from runeflow.features import INFERENCE_WARMUP_DAYS, build_pipeline
from runeflow.ports.store import DataStore
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Price_EUR_MWh"


class WarmupService:
    """
    Assembles the most recent INFERENCE_WARMUP_DAYS × 24 hours of
    raw (un-engineered) data and persists it to the store.

    The inference loop prepends this window to each forecast step so that
    all rolling-window features have sufficient history.
    """

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),
        store: DataStore = inject.attr(DataStore),
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store

    # ------------------------------------------------------------------
    def run(self, force: bool = False) -> pd.DataFrame:
        zone = self._zone_cfg.zone

        if not force:
            cached = self._store.load_warmup_cache(zone)
            if cached is not None and not cached.empty:
                logger.info("Warmup cache already exists (%d rows), skipping", len(cached))
                return cached

        logger.info("Building warmup cache for zone=%s", zone)

        price_series = self._store.load_prices(zone)
        weather_series = self._store.load_weather(zone)
        if price_series is None or weather_series is None:
            raise RuntimeError(f"Missing data for zone={zone}. Run 'update-data' first.")

        df_prices = price_series.to_dataframe()
        idx = pd.DatetimeIndex(df_prices.index)
        df_prices.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        df_prices = df_prices[["Price_EUR_MWh"]]

        df_weather = weather_series.df.astype("float64")
        idx = pd.DatetimeIndex(df_weather.index)
        df_weather.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

        df = df_weather.join(df_prices, how="left")

        supp = self._store.load_supplemental(zone, "historical")
        if supp is not None and not supp.empty:
            idx = pd.DatetimeIndex(supp.index)
            supp.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            df = df.join(supp, how="left")

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # Keep only the most recent warmup window
        n_rows = INFERENCE_WARMUP_DAYS * 24
        df = df.tail(n_rows)

        self._store.save_warmup_cache(df, zone)
        logger.info("Warmup cache saved (%d rows)", len(df))
        return df