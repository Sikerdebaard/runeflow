# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
ExportTariffsService — converts a ForecastResult into a tariff
JSON file (list of rate slots) that home-automation systems can consume.

Actual ENTSO-E prices (yesterday + today + day-ahead) are spliced in
where available so the first hours of the export use known market data
rather than model predictions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import inject
import pandas as pd

from runeflow.domain.tariff import TariffRateSlot
from runeflow.domain.forecast import ForecastResult
from runeflow.ports.price import PricePort
from runeflow.ports.store import DataStore
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)


class ExportTariffsService:
    """
    Reads the latest forecast from the store, applies a tariff formula
    (retailer markup + taxes) and writes a tariff JSON file.
    """

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),
        store: DataStore = inject.attr(DataStore),
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store
        # Optional price adapter (for actual/day-ahead splicing)
        try:
            self._price_port: PricePort | None = inject.instance(PricePort)
        except inject.InjectorException:
            self._price_port = None

    # ------------------------------------------------------------------
    def run(
        self,
        provider: str,
        output_path: Path | None = None,
    ) -> list[TariffRateSlot]:
        zone = self._zone_cfg.zone
        logger.info("ExportTariffs: zone=%s provider=%s", zone, provider)

        forecast = self._store.load_latest_forecast(zone)
        if forecast is None:
            raise RuntimeError(f"No forecast found for zone={zone}. Run 'inference' first.")

        tariff_formula = self._zone_cfg.tariff_formulas.get(provider)
        if tariff_formula is None:
            available = list(self._zone_cfg.tariff_formulas.keys())
            raise ValueError(
                f"Provider '{provider}' not found for zone={zone}. "
                f"Available: {available}"
            )

        # Build forecast price map (timestamp → EUR/MWh wholesale)
        forecast_map: dict[pd.Timestamp, float] = {
            pt.timestamp: pt.prediction for pt in forecast.points
        }

        # Load actual ENTSO-E prices (yesterday + today + published day-ahead)
        actual_map = self._load_actual_prices(zone)
        if actual_map:
            logger.info("Splicing %d actual price hours into export", len(actual_map))

        # Combine: actuals overwrite forecast where available
        combined_map = {**forecast_map, **actual_map}

        slots: list[TariffRateSlot] = []
        for ts in sorted(combined_map):
            price_mwh = combined_map[ts]
            price_kwh_wholesale = price_mwh / 1000.0  # EUR/MWh → EUR/kWh
            price_kwh = tariff_formula.apply(price_kwh_wholesale, ts.date())
            end_ts = ts + pd.Timedelta(hours=1)
            slots.append(TariffRateSlot(
                start=ts.isoformat(),
                end=end_ts.isoformat(),
                price=round(price_kwh, 6),
            ))

        if output_path is None:
            output_path = Path(f"tariffs_{zone.lower()}.json")

        payload = {
            "type": "fixed",
            "zones": [
                {
                    "price": slot.price,
                    "start": slot.start,
                }
                for slot in slots
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %d tariff rate slots to %s", len(slots), output_path)
        return slots

    # ------------------------------------------------------------------
    def _load_actual_prices(self, zone: str) -> dict[pd.Timestamp, float]:
        """Fetch yesterday + today + day-ahead actual ENTSO-E prices."""
        import datetime as dt_mod

        prices: dict[pd.Timestamp, float] = {}
        now_utc = pd.Timestamp.now("UTC").floor("h")

        # Try cached historical prices first (yesterday + today)
        yesterday = (now_utc - pd.Timedelta(days=1)).normalize()
        price_series = self._store.load_prices(zone)
        if price_series is not None:
            df = price_series.to_dataframe()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            price_col = "Price_EUR_MWh" if "Price_EUR_MWh" in df.columns else df.columns[0]
            # Resample to hourly (NL has 15-min settlement since 2023)
            df_h = df[[price_col]].resample("1h").mean().dropna()
            recent = df_h[df_h.index >= yesterday]
            for ts, row in recent.iterrows():
                prices[pd.Timestamp(ts)] = float(row[price_col])

        # Try downloading day-ahead prices if price port is available
        if self._price_port is not None:
            try:
                da = self._price_port.download_day_ahead(zone)
                if da is not None:
                    da_df = da.to_dataframe()
                    if da_df.index.tz is None:
                        da_df.index = da_df.index.tz_localize("UTC")
                    else:
                        da_df.index = da_df.index.tz_convert("UTC")
                    da_col = "Price_EUR_MWh" if "Price_EUR_MWh" in da_df.columns else da_df.columns[0]
                    # Resample to hourly to match production
                    da_df = da_df[[da_col]].resample("1h").mean().dropna()
                    for ts, row in da_df.iterrows():
                        prices[pd.Timestamp(ts)] = float(row[da_col])
                    logger.info("Day-ahead prices: %d hours", len(da_df))
            except Exception as exc:
                logger.debug("Day-ahead prices not yet available: %s", exc)

        return prices