# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
ExportTariffsService — converts a ForecastResult into tariff JSON and CSV
files that home-automation systems (evcc) and data tools (pandas) can consume.

Actual ENTSO-E prices (yesterday + today + day-ahead) are spliced in
where available so the first hours of the export use known market data
rather than model predictions.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import inject
import pandas as pd

from runeflow.domain.forecast import ForecastResult
from runeflow.domain.tariff import TariffRateSlot
from runeflow.ports.price import PricePort
from runeflow.ports.store import DataStore
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RichSlot:
    """Internal slot with full context before writing."""

    start: str
    end: str
    price: float  # All-in EUR/kWh (retailer tariff)
    is_actual: bool
    # Wholesale EUR/kWh fields (NaN for actual-only hours)
    prediction_eur_kwh: float
    lower_eur_kwh: float
    upper_eur_kwh: float
    ensemble_p50_eur_kwh: float


class ExportTariffsService:
    """
    Reads the latest forecast from the store, applies a tariff formula
    (retailer markup + taxes) and writes a tariff JSON file and CSV file.
    """

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),  # type: ignore[assignment]  # noqa: B008
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store
        # Optional price adapter (for actual/day-ahead splicing)
        try:
            self._price_port: PricePort | None = inject.instance(PricePort)  # type: ignore[assignment]
        except inject.InjectorException:
            self._price_port = None

    # ------------------------------------------------------------------
    def run(
        self,
        provider: str,
        output_path: Path | None = None,
    ) -> list[TariffRateSlot]:
        """Export tariff to JSON (and CSV alongside it).

        Args:
            provider: Tariff provider ID (e.g. "vattenfall", "wholesale").
            output_path: Path for the JSON output file.  CSV is written
                next to it with the same stem.  Defaults to
                ``tariffs_{zone}.json`` in the current directory.

        Returns:
            List of :class:`TariffRateSlot` (all-in EUR/kWh per hour).
        """
        zone = self._zone_cfg.zone
        logger.info("ExportTariffs: zone=%s provider=%s", zone, provider)

        forecast = self._store.load_latest_forecast(zone)
        if forecast is None:
            raise RuntimeError(f"No forecast found for zone={zone}. Run 'inference' first.")

        tariff_formula = self._zone_cfg.tariff_formulas.get(provider)
        if tariff_formula is None:
            available = list(self._zone_cfg.tariff_formulas.keys())
            raise ValueError(
                f"Provider '{provider}' not found for zone={zone}. Available: {available}"
            )

        if output_path is None:
            output_path = Path(f"tariffs_{zone.lower()}.json")

        rich_slots = self._build_rich_slots(forecast, tariff_formula)

        self._write_json(rich_slots, zone, provider, forecast, output_path)
        self._write_csv(rich_slots, output_path.with_suffix(".csv"))

        logger.info("Wrote %d tariff rate slots to %s", len(rich_slots), output_path)
        return [TariffRateSlot(start=s.start, end=s.end, price=s.price) for s in rich_slots]

    # ------------------------------------------------------------------
    def _build_rich_slots(
        self,
        forecast: ForecastResult,
        tariff_formula: object,
    ) -> list[_RichSlot]:
        """Merge forecast + actuals into rich slots with is_actual tracking."""
        from runeflow.domain.tariff import TariffFormula

        formula: TariffFormula = tariff_formula  # type: ignore[assignment]

        # Build forecast maps: timestamp → value (EUR/MWh)
        forecast_price_map: dict[pd.Timestamp, float] = {
            pt.timestamp: pt.prediction for pt in forecast.points
        }
        forecast_lower_map: dict[pd.Timestamp, float] = {
            pt.timestamp: pt.lower for pt in forecast.points
        }
        forecast_upper_map: dict[pd.Timestamp, float] = {
            pt.timestamp: pt.upper for pt in forecast.points
        }
        forecast_p50_map: dict[pd.Timestamp, float] = {
            pt.timestamp: pt.ensemble_p50 for pt in forecast.points
        }

        # Load actual ENTSO-E prices (yesterday + today + published day-ahead)
        actual_map = self._load_actual_prices(self._zone_cfg.zone)
        if actual_map:
            logger.info("Splicing %d actual price hours into export", len(actual_map))

        # All timestamps (forecast ∪ actuals)
        all_timestamps = sorted(set(forecast_price_map) | set(actual_map))

        slots: list[_RichSlot] = []
        for ts in all_timestamps:
            end_ts = ts + pd.Timedelta(hours=1)
            is_actual = ts in actual_map
            price_mwh = actual_map[ts] if is_actual else forecast_price_map.get(ts, float("nan"))

            price_kwh_wholesale = price_mwh / 1000.0
            price_kwh = formula.apply(price_kwh_wholesale, ts.date())

            # Wholesale forecast bounds (EUR/kWh) — NaN for actual hours
            if ts in forecast_price_map:
                pred_kwh = forecast_price_map[ts] / 1000.0
                lower_kwh = forecast_lower_map[ts] / 1000.0
                upper_kwh = forecast_upper_map[ts] / 1000.0
                p50_kwh = forecast_p50_map[ts] / 1000.0
            else:
                pred_kwh = float("nan")
                lower_kwh = float("nan")
                upper_kwh = float("nan")
                p50_kwh = float("nan")

            slots.append(
                _RichSlot(
                    start=ts.isoformat(),
                    end=end_ts.isoformat(),
                    price=round(price_kwh, 6),
                    is_actual=is_actual,
                    prediction_eur_kwh=(
                        round(pred_kwh, 6) if not pd.isna(pred_kwh) else float("nan")
                    ),
                    lower_eur_kwh=(round(lower_kwh, 6) if not pd.isna(lower_kwh) else float("nan")),
                    upper_eur_kwh=(round(upper_kwh, 6) if not pd.isna(upper_kwh) else float("nan")),
                    ensemble_p50_eur_kwh=(
                        round(p50_kwh, 6) if not pd.isna(p50_kwh) else float("nan")
                    ),
                )
            )

        return slots

    # ------------------------------------------------------------------
    def _write_json(
        self,
        slots: list[_RichSlot],
        zone: str,
        provider: str,
        forecast: ForecastResult,
        output_path: Path,
    ) -> None:
        """Write evcc forecast-compatible tariff JSON."""
        generated_at = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {
            "type": "forecast",
            "provider": provider,
            "zone": zone,
            "currency": "EUR",
            "unit": "kWh",
            "generated_at": generated_at,
            "model_version": forecast.model_version,
            "rates": [
                {
                    "start": slot.start.replace("+00:00", "Z"),
                    "end": slot.end.replace("+00:00", "Z"),
                    "value": slot.price,
                    "is_actual": slot.is_actual,
                }
                for slot in slots
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def _write_csv(self, slots: list[_RichSlot], csv_path: Path) -> None:
        """Write pandas-friendly flat CSV with all columns."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "start",
            "end",
            "price_eur_kwh",
            "is_actual",
            "prediction_eur_kwh",
            "lower_eur_kwh",
            "upper_eur_kwh",
            "ensemble_p50_eur_kwh",
        ]

        def _fmt(v: float) -> str:
            return "" if pd.isna(v) else str(v)

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for slot in slots:
                writer.writerow(
                    {
                        "start": slot.start,
                        "end": slot.end,
                        "price_eur_kwh": slot.price,
                        "is_actual": str(slot.is_actual).lower(),
                        "prediction_eur_kwh": _fmt(slot.prediction_eur_kwh),
                        "lower_eur_kwh": _fmt(slot.lower_eur_kwh),
                        "upper_eur_kwh": _fmt(slot.upper_eur_kwh),
                        "ensemble_p50_eur_kwh": _fmt(slot.ensemble_p50_eur_kwh),
                    }
                )

    # ------------------------------------------------------------------
    def _load_actual_prices(self, zone: str) -> dict[pd.Timestamp, float]:
        """Fetch yesterday + today + day-ahead actual ENTSO-E prices."""

        prices: dict[pd.Timestamp, float] = {}
        now_utc = pd.Timestamp.now("UTC").floor("h")

        # Try cached historical prices first (yesterday + today)
        yesterday = (now_utc - pd.Timedelta(days=1)).normalize()
        price_series = self._store.load_prices(zone)
        if price_series is not None:
            df = price_series.to_dataframe()
            if df.index.tz is None:  # type: ignore[attr-defined]
                df.index = df.index.tz_localize("UTC")  # type: ignore[attr-defined]
            else:
                df.index = df.index.tz_convert("UTC")  # type: ignore[attr-defined]
            price_col = "Price_EUR_MWh" if "Price_EUR_MWh" in df.columns else df.columns[0]
            # Resample to hourly (NL has 15-min settlement since 2023)
            df_h = df[[price_col]].resample("1h").mean().dropna()
            recent = df_h[df_h.index >= yesterday]
            for ts, row in recent.iterrows():
                prices[pd.Timestamp(ts)] = float(row[price_col])  # type: ignore[arg-type]

        # Try downloading day-ahead prices if price port is available
        if self._price_port is not None:
            try:
                da = self._price_port.download_day_ahead(zone)
                if da is not None:
                    da_df = da.to_dataframe()
                    if da_df.index.tz is None:  # type: ignore[attr-defined]
                        da_df.index = da_df.index.tz_localize("UTC")  # type: ignore[attr-defined]
                    else:
                        da_df.index = da_df.index.tz_convert("UTC")  # type: ignore[attr-defined]
                    da_col = (
                        "Price_EUR_MWh" if "Price_EUR_MWh" in da_df.columns else da_df.columns[0]
                    )
                    # Resample to hourly to match production
                    da_df = da_df[[da_col]].resample("1h").mean().dropna()
                    for ts, row in da_df.iterrows():
                        prices[pd.Timestamp(ts)] = float(row[da_col])  # type: ignore[arg-type]
                    logger.info("Day-ahead prices: %d hours", len(da_df))
            except Exception as exc:
                logger.debug("Day-ahead prices not yet available: %s", exc)

        return prices
