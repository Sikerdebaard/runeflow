# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""FallbackPriceAdapter — tries adapters in order, gap-fills from backup adapters."""
from __future__ import annotations

import datetime

import pandas as pd
from loguru import logger

from runeflow.domain.price import PriceSeries
from runeflow.exceptions import DataUnavailableError
from runeflow.ports.price import PricePort


def _to_utc(dt: datetime.date | pd.Timestamp) -> pd.Timestamp:
    """Normalise any date/datetime/Timestamp to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class FallbackPriceAdapter(PricePort):
    """
    Tries each adapter in order; first success wins.

    After the primary fetch, attempts to fill any hourly gaps using the
    remaining (non-primary) adapters.
    """

    def __init__(self, adapters: list[PricePort]) -> None:
        self.adapters = adapters

    @property
    def name(self) -> str:
        return "Fallback(" + ", ".join(a.name for a in self.adapters) + ")"

    def supports_zone(self, zone: str) -> bool:
        return any(a.supports_zone(zone) for a in self.adapters)

    def download_historical(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> PriceSeries:
        last_exc: Exception | None = None

        for adapter in self.adapters:
            if not adapter.supports_zone(zone):
                logger.debug(f"[Fallback] {adapter.name} does not support {zone}, skipping.")
                continue
            try:
                series = adapter.download_historical(zone, start, end)
                if series and len(series) > 0:
                    logger.info(
                        f"[Fallback] Primary fetch via {adapter.name} succeeded "
                        f"({len(series)} records)."
                    )
                    filled = self._fill_gaps(series, zone, start, end, skip={adapter.name})
                    return filled
                logger.warning(f"[Fallback] {adapter.name} returned empty data, trying next.")
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[Fallback] {adapter.name} failed: {exc}. Trying next.")

        if last_exc is not None:
            raise last_exc
        raise DataUnavailableError(
            f"All price adapters failed or returned no data for {zone} ({start} → {end})."
        )

    def download_day_ahead(self, zone: str) -> PriceSeries | None:
        for adapter in self.adapters:
            if not adapter.supports_zone(zone):
                continue
            result = adapter.download_day_ahead(zone)
            if result is not None and len(result) > 0:
                return result
        return None

    # ── Gap filling ───────────────────────────────────────────────────────────

    def _fill_gaps(
        self,
        series: PriceSeries,
        zone: str,
        start: datetime.date,
        end: datetime.date,
        skip: set[str],
    ) -> PriceSeries:
        df = series.to_dataframe().reset_index()
        gaps = self._find_missing_ranges(df, start, end)
        if not gaps:
            return series

        logger.info(f"[Fallback] Found {len(gaps)} gap(s) to fill for {zone}.")
        patch_frames: list[pd.DataFrame] = []

        for adapter in self.adapters:
            if adapter.name in skip:
                continue
            if not adapter.supports_zone(zone):
                continue
            for gap_start, gap_end in gaps:
                try:
                    patch = adapter.download_historical(zone, gap_start, gap_end)
                    if patch and len(patch) > 0:
                        logger.info(
                            f"[Fallback] Filled gap {gap_start}→{gap_end} "
                            f"via {adapter.name}."
                        )
                        patch_frames.append(patch.to_dataframe().reset_index())
                except Exception as exc:
                    logger.warning(
                        f"[Fallback] {adapter.name} failed to fill gap "
                        f"{gap_start}→{gap_end}: {exc}"
                    )
            if patch_frames:
                break  # Got patches — stop trying further adapters

        if not patch_frames:
            logger.warning("[Fallback] Could not fill all gaps.")
            return series

        combined = pd.concat([df] + patch_frames, ignore_index=True)
        combined.drop_duplicates(subset=["date"], keep="first", inplace=True)
        combined.sort_values("date", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        return PriceSeries.from_dataframe(combined, zone=zone, source=self.name)

    @staticmethod
    def _find_missing_ranges(
        df: pd.DataFrame,
        start: datetime.date,
        end: datetime.date,
    ) -> list[tuple[datetime.date, datetime.date]]:
        """Return a list of (gap_start, gap_end) date pairs for missing hourly slots."""
        if df.empty:
            return [(start, end)]

        dates = pd.to_datetime(df["date"], utc=True)
        full_range = pd.date_range(
            start=_to_utc(start),
            end=_to_utc(end) + pd.Timedelta("23h"),
            freq="h",
        )
        missing = full_range.difference(dates)
        if missing.empty:
            return []

        # Collapse consecutive missing hours into date ranges
        gaps: list[tuple[datetime.date, datetime.date]] = []
        gap_start = missing[0]
        prev = missing[0]

        for ts in missing[1:]:
            if (ts - prev).total_seconds() > 3600:  # gap > 1 hour
                gaps.append((gap_start.date(), prev.date()))
                gap_start = ts
            prev = ts
        gaps.append((gap_start.date(), prev.date()))
        return gaps