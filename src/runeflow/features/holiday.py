# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Holiday feature group using the `holidays` package."""

from __future__ import annotations

import logging

import holidays as hols
import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup

logger = logging.getLogger(__name__)


# Map `holidays` library names → `workalendar` names used by the
# production model so that the feature columns stay identical.
_HOLIDAY_NAME_MAP: dict[str, str] = {
    "ascension_day": "ascension_thursday",
    "new_year's_day": "new_year",
    "second_day_of_christmas": "boxing_day",
}


def _str_sanitize(s: str) -> str:
    key = s.strip().lower().replace(" ", "_").replace("-", "_")
    return _HOLIDAY_NAME_MAP.get(key, key)


class HolidayFeatures(FeatureGroup):
    """Public-holiday indicator + one-hot encoded holiday names."""

    name = "holiday"

    @property
    def produces(self) -> tuple[str, ...]:
        return ("is_holiday", "holiday_tomorrow", "holiday_yesterday", "holiday_week")

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        country = zone_cfg.workalendar_country
        years = sorted(int(y) for y in df.index.year.unique() if pd.notna(y))  # type: ignore[attr-defined]

        # Build holiday set year-by-year, skipping any years the package chokes on.
        holiday_dates: set = set()
        holiday_names: dict = {}
        for y in years:
            try:
                year_cal = hols.country_holidays(country, years=[y])
                # Force full evaluation of this year's entries
                for d, name in list(year_cal.items()):
                    holiday_dates.add(d)
                    holiday_names[d] = name
            except Exception as exc:
                logger.warning("holidays: skipping year %s for %s (%s)", y, country, exc)

        dates = df.index.to_series()

        df["is_holiday"] = dates.apply(lambda x: int(x.date() in holiday_dates)).astype(int)
        df["holiday_name"] = dates.apply(
            lambda x: _str_sanitize(holiday_names[x.date()]) if x.date() in holiday_dates else None
        )

        # One-hot encode holiday names
        df = pd.get_dummies(df, columns=["holiday_name"], prefix="holiday", dummy_na=False)

        # Proximity effects
        df["holiday_tomorrow"] = df["is_holiday"].shift(-24).fillna(0).astype(int)
        df["holiday_yesterday"] = df["is_holiday"].shift(24).fillna(0).astype(int)
        df["holiday_week"] = df["is_holiday"].rolling(168, min_periods=1, center=True).max()

        return df
