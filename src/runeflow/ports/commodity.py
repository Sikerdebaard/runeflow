# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""CommodityPricePort — global energy commodity price data (oil, gas, coal)."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

import pandas as pd


class CommodityPricePort(ABC):
    """Global energy commodity price source (oil, natural gas, coal).

    Implementations return an hourly-indexed DataFrame with columns:

    * ``commodity_oil_usd_bbl``  — crude oil spot price (USD/barrel)
    * ``commodity_gas_usd_mmbtu`` — natural gas price (USD/MMBtu)
    * ``commodity_coal_usd_t``   — coal price (USD/metric ton)

    Missing observations are forward-filled by the adapter so that every
    hour has a value; the feature engineering layer adds further lag and
    rolling-window derivations.
    """

    @abstractmethod
    def download(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> pd.DataFrame | None:
        """Return hourly forward-filled commodity prices for *start*–*end*.

        Returns ``None`` if no data could be obtained for any commodity.
        Individual commodity columns may contain NaN if that specific
        series failed to download.
        """

    @abstractmethod
    def download_latest(self) -> pd.DataFrame | None:
        """Return the most recently available commodity prices.

        Typically covers rolling 12 months ending today.  Returns ``None``
        on failure.
        """
