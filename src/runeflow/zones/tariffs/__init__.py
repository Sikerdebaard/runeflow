# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Per-country tariff formula registries."""

from __future__ import annotations

from runeflow.domain.tariff import TariffFormula
from runeflow.zones.tariffs.de import DE_TARIFF_FORMULAS
from runeflow.zones.tariffs.nl import NL_TARIFF_FORMULAS
from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

_ALL: dict[str, dict[str, TariffFormula]] = {
    "NL": NL_TARIFF_FORMULAS,
    "DE_LU": DE_TARIFF_FORMULAS,
    "DE": DE_TARIFF_FORMULAS,
}


def get_tariff_formula(zone: str, provider: str) -> TariffFormula:
    """
    Look up a tariff formula by zone and provider ID.

    Falls back to the universal wholesale formula if the zone/provider
    combination is not found.
    """
    zone_formulas = _ALL.get(zone.upper(), {})
    if provider in zone_formulas:
        return zone_formulas[provider]
    if provider == "wholesale":
        return WHOLESALE_FORMULA
    # Last resort: wholesale
    return WHOLESALE_FORMULA


__all__ = [
    "NL_TARIFF_FORMULAS",
    "DE_TARIFF_FORMULAS",
    "WHOLESALE_FORMULA",
    "get_tariff_formula",
]
