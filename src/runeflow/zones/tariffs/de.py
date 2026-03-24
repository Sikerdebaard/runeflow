# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""German tariff formulas."""

from __future__ import annotations

import datetime

from runeflow.domain.tariff import TariffFormula

# ── Constants ─────────────────────────────────────────────────────────────────

DE_VAT = 1.19

# German energy tax (Stromsteuer) — fixed at 2.05 ct/kWh since 2003
_DE_STROMSTEUER = 0.0205  # EUR/kWh

# EEG-Umlage abolished from 2022-07-01; previously was ~3-7 ct/kWh
# aWATTar, Tibber: pure spot + network fees; network bundestag varies by region
# Using typical household network costs (Netzentgelt) ~8 ct/kWh as surrogate
_DE_NETWORK_COST_EUR_KWH = 0.080  # approximate grid + levies, excl. spot & tax


def _apply_tibber_de(p: float, dt: datetime.date) -> float:
    """Tibber DE: spot + 0.99 ct/kWh markup + Stromsteuer incl. VAT."""
    return (p + 0.0099 / DE_VAT + _DE_STROMSTEUER + _DE_NETWORK_COST_EUR_KWH) * DE_VAT


def _apply_awattar_de(p: float, dt: datetime.date) -> float:
    """aWATTar DE: pure spot incl. Stromsteuer + network, no markup."""
    return (p + _DE_STROMSTEUER + _DE_NETWORK_COST_EUR_KWH) * DE_VAT


def _apply_ostrom_de(p: float, dt: datetime.date) -> float:
    """Ostrom DE: spot + 0.49 ct/kWh markup."""
    return (p + 0.0049 / DE_VAT + _DE_STROMSTEUER + _DE_NETWORK_COST_EUR_KWH) * DE_VAT


def _apply_wholesale(p: float, dt: datetime.date) -> float:  # noqa: ARG001
    """Wholesale passthrough — no markup."""
    return p


# ── Registry ──────────────────────────────────────────────────────────────────

DE_TARIFF_FORMULAS: dict[str, TariffFormula] = {
    "wholesale": TariffFormula("wholesale", "DE", "Wholesale (excl. tax)", _apply_wholesale),
    "tibber": TariffFormula("tibber", "DE", "Tibber DE", _apply_tibber_de),
    "awattar": TariffFormula("awattar", "DE", "aWATTar DE", _apply_awattar_de),
    "ostrom": TariffFormula("ostrom", "DE", "Ostrom DE", _apply_ostrom_de),
}
