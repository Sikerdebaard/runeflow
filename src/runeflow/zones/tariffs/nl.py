# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Dutch tariff formulas."""
from __future__ import annotations

import datetime

from runeflow.domain.tariff import TariffFormula

# ── Constants ─────────────────────────────────────────────────────────────────

NL_VAT = 1.21

# Dutch energy tax (energiebelasting) per kWh, by year.
# Source: Belastingdienst (2024–2026 confirmed; earlier years kept for completeness)
_NL_ENERGIEBELASTING: dict[int, float] = {
    2020: 0.08620,
    2021: 0.08693,
    2022: 0.04359,   # Emergency cut mid-year; use average
    2023: 0.12599,
    2024: 0.10154,
    2025: 0.10154,
    2026: 0.09161157,
}

_NL_EB_DEFAULT = 0.10154   # Fallback for unknown years


def _get_eb(dt: datetime.date) -> float:
    """Energy tax rate for *dt*'s year, with fallback."""
    return _NL_ENERGIEBELASTING.get(dt.year, _NL_EB_DEFAULT)


# ── Individual formulas ───────────────────────────────────────────────────────

def _apply_zonneplan(p: float, dt: datetime.date) -> float:
    """Zonneplan: spot + 2 ct/kWh markup incl. VAT."""
    return (p + 0.02 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_tibber_nl(p: float, dt: datetime.date) -> float:
    """Tibber NL: spot + 1.5 ct/kWh markup incl. VAT."""
    return (p + 0.015 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_easy_energy(p: float, dt: datetime.date) -> float:
    """EasyEnergy: spot + 1.99 ct/kWh markup."""
    return (p + 0.0199 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_greenchoice(p: float, dt: datetime.date) -> float:
    """Greenchoice: spot + 3.0 ct/kWh markup."""
    return (p + 0.03 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_vattenfall_nl(p: float, dt: datetime.date) -> float:
    """Vattenfall NL: spot + 2.5 ct/kWh markup."""
    return (p + 0.025 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_eneco_nl(p: float, dt: datetime.date) -> float:
    """Eneco NL: spot + 3.5 ct/kWh markup."""
    return (p + 0.035 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_essent_nl(p: float, dt: datetime.date) -> float:
    """Essent NL: spot + 3.5 ct/kWh markup."""
    return (p + 0.035 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_anwb_energie(p: float, dt: datetime.date) -> float:
    """ANWB Energie: spot + 2.0 ct/kWh markup."""
    return (p + 0.020 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_leapp(p: float, dt: datetime.date) -> float:
    """Leapp: spot + 1.0 ct/kWh markup."""
    return (p + 0.010 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_energie_van_ons(p: float, dt: datetime.date) -> float:
    """Energie van Ons: spot + 2.5 ct/kWh markup."""
    return (p + 0.025 / NL_VAT + _get_eb(dt)) * NL_VAT


def _apply_wholesale(p: float, dt: datetime.date) -> float:  # noqa: ARG001
    """Wholesale passthrough — no markup."""
    return p


# ── Registry ──────────────────────────────────────────────────────────────────

NL_TARIFF_FORMULAS: dict[str, TariffFormula] = {
    "wholesale": TariffFormula(
        "wholesale", "NL", "Wholesale (excl. tax)",
        _apply_wholesale,
    ),
    "zonneplan": TariffFormula("zonneplan", "NL", "Zonneplan", _apply_zonneplan),
    "tibber": TariffFormula("tibber", "NL", "Tibber NL", _apply_tibber_nl),
    "easy_energy": TariffFormula("easy_energy", "NL", "EasyEnergy", _apply_easy_energy),
    "greenchoice": TariffFormula("greenchoice", "NL", "Greenchoice", _apply_greenchoice),
    "vattenfall": TariffFormula("vattenfall", "NL", "Vattenfall NL", _apply_vattenfall_nl),
    "eneco": TariffFormula("eneco", "NL", "Eneco NL", _apply_eneco_nl),
    "essent": TariffFormula("essent", "NL", "Essent NL", _apply_essent_nl),
    "anwb": TariffFormula("anwb", "NL", "ANWB Energie", _apply_anwb_energie),
    "leapp": TariffFormula("leapp", "NL", "Leapp", _apply_leapp),
    "energie_van_ons": TariffFormula(
        "energie_van_ons", "NL", "Energie van Ons", _apply_energie_van_ons
    ),
}