"""
Dutch electricity tariff constants and provider price formulas.

This is the single source of truth for all provider markup values.
All other modules should import from here — do not define price constants
or formulas in other files.

Markup values are sourced from epexprijzen.nl (verified 2026-03-22).
Update this file whenever provider rates change.

Formula structure
-----------------
All prices are in EUR/kWh, the input ``p`` is the EPEX wholesale spot
price *excluding* BTW (as delivered by ENTSO-e, converted from EUR/MWh).

    all_in_price = (p + opslag_excl_btw + energiebelasting_excl_btw) * 1.21

Provider markups on epexprijzen.nl are advertised *including* BTW, so the
raw advertised figure must be divided by 1.21 to get the excl-BTW value
before adding it to the wholesale price.  Two providers (Budget Energie,
Engie) happen to store their markup already excl. BTW in the source — those
are used directly.
"""

from __future__ import annotations

import datetime
from typing import Callable

# ---------------------------------------------------------------------------
# VAT
# ---------------------------------------------------------------------------

NL_VAT = 1.21  # 21% BTW multiplier

# ---------------------------------------------------------------------------
# Energiebelasting (energy tax) — Netherlands, per calendar year
# ---------------------------------------------------------------------------
# Energiebelasting (incl. ODE, merged 2023) excl. 21% BTW.
# Source: Wet belastingen op milieugrondslag, Art. 59
# Applies to the first 10 000 kWh/year of household consumption.
#
# To add a new year: insert one line in the dict below and update the
# fallback comment.  Nothing else needs to change.
_NL_ENERGIEBELASTING_BY_YEAR: dict[int, float] = {
    # year : EUR/kWh excl. BTW
    2024: 0.10154,
    2025: 0.10154,    # unchanged from 2024
    2026: 0.09161157, # reduced as of 2026-01-01 (source: epexprijzen.nl)
    # 2027: 0.0????,  # ← add here when known (published ~December each year)
}

# Convenience alias for the current year — kept for backward compatibility.
NL_ENERGIEBELASTING_2026 = _NL_ENERGIEBELASTING_BY_YEAR[2026]


def get_energiebelasting(dt: datetime.date | None = None) -> float:
    """
    Return the Energiebelasting rate (excl. BTW) for the given date.

    Falls back to the most recent known rate when the year is not yet in the
    table (e.g. early 2027 before the new rate is published).

    Args:
        dt: Date to look up.  Defaults to today (Amsterdam time).

    Returns:
        EUR/kWh excl. BTW for the first 10 000 kWh/year bracket.
    """
    if dt is None:
        dt = datetime.datetime.now(tz=datetime.timezone.utc).date()
    year = dt.year if isinstance(dt, datetime.date) else dt
    # Exact match first, then walk backwards to find the most recent known rate.
    while year >= min(_NL_ENERGIEBELASTING_BY_YEAR):
        if year in _NL_ENERGIEBELASTING_BY_YEAR:
            return _NL_ENERGIEBELASTING_BY_YEAR[year]
        year -= 1
    # Should never be reached given the table starts at 2024.
    return _NL_ENERGIEBELASTING_BY_YEAR[min(_NL_ENERGIEBELASTING_BY_YEAR)]


# ---------------------------------------------------------------------------
# Provider price formulas
# ---------------------------------------------------------------------------
# Keys match the provider IDs used by epexprijzen.nl so they can also be
# used as the ``provider`` parameter for the tariff export.
#
# ``p``  — EPEX wholesale spot price, EUR/kWh excl. BTW
# return — all-in end-user price, EUR/kWh incl. BTW
#
# IMPORTANT: the lambdas call get_energiebelasting() at *evaluation time*,
# not at import time, so the correct rate is used automatically after a
# year rollover without any code changes.
#
# Advertised markup (incl. BTW) → divide by 1.21 before applying VAT.
# Markup already excl. BTW     → use value directly (noted in comment).

PRICE_FORMULAS: dict[str, Callable[[float], float]] = {
    # Raw wholesale price — no taxes, no markup
    "wholesale": lambda p: p,

    # --- Dutch dynamic electricity providers, sorted by opslag (low → high) ---
    # Advertised opslag incl. BTW: €0.018/kWh
    "anwb-energie":       lambda p: (p + 0.018   / 1.21 + get_energiebelasting()) * NL_VAT,
    "frank-energie":      lambda p: (p + 0.0182  / 1.21 + get_energiebelasting()) * NL_VAT,
    "frank-energie-slim": lambda p: (p + 0.0182  / 1.21 + get_energiebelasting()) * NL_VAT,
    # Engie stores markup excl. BTW directly (→ €0.019 incl. BTW)
    "engie":              lambda p: (p + 0.0157          + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.020/kWh
    "coolblue-energie":   lambda p: (p + 0.02    / 1.21 + get_energiebelasting()) * NL_VAT,
    "zonneplan":          lambda p: (p + 0.02    / 1.21 + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.021/kWh
    "tibber":             lambda p: (p + 0.021   / 1.21 + get_energiebelasting()) * NL_VAT,
    # Budget Energie stores markup excl. BTW directly (→ €0.021 incl. BTW)
    "budget-energie":     lambda p: (p + 0.0174          + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.022/kWh
    "easyenergy":         lambda p: (p + 0.02178 / 1.21 + get_energiebelasting()) * NL_VAT,
    "nextenergy":         lambda p: (p + 0.0219  / 1.21 + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.023/kWh
    "eneco":              lambda p: (p + 0.02257 / 1.21 + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.025/kWh
    "innova":             lambda p: (p + 0.02508 / 1.21 + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.026/kWh
    "vandebron":          lambda p: (p + 0.02571 / 1.21 + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.029/kWh
    "energie-vanons":     lambda p: (p + 0.029   / 1.21 + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.030/kWh
    "atoom-alliantie":    lambda p: (p + 0.03025 / 1.21 + get_energiebelasting()) * NL_VAT,
    # Advertised opslag incl. BTW: €0.034/kWh
    "energyzero":         lambda p: (p + 0.03388 / 1.21 + get_energiebelasting()) * NL_VAT,
    "hegg":               lambda p: (p + 0.03388 / 1.21 + get_energiebelasting()) * NL_VAT,
    "samsam":             lambda p: (p + 0.0339  / 1.21 + get_energiebelasting()) * NL_VAT,
}


def get_price_formula(provider: str = "wholesale") -> Callable[[float], float]:
    """
    Return the price formula for *provider*, falling back to 'wholesale'.

    Args:
        provider: Key from PRICE_FORMULAS, e.g. 'zonneplan', 'tibber',
                  'anwb-energie'.  Case-insensitive.

    Returns:
        Callable ``f(p) -> all_in_price`` where both values are EUR/kWh.
    """
    return PRICE_FORMULAS.get(provider.lower(), PRICE_FORMULAS["wholesale"])
