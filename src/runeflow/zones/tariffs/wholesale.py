# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Universal wholesale tariff (no markup, zone-independent)."""

from __future__ import annotations

import datetime

from runeflow.domain.tariff import TariffFormula


def _apply_wholesale(p: float, dt: datetime.date) -> float:  # noqa: ARG001
    """Wholesale passthrough — no markup."""
    return p


WHOLESALE_FORMULA = TariffFormula(
    provider_id="wholesale",
    country="",
    label="Wholesale (no markup)",
    apply=_apply_wholesale,
)
