# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Ensemble strategy registry."""

from __future__ import annotations

from runeflow.ports.ensemble import EnsembleStrategy

from .condition_gated import ConditionGatedStrategy
from .simple_weighted import SimpleWeightedStrategy

ENSEMBLE_REGISTRY: dict[str, type[EnsembleStrategy]] = {
    "condition_gated": ConditionGatedStrategy,
    "simple_weighted": SimpleWeightedStrategy,
}

__all__ = ["ENSEMBLE_REGISTRY", "ConditionGatedStrategy", "SimpleWeightedStrategy"]
