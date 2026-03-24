# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Model registry."""
from __future__ import annotations

from runeflow.ports.model import ModelPort

from .xgboost_quantile import XGBoostQuantileModel
from .extreme_high import ExtremeHighModel
from .extreme_low import ExtremeLowModel

MODEL_REGISTRY: dict[str, type[ModelPort]] = {
    "xgboost_quantile": XGBoostQuantileModel,
    "extreme_high": ExtremeHighModel,
    "extreme_low": ExtremeLowModel,
}

__all__ = [
    "MODEL_REGISTRY",
    "XGBoostQuantileModel",
    "ExtremeHighModel",
    "ExtremeLowModel",
]