# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Feature-block feature flags (single source of truth).

Experimental feature blocks introduced in PR #18 are **off by default**. They are
only enabled when the corresponding environment variable is explicitly set to a
truthy value. This keeps the production model on the known-good feature set
unless an experiment (such as the ablation runner) opts in explicitly.
"""

from __future__ import annotations

import os

# Map: environment variable -> short label shared with TrainService logging.
FLAG_LABELS: dict[str, str] = {
    "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": "cyclical_temporal",
    "RUNEFLOW_ENABLE_SPREAD_FEATURES": "spread_features",
    "RUNEFLOW_ENABLE_MARKET_STRESS": "market_stress",
    "RUNEFLOW_ENABLE_DATACENTER_PROXY": "datacenter_proxy",
}

_FALSY = {"0", "false", "no", "off"}


def feature_flag_enabled(name: str, *, default: bool = False) -> bool:
    """Return True if *name* is explicitly enabled via env, else *default*.

    Defaults to False so new/experimental feature blocks stay off in production
    unless a deployment explicitly enables them.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSY


def enabled_flag_labels() -> list[str]:
    """Return the short labels of the currently enabled feature flags."""
    return [
        label for name, label in FLAG_LABELS.items() if feature_flag_enabled(name, default=False)
    ]
