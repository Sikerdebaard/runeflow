# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Base classes for the feature engineering pipeline."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)

# Number of warmup days required for inference feature engineering.
# Must be > max lag (7 days same-hour) + forecast horizon (9 days).
INFERENCE_WARMUP_DAYS = 16


class FeatureGroup(ABC):
    """Abstract base class for a single feature group."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used in ZoneConfig.feature_groups."""

    @property
    def requires(self) -> tuple[str, ...]:
        """Column names (or prefixes) that must exist in the DataFrame."""
        return ()

    @property
    def produces(self) -> tuple[str, ...]:
        """Column names this group adds (documentation only; not enforced)."""
        return ()

    @abstractmethod
    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        """
        Add features to *df* and return the enriched copy.

        Never mutates the incoming DataFrame — always works on a copy.
        """

    def _copy(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


class FeaturePipeline:
    """Executes a sequence of FeatureGroup transforms in dependency order."""

    def __init__(self, groups: list[FeatureGroup]) -> None:
        self._groups = groups
        self._validate_dependencies()

    # ------------------------------------------------------------------
    def _validate_dependencies(self) -> None:
        available: set[str] = set()
        for group in self._groups:
            missing = [r for r in group.requires if not any(r in a for a in available)]
            if missing:
                logger.debug(
                    "Group '%s' has unresolved requires at validation time: %s "
                    "(may be satisfied by runtime data)",
                    group.name,
                    missing,
                )
            for col in group.produces:
                available.add(col)

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        """Apply all groups sequentially, skipping any that raise an exception."""
        for group in self._groups:
            try:
                df = group.transform(df, zone_cfg)
            except Exception:
                logger.exception("Feature group '%s' failed — skipping", group.name)
        return df

    # ------------------------------------------------------------------
    @property
    def groups(self) -> list[FeatureGroup]:
        return list(self._groups)