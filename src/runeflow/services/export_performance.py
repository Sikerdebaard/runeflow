# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""ExportPerformanceService — writes api/performance.json."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import inject
import pandas as pd

from runeflow.domain.performance import ZonePerformance
from runeflow.ports.store import DataStore
from runeflow.zones.registry import ZoneRegistry

logger = logging.getLogger(__name__)


def _safe(v: object) -> object:
    """Convert NaN/Inf floats to None for clean JSON serialisation."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, float):
        return round(v, 4)
    return v


class ExportPerformanceService:
    """Compute performance metrics for all zones and write ``performance.json``."""

    @inject.autoparams()
    def __init__(
        self,
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
    ) -> None:
        self._store = store

    # ------------------------------------------------------------------
    def run(
        self,
        output_path: Path,
        zones: list[str] | None = None,
    ) -> dict:
        """Compute and write performance JSON.

        Args:
            output_path: Destination path (e.g. ``site/api/performance.json``).
            zones: Zone codes to include.  Defaults to all registered zones.

        Returns:
            The payload dict.
        """
        zone_codes = zones or ZoneRegistry.list_zones()
        payload: dict = {}

        for zone in zone_codes:
            try:
                from runeflow.binder import configure_injector

                configure_injector(zone, allow_override=True)
                from runeflow.services.performance import PerformanceService

                svc = PerformanceService()
                perf = svc.compute_zone_performance()
                payload[zone] = self._serialize_zone(perf)
            except Exception as exc:
                logger.warning("ExportPerformance: %s failed: %s", zone, exc)

        # Rankings sorted by MAE ascending (lower is better)
        ranked = sorted(
            [
                (z, d.get("overall_mae"))
                for z, d in payload.items()
                if isinstance(d, dict) and d.get("overall_mae") is not None
            ],
            key=lambda x: x[1],  # type: ignore[arg-type, return-value]
        )
        payload["_rankings"] = [
            {"zone": z, "mae": _safe(mae), "rank": i + 1} for i, (z, mae) in enumerate(ranked)
        ]
        payload["_generated_at"] = pd.Timestamp.now("UTC").isoformat()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info(
            "ExportPerformance: wrote %d zones to %s",
            len(zone_codes),
            output_path,
        )
        return payload

    # ------------------------------------------------------------------
    def _serialize_zone(self, perf: ZonePerformance) -> dict:
        # Weighted directional accuracy across all archived forecasts
        total_h = sum(a.n_comparable_hours for a in perf.forecast_accuracies)
        overall_directional_accuracy: float | None = (
            sum(a.directional_accuracy * a.n_comparable_hours for a in perf.forecast_accuracies)
            / total_h
            if total_h > 0
            else None
        )

        # Latest R² from training history (most recent training run)
        latest_r2: float | None = None
        for row in reversed(perf.training_history):
            v = row.get("r2")
            if v is not None:
                latest_r2 = _safe(float(v))  # type: ignore[assignment]
                break

        return {
            "zone": perf.zone,
            "zone_name": perf.zone_name,
            "overall_mae": _safe(perf.overall_mae),
            "overall_rmse": _safe(perf.overall_rmse),
            "overall_directional_accuracy": _safe(overall_directional_accuracy),
            "latest_r2": latest_r2,
            "ensemble_coverage_pct": _safe(perf.ensemble_coverage_pct),
            "n_archived_forecasts": perf.n_archived_forecasts,
            "n_comparable_hours": perf.n_comparable_hours,
            "training_history": [
                {k: _safe(v) if isinstance(v, float) else v for k, v in row.items()}
                for row in perf.training_history
            ],
            "horizon_metrics": [
                {
                    "label": h.horizon_label,
                    "n_hours": h.n_hours,
                    "mae": _safe(h.mae),
                    "rmse": _safe(h.rmse),
                }
                for h in perf.horizon_metrics
            ],
            "forecast_accuracies": [
                {
                    "created_at": a.forecast_created_at.isoformat(),
                    "model_version": a.model_version,
                    "n_hours": a.n_comparable_hours,
                    "mae": _safe(a.mae),
                    "rmse": _safe(a.rmse),
                    "mape": _safe(a.mape),
                    "directional_accuracy": _safe(a.directional_accuracy),
                    "mean_model_agreement": _safe(a.mean_model_agreement),
                }
                for a in perf.forecast_accuracies
            ],
            "generated_at": perf.generated_at.isoformat(),
        }
