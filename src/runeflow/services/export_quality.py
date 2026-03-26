# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
ExportQualityService — writes site/api/quality.json with live forecast
quality metrics derived from the latest ForecastResult.

Training metrics (MAE, R², coverage) are optionally loaded from a
train-result JSON sidecar written by TrainService.  When that sidecar
is absent the training section is omitted.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import inject

from runeflow.ports.store import DataStore
from runeflow.zones.registry import ZoneRegistry

logger = logging.getLogger(__name__)


def _safe(v: float) -> float | None:
    """Convert NaN/inf to None so JSON serialises cleanly."""
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, 4)


class ExportQualityService:
    """Compute and write ``quality.json`` for all registered zones."""

    @inject.autoparams()
    def __init__(
        self,
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
    ) -> None:
        self._store = store

    # ------------------------------------------------------------------
    def run(self, output_path: Path, zones: list[str] | None = None) -> dict[str, object]:
        """Compute quality metrics and write quality.json.

        Args:
            output_path: Destination path (e.g. ``site/api/quality.json``).
            zones: Zone codes to include.  Defaults to all registered zones.

        Returns:
            The payload dict.
        """
        zone_codes = zones or ZoneRegistry.list_zones()
        payload: dict[str, object] = {}

        for zone in zone_codes:
            zone_data = self._compute_zone_quality(zone)
            if zone_data:
                payload[zone] = zone_data

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("ExportQuality: wrote quality for %d zones to %s", len(payload), output_path)
        return payload

    # ------------------------------------------------------------------
    def _compute_zone_quality(self, zone: str) -> dict[str, object] | None:
        forecast = self._store.load_latest_forecast(zone)
        if forecast is None:
            logger.warning("ExportQuality: no forecast for zone=%s, skipping", zone)
            return None

        pts = forecast.points
        if not pts:
            return None

        # ── Live metrics from forecast points ─────────────────────────────────
        spreads = [p.upper - p.lower for p in pts if p.upper > p.lower]
        agreements = [p.model_agreement for p in pts if 0 < p.model_agreement <= 1]
        ensemble_p50s = [p.ensemble_p50 for p in pts if p.ensemble_p50 > 0]

        mean_spread = float(sum(spreads) / len(spreads)) if spreads else 0.0
        mean_agreement = float(sum(agreements) / len(agreements)) if agreements else 0.0
        mean_p50 = float(sum(ensemble_p50s) / len(ensemble_p50s)) if ensemble_p50s else 0.0

        # Composite grade (0–10): agreement 60 %, spread score 40 %
        # Spread score: 10 when spread ≤ 10 EUR/MWh, 0 when ≥ 60 EUR/MWh
        spread_score = max(0.0, min(10.0, 10.0 - (mean_spread - 10.0) / 5.0))
        composite = round(mean_agreement * 10.0 * 0.6 + spread_score * 0.4, 2)

        # Grade label
        if composite >= 9.0:
            grade_label = "Excellent"
        elif composite >= 7.5:
            grade_label = "Very Good"
        elif composite >= 6.0:
            grade_label = "Good"
        elif composite >= 4.0:
            grade_label = "Fair"
        else:
            grade_label = "Poor"

        live: dict[str, object] = {
            "mean_ensemble_spread_eur_mwh": _safe(mean_spread),
            "mean_model_agreement": _safe(mean_agreement),
            "mean_ensemble_p50_eur_mwh": _safe(mean_p50),
            "forecast_hours": len(pts),
            "generated_at": forecast.created_at.isoformat(),
        }

        # ── Training sidecar (optional) ────────────────────────────────────────
        training = self._load_training_sidecar(zone)

        result: dict[str, object] = {
            "composite_grade": composite,
            "grade_label": grade_label,
            "live": live,
            "model_version": forecast.model_version,
        }
        if training:
            result["training"] = training

        return result

    # ------------------------------------------------------------------
    def _load_training_sidecar(self, zone: str) -> dict[str, object] | None:
        """Load training metrics from the JSON sidecar written by TrainService."""
        sidecar = self._store.load_supplemental(zone, "train_result")
        if sidecar is None or sidecar.empty:
            return None
        try:
            row = sidecar.iloc[0]
            return {
                "mae": _safe(float(row.get("mae", float("nan")))),
                "r2": _safe(float(row.get("r2", float("nan")))),
                "coverage": _safe(float(row.get("coverage", float("nan")))),
                "trained_at": str(row.get("trained_at", "")),
            }
        except Exception as exc:
            logger.debug("ExportQuality: could not parse training sidecar: %s", exc)
            return None
