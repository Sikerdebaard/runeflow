# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""AblationService — cheap-first E0..E7 experiment runner."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import inject

from runeflow.ports.store import DataStore
from runeflow.services.inference import InferenceService
from runeflow.services.performance import PerformanceService
from runeflow.services.train import TrainService
from runeflow.services.warmup import WarmupService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StageConfig:
    deterministic_only: bool
    ensemble_members: int | None
    days_back: int


@dataclass(frozen=True)
class _Profile:
    key: str
    description: str
    flags: dict[str, bool]


_STAGE_CONFIGS: dict[str, _StageConfig] = {
    "A": _StageConfig(deterministic_only=True, ensemble_members=0, days_back=7),
    "B": _StageConfig(deterministic_only=True, ensemble_members=0, days_back=30),
    "C": _StageConfig(deterministic_only=False, ensemble_members=5, days_back=30),
    "D": _StageConfig(deterministic_only=False, ensemble_members=None, days_back=30),
}

_PROFILES: dict[str, _Profile] = {
    # Legacy-ish baseline (new feature blocks disabled)
    "E0": _Profile(
        key="E0",
        description="baseline_without_new_blocks",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": False,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": False,
            "RUNEFLOW_ENABLE_DIESEL_CRACK": False,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E1": _Profile(
        key="E1",
        description="seasonality_only",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": True,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": False,
            "RUNEFLOW_ENABLE_DIESEL_CRACK": False,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E2": _Profile(
        key="E2",
        description="spread_only",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": False,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_DIESEL_CRACK": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E3": _Profile(
        key="E3",
        description="stress_only",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": False,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_DIESEL_CRACK": False,
            "RUNEFLOW_ENABLE_MARKET_STRESS": True,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E4": _Profile(
        key="E4",
        description="datacenter_only",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": False,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": False,
            "RUNEFLOW_ENABLE_DIESEL_CRACK": False,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": True,
        },
    ),
    "E5": _Profile(
        key="E5",
        description="seasonality_plus_spread",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": True,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_DIESEL_CRACK": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E6": _Profile(
        key="E6",
        description="seasonality_plus_spread_plus_stress",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": True,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_DIESEL_CRACK": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": True,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E7": _Profile(
        key="E7",
        description="all_new_feature_blocks_enabled",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": True,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_DIESEL_CRACK": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": True,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": True,
        },
    ),
}


@contextmanager
def _env_flags(flags: dict[str, bool]) -> Iterator[None]:
    old: dict[str, str | None] = {k: os.getenv(k) for k in flags}
    try:
        for key, value in flags.items():
            os.environ[key] = "1" if value else "0"
        yield
    finally:
        for key, prev in old.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


class AblationService:
    """Run predefined ablation experiments under staged compute budgets."""

    @staticmethod
    def available_profiles() -> list[str]:
        return list(_PROFILES.keys())

    @staticmethod
    def available_stages() -> list[str]:
        return list(_STAGE_CONFIGS.keys())

    def run(
        self,
        *,
        zone: str,
        stage: str,
        experiments: list[str],
        output_path: Path,
    ) -> dict[str, Any]:
        stage = stage.upper()
        if stage not in _STAGE_CONFIGS:
            expected = ", ".join(_STAGE_CONFIGS)
            raise ValueError(f"Unknown stage '{stage}'. Expected one of: {expected}")

        stage_cfg = _STAGE_CONFIGS[stage]
        now = datetime.now(UTC)

        report: dict[str, Any] = {
            "zone": zone,
            "stage": stage,
            "stage_config": {
                "deterministic_only": stage_cfg.deterministic_only,
                "ensemble_members": stage_cfg.ensemble_members,
                "days_back": stage_cfg.days_back,
            },
            "started_at": now.isoformat(),
            "experiments": [],
        }

        store: DataStore = inject.instance(DataStore)  # type: ignore[assignment]

        for key in experiments:
            profile = _PROFILES.get(key)
            if profile is None:
                report["experiments"].append(
                    {
                        "key": key,
                        "status": "skipped",
                        "reason": "unknown_experiment_key",
                    }
                )
                continue

            logger.info("Ablation %s (%s): starting", profile.key, profile.description)
            enabled_flags = [name for name, enabled in profile.flags.items() if enabled]
            logger.info(
                "Ablation %s: active feature flags = %s",
                profile.key,
                ", ".join(enabled_flags) if enabled_flags else "none",
            )
            exp_rec: dict[str, Any] = {
                "key": profile.key,
                "description": profile.description,
                "flags": profile.flags,
            }

            try:
                with _env_flags(profile.flags):
                    train_result = TrainService().run()
                    WarmupService().run(force=True)

                    forecast = InferenceService().run(
                        deterministic_only=stage_cfg.deterministic_only,
                        ensemble_member_limit=stage_cfg.ensemble_members,
                    )
                    store.save_forecast(forecast)

                    perf = PerformanceService().compute_zone_performance(
                        days_back=stage_cfg.days_back,
                    )

                xgb_metrics = train_result.metrics.get("xgboost_quantile", {})
                horizon = {
                    h.horizon_label: {
                        "mae": h.mae,
                        "rmse": h.rmse,
                        "n_hours": h.n_hours,
                    }
                    for h in perf.horizon_metrics
                }

                exp_rec.update(
                    {
                        "status": "ok",
                        "train": {
                            "model_version": train_result.model_version,
                            "mae": xgb_metrics.get("mae"),
                            "r2": xgb_metrics.get("r2"),
                            "coverage": xgb_metrics.get("coverage"),
                        },
                        "performance": {
                            "overall_mae": perf.overall_mae,
                            "overall_rmse": perf.overall_rmse,
                            "ensemble_coverage_pct": perf.ensemble_coverage_pct,
                            "n_archived_forecasts": perf.n_archived_forecasts,
                            "n_comparable_hours": perf.n_comparable_hours,
                            "horizon": horizon,
                        },
                    }
                )
            except Exception as exc:
                logger.exception("Ablation %s failed", profile.key)
                exp_rec.update(
                    {
                        "status": "failed",
                        "error": str(exc),
                    }
                )

            report["experiments"].append(exp_rec)

        report["finished_at"] = datetime.now(UTC).isoformat()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Ablation report written to %s", output_path)
        return report
