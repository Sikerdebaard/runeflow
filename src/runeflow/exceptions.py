# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Application exception hierarchy."""
from __future__ import annotations


class RuneflowError(Exception):
    """Base class for all runeflow exceptions."""


# ── Adapter errors ────────────────────────────────────────────────────────────

class AdapterError(RuneflowError):
    """Base class for adapter (I/O) errors."""


class DataUnavailableError(AdapterError):
    """Requested data does not exist for this zone/period."""


class DownloadError(AdapterError):
    """Network or API error while downloading data."""


class RateLimitError(AdapterError):
    """API rate limit exceeded."""


class AuthenticationError(AdapterError):
    """API key missing or invalid."""


# ── Zone errors ───────────────────────────────────────────────────────────────

class ZoneError(RuneflowError):
    """Base class for zone configuration errors."""


class UnsupportedZoneError(ZoneError):
    """Requested zone is not registered."""


# ── Validation errors ─────────────────────────────────────────────────────────

class ValidationError(RuneflowError):
    """Data quality check failed."""


# ── Config errors ─────────────────────────────────────────────────────────────

class ConfigError(RuneflowError):
    """Invalid application configuration."""


# ── Model errors ──────────────────────────────────────────────────────────────

class ModelError(RuneflowError):
    """Base class for ML model errors."""


class ModelNotTrainedError(ModelError):
    """Attempted to predict before training the model."""


class ModelLoadError(ModelError):
    """Could not load model artifacts from storage."""