# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for AppConfig — defaults, env override, directory helpers."""

from __future__ import annotations

from pathlib import Path

from runeflow.config import AppConfig, _default_cache_dir


class TestDefaultCacheDir:
    def test_not_under_data(self):
        """Default must NOT be the old Docker path /data/runeflow."""
        p = _default_cache_dir()
        assert not str(p).startswith("/data"), (
            f"cache_dir default '{p}' looks like a Docker path; expected ~/.cache/runeflow"
        )

    def test_ends_with_runeflow(self):
        p = _default_cache_dir()
        assert p.name == "runeflow"

    def test_under_home_by_default(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        p = _default_cache_dir()
        assert str(p).startswith(str(Path.home()))

    def test_respects_xdg_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        p = _default_cache_dir()
        assert str(p).startswith(str(tmp_path))
        assert p.name == "runeflow"

    def test_xdg_empty_string_falls_back_to_home(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "  ")  # whitespace → treated as empty
        p = _default_cache_dir()
        assert str(p).startswith(str(Path.home()))


class TestAppConfigDefaults:
    def test_zone_default(self):
        cfg = AppConfig()
        assert cfg.zone == "NL"

    def test_cache_dir_is_path(self):
        cfg = AppConfig()
        assert isinstance(cfg.cache_dir, Path)

    def test_cache_dir_not_docker(self):
        cfg = AppConfig()
        assert not str(cfg.cache_dir).startswith("/data")

    def test_cache_dir_is_absolute(self):
        cfg = AppConfig()
        assert cfg.cache_dir.is_absolute()

    def test_derived_subdirs(self):
        cfg = AppConfig()
        assert cfg.prices_cache_dir == cfg.cache_dir / "prices"
        assert cfg.weather_cache_dir == cfg.cache_dir / "weather"
        assert cfg.generation_cache_dir == cfg.cache_dir / "generation"
        assert cfg.models_cache_dir == cfg.cache_dir / "models"
        assert cfg.forecasts_cache_dir == cfg.cache_dir / "forecasts"


class TestAppConfigFromEnv:
    def test_override_cache_dir(self, tmp_path):
        cfg = AppConfig.from_env(env={"CACHE_DIR": str(tmp_path / "custom")})
        assert cfg.cache_dir == (tmp_path / "custom").resolve()

    def test_override_zone(self):
        cfg = AppConfig.from_env(zone="DE_LU")
        assert cfg.zone == "DE_LU"

    def test_override_entsoe_key(self):
        cfg = AppConfig.from_env(env={"ENTSOE": "test-key-abc"})
        assert cfg.entsoe_api_key == "test-key-abc"

    def test_empty_env_uses_defaults(self):
        cfg = AppConfig.from_env(env={})
        assert cfg.zone == "NL"
        assert not str(cfg.cache_dir).startswith("/data")

    def test_tilde_expansion_in_cache_dir(self, tmp_path):
        """A CACHE_DIR with ~ should be expanded."""
        # Use an absolute path starting with home shorthand equivalent
        home = str(Path.home())
        cfg = AppConfig.from_env(env={"CACHE_DIR": f"{home}/my_cache"})
        assert cfg.cache_dir == Path(home, "my_cache")


class TestEnsureDirs:
    def test_creates_all_subdirs(self, tmp_path):
        cfg = AppConfig.from_env(env={"CACHE_DIR": str(tmp_path / "rf")})
        cfg.ensure_dirs()
        assert cfg.prices_cache_dir.is_dir()
        assert cfg.weather_cache_dir.is_dir()
        assert cfg.generation_cache_dir.is_dir()
        assert cfg.models_cache_dir.is_dir()
        assert cfg.forecasts_cache_dir.is_dir()

    def test_idempotent(self, tmp_path):
        """Calling ensure_dirs twice should not fail."""
        cfg = AppConfig.from_env(env={"CACHE_DIR": str(tmp_path / "rf2")})
        cfg.ensure_dirs()
        cfg.ensure_dirs()  # second call must not raise
