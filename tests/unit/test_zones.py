# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for ZoneRegistry and zone configurations."""

from __future__ import annotations

import datetime as _dt

import pytest

from runeflow.exceptions import UnsupportedZoneError
from runeflow.zones.registry import ZoneRegistry


class TestZoneRegistry:
    def test_lists_nl_and_de_lu(self):
        zones = ZoneRegistry.list_zones()
        assert "NL" in zones
        assert "DE_LU" in zones

    def test_list_zones_sorted(self):
        zones = ZoneRegistry.list_zones()
        assert zones == sorted(zones)

    def test_get_nl(self):
        cfg = ZoneRegistry.get("NL")
        assert cfg.zone == "NL"

    def test_get_de_lu(self):
        cfg = ZoneRegistry.get("DE_LU")
        assert cfg.zone == "DE_LU"

    def test_unsupported_zone_raises(self):
        with pytest.raises(UnsupportedZoneError, match="Zone 'XX'"):
            ZoneRegistry.get("XX")

    def test_case_sensitive(self):
        """Zone code is case-sensitive; lowercase should raise."""
        with pytest.raises(UnsupportedZoneError):
            ZoneRegistry.get("nl")


class TestNLZoneConfig:
    @pytest.fixture(autouse=True)
    def cfg(self, zone_cfg_nl):
        self.cfg = zone_cfg_nl

    def test_name(self):
        assert "Netherlands" in self.cfg.name

    def test_timezone(self):
        assert self.cfg.timezone == "Europe/Amsterdam"

    def test_has_energyzero(self):
        assert self.cfg.has_energyzero is True

    def test_has_ned(self):
        assert self.cfg.has_ned is True

    def test_solar_capacity_positive(self):
        assert self.cfg.installed_solar_capacity_mw > 0

    def test_wind_capacity_positive(self):
        assert self.cfg.installed_wind_capacity_mw > 0

    def test_feature_groups_contain_temporal(self):
        assert "temporal" in self.cfg.feature_groups

    def test_feature_groups_valid_names(self):
        """All feature groups should match names in the registry."""
        from runeflow.features.registry import FEATURE_REGISTRY

        unknown = [g for g in self.cfg.feature_groups if g not in FEATURE_REGISTRY]
        assert unknown == [], f"Unknown feature groups in NL config: {unknown}"

    def test_models_include_xgboost(self):
        assert "xgboost_quantile" in self.cfg.models

    def test_ensemble_strategy(self):
        assert self.cfg.ensemble_strategy == "condition_gated"

    def test_has_tariff_formulas(self):
        assert len(self.cfg.tariff_formulas) > 0

    def test_weather_locations_present(self):
        assert len(self.cfg.weather_locations) >= 1

    def test_primary_weather_location_coords(self):
        loc = self.cfg.primary_weather_location
        # De Bilt is roughly 52.1°N, 5.2°E
        assert 50 < loc.lat < 54
        assert 4 < loc.lon < 7

    def test_historical_years_non_empty(self):
        assert len(self.cfg.historical_years) > 0

    def test_min_training_years(self):
        assert self.cfg.min_training_years >= 1


class TestDELUZoneConfig:
    @pytest.fixture(autouse=True)
    def cfg(self, zone_cfg_de_lu):
        self.cfg = zone_cfg_de_lu

    def test_name(self):
        assert "Germany" in self.cfg.name or "DE" in self.cfg.zone

    def test_no_energyzero(self):
        """DE_LU does not have EnergyZero (NL-specific retailer)."""
        assert self.cfg.has_energyzero is False

    def test_no_ned(self):
        """DE_LU does not have NED (Dutch-specific)."""
        assert self.cfg.has_ned is False

    def test_feature_groups_valid_names(self):
        from runeflow.features.registry import FEATURE_REGISTRY

        unknown = [g for g in self.cfg.feature_groups if g not in FEATURE_REGISTRY]
        assert unknown == [], f"Unknown feature groups in DE_LU config: {unknown}"

    def test_solar_capacity_large(self):
        """Germany has very large solar capacity (>50 GW)."""
        assert self.cfg.installed_solar_capacity_mw > 50_000

    def test_timezone(self):
        assert self.cfg.timezone == "Europe/Berlin"


# ---------------------------------------------------------------------------
# Tariff formula functions — NL
# ---------------------------------------------------------------------------

_DATE_2024 = _dt.date(2024, 6, 1)
_DATE_2026 = _dt.date(2026, 6, 1)
_DATE_FUTURE = _dt.date(2099, 1, 1)  # unknown year → fallback
_WHOLESALE = 0.10  # EUR/kWh


class TestNLTariffFormulas:
    """Each NL provider formula should add markup + taxes over the wholesale price."""

    @pytest.fixture(autouse=True)
    def _formulas(self):
        from runeflow.zones.tariffs.nl import NL_TARIFF_FORMULAS

        self.formulas = NL_TARIFF_FORMULAS

    def _apply(self, provider: str, p: float = _WHOLESALE, dt: _dt.date = _DATE_2024) -> float:
        formula = self.formulas[provider]
        return formula.apply(p, dt)

    def test_wholesale_passthrough(self):
        result = self._apply("wholesale", p=0.1)
        assert result == pytest.approx(0.1, abs=1e-9)

    def test_wholesale_zero(self):
        result = self._apply("wholesale", p=0.0)
        assert result == 0.0

    def test_zonneplan_higher_than_wholesale(self):
        assert self._apply("zonneplan") > _WHOLESALE

    def test_tibber_higher_than_wholesale(self):
        assert self._apply("tibber") > _WHOLESALE

    def test_easy_energy_higher_than_wholesale(self):
        assert self._apply("easy_energy") > _WHOLESALE

    def test_greenchoice_higher_than_wholesale(self):
        assert self._apply("greenchoice") > _WHOLESALE

    def test_vattenfall_higher_than_wholesale(self):
        assert self._apply("vattenfall") > _WHOLESALE

    def test_eneco_higher_than_wholesale(self):
        assert self._apply("eneco") > _WHOLESALE

    def test_essent_higher_than_wholesale(self):
        assert self._apply("essent") > _WHOLESALE

    def test_anwb_higher_than_wholesale(self):
        assert self._apply("anwb") > _WHOLESALE

    def test_leapp_higher_than_wholesale(self):
        assert self._apply("leapp") > _WHOLESALE

    def test_energie_van_ons_higher_than_wholesale(self):
        assert self._apply("energie_van_ons") > _WHOLESALE

    def test_greenchoice_higher_markup_than_tibber(self):
        """Greenchoice (3 ct) > Tibber (1.5 ct) for same input."""
        assert self._apply("greenchoice") > self._apply("tibber")

    def test_eneco_higher_markup_than_zonneplan(self):
        """Eneco (3.5 ct) > Zonneplan (2 ct)."""
        assert self._apply("eneco") > self._apply("zonneplan")

    def test_2026_energy_tax_different_from_2024(self):
        """Energy tax changes year-to-year."""
        r2024 = self._apply("vattenfall", dt=_DATE_2024)
        r2026 = self._apply("vattenfall", dt=_DATE_2026)
        assert r2024 != pytest.approx(r2026, abs=1e-6)

    def test_unknown_year_uses_fallback(self):
        """Future years not in the lookup table use the fallback rate."""
        result = self._apply("zonneplan", dt=_DATE_FUTURE)
        assert result > 0

    def test_all_providers_registered(self):
        providers = set(self.formulas.keys())
        for p in [
            "wholesale",
            "zonneplan",
            "tibber",
            "easy_energy",
            "greenchoice",
            "vattenfall",
            "eneco",
            "essent",
            "anwb",
            "leapp",
            "energie_van_ons",
        ]:
            assert p in providers, f"Provider '{p}' missing from NL_TARIFF_FORMULAS"


class TestDETariffFormulas:
    """DE/DE_LU tariff formula functions."""

    @pytest.fixture(autouse=True)
    def _formulas(self):
        from runeflow.zones.tariffs.de import DE_TARIFF_FORMULAS

        self.formulas = DE_TARIFF_FORMULAS

    def _apply(self, provider: str, p: float = 0.05) -> float:
        return self.formulas[provider].apply(p, _DATE_2024)

    def test_wholesale_passthrough(self):
        assert self._apply("wholesale") == pytest.approx(0.05, abs=1e-9)

    def test_tibber_higher_than_wholesale(self):
        assert self._apply("tibber") > 0.05

    def test_awattar_higher_than_wholesale(self):
        assert self._apply("awattar") > 0.05

    def test_ostrom_higher_than_wholesale(self):
        assert self._apply("ostrom") > 0.05

    def test_tibber_slightly_above_awattar(self):
        """Tibber adds 0.99 ct markup on top of aWATTar's base."""
        assert self._apply("tibber") > self._apply("awattar")

    def test_all_providers_registered(self):
        for p in ["wholesale", "tibber", "awattar", "ostrom"]:
            assert p in self.formulas


class TestWholesaleTariff:
    """Universal wholesale formula."""

    def test_passthrough(self):
        from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

        result = WHOLESALE_FORMULA.apply(0.123, _DATE_2024)
        assert result == pytest.approx(0.123)

    def test_zero(self):
        from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

        assert WHOLESALE_FORMULA.apply(0.0, _DATE_2024) == 0.0

    def test_negative(self):
        from runeflow.zones.tariffs.wholesale import WHOLESALE_FORMULA

        assert WHOLESALE_FORMULA.apply(-0.05, _DATE_2024) == pytest.approx(-0.05)


class TestGetTariffFormula:
    """Tests for the get_tariff_formula registry lookup."""

    def test_nl_vattenfall(self):
        from runeflow.zones.tariffs import get_tariff_formula

        formula = get_tariff_formula("NL", "vattenfall")
        assert formula.provider_id == "vattenfall"

    def test_de_lu_tibber(self):
        from runeflow.zones.tariffs import get_tariff_formula

        formula = get_tariff_formula("DE_LU", "tibber")
        assert formula.provider_id == "tibber"

    def test_unknown_zone_returns_wholesale(self):
        from runeflow.zones.tariffs import get_tariff_formula

        formula = get_tariff_formula("XX", "vattenfall")
        assert formula.provider_id == "wholesale"

    def test_unknown_provider_returns_wholesale(self):
        from runeflow.zones.tariffs import get_tariff_formula

        formula = get_tariff_formula("NL", "no_such_provider")
        assert formula.provider_id == "wholesale"

    def test_wholesale_provider_explicit(self):
        from runeflow.zones.tariffs import get_tariff_formula

        formula = get_tariff_formula("NL", "wholesale")
        assert formula.provider_id == "wholesale"

    def test_lowercase_zone_normalised(self):
        from runeflow.zones.tariffs import get_tariff_formula

        formula = get_tariff_formula("nl", "tibber")
        assert formula.provider_id == "tibber"

    def test_de_zone_alias(self):
        """'DE' and 'DE_LU' share the same formula registry."""
        from runeflow.zones.tariffs import get_tariff_formula

        f_de = get_tariff_formula("DE", "awattar")
        f_de_lu = get_tariff_formula("DE_LU", "awattar")
        assert f_de.provider_id == f_de_lu.provider_id

    def test_unknown_zone_with_wholesale_provider(self):
        """Unknown zone + 'wholesale' provider hits the explicit wholesale branch."""
        from runeflow.zones.tariffs import WHOLESALE_FORMULA, get_tariff_formula

        # "XX" not in _ALL → zone_formulas={} → "wholesale" not in {}
        # → if provider=="wholesale" → True
        formula = get_tariff_formula("XX", "wholesale")
        assert formula is WHOLESALE_FORMULA


class TestZoneRegistryClear:
    def test_clear_removes_all_zones_and_restores(self):
        """ZoneRegistry.clear() covers registry.py line 47."""
        from runeflow.zones.registry import ZoneRegistry

        # Ensure zones are loaded first
        existing = dict(ZoneRegistry._zones)
        assert len(existing) > 0  # sanity: zones are registered

        ZoneRegistry.clear()
        assert len(ZoneRegistry._zones) == 0

        # Restore so other tests are not affected
        ZoneRegistry._zones.update(existing)
        assert "NL" in ZoneRegistry._zones


# ---------------------------------------------------------------------------
# All ENTSO-E zones — shared parametrized suite
# ---------------------------------------------------------------------------

ALL_ZONE_CODES = [
    # Core
    "NL",
    "DE_LU",
    # Western / Central Europe
    "FR",
    "BE",
    "AT",
    "CH",
    # Iberia
    "ES",
    "PT",
    # Italy (regional bidding zones)
    "IT_NORD",
    "IT_CNOR",
    "IT_CSUD",
    "IT_SUD",
    "IT_SICI",
    "IT_SARD",
    # Central-Eastern Europe
    "PL",
    "CZ",
    "SK",
    "HU",
    "RO",
    "BG",
    "GR",
    # South-Eastern Europe
    "SI",
    "HR",
    "RS",
    "BA",
    "ME",
    "MK",
    "XK",
    # Nordic
    "DK_1",
    "DK_2",
    "FI",
    "NO_1",
    "NO_2",
    "NO_3",
    "NO_4",
    "NO_5",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
    # Baltic states
    "EE",
    "LV",
    "LT",
    # British Isles
    "GB",
    "IE",
    # Island / isolated systems
    "MT",
    "CY",
]


class TestZoneRegistryAllZones:
    def test_all_zone_codes_registered(self):
        zones = ZoneRegistry.list_zones()
        missing = [z for z in ALL_ZONE_CODES if z not in zones]
        assert missing == [], f"Zones not registered: {missing}"

    def test_list_zones_contains_all(self):
        zones = set(ZoneRegistry.list_zones())
        assert set(ALL_ZONE_CODES) == zones

    def test_list_zones_sorted(self):
        zones = ZoneRegistry.list_zones()
        assert zones == sorted(zones)


@pytest.mark.parametrize("zone_code", ALL_ZONE_CODES)
class TestAllZonesCommon:
    """Mandatory invariants that every zone config must satisfy."""

    def test_zone_code_matches(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.zone == zone_code

    def test_solar_capacity_positive(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.installed_solar_capacity_mw > 0

    def test_wind_capacity_non_negative(self, zone_code):
        """Wind capacity must be non-negative; island zones (MT, CY) may have zero."""
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.installed_wind_capacity_mw >= 0

    def test_typical_load_positive(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.typical_load_mw > 0

    def test_feature_groups_contain_temporal(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert "temporal" in cfg.feature_groups

    def test_feature_groups_all_valid(self, zone_code):
        from runeflow.features.registry import FEATURE_REGISTRY

        cfg = ZoneRegistry.get(zone_code)
        unknown = [g for g in cfg.feature_groups if g not in FEATURE_REGISTRY]
        assert unknown == [], f"{zone_code}: unknown feature groups {unknown}"

    def test_models_include_xgboost_quantile(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert "xgboost_quantile" in cfg.models

    def test_ensemble_strategy(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.ensemble_strategy == "condition_gated"

    def test_has_energyzero_only_nl(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.has_energyzero == (zone_code == "NL")

    def test_has_ned_only_nl(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.has_ned == (zone_code == "NL")

    def test_tariff_formulas_non_empty(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert len(cfg.tariff_formulas) > 0

    def test_tariff_formulas_contains_wholesale(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert "wholesale" in cfg.tariff_formulas

    def test_weather_locations_present(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert len(cfg.weather_locations) >= 1

    def test_primary_weather_location_in_europe(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        loc = cfg.primary_weather_location
        # All ENTSO-E zones lie between 27°N–72°N and 25°W–45°E
        assert 27 < loc.lat < 72, f"{zone_code}: lat={loc.lat} outside Europe"
        assert -25 < loc.lon < 45, f"{zone_code}: lon={loc.lon} outside Europe"

    def test_historical_years_at_least_three(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert len(cfg.historical_years) >= 3

    def test_min_training_years_positive(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.min_training_years >= 1


# ---------------------------------------------------------------------------
# Notable-zone specifics
# ---------------------------------------------------------------------------


class TestFRZoneConfig:
    @pytest.fixture(autouse=True)
    def cfg(self):
        self.cfg = ZoneRegistry.get("FR")

    def test_name(self):
        assert "France" in self.cfg.name

    def test_timezone(self):
        assert self.cfg.timezone == "Europe/Paris"

    def test_large_nuclear_reflected_in_load(self):
        """France has the largest nuclear fleet in the EU — very high typical load."""
        assert self.cfg.typical_load_mw > 50_000

    def test_has_cross_border_feature(self):
        assert "cross_border" in self.cfg.feature_groups

    def test_neighbors_non_empty(self):
        assert len(self.cfg.neighbors) > 0


class TestESZoneConfig:
    @pytest.fixture(autouse=True)
    def cfg(self):
        self.cfg = ZoneRegistry.get("ES")

    def test_name(self):
        assert "Spain" in self.cfg.name or "ES" in self.cfg.zone

    def test_timezone(self):
        assert self.cfg.timezone == "Europe/Madrid"

    def test_large_solar_capacity(self):
        """Spain has substantial solar capacity (20+ GW as of 2024)."""
        assert self.cfg.installed_solar_capacity_mw > 15_000

    def test_solar_feature(self):
        assert "solar_power" in self.cfg.feature_groups


class TestGBZoneConfig:
    @pytest.fixture(autouse=True)
    def cfg(self):
        self.cfg = ZoneRegistry.get("GB")

    def test_name(self):
        assert "Britain" in self.cfg.name or "GB" in self.cfg.zone

    def test_timezone(self):
        assert self.cfg.timezone == "Europe/London"

    def test_large_offshore_wind(self):
        """GB leads Europe in offshore wind — >20 GW."""
        assert self.cfg.installed_wind_capacity_mw > 20_000

    def test_wind_feature(self):
        assert "wind" in self.cfg.feature_groups


class TestNordicHydroZones:
    """Norway and Sweden have hydro-dominated markets — precipitation matters."""

    @pytest.mark.parametrize("zone_code", ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"])
    def test_precipitation_feature_norway(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert "precipitation" in cfg.feature_groups, (
            f"{zone_code} should have 'precipitation' (hydro-dominated)"
        )

    @pytest.mark.parametrize("zone_code", ["SE_1", "SE_2", "SE_3", "SE_4"])
    def test_precipitation_feature_sweden(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert "precipitation" in cfg.feature_groups, (
            f"{zone_code} should have 'precipitation' (significant hydro)"
        )

    def test_no_1_timezone(self):
        cfg = ZoneRegistry.get("NO_1")
        assert cfg.timezone == "Europe/Oslo"


class TestItalianZones:
    """Italy has regional bidding zones — all should be independently registered."""

    IT_ZONES = ["IT_NORD", "IT_CNOR", "IT_CSUD", "IT_SUD", "IT_SICI", "IT_SARD"]

    def test_all_it_zones_registered(self):
        zones = ZoneRegistry.list_zones()
        for code in self.IT_ZONES:
            assert code in zones

    @pytest.mark.parametrize("zone_code", IT_ZONES)
    def test_timezone_is_rome(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert cfg.timezone == "Europe/Rome"

    def test_nord_has_largest_capacity(self):
        """IT_NORD (industrial north) should have the highest typical load."""
        nord = ZoneRegistry.get("IT_NORD")
        sud = ZoneRegistry.get("IT_SUD")
        assert nord.typical_load_mw > sud.typical_load_mw


class TestBalkanZones:
    """Small, data-sparse Balkan zones should still satisfy all common invariants."""

    @pytest.mark.parametrize("zone_code", ["RS", "BA", "ME", "MK", "XK"])
    def test_tariff_is_wholesale(self, zone_code):
        cfg = ZoneRegistry.get(zone_code)
        assert "wholesale" in cfg.tariff_formulas
        assert cfg.has_energyzero is False
        assert cfg.has_ned is False


class TestWholesaleTariffNewZones:
    """Wholesale passthrough should work for every new zone's formula."""

    @pytest.mark.parametrize("zone_code", [z for z in ALL_ZONE_CODES if z not in ("NL", "DE_LU")])
    def test_wholesale_passthrough(self, zone_code):
        import datetime as dt

        cfg = ZoneRegistry.get(zone_code)
        formula = cfg.tariff_formulas["wholesale"]
        p = 0.085
        result = formula.apply(p, dt.date(2025, 6, 1))
        assert result == pytest.approx(p, abs=1e-9)

    @pytest.mark.parametrize("zone_code", [z for z in ALL_ZONE_CODES if z not in ("NL", "DE_LU")])
    def test_wholesale_negative_price_passthrough(self, zone_code):
        """Negative prices (grid congestion) must pass through unchanged."""
        import datetime as dt

        cfg = ZoneRegistry.get(zone_code)
        formula = cfg.tariff_formulas["wholesale"]
        result = formula.apply(-0.05, dt.date(2025, 6, 1))
        assert result == pytest.approx(-0.05, abs=1e-9)
