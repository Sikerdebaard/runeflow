# Disabled Zones

Zones listed here are registered in `ZoneRegistry` (so `ZoneRegistry.get()` still works and
their configs are validated by tests), but they are **excluded from `ZoneRegistry.list_zones()`**
and therefore never processed by the inference or export pipelines.

The `disabled_reason` field on each `ZoneConfig` is the programmatic record; this file provides
additional context and a suggested path to re-enable each zone.

---

## BA — Bosnia-Herzegovina

**Reason:** ENTSO-E does not publish day-ahead electricity prices for the Bosnian bidding zone on
the Transparency Platform. Confirmed with `NoMatchingDataError` during testing (2025).

**To re-enable:**
1. Confirm ENTSO-E starts publishing BA day-ahead prices, **or** identify an alternative public API
   (e.g., EPEX SPOT, OTE, HOPS data portal).
2. Add/update the price adapter in `src/runeflow/adapters/price/`.
3. Remove `disabled_reason` from `src/runeflow/zones/definitions/balkans.py`.
4. Update `ACTIVE_ZONE_CODES` / `DISABLED_ZONE_CODES` in `tests/unit/test_zones.py`.

---

## CY — Cyprus

**Reason:** Cyprus operates an isolated island grid with no AC interconnection to the continental
European network. No day-ahead prices are published on the ENTSO-E Transparency Platform.
Confirmed with `NoMatchingDataError` during testing (2025).

**To re-enable:**
1. Identify a Cypriot market operator API (e.g., TSOC — Transmission System Operator of Cyprus).
2. Implement a dedicated price adapter in `src/runeflow/adapters/price/`.
3. Remove `disabled_reason` from `src/runeflow/zones/definitions/mt_cy.py`.
4. Update `ACTIVE_ZONE_CODES` / `DISABLED_ZONE_CODES` in `tests/unit/test_zones.py`.

---

## GB — Great Britain

**Reason:** Great Britain left ENTSO-E day-ahead market coupling after Brexit (effective 2021).
Electricity prices are no longer published on the ENTSO-E Transparency Platform for any GB area
code (`10YGB----------A` and variants). Confirmed with `NoMatchingDataError` during testing (2025).

**To re-enable:**
1. Implement a dedicated adapter using the **BMRS (Balancing Mechanism Reporting Service)**
   or the **Elexon Data Portal API** (both are public and well-documented).
2. Add the adapter in `src/runeflow/adapters/price/` (e.g., `bmrs.py`).
3. Remove `disabled_reason` from `src/runeflow/zones/definitions/gb_ie.py`.
4. Update `ACTIVE_ZONE_CODES` / `DISABLED_ZONE_CODES` in `tests/unit/test_zones.py`.

**Resources:**
- Elexon Data Portal: https://developer.data.elexon.co.uk/
- BMRS API: https://www.bmreports.com/bmrs/?q=help/api-documentation

---

## IE — Ireland

**Reason:** Ireland's electricity market uses the Single Electricity Market (SEM), operated by
SEMO/SEMOPX. The IE_SEM area code exists in entsoe-py, but the ENTSO-E Transparency Platform
does not publish historical day-ahead prices for this area. `NoMatchingDataError` is returned
for all years 2020-2026 (confirmed during testing, April 2026).

Note: the `IE` → `IE_SEM` alias is registered in `entsoe.py` and works for current-day
queries, but historical data is unavailable.

**To re-enable:**
1. Implement a dedicated adapter for the **SEMOPX (Single Electricity Market Operator PX)**
   or **SEMO** data portal — both publish historical day-ahead prices publicly.
2. Add the adapter in `src/runeflow/adapters/price/` (e.g., `semopx.py`).
3. Register it in the fallback chain for IE in `src/runeflow/adapters/price/fallback.py`.
4. Remove `disabled_reason` from `src/runeflow/zones/definitions/gb_ie.py`.
5. Update `ACTIVE_ZONE_CODES` / `DISABLED_ZONE_CODES` in `tests/unit/test_zones.py`.

**Resources:**
- SEMOPX Public Reports: https://www.semopx.com/market-data/

---

## MT — Malta

**Reason:** Malta is an island market connected only via HVDC interconnector to Sicily (IT_SICI).
No day-ahead prices are published on the ENTSO-E Transparency Platform.
Confirmed with `NoMatchingDataError` during testing (2025).

**To re-enable:**
1. Identify a Maltese market operator API (e.g., Enemalta or the Malta Resources Authority).
2. Implement a dedicated price adapter in `src/runeflow/adapters/price/`.
3. Remove `disabled_reason` from `src/runeflow/zones/definitions/mt_cy.py`.
4. Update `ACTIVE_ZONE_CODES` / `DISABLED_ZONE_CODES` in `tests/unit/test_zones.py`.

---

## XK — Kosovo

**Reason:** Kosovo does not publish day-ahead electricity prices on the ENTSO-E Transparency
Platform. Confirmed with `NoMatchingDataError` during testing (2025).

**To re-enable:**
1. Confirm ENTSO-E starts publishing XK day-ahead prices, **or** identify an alternative source
   (e.g., KOSTT — Kosovo Transmission System and Market Operator).
2. Add/update the price adapter in `src/runeflow/adapters/price/`.
3. Remove `disabled_reason` from `src/runeflow/zones/definitions/balkans.py`.
4. Update `ACTIVE_ZONE_CODES` / `DISABLED_ZONE_CODES` in `tests/unit/test_zones.py`.

---

## Programmatic access

```python
from runeflow.zones.registry import ZoneRegistry

# List all active zones (BA, CY, GB, MT, XK excluded)
active = ZoneRegistry.list_zones()

# List disabled zones with reasons
for code, reason in ZoneRegistry.list_disabled_zones():
    print(f"{code}: {reason}")

# Config is still accessible for disabled zones
gb_cfg = ZoneRegistry.get("GB")
print(gb_cfg.disabled_reason)
```
