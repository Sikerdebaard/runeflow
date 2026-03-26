#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.
#
# cloudflare-setup.sh — Configure Cloudflare rules for the runeflow dashboard
#
# Run this script ONCE after the Cloudflare Tunnel is working to:
#   1. Disable Bot Fight Mode + Browser Integrity Check on /**.json and /**.csv
#      so that external scripts/bots can freely download price data.
#
# Requirements:
#   - curl, jq
#   - A Cloudflare API token with Edit Zone permissions:
#       Zone > Zone Settings > Edit
#       Zone > Firewall Services > Edit
#
# Usage:
#   export CF_API_TOKEN="your-api-token"
#   export CF_ZONE_ID="your-zone-id"           # found in the zone Overview page
#   bash scripts/cloudflare-setup.sh
#
# To find your Zone ID:
#   Open dash.cloudflare.com → select your domain → right-hand sidebar shows Zone ID.
# To create an API token:
#   My Profile → API Tokens → Create Token → use "Edit zone" template.

set -euo pipefail

CF_API="https://api.cloudflare.com/client/v4"

# ── Validate env ──────────────────────────────────────────────────────────────
: "${CF_API_TOKEN:?CF_API_TOKEN must be set}"
: "${CF_ZONE_ID:?CF_ZONE_ID must be set}"

AUTH=(-H "Authorization: Bearer $CF_API_TOKEN" -H "Content-Type: application/json")

die() { echo "ERROR: $*" >&2; exit 1; }
ok()  { echo "✓ $*"; }

# ── Helper: check Cloudflare response ────────────────────────────────────────
cf_check() {
  local resp="$1" label="$2"
  local success
  success=$(echo "$resp" | jq -r '.success')
  if [[ "$success" != "true" ]]; then
    echo "FAILED: $label"
    echo "$resp" | jq '.errors' >&2
    exit 1
  fi
}

echo ""
echo "Runeflow — Cloudflare setup"
echo "Zone: $CF_ZONE_ID"
echo ""

# ── 1. Disable Browser Integrity Check globally ───────────────────────────────
# Browser Integrity Check rejects requests without browser-like headers.
# JSON/CSV consumers (evcc, scripts, pandas) don't send those headers.
echo "→ Disabling Browser Integrity Check for the zone…"
resp=$(curl -s -X PATCH "$CF_API/zones/$CF_ZONE_ID/settings/browser_check" \
  "${AUTH[@]}" \
  --data '{"value": "off"}')
cf_check "$resp" "browser_check"
ok "Browser Integrity Check disabled"

# ── 2. Configuration Rule — bypass security for data files ───────────────────
# Sets Security Level = essentially_off for all *.json and *.csv paths.
# This allows bots and scripts to GET the price files without CAPTCHA.
#
# Uses the Rulesets API (replaces the deprecated Page Rules API).

RULE_PAYLOAD=$(cat <<'JSON'
{
  "description": "Bypass security for runeflow data files (JSON / CSV)",
  "expression": "(http.request.uri.path matches \"\\.(json|csv)$\")",
  "action": "set_config",
  "action_parameters": {
    "security_level": "essentially_off"
  }
}
JSON
)

echo "→ Fetching existing Configuration Rules ruleset…"
resp=$(curl -s -X GET \
  "$CF_API/zones/$CF_ZONE_ID/rulesets/phases/http_config_settings/entrypoint" \
  "${AUTH[@]}")

ruleset_id=$(echo "$resp" | jq -r '.result.id // empty')

if [[ -z "$ruleset_id" ]]; then
  # No ruleset yet — create it from scratch with our one rule.
  echo "   No existing ruleset found, creating…"
  resp=$(curl -s -X PUT \
    "$CF_API/zones/$CF_ZONE_ID/rulesets/phases/http_config_settings/entrypoint" \
    "${AUTH[@]}" \
    --data "{
      \"name\": \"Zone-level configuration settings\",
      \"kind\": \"zone\",
      \"phase\": \"http_config_settings\",
      \"rules\": [$RULE_PAYLOAD]
    }")
  cf_check "$resp" "create_ruleset"
  ok "Configuration ruleset created with data-file bypass rule"
else
  # Ruleset exists — check if the rule is already there.
  existing=$(echo "$resp" | jq -r \
    '.result.rules[]? | select(.description == "Bypass security for runeflow data files (JSON / CSV)") | .id' \
    2>/dev/null || true)

  if [[ -n "$existing" ]]; then
    ok "Bypass rule already exists (id: $existing) — skipping"
  else
    echo "   Adding rule to existing ruleset ($ruleset_id)…"
    resp=$(curl -s -X POST \
      "$CF_API/zones/$CF_ZONE_ID/rulesets/$ruleset_id/rules" \
      "${AUTH[@]}" \
      --data "$RULE_PAYLOAD")
    cf_check "$resp" "add_rule"
    ok "Bypass rule added to existing ruleset"
  fi
fi

# ── 3. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "Done! Cloudflare is now configured:"
echo "  • Browser Integrity Check  : off (zone-wide)"
echo "  • Security Level           : essentially_off for *.json and *.csv"
echo ""
echo "Consumers such as evcc, pandas, and curl can now freely download"
echo "the tariff data without being challenged or blocked."
echo ""
echo "To verify, test with a plain user-agent:"
echo "  curl -A 'python-requests/2.32' https://YOUR_DOMAIN/NL/wholesale/tariff.json | head"
