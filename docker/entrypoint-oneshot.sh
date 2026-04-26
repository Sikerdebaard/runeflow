#!/bin/bash
# runeflow One-Shot Pipeline Entrypoint
#
# Runs the full pipeline exactly once (no cron, no daemon) and exits.
# Intended for dev/test runs with --entrypoint, or CI pipelines.
#
# Steps (per zone): update-data → train → warmup-cache → inference →
#        export-tariffs → plot-uncertainty
# Final step (all zones): build-site
#
# Required env: ENTSOE
# Optional env: ZONES (NL), ZONE (deprecated single-zone alias for ZONES),
#               PRICE_PROVIDER (wholesale),
#               OUTPUT_DIR (/outputs),
#               SITE_OUTPUT_DIR (/outputs/site), LOG_LEVEL (INFO)

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required env
# ---------------------------------------------------------------------------
if [ -z "${ENTSOE:-}" ]; then
    echo "ERROR: ENTSOE environment variable (API key) is required"
    echo "Get your API key from: https://transparency.entsoe.eu/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ZONES="${ZONES:-${ZONE:-ALL}}"
# Expand the special sentinel ALL to the full list of enabled zones from the registry
if [ "$ZONES" = "ALL" ]; then
    ZONES=$(python3 -c "
from runeflow.zones.registry import ZoneRegistry
print(','.join(ZoneRegistry.list_zones()))
")
fi
PRICE_PROVIDER="${PRICE_PROVIDER:-wholesale}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"
SITE_OUTPUT_DIR="${SITE_OUTPUT_DIR:-/outputs/site}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

export ENTSOE ZONES PRICE_PROVIDER OUTPUT_DIR SITE_OUTPUT_DIR LOG_LEVEL
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/app/.cache}"
export HOME=/root

IFS=',' read -ra ZONE_ARRAY <<< "$ZONES"

echo "==================================="
echo " runeflow One-Shot Pipeline"
echo "==================================="
echo "  Zones:          $ZONES"
echo "  Price Provider: $PRICE_PROVIDER"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Site Dir:       $SITE_OUTPUT_DIR"
echo "==================================="

mkdir -p "$OUTPUT_DIR" "$SITE_OUTPUT_DIR" /app/.cache/runeflow /var/log/runeflow

cd /app

FAILED_ZONES=()

for ZONE in "${ZONE_ARRAY[@]}"; do
    ZONE=$(echo "$ZONE" | xargs)
    OUTPUT_FILE="${OUTPUT_DIR}/tariffs_${ZONE}.json"
    CHART_FILE="${OUTPUT_DIR}/runeflow_uncertainty_${ZONE}.png"

    ZONE_LOGFILE="/var/log/runeflow/zone_${ZONE}.log"

    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo " Zone: $ZONE"
    echo "══════════════════════════════════════════════════════════════════"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ── Zone: $ZONE — start ──" >> "$ZONE_LOGFILE"

    # -----------------------------------------------------------------------
    # 1. Fetch / update historical data
    # -----------------------------------------------------------------------
    echo ""
    echo "── Step 1/6: Updating data ──────────────────────────────────────────"
    runeflow update-data --zone "$ZONE" || {
        echo "WARNING: Data update had issues, continuing..."
    }

    # -----------------------------------------------------------------------
    # 2. Train model
    # -----------------------------------------------------------------------
    echo ""
    echo "── Step 2/6: Training model ─────────────────────────────────────────"
    runeflow train --zone "$ZONE" || {
        echo "ERROR: Training failed for zone $ZONE — skipping"
        FAILED_ZONES+=("$ZONE")
        continue
    }

    # -----------------------------------------------------------------------
    # 3. Warm up feature cache
    # -----------------------------------------------------------------------
    echo ""
    echo "── Step 3/6: Warming up cache ───────────────────────────────────────"
    runeflow warmup-cache --zone "$ZONE" || {
        echo "WARNING: Warmup had issues, continuing..."
    }

    # -----------------------------------------------------------------------
    # 4. Run inference (produce forecast)
    # -----------------------------------------------------------------------
    echo ""
    echo "── Step 4/6: Running inference ──────────────────────────────────────"
    runeflow inference --zone "$ZONE" || {
        echo "ERROR: Inference failed for zone $ZONE — skipping"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Inference failed" >> "$ZONE_LOGFILE"
        FAILED_ZONES+=("$ZONE")
        continue
    }

    # -----------------------------------------------------------------------
    # 5. Export tariffs JSON
    # -----------------------------------------------------------------------
    echo ""
    echo "── Step 5/6: Exporting tariffs ──────────────────────────────────────"
    runeflow export-tariffs \
        --zone "$ZONE" \
        --provider "$PRICE_PROVIDER" \
        --output "$OUTPUT_FILE" || {
        echo "ERROR: Export failed for zone $ZONE (non-critical, continuing)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Export failed (non-critical)" >> "$ZONE_LOGFILE"
    }

    # -----------------------------------------------------------------------
    # 6. Generate uncertainty chart
    # -----------------------------------------------------------------------
    echo ""
    echo "── Step 6/6: Generating chart ───────────────────────────────────────"
    runeflow plot-uncertainty \
        --zone "$ZONE" \
        --output "$CHART_FILE" || {
        echo "WARNING: Chart generation failed (non-critical)"
    }

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Zone $ZONE complete." >> "$ZONE_LOGFILE"
done

# ---------------------------------------------------------------------------
# 7. Build static site (all zones in one call)
# ---------------------------------------------------------------------------
echo ""
echo "── Step 7/7: Building static site ──────────────────────────────────"
runeflow build-site \
    --zones "$ZONES" \
    --output "$SITE_OUTPUT_DIR"

echo ""
echo "==================================="
echo " Pipeline complete"
if [ ${#FAILED_ZONES[@]} -gt 0 ]; then
    echo "  ⚠ Failed zones: ${FAILED_ZONES[*]}"
fi
echo "  output dir : $OUTPUT_DIR"
echo "  site       : $SITE_OUTPUT_DIR"
echo "==================================="
