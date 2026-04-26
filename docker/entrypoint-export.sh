#!/bin/bash
set -e

# runeflow Tariff Exporter Entrypoint
# Runs inference on a schedule and writes tariff JSON.
#
# Scheduling (Amsterdam / Europe/Amsterdam time):
#   - 08:05  Full run: update-data → train → inference → export → site
#   - 20:05  Full run: update-data → train → inference → export → site
#   - 12:00–16:00 every 15 min: price watcher → if new day-ahead prices
#             detected, run update-data → inference → export → site (no retrain)

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
if [ -z "$ENTSOE" ]; then
    echo "ERROR: ENTSOE environment variable (API key) is required"
    echo "Get your API key from: https://transparency.entsoe.eu/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Optional environment variables with defaults
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

echo "==================================="
echo "runeflow Tariff Exporter"
echo "==================================="
echo "Zones:          $ZONES"
echo "Price Provider: $PRICE_PROVIDER"
echo "Output Dir:     $OUTPUT_DIR"
echo "Site Dir:       $SITE_OUTPUT_DIR"
echo "Timezone:       $(cat /etc/timezone)"
echo "==================================="

# ---------------------------------------------------------------------------
# Shared environment file (used by cron-launched scripts)
# ---------------------------------------------------------------------------
cat > /app/.env.docker << EOF
PATH=/usr/local/bin:/usr/bin:/bin
ENTSOE=$ENTSOE
ZONES=$ZONES
PRICE_PROVIDER=$PRICE_PROVIDER
OUTPUT_DIR=$OUTPUT_DIR
SITE_OUTPUT_DIR=$SITE_OUTPUT_DIR
LOG_LEVEL=$LOG_LEVEL
HOME=/root
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/app/.cache}
EOF

export ENTSOE ZONES PRICE_PROVIDER OUTPUT_DIR SITE_OUTPUT_DIR LOG_LEVEL
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/app/.cache}"

# ---------------------------------------------------------------------------
# Full run: update-data → train → warmup-cache → inference → export + plot
# ---------------------------------------------------------------------------
cat > /app/run-inference.sh << 'SCRIPT'
#!/bin/bash
set -e

source /app/.env.docker
export PATH ENTSOE XDG_CACHE_HOME HOME

LOGFILE="/var/log/runeflow/inference.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
cd /app

IFS=',' read -ra ZONE_ARRAY <<< "$ZONES"

echo "[$TIMESTAMP] ── Full run (update + train + inference) ── zones=${ZONES}" >> "$LOGFILE"

FAILED_ZONES=()

for ZONE in "${ZONE_ARRAY[@]}"; do
    ZONE=$(echo "$ZONE" | xargs)  # trim whitespace
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    ZONE_LOGFILE="/var/log/runeflow/zone_${ZONE}.log"
    echo "[$TIMESTAMP] ── Zone: $ZONE ──" | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"

    echo "[$TIMESTAMP] [$ZONE] Updating data..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow update-data --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] ERROR: [$ZONE] Data update failed — skipping zone." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
        FAILED_ZONES+=("$ZONE")
        continue
    }

    echo "[$TIMESTAMP] [$ZONE] Training model..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow train --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] ERROR: [$ZONE] Training failed" | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
        FAILED_ZONES+=("$ZONE")
        continue
    }

    echo "[$TIMESTAMP] [$ZONE] Warming up cache..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow warmup-cache --zone "$ZONE" >> "$LOGFILE" 2>&1 || \
        echo "[$TIMESTAMP] WARNING: [$ZONE] Warmup had issues, continuing..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"

    echo "[$TIMESTAMP] [$ZONE] Running inference..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow inference --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] ERROR: [$ZONE] Inference failed" | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
        FAILED_ZONES+=("$ZONE")
        continue
    }

    OUTPUT_FILE="${OUTPUT_DIR}/tariffs_${ZONE}.json"
    CHART_FILE="${OUTPUT_DIR}/runeflow_uncertainty_${ZONE}.png"

    echo "[$TIMESTAMP] [$ZONE] Exporting tariffs..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow export-tariffs --zone "$ZONE" \
        --provider "$PRICE_PROVIDER" \
        --output "$OUTPUT_FILE" >> "$LOGFILE" 2>&1 || {
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] ERROR: [$ZONE] Export failed (non-critical)" | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    }

    echo "[$TIMESTAMP] [$ZONE] Generating uncertainty chart..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow plot-uncertainty --zone "$ZONE" \
        --output "$CHART_FILE" >> "$LOGFILE" 2>&1 || \
        echo "[$TIMESTAMP] WARNING: [$ZONE] Chart generation failed (non-critical)" | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"

    # Stamp price-watcher state file if tomorrow's prices are already present
    STATEFILE="/app/.cache/runeflow/entsoe_last_processed_${ZONE}.txt"
    TOMORROW_DATE=$(date -d "tomorrow" '+%Y-%m-%d')
    PRICES_PRESENT=$(python3 - "$ZONE" << 'PYTHON'
import os, sys
zone = sys.argv[1]
os.environ.setdefault("XDG_CACHE_HOME", "/app/.cache")
try:
    from runeflow.binder import configure_injector
    configure_injector(zone)
    import inject
    from runeflow.ports.store import DataStore
    from datetime import date, timedelta
    store = inject.instance(DataStore)
    tomorrow = date.today() + timedelta(days=1)
    series = store.load_prices(zone)
    if series is not None:
        df = series.to_dataframe()
        print("yes" if len(df[df.index.date == tomorrow]) >= 20 else "no")
    else:
        print("no")
except Exception:
    print("no")
PYTHON
    )
    if [ "$PRICES_PRESENT" = "yes" ]; then
        mkdir -p "$(dirname "$STATEFILE")"
        echo "$TOMORROW_DATE" > "$STATEFILE"
        echo "[$TIMESTAMP] [$ZONE] Day-ahead prices for $TOMORROW_DATE already present — price watcher suppressed." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    fi

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] [$ZONE] ✓ Zone complete." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
done

# Build static dashboard for ALL zones (single call, renders cross-zone index)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Building static dashboard for all zones..." >> "$LOGFILE"
runeflow build-site --zones "$ZONES" --output "$SITE_OUTPUT_DIR" >> "$LOGFILE" 2>&1 || \
    echo "[$TIMESTAMP] WARNING: build-site failed (non-critical)" >> "$LOGFILE"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
if [ ${#FAILED_ZONES[@]} -gt 0 ]; then
    echo "[$TIMESTAMP] ⚠ Full run complete with failures: ${FAILED_ZONES[*]}" >> "$LOGFILE"
else
    echo "[$TIMESTAMP] ✓ Full run complete — all zones OK" >> "$LOGFILE"
fi
echo "" >> "$LOGFILE"
SCRIPT
chmod +x /app/run-inference.sh

# ---------------------------------------------------------------------------
# Inference-only run: warmup-cache → inference → export + plot (no training)
# Used for the mid-day day-ahead price update (~13:00 CET)
# ---------------------------------------------------------------------------
cat > /app/run-inference-only.sh << 'SCRIPT'
#!/bin/bash
set -e

source /app/.env.docker
export PATH ENTSOE XDG_CACHE_HOME HOME

LOGFILE="/var/log/runeflow/inference.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
cd /app

IFS=',' read -ra ZONE_ARRAY <<< "$ZONES"

echo "[$TIMESTAMP] ── Price-update inference run (no retrain) ── zones=${ZONES}" >> "$LOGFILE"

FAILED_ZONES=()

for ZONE in "${ZONE_ARRAY[@]}"; do
    ZONE=$(echo "$ZONE" | xargs)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    ZONE_LOGFILE="/var/log/runeflow/zone_${ZONE}.log"
    echo "[$TIMESTAMP] ── Zone: $ZONE ──" | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"

    echo "[$TIMESTAMP] [$ZONE] Updating data..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow update-data --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] ERROR: [$ZONE] Data update failed — skipping." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
        FAILED_ZONES+=("$ZONE")
        continue
    }

    echo "[$TIMESTAMP] [$ZONE] Warming up cache..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow warmup-cache --zone "$ZONE" >> "$LOGFILE" 2>&1 || \
        echo "[$TIMESTAMP] WARNING: [$ZONE] Warmup had issues, continuing..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"

    echo "[$TIMESTAMP] [$ZONE] Running inference..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow inference --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] ERROR: [$ZONE] Inference failed — skipping." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
        FAILED_ZONES+=("$ZONE")
        continue
    }

    OUTPUT_FILE="${OUTPUT_DIR}/tariffs_${ZONE}.json"
    CHART_FILE="${OUTPUT_DIR}/runeflow_uncertainty_${ZONE}.png"

    echo "[$TIMESTAMP] [$ZONE] Exporting tariffs..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow export-tariffs --zone "$ZONE" \
        --provider "$PRICE_PROVIDER" \
        --output "$OUTPUT_FILE" >> "$LOGFILE" 2>&1 || {
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] ERROR: [$ZONE] Export failed (non-critical)" | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    }

    echo "[$TIMESTAMP] [$ZONE] Generating uncertainty chart..." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
    runeflow plot-uncertainty --zone "$ZONE" \
        --output "$CHART_FILE" >> "$LOGFILE" 2>&1 || \
        echo "[$TIMESTAMP] WARNING: [$ZONE] Chart generation failed (non-critical)" | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] [$ZONE] ✓ Zone complete." | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
done

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Building static dashboard for all zones..." >> "$LOGFILE"
runeflow build-site --zones "$ZONES" --output "$SITE_OUTPUT_DIR" >> "$LOGFILE" 2>&1 || \
    echo "[$TIMESTAMP] WARNING: build-site failed (non-critical)" >> "$LOGFILE"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
if [ ${#FAILED_ZONES[@]} -gt 0 ]; then
    echo "[$TIMESTAMP] ⚠ Inference-only run complete with failures: ${FAILED_ZONES[*]}" >> "$LOGFILE"
else
    echo "[$TIMESTAMP] ✓ Inference-only run complete — all zones OK" >> "$LOGFILE"
fi
echo "" >> "$LOGFILE"
SCRIPT
chmod +x /app/run-inference-only.sh

# ---------------------------------------------------------------------------
# ENTSO-E day-ahead price watcher
# Checks once whether tomorrow's prices are out; if so, runs inference-only.
# Called every 15 min via cron (12:00–16:00). Uses a state file to avoid
# re-running once tomorrow's batch has already been processed.
# ---------------------------------------------------------------------------
cat > /app/check-prices.sh << 'SCRIPT'
#!/bin/bash
set -e

source /app/.env.docker
export PATH ENTSOE XDG_CACHE_HOME HOME

LOGFILE="/var/log/runeflow/entsoe-check.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Prevent concurrent runs
exec 200>/tmp/runeflow-check.lock
flock -n 200 || exit 0

IFS=',' read -ra ZONE_ARRAY <<< "$ZONES"
TOMORROW=$(date -d "tomorrow" '+%Y-%m-%d')
ANY_NEW=0

for ZONE in "${ZONE_ARRAY[@]}"; do
    ZONE=$(echo "$ZONE" | xargs)
    STATEFILE="/app/.cache/runeflow/entsoe_last_processed_${ZONE}.txt"

    if [ -f "$STATEFILE" ] && [ "$(cat "$STATEFILE")" = "$TOMORROW" ]; then
        continue  # Already processed this zone today
    fi

    PRICES_AVAILABLE=$(python3 - "$ZONE" << 'PYTHON'
import os, sys
zone = sys.argv[1]
os.environ.setdefault("XDG_CACHE_HOME", "/app/.cache")
os.environ.setdefault("ENTSOE", os.environ["ENTSOE"])
try:
    from runeflow.binder import configure_injector
    configure_injector(zone)
    import inject
    from runeflow.ports.store import DataStore
    from datetime import date, timedelta
    store = inject.instance(DataStore)
    tomorrow = date.today() + timedelta(days=1)
    series = store.load_prices(zone)
    if series is not None:
        df = series.to_dataframe()
        tomorrow_data = df[df.index.date == tomorrow]
        print("yes" if len(tomorrow_data) >= 20 else "no")
    else:
        print("no")
except Exception as e:
    print(f"check error: {e}", file=sys.stderr)
    print("no")
PYTHON
    )

    if [ "$PRICES_AVAILABLE" = "yes" ]; then
        echo "[$TIMESTAMP] [$ZONE] Day-ahead prices detected" >> "$LOGFILE"
        mkdir -p "$(dirname "$STATEFILE")"
        echo "$TOMORROW" > "$STATEFILE"
        ANY_NEW=1
    else
        echo "[$TIMESTAMP] [$ZONE] Prices not yet available" >> "$LOGFILE"
    fi
done

if [ "$ANY_NEW" = "1" ]; then
    echo "[$TIMESTAMP] New prices detected — running inference for all zones..." >> "$LOGFILE"
    /app/run-inference-only.sh
    echo "[$TIMESTAMP] ✓ Day-ahead update done" >> "$LOGFILE"
fi
SCRIPT
chmod +x /app/check-prices.sh

# ---------------------------------------------------------------------------
# Weather data refresh
# Runs every hour at :20. Re-attempts update-data for any zone whose
# historical weather is missing or incomplete (e.g. after an hourly API
# rate limit was hit during the main run).  Zones whose data is already
# current are skipped quickly by the gap-detection in UpdateDataService.
# ---------------------------------------------------------------------------
cat > /app/run-update-weather.sh << 'SCRIPT'
#!/bin/bash
set -e

source /app/.env.docker
export PATH ENTSOE XDG_CACHE_HOME HOME

LOGFILE="/var/log/runeflow/inference.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Prevent concurrent runs
exec 201>/tmp/runeflow-weather.lock
flock -n 201 || { echo "[$TIMESTAMP] Weather refresh already running, skipping." >> "$LOGFILE"; exit 0; }

echo "[$TIMESTAMP] ── Weather refresh run ──" >> "$LOGFILE"

IFS=',' read -ra ZONE_ARRAY <<< "$ZONES"
FAILED_ZONES=()

for ZONE in "${ZONE_ARRAY[@]}"; do
    ZONE=$(echo "$ZONE" | xargs)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    ZONE_LOGFILE="/var/log/runeflow/zone_${ZONE}.log"

    runeflow update-data --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] WARNING: [$ZONE] Weather refresh failed (will retry next hour)" \
            | tee -a "$ZONE_LOGFILE" >> "$LOGFILE"
        FAILED_ZONES+=("$ZONE")
    }
done

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
if [ ${#FAILED_ZONES[@]} -gt 0 ]; then
    echo "[$TIMESTAMP] ⚠ Weather refresh complete with failures: ${FAILED_ZONES[*]}" >> "$LOGFILE"
else
    echo "[$TIMESTAMP] ✓ Weather refresh complete" >> "$LOGFILE"
fi
SCRIPT
chmod +x /app/run-update-weather.sh

# ---------------------------------------------------------------------------
# Cron schedule
# ---------------------------------------------------------------------------
# Full run (data + train + inference) at 08:05, 14:30 and 20:05 Amsterdam time
cat > /etc/cron.d/runeflow << 'CRON'
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin

# Full run: update-data → train → warmup → inference → export → site
5 8  * * * root /app/run-inference.sh      >> /var/log/runeflow/cron.log 2>&1
30 14 * * * root /app/run-inference.sh      >> /var/log/runeflow/cron.log 2>&1
5 20 * * * root /app/run-inference.sh      >> /var/log/runeflow/cron.log 2>&1
# Day-ahead price watcher (12:00–16:00 Amsterdam, every 15 min)
# Triggers update-data → inference → export → site when new prices are detected
*/15 12-16 * * * root /app/check-prices.sh >> /var/log/runeflow/cron.log 2>&1
# Hourly weather refresh: retries update-data for zones with missing/incomplete
# historical weather (e.g. after an Open-Meteo hourly rate limit).
# Zones with up-to-date data are skipped automatically.
20 * * * * root /app/run-update-weather.sh >> /var/log/runeflow/cron.log 2>&1
CRON
chmod 0644 /etc/cron.d/runeflow

# ---------------------------------------------------------------------------
# Init log files and output directory
# ---------------------------------------------------------------------------
touch /var/log/runeflow/inference.log \
      /var/log/runeflow/entsoe-check.log \
      /var/log/runeflow/cron.log
mkdir -p "$OUTPUT_DIR" /app/.cache/runeflow

# ---------------------------------------------------------------------------
# Startup: run once immediately so the output file is available right away
# ---------------------------------------------------------------------------
echo "Running initial full pipeline on startup (update-data → train → inference)..."
echo "On a fresh server with an empty cache this will take several minutes."
/app/run-inference.sh || {
    echo ""
    echo "WARNING: Initial pipeline run failed."
    echo "This is normal on a completely fresh install where the data cache is empty."
    echo "The most common cause is a transient API error during 'update-data'."
    echo "The system will retry automatically at the next scheduled run (08:05, 14:30 and 20:05 AMS)."
    echo "Check /var/log/runeflow/inference.log for details."
}

echo ""
echo "Cron jobs installed:"
echo "  - Full run (update + train + infer) at 08:05, 14:30 and 20:05 Amsterdam"
echo "  - Day-ahead price watcher every 15 min (12:00–16:00)"
echo "  - Hourly weather refresh at :20 (retries zones with missing weather data)"
echo ""
echo "Zones: $ZONES"
echo "Logs: /var/log/runeflow/"
echo "Output: $OUTPUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Start cron (foreground) or exec arbitrary command
# ---------------------------------------------------------------------------
if [ "$1" = "cron" ]; then
    tail -F /var/log/runeflow/inference.log \
             /var/log/runeflow/cron.log &
    exec cron -f
else
    exec "$@"
fi
