#!/bin/bash
# runeflow Archive Cache Warmer
#
# Downloads the full historical weather archive for all configured zones,
# stored in quarterly chunks with NEVER_EXPIRE HTTP caching.  Already-cached
# quarters are skipped instantly, so this container is safe to restart at any
# time — it always resumes where it left off.
#
# Purpose: warm the Open-Meteo HTTP cache before the export container runs its
# first full pipeline, preventing repeated rate-limited archive downloads from
# stalling the daily inference cycle.
#
# No ENTSOE key is required.
#
# Environment variables (all optional):
#   ZONES              comma-separated zone codes, or ALL (default: ALL)
#   LOG_LEVEL          loguru level (default: INFO)
#   SLEEP_ON_FAILURE   seconds to wait after a partial failure (default: 3600)
#                      Typical cause: Open-Meteo hourly rate limit exhausted.
#   SLEEP_ON_SUCCESS   seconds to sleep after all zones are fully cached
#                      before rechecking (default: 86400 = 24 h).
#                      Catches new zone configs or extended historical ranges
#                      without requiring a container restart.
#
# Status page: http://localhost:7072

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ZONES="${ZONES:-${ZONE:-ALL}}"
if [ "$ZONES" = "ALL" ]; then
    ZONES=$(python3 -c "
from runeflow.zones.registry import ZoneRegistry
print(','.join(ZoneRegistry.list_zones()))
")
fi

LOG_LEVEL="${LOG_LEVEL:-INFO}"
SLEEP_ON_FAILURE="${SLEEP_ON_FAILURE:-3600}"    # 1 h — rate-limit recovery window
SLEEP_ON_SUCCESS="${SLEEP_ON_SUCCESS:-86400}"   # 24 h — daily re-check

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/app/.cache}"
export LOG_LEVEL ZONES

# ---------------------------------------------------------------------------
# Status HTTP server (port 7072)
# ---------------------------------------------------------------------------
STATUS_DIR="/tmp/runeflow-status"
mkdir -p "$STATUS_DIR"
cp /app/archive-status.html "$STATUS_DIR/index.html"

python3 -m http.server --bind 0.0.0.0 --directory "$STATUS_DIR" 7072 \
    > /dev/null 2>&1 &
HTTP_PID=$!

# Stop the HTTP server when this script exits.
trap 'kill "$HTTP_PID" 2>/dev/null; exit' TERM INT EXIT

echo "==========================================="
echo "runeflow Archive Cache Warmer"
echo "==========================================="
echo "Zones:            $ZONES"
echo "On failure sleep: ${SLEEP_ON_FAILURE}s"
echo "On success sleep: ${SLEEP_ON_SUCCESS}s"
echo "Cache dir:        $XDG_CACHE_HOME"
echo "Status page:      http://localhost:7072"
echo "==========================================="
echo ""

# ---------------------------------------------------------------------------
# Main loop — run until stopped
# ---------------------------------------------------------------------------
PASS=0
while true; do
    PASS=$((PASS + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Pass ${PASS} ==="

    runeflow prefetch-archive \
        --zones "$ZONES" \
        --status-file "$STATUS_DIR/status.json" \
        --pass "$PASS"
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All zones cached."
        echo "Sleeping ${SLEEP_ON_SUCCESS}s, then rechecking for new zones or year-range extensions…"
        sleep "$SLEEP_ON_SUCCESS"
    elif [ "$EXIT_CODE" -eq 2 ]; then
        # Exit code 2 = daily API quota exhausted.  Compute seconds until midnight
        # Amsterdam time (Open-Meteo quota resets at midnight UTC; Amsterdam is UTC+2
        # in summer and UTC+1 in winter — sleeping until 00:10 local is safe).
        NOW_UTC=$(date -u '+%s')
        MIDNIGHT_UTC=$(date -u -d 'tomorrow 00:00:00' '+%s' 2>/dev/null \
            || python3 -c "
import datetime, calendar
tomorrow = datetime.datetime.utcnow().date() + datetime.timedelta(days=1)
print(calendar.timegm(datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 10).timetuple()))
")
        WAIT=$((MIDNIGHT_UTC - NOW_UTC + 600))   # +10 min grace period
        [ "$WAIT" -lt 60 ] && WAIT=3600          # never sleep less than 1 h (clock skew)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daily quota exhausted — sleeping ${WAIT}s until after midnight UTC."
        sleep "$WAIT"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Some zones incomplete (rate limit or transient error)."
        echo "Sleeping ${SLEEP_ON_FAILURE}s before retry…"
        sleep "$SLEEP_ON_FAILURE"
    fi
done
