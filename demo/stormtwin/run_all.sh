#!/bin/sh
# Build the demo farms: storm runs, fatigue history, modal tracking; then the page.
# Usage: ./run_all.sh [A B C]   (record lengths via STORM_RECORD / HIST_RECORD)
D=$(cd "$(dirname "$0")" && pwd)
FARMS=${*:-A B C}
for F in $FARMS; do
  if [ -z "$SKIP_STORM" ]; then
    STORMTWIN_FARM=$F STORMTWIN_RECORD=${STORM_RECORD:-1200} python3 -W ignore "$D/build_data.py" > "$D/farm_$F.log" 2>&1 || exit 1
  fi
  STORMTWIN_FARM=$F STORMTWIN_RECORD=${HIST_RECORD:-300} python3 -W ignore "$D/build_history.py" > "$D/history_$F.log" 2>&1 || exit 1
  STORMTWIN_FARM=$F python3 -W ignore "$D/build_modal.py" > "$D/modal_$F.log" 2>&1 || exit 1
done
python3 "$D/build_rna.py" && python3 "$D/build_live.py" && python3 "$D/make_page.py"
