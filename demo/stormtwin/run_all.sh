#!/bin/sh
# Build all three demo farms: storm runs, then fatigue history; then the page.
D=$(cd "$(dirname "$0")" && pwd)
for F in A B C; do
  STORMTWIN_FARM=$F python3 -W ignore "$D/build_data.py" > "$D/farm_$F.log" 2>&1 || exit 1
  STORMTWIN_FARM=$F python3 -W ignore "$D/build_history.py" > "$D/history_$F.log" 2>&1 || exit 1
done
python3 "$D/make_page.py"
