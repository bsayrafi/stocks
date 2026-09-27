#!/bin/bash
# Wrapper for launchd: run_mds.sh [entry|daily] [extra args...]
#   entry (default) -- 15m trigger check; exits by itself outside US market hours,
#                      so launchd can fire it every 15 minutes
#   daily           -- after-close scan that rebuilds live/mds_watchlist.json
cd "$(dirname "$0")"
mkdir -p live reports cache
if [[ -f .env ]]; then set -a; source .env; set +a; fi
mode="${1:-entry}"
if [[ $# -gt 0 ]]; then shift; fi
./.venv/bin/python3 run_mds_scanner.py "$mode" ${1+"$@"} >> "live/mds_${mode}.log" 2>&1
