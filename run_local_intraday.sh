#!/usr/bin/env bash
# run_local.sh -- run the screener on your own machine with the API keys set.
#
# Keys are read from a local ".env" file (never committed -- see .gitignore).
# Any key that is missing from .env is asked for at the prompt (typing hidden)
# and used for this run only.
#
# Usage:
#   ./run_local.sh                  # runs run_intraday_vwap.py
#   ./run_local.sh other_script.py  # runs a different script with the same keys
#
# First time:
#   cp .env.example .env    # then put your keys in .env
#   chmod +x run_local.sh

set -euo pipefail
cd "$(dirname "$0")"

# 1. Load .env if present (every KEY=value line becomes an environment variable)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# 2. Ask for anything still missing
for var in APCA_API_KEY_ID APCA_API_SECRET_KEY FINNHUB_API_KEY HF_TOKEN; do
  if [[ -z "${!var:-}" ]]; then
    read -r -s -p "$var (Enter to skip): " value
    echo
    if [[ -n "$value" ]]; then
      export "$var=$value"
    fi
  fi
done

# 3. Show what is set (never the values themselves)
for var in APCA_API_KEY_ID APCA_API_SECRET_KEY FINNHUB_API_KEY HF_TOKEN; do
  if [[ -n "${!var:-}" ]]; then echo "  $var: set"; else echo "  $var: NOT set"; fi
done

# 4. Same folders the GitHub workflow creates, then run
mkdir -p reports data cache
PYTHON="${PYTHON:-python3}"
script="${1:-run_intraday_vwap.py}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PYTHON" "$script" ${1+"$@"}   # ${1+"$@"}: safe with no extra args on macOS's bash 3.2
