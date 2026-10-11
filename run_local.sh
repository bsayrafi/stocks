#!/usr/bin/env bash
# run_local.sh -- run any script of this project on your own machine with the API keys set.
#
# Keys are read from a local ".env" file (never committed -- see .gitignore).
# Any key still missing is asked for at the prompt (typing hidden) and used for this run only.
#
# Usage:
#   ./run_local.sh run_buy_entry.py
#   ./run_local.sh run_trend_entry.py refresh Large
#   ./run_local.sh fvg_live_signal.py --sector Technology    # scripts in subfolders are found by name
#   ./run_local.sh                                           # lists the run_*.py scripts
#
# First time:
#   cp env.example .env    # then put your keys in .env
#   chmod +x run_local.sh

set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -eq 0 ]]; then
  echo "usage: ./run_local.sh <script.py> [args...]"
  echo "scripts:"; ls run_*.py | sed 's/^/  /'
  exit 1
fi

# 1. Find the script: as given, or by name in a subfolder ("fvg_live_signal.py" -> "fvg/fvg_live_signal.py")
script="$1"; shift
if [[ ! -f "$script" ]]; then
  for cand in */"$script" */*/"$script"; do [[ -f "$cand" ]] && { script="$cand"; break; }; done
fi
[[ -f "$script" ]] || { echo "not found: $script"; exit 1; }

# 2. Load .env if present (every KEY=value line becomes an environment variable)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# 3. Ask for anything still missing, then show what is set (never the values themselves)
KEYS="APCA_API_KEY_ID APCA_API_SECRET_KEY FINNHUB_API_KEY HF_TOKEN NTFY_TOPIC"
for var in $KEYS; do
  if [[ -z "${!var:-}" ]]; then
    read -r -s -p "$var (Enter to skip): " value
    echo
    if [[ -n "$value" ]]; then export "$var=$value"; fi
  fi
done
for var in $KEYS; do
  if [[ -n "${!var:-}" ]]; then echo "  $var: set"; else echo "  $var: NOT set"; fi
done

# 4. Same folders the GitHub workflows create, then run
mkdir -p reports data cache
PYTHON="${PYTHON:-python3}"
[[ -x .venv/bin/python3 && "$PYTHON" == python3 ]] && PYTHON=.venv/bin/python3
exec "$PYTHON" "$script" ${1+"$@"}   # ${1+"$@"}: safe with no extra args on macOS's bash 3.2
