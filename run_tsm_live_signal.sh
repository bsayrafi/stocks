#!/bin/bash
# Wrapper for launchd: sets a real cwd/env (launchd gives neither) and logs output.
cd "$(dirname "$0")"
mkdir -p live
./.venv/bin/python3 trend_tsm_live_signal.py \
  --sectors Technology "Health Care" Financials \
  --capital 10000 \
  >> live/tsm_live_signal.log 2>&1
