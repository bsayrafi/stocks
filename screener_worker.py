#!/usr/bin/env python3
"""
screener_worker.py - a long-lived Python process that report_server.py keeps running in the background.

It imports run_trend_entry.py (and with it pandas, yfinance, enrich_html_v2 ...) ONCE, then waits for requests,
so a "Refresh now" click does not pay the 2-3 s Python start-up again. You do not run this file yourself.

Protocol (one line per message on stdin / stdout):
    in : {"mode": "refresh", "lists": ["Large"]}          (or "full", lists [] = all)
    out: everything the run prints, then  "@@DONE <exit code> <seconds>"
    "@@READY <seconds>" is printed once when the imports are done.
report_server.py restarts this worker when any .py file (or .env) in the folder changed, so code edits are picked up.
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

t0 = time.time()
import run_trend_entry as R          # noqa: E402  (heavy imports happen here, once)

print(f"@@READY {time.time() - t0:.1f}", flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        cmd = json.loads(line)
    except ValueError:
        print(f"worker: ignoring a bad request: {line[:100]}", flush=True)
        continue
    t = time.time()
    code = 0
    try:
        R.run(cmd.get("mode", "refresh"), cmd.get("lists") or None)
    except SystemExit as e:              # run() uses SystemExit for bad input; never let it end the worker
        if isinstance(e.code, int):
            code = e.code
        else:
            print(e.code, flush=True)
            code = 1
    except Exception:
        traceback.print_exc(file=sys.stdout)
        code = 1
    sys.stdout.flush()
    print(f"@@DONE {code} {time.time() - t:.1f}", flush=True)
