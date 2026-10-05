import os
import sys
import time
from datetime import datetime, timedelta, timezone
import pandas as pd

# 1. Setup Working Directory for GitHub Actions
WORK_DIR = os.getcwd() 
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

# Import local modules (Ensure these files are uploaded to your GitHub repo!)
from benchmarks import benchmark_symbols, get_benchmark_map
import constants
import indicators
import tickers
import excel_writer
import finvizfinance.screener.overview 
import enrich_html_v2
import event_catalysts
import buy_entry
from buy_entry import *
from event_catalysts import *
from flow_common import fetch_intraday_bars
from dip_confirm import score_dips
from trend_confirm import score_trends
import tickersV2



# 2. Securely load API Token
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

debug_enrichment_tickers = [

  "META", "AAPL", "AMZN", "GOOG", "MSFT", "TSLA", "NVDA", "QCOM", "ARLO", "PSX"
    ]

my_filters = {
          'Country': 'USA',
          'Market Cap.': '+Small (over $300mln)',
          'Float Short': 'Under 20%',
          'Analyst Recom.': 'Hold or better',
          'Average Volume': 'Over 750K',
          #'P/E': 'Under 50',
          #'Forward P/E': 'Under 50',
          'InstitutionalTransactions': 'Positive (>0%)',
    }


# Mode: "full" (default) = Finviz list + every ticker -> new _up / _down reports.
#       "refresh-setups"   = like refresh, but only the setup tickers (WATCH / forming / turning up) - faster
#       "refresh"          = re-screen only the tickers of each list's LAST _up report and write over that same file
#                            (no Finviz call; seconds instead of minutes with today's cache). A list whose last full
#                            scan is not from today (or that has none) gets a FULL scan instead.
# Set it with   python run_trend_entry.py refresh [Large|Small,Med]   or the HTMLV2_MODE / HTMLV2_LISTS environment
# variables (the GitHub workflow's "mode"). report_server.py's worker imports this file and calls run() directly.
FILEAPPS = {0: "Small", 1: "Med", 2: "Large"}
lowTickers, midTickers, highTickers = [], [], []     # filled by tickersV2.fetch_tickers() in run()


def run_enrichment(num ):
    enrichment_tickers = []
    if num==0:
        enrichment_tickers = lowTickers
    elif num==1:
        enrichment_tickers = midTickers
    elif num==2:
        enrichment_tickers = highTickers
    elif num==3:
        enrichment_tickers = debug_enrichment_tickers
    
    
    
    fileapp = FILEAPPS.get(num, "Debug")
    
    
    if not enrichment_tickers:
        print("No tickers")
    else:
        start = time.time()
        constants.set_pkl_path(num)
        enrich_html_v2.main(enrichment_tickers, fileapp)          
        elapsed = time.time() - start

        print(f"Entry run took {elapsed:.1f}s for {len(enrichment_tickers)} ticker(s) "
                f"({elapsed/max(len(enrichment_tickers),1):.2f}s/ticker)")


def run_refresh(num, scope="all"):
    """Update the last _up report of this list in place (see enrich_html_v2.refresh). scope "setups" = only the
    setup tickers (WATCH / forming / turning up), "all" = every ticker in the report."""
    fileapp = FILEAPPS.get(num, "Debug")
    start = time.time()
    constants.set_pkl_path(num)
    enrich_html_v2.refresh(fileapp, scope=scope)
    print(f"Refresh of {fileapp} took {time.time() - start:.1f}s")


def run(mode="full", only=None):
    """mode "full" or "refresh"; only = list names (e.g. ["Large"]) or None for all."""
    global lowTickers, midTickers, highTickers
    MODE = (mode or "full").strip().lower()
    ONLY = [x.strip() for x in (only or []) if x and x.strip()]
    num = 0 #small
    #num=4;

    groups = [3] if num == 4 else [0, 1, 2]
    if ONLY:
        by_name = {v.lower(): k for k, v in {**FILEAPPS, 3: "Debug"}.items()}
        unknown = [x for x in ONLY if x.lower() not in by_name]
        if unknown:
            raise SystemExit(f"unknown list(s) {unknown}: use {sorted(set(FILEAPPS.values()) | {'Debug'})}")
        groups = [by_name[x.lower()] for x in ONLY]
    if MODE not in ("full", "refresh", "refresh-setups"):
        raise SystemExit(f"unknown mode {MODE!r}: use 'full', 'refresh' or 'refresh-setups'")
    scope = "setups" if MODE == "refresh-setups" else "all"

    # A refresh only updates a report whose full scan ran TODAY. Any list without one gets a full scan instead,
    # so a full scan always happens at least once a day.
    full_groups, refresh_groups = list(groups), []
    if MODE.startswith("refresh"):
        full_groups = []
        for g in groups:
            why = enrich_html_v2.refresh_blocked_reason(FILEAPPS.get(g, "Debug"))
            if why:
                print(f"{FILEAPPS.get(g, 'Debug')}: refresh not possible ({why}) -> FULL scan instead")
                full_groups.append(g)
            else:
                refresh_groups.append(g)

    if full_groups:
        if any(g != 3 for g in full_groups):
            lowTickers, midTickers, highTickers = tickersV2.fetch_tickers(my_filters)
        for g in full_groups:
            run_enrichment(g)
    for g in refresh_groups:
        run_refresh(g, scope)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HTMLV2_MODE", "full"),
        (sys.argv[2] if len(sys.argv) > 2 else os.environ.get("HTMLV2_LISTS", "")).split(","))
