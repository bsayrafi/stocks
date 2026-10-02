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
    
    
    
    fileapp = {
        0: "Small",
        1: "Med",
        2: "Large"
    }.get(num, "Debug")
    
    
    if not enrichment_tickers:
        print("No tickers")
    else:
        start = time.time()
        constants.set_pkl_path(num)
        enrich_html_v2.main(enrichment_tickers, fileapp)          
        elapsed = time.time() - start

        print(f"Entry run took {elapsed:.1f}s for {len(enrichment_tickers)} ticker(s) "
                f"({elapsed/max(len(enrichment_tickers),1):.2f}s/ticker)")



lowTickers, midTickers, highTickers = tickersV2.fetch_tickers(my_filters)

num = 0 #small
#num=4;

if num==4:
   run_enrichment(3)
else:
    run_enrichment(0)
    run_enrichment(1)
    run_enrichment(2)
    #run_enrichment(3)
