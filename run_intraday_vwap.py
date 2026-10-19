import os
import sys
import time
from datetime import datetime, timedelta, timezone
import pandas as pd

from common import tickersV2

# 1. Setup Working Directory for GitHub Actions
WORK_DIR = os.getcwd() 
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

# Import local modules (Ensure these files are uploaded to your GitHub repo!)
from common import constants
from common import indicators
from common import tickers
from common import excel_writer
import finvizfinance.screener.overview 
from screener import enrich_html
from common import event_catalysts
from screener import buy_entry
from screener import screener
from common import data_loader
from common.profiling import reset_profile, print_profile_summary, profile_step
from screener import external_data
import requests
from screener.sector_valuation import *
from screener.strategy import *
from screener import screener #type:ignore
from requests.adapters import HTTPAdapter
from screener.buy_entry import *
from common.event_catalysts import *

# 2. Securely load API Token
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")


MAX_WORKERS = 4   # parallel ticker workers (was 6; watch the summary for Yahoo rate-limit outliers)

# Same-day disk caches (see cached_ticker.py). The first run each day fetches fresh
# data; later runs that day reuse it. Skipped for tickers within 2 days of earnings.
#   CACHE_INFO_DAILY:    .info -> price-based ratios (market cap, P/E, EV/EBITDA, FCF yield)
#                        reflect the first run of the day. Set to 0 to always fetch fresh.
#   CACHE_ANALYST_DAILY: recommendations, EPS revisions / trend / estimates.
constants.CONFIG["CACHE_INFO_DAILY"] = 1
constants.CONFIG["CACHE_ANALYST_DAILY"] = 1
constants.CONFIG["EARNINGS_FRESH_WINDOW_DAYS"] = 2

lowTickers, midTickers, highTickers = [], [], []     # filled by tickersV2.fetch_tickers() in run()
my_filters = {
          'Country': 'USA',
          'Market Cap.': '+Small (over $300mln)',
          'Float Short': 'Under 20%',
          #'Sector': 'Technology',
          'Analyst Recom.': 'Hold or better',
          'Average Volume': 'Over 1M',
          #'P/E': 'Under 50',
          #'Forward P/E': 'Under 60',
          #'InstitutionalTransactions': 'Positive (>0%)',
          #'Price': 'Under $2',
    }


debugTickers = [
    "S", "CCC","WLTH",
    "LDOS","DOCN","NOW","INTU","AVT","CRM","FICO","AAPL","MSFT","IBM","DDOG","WDAY",

        ]
  # 2 Large
  # 1 Medium
  # 0 small



def getTickers(num):

  with profile_step("main: finviz ticker list"):
    if num!=3:
        #filteredTickers = tickers.get_tickers(my_filters)
        lowTickers, midTickers, highTickers = tickersV2.fetch_tickers(my_filters)

    else:
        filteredTickers = debugTickers

    print(f"Ticker counts: Small={len(lowTickers)}, Medium={len(midTickers)}, Large={len(highTickers)}")
    return lowTickers, midTickers, highTickers


def loadData(num, filteredTickers, force_redownload=False):

  constants.set_pkl_path(num)

  with profile_step("main: daily price download"):
     _, data = data_loader.load_or_download_market_data(filteredTickers,force_redownload)
     return data


def     setEnable(num):
    constants.CONFIG["ENABLE_INTRADAY"] = num
    constants.CONFIG["SHOW_PREMARKET_PRICE"] = num
    constants.CONFIG["ENABLE_ANALYST_DATA"] = num
    constants.CONFIG["ENABLE_FINVIZ"] = 0          # always off (see top of file)
    constants.CONFIG["ENABLE_RAW_STATEMENTS"] = num
    constants.CONFIG["ENABLE_ALTMAN_ZSCORE"] = 0   # always off: saves 2 statement requests per ticker
    constants.CONFIG["ENABLE_CMF"] = num
    constants.CONFIG["ENABLE_INTRADAY"] = 0
    constants.CONFIG["ENABLE_DIP_STRATEGY"] = num
    constants.CONFIG["ENABLE_VALUATION"] = num
    constants.CONFIG["ENABLE_SHORT_INTEREST"] = 0
    constants.CONFIG["ENABLE_EPS_DATA"] = num
    constants.CONFIG["ENABLE_EARNINGS_DATES"] = num
    constants.CONFIG["ENABLE_CASH_METRICS"] = num

def runCoreScreener(num=2, filteredTickers=None, force_redownload=True) :
        

    if not filteredTickers:
        print(f"[bucket {num}] No tickers - skipping.")
        return

    start = time.time()
    reset_profile()   # timings are per run (small caps and large caps reported separately)

    data = loadData(num, filteredTickers, force_redownload)
    if data is None or len(data) == 0:
        print(f"[bucket {num}] No price data downloaded - skipping.")
        return
    extra_data_store = {}

    print(f"Loaded {len(data)} tickers of {len(filteredTickers)} requested. ")
          
    # 1. Check the macro regime first
    with profile_step("main: SPY trend check"):
        market_bullish = screener.is_market_in_uptrend("SPY", 200)

    # 2. Halt or warn based on the regime
    if not market_bullish:
        print("WARNING: Broad market is in a downtrend. 80% of stocks follow the market.")
        print("Action: Suspending new fundamental dip-buys until SPY recovers.")
        # Option A: sys.exit("Stopping script execution to protect capital.")
        # Option B: Let it run, but append a global "MacroRisk" flag to your Excel output.

    #results2 = screener.evaluate_tickers(filteredTickers, data, extra_data_store=extra_data_store)
    results2 = screener.evaluate_tickers_parallel(
        filteredTickers, data,
        max_workers=MAX_WORKERS,
        extra_data_store=extra_data_store,
    )

    results_df2 = pd.DataFrame(results2)

    # ---> START OF REPLACEMENT BLOCK <---
    if not results_df2.empty:

        # 1. Sector Valuation
        if constants.CONFIG["ENABLE_COMPANY_INFO"] == 1 and constants.CONFIG["ENABLE_VALUATION"] == 1:
            with profile_step("main: sector valuation"):
                results_df2, sector_medians = add_sector_relative_valuation(results_df2)
            print("\nSector median valuations (this batch):")
            print(sector_medians)

        # 2. Final Quality Scoring
        if constants.CONFIG["ENABLE_DIP_STRATEGY"] == 1:
            with profile_step("main: quality scores"):
                results_df2 = combine_final_quality_scores(results_df2)

    else:
        print("No tickers passed the technical screening criteria. DataFrame is empty.")
    # ---> END OF REPLACEMENT BLOCK <---


    if not results_df2.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        #print(results_df2.to_string(index=False))
        csv_filename = constants.CONFIG["PKL_PATH"].replace(".pkl","")
        csv_filename = csv_filename + "_" + constants.get_dayprefix() + ".xlsx"
        #results_df2.to_excel(csv_filename, index=False, sheet_name="Results",engine="openpyxl")
        status_colors = {
            "1": "E2EFDA",  # Soft Green
            "0": "FFF2F2",     # Soft Red
            #"Pending": "FFF2CC"     # Soft Yellow
        }
        with profile_step("main: excel export"):
          excel_writer.export_df_with_row_colors(
            df=results_df2,
            file_path=csv_filename,
            target_col="G1 ",
            sheet_name="Sheet1",
            color_map=status_colors,
            header_bg="1F4E78",    # Dark Blue Header (Change to e.g. "2F5597", "333333", etc.)
            header_text="FFFFFF",   # White text
        )
    else:
        print("Error!")

    elapsed = time.time() - start

    print(f"Run took {elapsed:.1f}s for {len(filteredTickers)} ticker(s) "
                        f"({elapsed/max(len(filteredTickers),1):.2f}s/ticker)")
    print_profile_summary(elapsed, len(filteredTickers), max_workers=MAX_WORKERS)


   



    
    


lowTickers, midTickers, highTickers = getTickers(0)


setEnable(1)
force_redownload=True
#runCoreScreener(num=3, force_redownload=False)
runCoreScreener(0, lowTickers,force_redownload)
runCoreScreener(1, midTickers,force_redownload)
runCoreScreener(2, highTickers,force_redownload)
setEnable(0)


