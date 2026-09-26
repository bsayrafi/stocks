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
import constants
import indicators
import tickers
import excel_writer
import finvizfinance.screener.overview 
import enrich_html
import event_catalysts
import buy_entry
import screener
import data_loader
from profiling import reset_profile, print_profile_summary, profile_step
import external_data
import requests
from sector_valuation import *
from strategy import *
import screener #type:ignore
from requests.adapters import HTTPAdapter
from buy_entry import *
from event_catalysts import *

# 2. Securely load API Token
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

constants.CONFIG["ENABLE_INTRADAY"] = 1

constants.CONFIG["SHOW_PREMARKET_PRICE"] = 1
constants.CONFIG["ENABLE_COMPANY_INFO"] = 1
constants.CONFIG["ENABLE_ANALYST_DATA"] = 1
constants.CONFIG["ENABLE_SHORT_INTEREST"] = 1
constants.CONFIG["ENABLE_EPS_DATA"] = 1
constants.CONFIG["ENABLE_EARNINGS_DATES"] = 1
constants.CONFIG["ENABLE_CASH_METRICS"] = 1
constants.CONFIG["ENABLE_FINVIZ"] = 0          # off: slow per-ticker scrape; TargetMean / ShortPctFloat come from yfinance
constants.CONFIG["ENABLE_CMF"] = 1
constants.CONFIG["ENABLE_DIP_STRATEGY"]= 1
constants.CONFIG["ENABLE_VALUATION"]= 1
constants.CONFIG["DIP_LOOKBACK_DAYS"]= 5
constants.CONFIG["DIP_QUALITY_MIN_SCORE"]= 50

MAX_WORKERS = 6   # parallel ticker workers (was 6; watch the summary for Yahoo rate-limit outliers)

# Same-day disk caches (see cached_ticker.py). The first run each day fetches fresh
# data; later runs that day reuse it. Skipped for tickers within 2 days of earnings.
#   CACHE_INFO_DAILY:    .info -> price-based ratios (market cap, P/E, EV/EBITDA, FCF yield)
#                        reflect the first run of the day. Set to 0 to always fetch fresh.
#   CACHE_ANALYST_DAILY: recommendations, EPS revisions / trend / estimates.
constants.CONFIG["CACHE_INFO_DAILY"] = 1
constants.CONFIG["CACHE_ANALYST_DAILY"] = 1
constants.CONFIG["EARNINGS_FRESH_WINDOW_DAYS"] = 2



def loadData(num, force_redownload=False):

  # 2 Large
  # 1 Medium
  # 0 small

  constants.set_pkl_path(num)
  market_cap_options = {
          0: '-Small (under $2bln)',
          1: 'Mid ($2bln to $10bln)',
          2: '+Mid (over $2bln)',
          3: '+Micro (over $50mln)',
      }


  my_filters = {
          'Country': 'USA',
          #'Market Cap.': '+Mid (over $2bln)',
          #'Market Cap.': '-Small (under $2bln)',
          'Market Cap.': market_cap_options[num],
          'Float Short': 'Under 20%',
          'Analyst Recom.': 'Buy or better',

          'P/E': 'Profitable (>0)',
          'Forward P/E': 'Profitable (>0)',
          #'Current Ratio': 'Over 0.5',
          #'Quick Ratio': 'Over 0.5',
          #'PEG': 'Under 3',
          #'EPS growthqtr over qtr': 'Positive (>0%)',
          #'EPS growth ttm': 'Positive (>0%)',
          #'InstitutionalOwnership': 'Over 20%',


          '200-Day Simple Moving Average': 'Price above SMA200',
          #'Price': 'Under $50',
          #'RSI (14)': 'Not Overbought (<60)',
    }
  print(my_filters)
  with profile_step("main: finviz ticker list"):
    if num!=3:
        filteredTickers = tickers.get_tickers(my_filters)
    else:
        filteredTickers = debugTickers
  with profile_step("main: daily price download"):
    return data_loader.load_or_download_market_data(filteredTickers,force_redownload)


def     setEnable(num):
    constants.CONFIG["ENABLE_INTRADAY"] = num
    constants.CONFIG["SHOW_PREMARKET_PRICE"] = num
    constants.CONFIG["ENABLE_ANALYST_DATA"] = num
    constants.CONFIG["ENABLE_FINVIZ"] = 0          # always off (see top of file)
    constants.CONFIG["ENABLE_RAW_STATEMENTS"] = num
    constants.CONFIG["ENABLE_ALTMAN_ZSCORE"] = 0   # always off: saves 2 statement requests per ticker
    constants.CONFIG["ENABLE_CMF"] = num
    constants.CONFIG["ENABLE_INTRADAY"] = num
    constants.CONFIG["ENABLE_DIP_STRATEGY"] = num
    constants.CONFIG["ENABLE_VALUATION"] = num
    constants.CONFIG["ENABLE_SHORT_INTEREST"] = num
    constants.CONFIG["ENABLE_EPS_DATA"] = num
    constants.CONFIG["ENABLE_EARNINGS_DATES"] = num
    constants.CONFIG["ENABLE_CASH_METRICS"] = num

def runCoreScreener(num=2, force_redownload=True) :
        

    start = time.time()
    reset_profile()   # timings are per run (small caps and large caps reported separately)

    filteredTickers, data = loadData(num, force_redownload)
    extra_data_store = {}

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


   






debugTickers = [
        "A", "AAOI", "AAPL", "ABNB", "ACMR", "ADI", "AER", "AIR", "ALAB", "AMAT", "AMD", "AME", "AMRX", "AMZN", "ANET", "APH", "ARMK", "ARQT", "ATI", "ATRO", "AU", "AVGO", "AVNT", "AVPT", "AXTA", "BE", "BIIB", "BTSG", "BWA", "CART", "CAT", "CDE", "CDNA", "CDNS", "CDW", "CGNX", "CHRD", "CIEN", "COHR", "COP", "CORT", "CRDO", "CRM", "CRVW", "CRWD", "CSCO", "CTAS", "CTVA", "CVLT", "CVX", "DASH", "DDOG", "DELL", "DGX", "DHR", "DOCN", "DT", "ECL", "EMR", "ENTG", "EOG", "ESTC", "ETSY", "EXEL", "EXLS", "EXPE", "FCX", "FIGS", "FIVE", "FIVN", "FLS", "FLYW", "FORM", "FRSH", "FTI", "GEV", "GLW", "GOOG", "GOOGL", "GTES", "HALO", "HQY", "INGM", "INOD", "INSW", "INTC", "IOT", "IREN", "KEYS", "KLAC", "KO", "LECO", "LITE", "LLY", "LRCX", "MANH", "MDB", "META", "MNST", "MPC", "MRK", "MRVL", "MSFT", "MTCH", "MTSI", "MU", "NBIS", "NEM", "NESR", "NOW", "NTAP", "NVDA", "NWS", "NWSA", "OKTA", "ONTO", "ORCL", "P", "PAA", "PANW", "PARR", "PAY", "PCTY", "PDFS", "PH", "PLTR", "PR", "PSX", "Q", "QCOM", "REGN", "RGLD", "RKLB", "ROK", "ROST", "SANM", "SCHW", "SHC", "SITM", "SKHY", "SLB", "SMTC", "SNDK", "SNX", "SOFI", "SPCX", "SSRM", "TER", "TKR", "TMO", "TOST", "TSLA", "TSM", "TTC", "TTEK", "TWLO", "TXN", "UBER", "VCYT", "VEEV", "VSH", "VST", "WAT", "WAY", "WDAY", "WK", "WSM", "XYZ", "ZBRA", "ZM"
        ]
  # 2 Large
  # 1 Medium
  # 0 small
setEnable(1)
#runCoreScreener(num=3, force_redownload=False)
runCoreScreener(num=0, force_redownload=True)
runCoreScreener(num=2, force_redownload=True)
setEnable(0)


