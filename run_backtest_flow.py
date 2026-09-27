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
import enrich_html
import event_catalysts
import buy_entry
from buy_entry import *
from event_catalysts import *
from flow_common import fetch_intraday_bars
from dip_confirm import score_dips
from trend_confirm import score_trends
from benchmarks import get_benchmark_map
from backtest_flow import run_backtest

# 2. Securely load API Token
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")




large_enrichment_tickers = [

    "A", "AAOI", "AAPL", "ABNB", "ACMR", "ADI", "AER", "AIR", "ALAB", "AMAT", 
    "AMD", "AME", "AMRX", "AMZN", "ANET", "APH", "ARMK", "ARQT", "ATI", "ATRO", 
    "AU", "AVGO", "AVNT", "AVPT", "AXTA", "BE", "BIIB", "BTSG", "BWA", "CART", 
    "CAT", "CDE", "CDNA", "CDNS", "CDW", "CGNX", "CHRD", "CIEN", "COHR", "COP", 
    "CORT", "CRDO", "CRM", "CRVW", "CRWD", "CSCO", "CTAS", "CTVA", "CVLT", "CVX", 
    "DASH", "DDOG", "DELL", "DGX", "DHR", "DOCN", "DT", "ECL", "EMR", "ENTG", 
    "EOG", "ESTC", "ETSY", "EXEL", "EXLS", "EXPE", "FCX", "FIGS", "FIVE", "FIVN", 
    "FLS", "FLYW", "FORM", "FRSH", "FTI", "GEV", "GLW", "GOOG", "GOOGL", "GTES", 
    "HALO", "HQY", "INGM", "INOD", "INSW", "INTC", "IOT", "IREN", "KEYS", "KLAC", 
    "KO", "LECO", "LITE", "LLY", "LRCX", "MANH", "MDB", "META", "MNST", "MPC", 
    "MRK", "MRVL", "MSFT", "MTCH", "MTSI", "MU", "NBIS", "NEM", "NESR", "NOW", 
    "NTAP", "NVDA", "NWS", "NWSA", "OKTA", "ONTO", "ORCL", "P", "PAA", "PANW", 
    "PARR", "PAY", "PCTY", "PDFS", "PH", "PLTR", "PR", "PSX", "Q", "QCOM", 
    "REGN", "RGLD", "RKLB", "ROK", "ROST", "SANM", "SCHW", "SHC", "SITM", "SKHY", 
    "SLB", "SMTC", "SNDK", "SNX", "SOFI", "SPCX", "SSRM", "TER", "TKR", "TMO", 
   "TOST", "TSLA", "TSM", "TTC", "TTEK", "TWLO", "TXN", "UBER", "VCYT", "VEEV", 
    "VSH", "VST", "WAT", "WAY", "WDAY", "WK", "WSM", "XYZ", "ZBRA", "ZM"

]


small_enrichment_tickers = []

def runmain(num=1, runtype="all"):
    enrichment_tickers = large_enrichment_tickers
    if num==0:
        enrichment_tickers = small_enrichment_tickers
    elif num==2:
        enrichment_tickers = large_enrichment_tickers

    
    
    
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
        

        buy_start = time.time()
        results2_entry = []
        RUN_BACKTEST = True   # False = normal live scan

        bench_map= get_benchmark_map(enrichment_tickers)

        #bars = fetch_intraday_bars(enrichment_tickers + benchmark_symbols(bench_map))
        #dips = score_dips(enrichment_tickers, bars=bars, benchmarks=bench_map)
        #trends = score_trends(enrichment_tickers, bars=bars, benchmarks=bench_map)

        trend_bt = run_backtest(enrichment_tickers, mode="trend", benchmarks=bench_map, save_csv="trend_bt.csv")
        dip_bt = run_backtest(enrichment_tickers, mode="dip", benchmarks=bench_map, save_csv="dip_bt.csv")

        print(f"time took {time.time() - buy_start:.1f}s")
        #all_df = results2_entry["all"]
        elapsed = time.time() - start
    
        

        print(f"run took {elapsed:.1f}s for {len(enrichment_tickers)} ticker(s) "
                f"({elapsed/max(len(enrichment_tickers),1):.2f}s/ticker)")


#num = 0 small
runtype = "all"
runmain(3, runtype)

#runmain(0, runtype)
#runmain(2, runtype)
