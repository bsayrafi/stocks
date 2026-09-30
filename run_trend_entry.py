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


# 2. Securely load API Token
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")


small_enrichment_tickers = [
   "AMN", "ARHS", "ARLO", "ASTH", "ATEN", "CARS", "DSP", "ETON", 
    "GCT", "GDYN", "GNK", "GOLD", "GPRE", "HCSG", "HIVE", "HOPE", 
    "HPE", "INVX", "IONQ", "IRWD", "KRP", "MG", "MGY", "MITK", 
    "MLKN", "MNTN", "NOK", "NX", "NXDR", "OMER", "OPRT", "PANL", 
    "PAYS", "PCRX", "PGNY", "PGY", "PRGS", "QBTS", "QNST", "QTWO", 
    "QUBT", "REPX", "RIGL", "SWBI", "SWIM", "THRM", "VFF", "WWW", 
    "XPRO", "ZVRA"
]

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


debug_enrichment_tickers = [

   "CTRI", "ARLO", "AMZN", "SOFI", "QBTS", "COHR", "ATEN", "ONTO", 
    "PONY", "TYL", "ATRO", "IREN", "VST", "KLAC", "HIVE",
    ]

def run_enrichment(num, runtype="all"):
    enrichment_tickers = []
    if num==0:
        enrichment_tickers = small_enrichment_tickers
    elif num==2:
        enrichment_tickers = large_enrichment_tickers
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


num = 0 #small
runtype = "all"
#num=4;

if num==4:
   run_enrichment(3, runtype)
else:
    run_enrichment(0, runtype)
    run_enrichment(2, runtype)
    run_enrichment(3, runtype)
