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
   "AMN", "ARHS", "ARLO", "ASTH", "ATEN", "CARS", "DSP", "ETON", "EGHT",
    "GCT", "GDYN", "GNK", "GOLD", "GPRE", "HCSG", "HIVE", "HOPE", 
    "HPE", "INVX", "IONQ", "IRWD", "KRP", "MG", "MGY", "MITK", "MDXG",
    "MLKN", "MNTN", "NOK", "NX", "NXDR", "OMER", "OPRT", "OMDA", "PANL", 
    "PAYS", "PCRX", "PGNY", "PGY", "PRGS", "PRTH", "QBTS", "QNST", "QTWO", 
    "QUBT", "REPX", "RIGL", "SWBI", "SWIM", "THRM", "VFF", "WWW", 
    "XPRO", "ZVRA"
]

large_enrichment_tickers = [

   "A", "AAOI", "AAPL", "ABBV", "ABNB", "ACMR", "ADI", "AER", "AIR", "ALAB",
    "AMAT", "AMD", "AME", "AMRX", "AMZN", "ANET", "APH", "ARMK", "ARQT", "ATI",
    "ATRC", "ATRO", "AU", "AVAH", "AVGO", "AVNT", "AVPT", "AXTA", "BDX", "BE",
    "BIIB", "BMRN", "BMY", "BTSG", "BULL", "BWA", "CAH", "CART", "CAT", "CDE",
    "CDNA", "CDNS", "CDW", "CGNX", "CHRD", "CIEN", "CNK", "COHR", "COP", "CORT",
    "CRBG", "CRDO", "CRM", "CRVW", "CRWD", "CSCO", "CTAS", "CTVA", "CVLT", "CVX",
    "CXW", "DAR", "DASH", "DDOG", "DE", "DELL", "DGX", "DHR", "DIS", "DK",
    "DOCN", "DT", "DVN", "DXCM", "ECL", "EL", "ELF", "ELV", "EMR", "ENTG",
    "EOG", "ESI", "ESTC", "ET", "ETSY", "EW", "EXEL", "EXLS", "EXPE", "FCX",
    "FIGS", "FIVE", "FIVN", "FLEX", "FLS", "FLYW", "FORM", "FRSH", "FTI", "GDDY",
    "GEV", "GLW", "GNRC", "GOOG", "GOOGL", "GPN", "GTES", "GTX", "HALO", "HPE",
    "HQY", "HSIC", "HTGC", "HUM", "IFF", "INCY", "INGM", "INOD", "INSW", "INTC",
    "IOT", "IQV", "IREN", "ITT", "JBL", "KDP", "KEYS", "KLAC", "KO", "LECO",
    "LITE", "LLY", "LNG", "LRCX", "MANH", "MCHP", "MDB", "META", "MGNI", "MMM",
    "MNST", "MPC", "MPLX", "MRK", "MRVL", "MSFT", "MTCH", "MTSI", "MU", "NBIS",
    "NEM", "NESR", "NOW", "NTAP", "NTNX", "NVDA", "NWS", "NWSA", "OKE", "OKTA",
    "ONTO", "ORCL", "OVV", "OXY", "P", "PAA", "PAGP", "PANW", "PARR", "PAY",
    "PCTY", "PDFS", "PG", "PH", "PLTR", "PR", "PSX", "Q", "QCOM", "REGN",
    "RELY", "RGEN", "RGLD", "RKLB", "ROK", "ROST", "SANM", "SCHW", "SHC", "SITM",
    "SKHY", "SLB", "SM", "SMTC", "SN", "SNDK", "SNX", "SOFI", "SPCX", "SSRM",
    "ST", "STT", "TER", "TKR", "TMO", "TOST", "TRGP", "TSLA", "TSM", "TTC",
    "TTEK", "TWLO", "TXN", "UBER", "UNH", "USFD", "VCYT", "VEEV", "VG", "VLO",
    "VSH", "VST", "WAT", "WAY", "WDAY", "WK", "WSM", "WT", "WTTR", "XOM",
    "XYZ", "ZBRA", "ZM"

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
