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


# $0.04 – $40.65
small_enrichment_tickers = [
    "ADEA", "AGEN", "AGIO", "AMPL", "AMRX", "ANGX", "APPS", "ARHS", "ARLO", "ARQT",
    "ASTH", "ATEN", "AVAH", "AVNT", "AVPT", "AXTA", "BHVN", "BOX", "BRZE", "BULL",
    "CCC", "CDE", "CHYM", "CLYM", "CNK", "CRBG", "CRGY", "CRVW", "CTVA", "CXW",
    "DC", "DSP", "EGHT", "ESI", "ET", "EWTX", "EXLS", "EXTR", "FA", "FATE",
    "FIGS", "FIVN", "FLYW", "FRSH", "FSLY", "FTRE", "GEO", "GPRE", "GTES", "GTX",
    "HIVE", "HOPE", "HP", "HTGC", "HYLN", "IMMX", "INGM", "IOT", "IREN", "IRWD",
    "KDP", "KOS", "KURA", "MG", "MGNI", "MGY", "MITK", "MLKN", "MNTN", "MQ",
    "MRVI", "MTCH", "NAVN", "NESR", "NIQ", "NNBR", "NOK", "NWS", "NWSA", "NXDR",
    "OMDA", "OMER", "OVID", "PAA", "PAGP", "PANL", "PAY", "PAYS", "PCRX", "PD",
    "PFE", "PR", "PRTH", "PTEN", "QBTS", "QNST", "QUBT", "RCUS", "REI", "RELY",
    "SHC", "SM", "SOFI", "SSRM", "SXC", "TALO", "TENB", "TOST", "TTEK", "VG",
    "VIR", "VSH", "VSTS", "VTRS", "WAY", "WNC", "WT", "WTI", "WTTR", "ZETA",
]

# $40.95 – $147.77
medium_enrichment_tickers = [
    "AAOI", "ACMR", "AER", "AIR", "ANF", "APA", "APH", "ARMK", "ATRC", "ATRO",
    "AU", "AVT", "AXTI", "BBY", "BILL", "BMRN", "BMY", "BTSG", "BWA", "CART",
    "CDNA", "CDW", "CF", "CGNX", "CGON", "CHRD", "COP", "CORT", "CSCO", "CVLT",
    "DAR", "DINO", "DIS", "DK", "DOCN", "DOCU", "DT", "DUOL", "DVN", "DXCM",
    "EL", "ELF", "EOG", "ESTC", "ETON", "ETSY", "EW", "EXEL", "FAST", "FCX",
    "FLEX", "FLR", "FLS", "FTI", "GDDY", "GILD", "GPN", "GTLB", "HALO", "HPE",
    "HQY", "HSIC", "IFF", "INCY", "INOD", "INSW", "INTC", "IONQ", "KNX", "KO",
    "KOD", "LNTH", "LSCC", "MCHP", "MNST", "MPLX", "MRK", "MTDR", "MXL", "NEM",
    "NOW", "NTNX", "OKE", "ORCL", "OVV", "OXY", "P", "PARR", "PCTY", "PDFS",
    "PENG", "PG", "PPLI", "PTC", "Q", "QTWO", "RIGL", "RKLB", "RPRX", "SCHW",
    "SLB", "SMCI", "SSNC", "ST", "TKR", "TTC", "TTMI", "TVTX", "U", "UBER",
    "UCTT", "URBN", "USFD", "VCYT", "VIAV", "VST", "WES", "WK", "XYZ", "ZM",
]

# $148.07 – $1787.69
large_enrichment_tickers = [
    "A", "AAPL", "ABBV", "ABNB", "ADI", "ADP", "ALAB", "AMAT", "AMD", "AME",
    "AMGN", "AMZN", "ANET", "ATI", "AVGO", "BDX", "BE", "BIIB", "CAH", "CAT",
    "CDNS", "CIEN", "COHR", "CRDO", "CRM", "CRWD", "CTAS", "CVX", "DASH", "DDOG",
    "DE", "DELL", "DGX", "DHR", "ECL", "ELV", "EMR", "ENTG", "EXPE", "FIVE",
    "FORM", "FTNT", "GEV", "GH", "GLW", "GNRC", "GOOG", "GOOGL", "GWRE", "HUM",
    "ILMN", "IQV", "ITT", "JBL", "JNJ", "KEYS", "KLAC", "LECO", "LITE", "LLY",
    "LNG", "LRCX", "MANH", "MDB", "META", "MMM", "MPC", "MRVL", "MSFT", "MTSI",
    "MU", "NBIS", "NTAP", "NTRA", "NVDA", "OKTA", "ONTO", "PANW", "PH", "PLTR",
    "PSX", "QCOM", "REGN", "RGEN", "RGLD", "ROK", "ROKU", "ROST", "RVMD", "RVTY",
    "SANM", "SITM", "SKHY", "SMTC", "SN", "SNDK", "SNPS", "SNX", "SPCX", "STT",
    "STX", "TEAM", "TER", "TMO", "TRGP", "TSLA", "TSM", "TWLO", "TXN", "UNH",
    "UTHR", "VEEV", "VLO", "WAT", "WDAY", "WDC", "WSM", "XOM", "ZBRA", "ZS",
]

debug_enrichment_tickers = [

  "META", "AAPL", "AMZN", "GOOG", "MSFT", "TSLA", "NVDA", "QCOM", "ARLO", "PSX"
    ]

def run_enrichment(num ):
    enrichment_tickers = []
    if num==0:
        enrichment_tickers = small_enrichment_tickers
    elif num==1:
        enrichment_tickers = medium_enrichment_tickers
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
#num=4;

if num==4:
   run_enrichment(3)
else:
    run_enrichment(0)
    run_enrichment(1)
    run_enrichment(2)
    #run_enrichment(3)
