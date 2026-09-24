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
from buy_entry import *
from event_catalysts import *

# 2. Securely load API Token
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")


small_enrichment_tickers = [
    "QTWO", "PGY", "RIGL", "ETON", 
    "MITK", "THRM", "GPRE", "PGY", "ARLO", "DSP", "MLKN", "PCRX", "HCSG", "HIVE", "GDYN",
    "WWW", "MG", "ASTH", "GCT", "NX", "SWBI", "ZVRA", "KRP", "PGNY", "XPRO",
    "GNK", "GOLD", "PRGS", "INVX", "MNTN", "MGY", "NXDR", "PANL", "ATEN", "IRWD", "ARHS", "PAYS",
    "VFF", "SWIM", "HOPE", "REPX", "AMN", "QUBT", "QBTS", "IONQ", "NOK", "HPE",
]

large_enrichment_tickers = [

    "A", "AAPL", "AAOI", "ABNB", "AER", "AIR", "ALAB", "AMAT", "AMD", "AME", "AMRX", "AMZN", "ANET", 
    "APH", "ARMK", "ARQT", "ATRO", "ATI", "AVGO", "AVNT", "AVPT", "AXTA", "BE", "BIIB", "BTSG",
    "CART", "CAT", "CDE", "CDNA", "CDW", "CGNX", "CHRD", "CIEN", "COHR", "COP", "CRM", "CRDO", "CRWD", "CRVW", 
    "CSCO", "CTAS", "CTVA", "CVLT", "CVX", "DASH", "DDOG", "DELL", "DGX", "DHR",
    "DOCN", "DT", "ECL", "EMR", "ESTC", "EXLS", "ENTG", "FCX", "FIVE", "FIVN", "FLS",
    "FLYW", "FORM", "FRSH", "FTI", "GOOG", "GEV", "GLW", "GTES", "HALO", "HQY", "INGM", "IREN",
    "IOT", "INTC", "KEYS", "KLAC", "KO", "LECO", "LITE", "KLH", "LLY", "LRCX", "MANH", "MDB", "META",
    "MNST", "MRK", "MRVL", "MSFT", "MTSI", "MU", "NEM", "NESR", "NOW", "NTAP", "NBIS",
    "NVDA", "OKTA", "ONTO", "ORCL", "P", "PAA", "PANW", "PARR", "PAY", "PCTY", "PDFS",
    "PH", "PLTR", "PR", "PSX", "Q", "QCOM", "REGN", "RGLD", "ROK", "RKLB", "SCHW", "SHC", "SOFI", "SKHY", "SMTC", 
    "SITM", "SLB", "SMTC", "SNDK", "SNX", "SSRM", "SPCX", "TER", "TKR", "TMO", "TTC", "TSLA", "TSM",
    "TTEK", "TWLO", "TXN", "UBER", "VCYT", "VEEV", "VSH", "VST", "WAT", "WAY", "WDAY", "WK",
    "WSM", "XYZ", "ZBRA", "ZM"

]



def run_enrichment(num):
    enrichment_tickers = []
    if num==0:
        enrichment_tickers = small_enrichment_tickers
    else:
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
        enrich_html.main(enrichment_tickers, fileapp)
    
        results2_entry = check_buy_zone_confirmation(enrichment_tickers)
        all_df = results2_entry["all"]
        elapsed = time.time() - start
    
        if not all_df.empty:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            tz_gmt3 = timezone(timedelta(hours=3))
            timestamp = constants.get_dayprefix() + "_" + constants.get_timeprefix()
            txt = f"_buy_{timestamp}.xlsx"
            csv_filename = constants.CONFIG["PKL_PATH"].replace(".pkl", txt)
    
            print(csv_filename)
            status_colors = {
                "1": "E2EFDA",  
                "0": "FFF2F2",     
            }
            
            excel_writer.export_df_with_row_colors(
                df=all_df,
                file_path=csv_filename,
                target_col="Ticker",
                sheet_name="Sheet1",
                color_map=status_colors,
                header_bg="1F4E78",    
                header_text="FFFFFF",   
            )
    
            # --- Intraday entry timing ---
            confirmed_df = results2_entry["confirmed"]
    
            if not confirmed_df.empty:
                intraday_results = []
                for _, row in confirmed_df.iterrows():
                    intraday_results.append(find_intraday_entry(row["Ticker"], entry_type=row["Entry_Type"]))
                intraday_df = pd.DataFrame(intraday_results)
    
                print(intraday_df.to_string(index=False))
    
                intraday_filename = csv_filename.replace(".xlsx", "_intraday.xlsx")
                excel_writer.export_df_with_row_colors(
                    df=intraday_df,
                    file_path=intraday_filename,
                    target_col="Ticker",
                    sheet_name="Sheet1",
                    color_map=status_colors,
                    header_bg="1F4E78",
                    header_text="FFFFFF",
                )
            else:
                print("No confirmed setups — skipping intraday check.")
        else:
            print("Error!")
    
        print(f"BuyEntry run took {elapsed:.1f}s for {len(enrichment_tickers)} ticker(s) "
              f"({elapsed/max(len(enrichment_tickers),1):.2f}s/ticker)")


#num = 2
run_enrichment(0)
run_enrichment(2)
