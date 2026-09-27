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

    "A", "AAOI", "AAPL", "ABNB", "ACMR", "ADI", "AER", "AIR", "ALAB", "AMAT", 
    "AMD", "AME", "AMRX", "AMZN", "ANET", "APH", "ARMK", "ARQT", "ATI", "ATRO",
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
        if runtype == "html":
            enrich_html.main(enrichment_tickers, fileapp)
        else:
            enrich_html.main(enrichment_tickers, fileapp)

            buy_start = time.time()
            #results2_entry = check_buy_zone_confirmation(enrichment_tickers)
            bars = fetch_intraday_bars(dip_list + trend_list)   # one download for both
            dips = score_dips(dip_list, bars=bars)
            trends = score_trends(trend_list, bars=bars)

            print(f"check_buy_zone_confirmation took {time.time() - buy_start:.1f}s")
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

                excel_writer.export_df_with_row_colors(
                    df=dips,
                    file_path="a"+csv_filename,
                    target_col="rank",
                    sheet_name="Sheet1",
                    color_map=status_colors,
                    header_bg="1F4E78",    
                    header_text="FFFFFF",   
                )
                excel_writer.export_df_with_row_colors(
                    df=trends,
                    file_path="b"+csv_filename,
                    target_col="rank",
                    sheet_name="Sheet1",
                    color_map=status_colors,
                    header_bg="1F4E78",    
                    header_text="FFFFFF",   
                )
                # --- Intraday entry timing ---
                confirmed_df = results2_entry["confirmed"]
        
                if not confirmed_df.empty:
                    intraday_df = find_intraday_entries(confirmed_df)
        
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


#num = 0 small
runtype = "all"
run_enrichment(3, runtype)

#run_enrichment(0, runtype)
#run_enrichment(2, runtype)
