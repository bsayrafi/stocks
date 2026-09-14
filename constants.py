import os            # <--- Add this line here!
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import logging
import scipy.signal as signal
from datetime import datetime
from zoneinfo import ZoneInfo


warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


WORK_DIR = os.getcwd()
SPKL_PATH = WORK_DIR + "/data/smarket_data.pkl"
MPKL_PATH = WORK_DIR + "/data/mmarket_data.pkl"
LPKL_PATH = WORK_DIR + "/data/lmarket_data.pkl"
DPKL_PATH = WORK_DIR + "/data/dmarket_data.pkl"

def set_work_dir(work_dir_path):
    global WORK_DIR
    global SPKL_PATH
    global MPKL_PATH
    global LPKL_PATH
    global DPKL_PATH
    WORK_DIR = work_dir_path
    SPKL_PATH = WORK_DIR + "/data/smarket_data.pkl"
    MPKL_PATH = WORK_DIR + "/data/mmarket_data.pkl"
    LPKL_PATH = WORK_DIR + "/data/lmarket_data.pkl"
    DPKL_PATH = WORK_DIR + "/data/dmarket_data.pkl"



    
#SPKL_PATH = "/content/drive/MyDrive/stock_screener/data/smarket_data.pkl"
#MPKL_PATH = "/content/drive/MyDrive/stock_screener/data/mmarket_data.pkl"
#LPKL_PATH = "/content/drive/MyDrive/stock_screener/data/lmarket_data.pkl"
#DPKL_PATH = "/content/drive/MyDrive/stock_screener/data/dmarket_data.pkl"







# --- Constants: core screener ---

DAYm1_IDX = -1
DAYm2_IDX = DAYm1_IDX - 1
DAYm3_IDX = DAYm1_IDX - 2
DAYm4_IDX = DAYm1_IDX - 3
DAYm5_IDX = DAYm1_IDX - 4

DAY_IDXS = [DAYm1_IDX, DAYm2_IDX, DAYm3_IDX, DAYm4_IDX, DAYm5_IDX]

DAY_LABELS = {idx: -idx for idx in DAY_IDXS}  # DAYm1_IDX(-1) -> 1, DAYm5_IDX(-5) -> 5
DAY_ORDER = sorted(DAY_IDXS)  # ascending index = oldest -> most recent


# --- Constants: external data sources (all OFF by default — see note below) ---
# These hit yfinance sub-endpoints, TradingView, and Finviz PER TICKER.
# Running them across 500 tickers is slow and can get you rate-limited, so each
# source has its own on/off switch. Flip to 1 only when you actually want that
# column, ideally after narrowing your ticker list down with the core screener.
CONFIG = {

    "MARKET_SIZE": 0,   # market size 0 small, 1 medium, 2 large
    "PKL_PATH = ": DPKL_PATH,

    "SHOW_PREMARKET_PRICE": 0,   # just adds the Pre-Market column
    "FILTER_PREMARKET_GAP_UP": 0,  # actually excludes non-gapping tickers
    "SMA_FAST": 9,
    "SMA_SLOW": 50,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "ADX_PERIOD": 14,
    "ADX_TREND_THRESHOLD": 25,
    "BB_PERIOD": 20,
    "BB_NUM_STD": 2,
    "OBV_LOOKBACK": 5,
    "VOLUME_SPIKE_MULTIPLIER": 1.5,
    "VOLUME_AVG_PERIOD": 20,
    "PCT_OFF_HIGH_THRESHOLD": 0.20,
    "NEAR_LOW_BOUNCE_THRESHOLD": 0.05,
    "WICK_THRESHOLD": 0.01,
    "BODY_THRESHOLD": 0.005,
    "STOCH_PERIOD": 14,
    "SMOOTH_K": 3,
    "SMOOTH_D": 3,
    "OVERSOLD_LEVEL": 20,

    "CMF_PERIOD": 20,
    "CMF_BUY_THRESHOLD": 0.0,   # CMF crossing above this = buy signal

    "SR_FRACTAL_WINDOW": 2,
    "SR_CLUSTER_PCT": 0.01,
    "SR_LOOKBACK_DAYS": 90,
    "SR_NUM_LEVELS": 3,

    "ENABLE_COMPANY_INFO": 1,
    "ENABLE_ANALYST_DATA": 0,
    "ENABLE_TRADINGVIEW": 0,
    "ENABLE_FINVIZ": 0,
    "ENABLE_NEWS_SENTIMENT": 0,
    "ENABLE_SHORT_INTEREST": 0,
    "ENABLE_EPS_DATA": 0,
    "ENABLE_EARNINGS_DATES": 0,
    "ENABLE_RAW_STATEMENTS": 0,
    "ENABLE_ALTMAN_ZSCORE": 0,
    "ENABLE_CMF": 0,
    "ENABLE_CASH_METRICS": 0,

    "OPTIONS_EXPIRY_INDEX": 0,
    "NEWS_HEADLINE_LIMIT": 3,
    "EPS_DATES_LIMIT": 8,
    "FINBERT_MODEL_NAME": "ProsusAI/finbert",
    "TRADINGVIEW_SCREENER": "america",
    "TRADINGVIEW_EXCHANGE_DEFAULT": "NASDAQ",  # fallback if a symbol isn't in tv_exchange_map

    "DIP_MIN_PCT": 0.02,             # D1 must be down at least 5% from the 5-day high to qualify
    "DIP_LOOKBACK_DAYS": 5,          # baseline window for "recent high" comparison
    "EARNINGS_EXCLUSION_DAYS": 3,    # exclude tickers within N days of an earnings event (past or future)
    "DIP_QUALITY_MIN_SCORE": 50,     # minimum fundamental quality score (0-100) to qualify as a "quality dip"
    "ENABLE_DIP_STRATEGY": 0,
    "ENABLE_VALUATION": 0,
}

def set_pkl_path(size=0):
    CONFIG["MARKET_SIZE"]=size
    
    match size:
        case 0:
            CONFIG["PKL_PATH"]= SPKL_PATH
        case 1:
            CONFIG["PKL_PATH"]= MPKL_PATH
        case 2:
            CONFIG["PKL_PATH"]= LPKL_PATH
        case 3:
            CONFIG["PKL_PATH"]= DPKL_PATH



def get_ordinal_suffix(day):
    # Handle teens (11th, 12th, 13th) vs single digits (1st, 2nd, 3rd)
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

def get_dayprefix():

    # 1. Get today's date
    now = datetime.now()

    # 2. Format month ("Sept") and day (6)
    month = now.strftime("%b")  # Generates "Sep" or "Sept"
    day = now.day
    suffix = get_ordinal_suffix(day)

    # 3. Concatenate into "Sept 6th"
    date_str = f"{month}_{day}{suffix}"

    # Usage with a string
    final_text = f"{date_str}"

    return final_text
    # Output: Report generated on Sept 6th



def get_timeprefix():
    # Replace with your timezone (e.g., 'Asia/Hebron' or 'Asia/Jerusalem' for EEST)
    my_timezone = ZoneInfo("Asia/Hebron") 
    
    # Get the time specifically for that timezone
    local_now = datetime.now(my_timezone)

    # Format as Hour_Minute (e.g., "10_56")
    time_str = local_now.strftime("%H_%M") 
    
    return time_str

extra_data_store = {}  # symbol -> dict of DataFrames (recs, upgrades, financi