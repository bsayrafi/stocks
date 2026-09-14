import os
import pickle
import yfinance as yf
from constants import *


def load_or_download_market_data(sp500_tickers,force_redownload=False):
    """
    Load cached {"tickers": [...], "data": DataFrame} from PKL_PATH, or
    download a fresh 1y pull and cache it if the file doesn't exist yet
    (or force_redownload=True).
    """
    PKL_PATH = CONFIG["PKL_PATH"]

    #print (PKL_PATH)
    if os.path.exists(PKL_PATH) and not force_redownload:
        with open(PKL_PATH, "rb") as f:
            cached = pickle.load(f)
        tickers = cached["tickers"]
        data = cached["data"]
        print(f"Loaded cached data from {PKL_PATH} ({len(data)} rows)")
        return tickers, data

    data = yf.download(sp500_tickers, period="1y", auto_adjust=False, progress=False)
    os.makedirs(os.path.dirname(PKL_PATH), exist_ok=True)
    with open(PKL_PATH, "wb") as f:
        pickle.dump({"tickers": sp500_tickers, "data": data}, f)
    print(f"Downloaded and cached data to {PKL_PATH}")
    # Get a list of unique ticker symbols from Level 1 of the columns
    tickers = data.columns.get_level_values(1).unique()



    return sp500_tickers, data
