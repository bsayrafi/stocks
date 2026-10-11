import pickle
import yfinance as yf
import warnings
import logging

from tickersAug28 import get_tickers

warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

OUTPUT_FILE = "/content/drive/MyDrive/stock_screener/market_data.pkl"


def fetch_data(tickers, period="1y"):
    """Download OHLCV data for all tickers and return the raw yfinance DataFrame."""
    print(f"Fetching {period} of data for {len(tickers)} tickers...")
    data = yf.download(tickers, period=period, auto_adjust=False, progress=False)
    return data


def main():
    tickers = get_tickers()

    data = fetch_data(tickers)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"tickers": tickers, "data": data}, f)

    print(f"Saved data for {len(tickers)} tickers to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
