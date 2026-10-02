
import io
import os
import json
import hashlib
from datetime import date, datetime
import pandas as pd
import requests
from finvizfinance.screener.overview import Overview
from constants import *
from benchmarks import build_benchmark_map


# Cache location, in priority order:
#   1. FINVIZ_CACHE_DIR environment variable (set by the GitHub Actions workflow)
#   2. CONFIG["FINVIZ_CACHE_DIR"]
#   3. ./cache next to this script
CACHE_DIR = (
    os.environ.get("FINVIZ_CACHE_DIR")
    or CONFIG.get("FINVIZ_CACHE_DIR")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
)


def _cache_path(my_filters):
    """One cache file per distinct filter set, so changing filters forces a refetch."""
    key = json.dumps(my_filters, sort_keys=True)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"screener_{digest}.json")


def _load_cache(my_filters):
    """Return the cached DataFrame if it was fetched today, else None."""
    path = _cache_path(my_filters)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("date") != date.today().isoformat():
            return None
        df = pd.DataFrame(payload["rows"])
        print(f"Using cached Finviz results from {payload.get('fetched_at')} ({len(df)} tickers)")
        return df
    except Exception as e:
        print(f"Ignoring unreadable cache {path} ({e})")
        return None


def _save_cache(my_filters, df):
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {
        "date": date.today().isoformat(),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "filters": my_filters,
        "rows": df[["Ticker", "Price"]].to_dict(orient="records"),
    }
    path = _cache_path(my_filters)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)  # atomic, so a crash never leaves a half-written cache


def _fetch_screener_df(my_filters):
    """Hit Finviz and return the screener DataFrame (or None/empty)."""
    screener = Overview()
    screener.set_filter(filters_dict=my_filters)

    # finvizfinance waits `sleep_sec` between result pages (20 tickers/page).
    # The default of 1s cost ~13s for ~250 tickers; 0.2s is usually fine. If
    # Finviz rejects the fast fetch (rate limit), retry once at the old 1s pace.
    fast_sleep = CONFIG.get("FINVIZ_PAGE_SLEEP_SEC", 0.2)
    try:
        return screener.screener_view(sleep_sec=fast_sleep)
    except Exception as e:
        print(f"Finviz fetch at {fast_sleep}s/page failed ({e}); retrying at 1s/page")
        return screener.screener_view(sleep_sec=1)


def fetch_tickers(my_filters, force_refresh=False):
    """
    Fetch tickers from the Finviz screener and split them into three
    equal-sized groups by closing price.

    Finviz is queried at most once per day per filter set; later calls on the
    same day read from a local cache. Pass force_refresh=True to bypass it.

    Args:
        my_filters (dict): Finviz screener filters.
        force_refresh (bool): Ignore today's cache and refetch from Finviz.

    Returns:
        tuple[list, list, list]: (low_price, mid_price, high_price) ticker lists.
            Each list is sorted by price ascending. When the total isn't divisible
            by 3, the lower groups get the extra ticker(s), so sizes differ by
            at most 1.
    """
    df = None if force_refresh else _load_cache(my_filters)

    if df is None:
        df = _fetch_screener_df(my_filters)
        if df is None or df.empty:
            print("No tickers matched the current filters.")
            return [], [], []
        _save_cache(my_filters, df)

    if df.empty:
        print("No tickers matched the current filters.")
        return [], [], []

    # Sort by closing price (Finviz 'Price' = last close / last trade)
    df = df.copy()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    missing = df['Price'].isna().sum()
    if missing:
        print(f"Skipping {missing} ticker(s) with no price.")
    df = df.dropna(subset=['Price']).sort_values('Price', kind='stable')
    tickers = df['Ticker'].tolist()

    # Split into 3 groups whose sizes differ by at most 1
    base, extra = divmod(len(tickers), 3)
    sizes = [base + (1 if i < extra else 0) for i in range(3)]
    low = tickers[:sizes[0]]
    mid = tickers[sizes[0]:sizes[0] + sizes[1]]
    high = tickers[sizes[0] + sizes[1]:]

    prices = df['Price'].tolist()
    def _range(start, size):
        if size == 0:
            return "n/a"
        return f"${prices[start]:.2f}–${prices[start + size - 1]:.2f}"

    print(f"Total Tickers ({len(tickers)}):")
    print(f"  Low  ({len(low)}): {_range(0, sizes[0])}")
    print(f"  Mid  ({len(mid)}): {_range(sizes[0], sizes[1])}")
    print(f"  High ({len(high)}): {_range(sizes[0] + sizes[1], sizes[2])}")
    return low, mid, high



my_filters = {
          'Country': 'USA',
          #'Market Cap.': '+Mid (over $2bln)',
          #'Market Cap.': '-Small (under $2bln)',
          'Market Cap.': '+Mid (over $2bln)',
          'Float Short': 'Under 20%',
          'Analyst Recom.': 'Hold or better',
          'Average Volume': 'Over 1M',
          #'P/E': 'Profitable (>0)',
          #'Forward P/E': 'Profitable (>0)',
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
lowTickers, midTickers, highTickers = fetch_tickers(my_filters)
filteredTickers = lowTickers + midTickers + highTickers
