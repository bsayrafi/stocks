
import io
import pandas as pd
import requests
from finvizfinance.screener.overview import Overview
from constants import *
from benchmarks import build_benchmark_map



def fetch_tickers(my_filters):
    """
    Fetch tickers from the Finviz screener and split them into three
    equal-sized groups by closing price.

    Args:
        my_filters (dict): Finviz screener filters.

    Returns:
        tuple[list, list, list]: (low_price, mid_price, high_price) ticker lists.
            Each list is sorted by price ascending. When the total isn't divisible
            by 3, the lower groups get the extra ticker(s), so sizes differ by
            at most 1.
    """

    # 1. Initialize the Overview screener
    screener = Overview()

    # 2. Apply filters
    screener.set_filter(filters_dict=my_filters)

    # 3. Fetch data
    # finvizfinance waits `sleep_sec` between result pages (20 tickers/page).
    # The default of 1s cost ~13s for ~250 tickers; 0.2s is usually fine. If
    # Finviz rejects the fast fetch (rate limit), retry once at the old 1s pace.
    fast_sleep = CONFIG.get("FINVIZ_PAGE_SLEEP_SEC", 0.2)
    try:
        df = screener.screener_view(sleep_sec=fast_sleep)
    except Exception as e:
        print(f"Finviz fetch at {fast_sleep}s/page failed ({e}); retrying at 1s/page")
        df = screener.screener_view(sleep_sec=1)
    if df is None or df.empty:
        print("No tickers matched the current filters.")
        return [], [], []

    # 4. Sort by closing price (Finviz 'Price' = last close / last trade)
    df = df.copy()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    missing = df['Price'].isna().sum()
    if missing:
        print(f"Skipping {missing} ticker(s) with no price.")
    df = df.dropna(subset=['Price']).sort_values('Price', kind='stable')
    tickers = df['Ticker'].tolist()

    # 5. Split into 3 groups whose sizes differ by at most 1
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




