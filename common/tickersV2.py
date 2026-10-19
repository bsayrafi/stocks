
import io
import os
import re
import time
import json
import hashlib
from datetime import date, datetime
import pandas as pd
import requests
from finvizfinance.screener.overview import Overview
from common.constants import *
from common.benchmarks import build_benchmark_map
from common.cache_store import daily_cached


# The Finviz result is kept in the shared daily cache (cache_store.py) - the same folder and rules as the screener's
# cache. Location: CACHE_DIR / FINVIZ_CACHE_DIR environment variable (e.g. the GitHub Actions workflow), else
# CONFIG["CACHE_DIR"] / CONFIG["FINVIZ_CACHE_DIR"], else an "htmlv2cache" folder next to the scripts.
# One file per filter set per day, so changing the filters forces a refetch.


def _filters_key(my_filters):
    key = json.dumps(my_filters, sort_keys=True)
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


_FINVIZ_URL = "https://finviz.com/screener.ashx"
_FINVIZ_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
# Finviz redesigned the screener in Oct 2026 (screener.ashx now redirects to /screener/stocks).
#  - Rows are <div class="fv-table_row" data-row-key="TICKER"> with cells <div data-column-id="price">.
#    finvizfinance (<= 1.5.0) still looks for the old <select id="pageSelect"> and <table>, finds
#    neither, logs "No ticker found." and returns None - so every bucket came back empty.
#  - Without a login, only the first 1000 results of a query can be paged (r=1001 and later -> 403).
#    Bigger result sets are split by sector, and each part is read from both ends (asc + desc).
_ROW_RE = re.compile(r'<div class="fv-table_row"[^>]*\bdata-row-key="([^"]+)"(.*?)(?=<div class="fv-table_row"|$)', re.S)
_PRICE_RE = re.compile(r'data-column-id="price"[^>]*>\s*([^<]*)<')
_TOTAL_RE = re.compile(r'([\d,]+)\s*Total')
_MAX_RESULTS = 1000                    # deepest row Finviz serves without a login
_SECTORS = ["basicmaterials", "communicationservices", "consumercyclical", "consumerdefensive", "energy",
            "financial", "healthcare", "industrials", "realestate", "technology", "utilities"]


def _get_page(session, params, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(_FINVIZ_URL, params=params, headers=_FINVIZ_HEADERS, timeout=30)
            if r.status_code in (403, 429) or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = (10, 30, 60)[min(attempt, 2)]          # 403/429 here = rate limit: back off properly
            print(f"  Finviz page r={params.get('r', 1)} failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def _parse_rows(html):
    rows = []
    for ticker, body in _ROW_RE.findall(html):
        m = _PRICE_RE.search(body)
        price = m.group(1).strip().replace(",", "") if m else None
        rows.append({"Ticker": ticker, "Price": price})
    return rows


def _parse_total(html, rows):
    m = _TOTAL_RE.search(html)
    return int(m.group(1).replace(",", "")) if m else len(rows)


class _LayoutError(Exception):
    pass


def _fetch_query(session, f_codes, sleep_sec, first_html=None):
    """All rows for one filter string. Reads up to 1000 rows ascending, then (if needed) up to
    1000 more descending, so a single query can return up to 2000 rows."""
    collected = {}
    total = None
    for order in ("ticker", "-ticker"):
        params = {"v": 111, "f": f_codes, "o": order}
        offset = 1
        while offset <= _MAX_RESULTS:
            if first_html is not None and order == "ticker" and offset == 1:
                html = first_html
            else:
                time.sleep(sleep_sec)
                html = _get_page(session, {**params, "r": offset})
            rows = _parse_rows(html)
            if total is None:
                total = _parse_total(html, rows)
                if total and not rows:
                    raise _LayoutError(f"Finviz reports {total} matches but no rows could be parsed")
            if not rows:
                break
            new = 0
            for r in rows:
                if r["Ticker"] not in collected:
                    collected[r["Ticker"]] = r
                    new += 1
            if len(collected) >= total or (order == "-ticker" and new == 0):
                break
            offset += len(rows)
        if len(collected) >= (total or 0):
            break
    return list(collected.values()), (total or 0)


def _fetch_screener_df_v2(my_filters):
    """Scrape the redesigned Finviz screener directly. Returns a DataFrame (possibly empty
    when Finviz says 0 matches), or None when the page layout isn't recognised."""
    screener = Overview()                       # only used to turn the filter dict into URL codes
    screener.set_filter(filters_dict=my_filters)
    f_codes = screener.request_params.get("f", "")
    sleep_sec = CONFIG.get("FINVIZ_PAGE_SLEEP_SEC", 0.3)

    with requests.Session() as session:
        html = _get_page(session, {"v": 111, "f": f_codes, "o": "ticker"})
        rows = _parse_rows(html)
        if not rows and not _TOTAL_RE.search(html):
            print(f"  Finviz returned an unrecognised page ({len(html)} bytes) - layout changed or blocked.")
            return None
        total = _parse_total(html, rows)
        print(f"  Finviz: {total} matches")
        if total == 0:
            return pd.DataFrame(columns=["Ticker", "Price"])

        if total <= _MAX_RESULTS or "Sector" in my_filters:
            all_rows, _ = _fetch_query(session, f_codes, sleep_sec, first_html=html)
        else:
            all_rows, seen = [], set()
            for sec in _SECTORS:
                part, part_total = _fetch_query(session, f"{f_codes},sec_{sec}" if f_codes else f"sec_{sec}", sleep_sec)
                if len(part) < part_total:
                    print(f"  WARNING: sector {sec}: got {len(part)} of {part_total}")
                all_rows += [r for r in part if r["Ticker"] not in seen]
                seen.update(r["Ticker"] for r in part)
                print(f"  Finviz: {len(all_rows)}/{total} tickers (after {sec})")

    if len(all_rows) < total:
        print(f"  WARNING: Finviz reported {total} matches but only {len(all_rows)} were fetched "
              f"(free accounts can page through at most {_MAX_RESULTS} results per query).")
    return pd.DataFrame(all_rows)


def _fetch_screener_df(my_filters):
    """Hit Finviz and return the screener DataFrame (or None/empty)."""
    try:
        df = _fetch_screener_df_v2(my_filters)
        if df is not None:
            return df
    except Exception as e:
        print(f"Finviz direct scrape failed ({type(e).__name__}: {e}); trying finvizfinance")

    # Fallback: the finvizfinance library (only understands the old Finviz layout).
    screener = Overview()
    screener.set_filter(filters_dict=my_filters)
    fast_sleep = CONFIG.get("FINVIZ_PAGE_SLEEP_SEC", 0.3)
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
    fetched = []

    def fetch():
        print("cache miss; fetching from Finviz...")
        fetched.append(True)
        raw = _fetch_screener_df(my_filters)
        if raw is None or raw.empty:
            return None
        return {"fetched_at": datetime.now().isoformat(timespec="seconds"), "filters": my_filters,
                "rows": raw[["Ticker", "Price"]].to_dict(orient="records")}

    payload = daily_cached("finviz", _filters_key(my_filters), fetch,
                           is_valid=lambda p: bool(p and p.get("rows")), refresh=force_refresh)
    if not payload or not payload.get("rows"):
        print("No tickers matched the current filters.")
        return [], [], []
    df = pd.DataFrame(payload["rows"])
    if not fetched:
        print(f"Using cached Finviz results from {payload.get('fetched_at')} ({len(df)} tickers)")

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
