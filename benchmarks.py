"""
benchmarks.py
Maps each ticker to the ETF it should be compared against, using the finviz
Sector and Industry fields.

Order of precedence:
  1. OVERRIDES      - mega-caps that dominate their own sector/industry ETF
                      (comparing them with that ETF is partly comparing them
                      with themselves)
  2. INDUSTRY_ETF   - a focused industry ETF where a good one exists
  3. SECTOR_ETF     - the SPDR sector ETF
  4. MARKET         - SPY

Usage:
    from benchmarks import get_benchmark_map
    bench_map = get_benchmark_map(["ADI", "AMD", "A"])   # {"ADI": "SMH", "A": "XLV", ...}

Sector/industry come from Finviz (one screener request per 20 tickers) and are
cached in a JSON file, so repeat runs and backtests don't hit Finviz.
If you already have a finviz screener df, build_benchmark_map(df) also works.

Names below are finviz's industry/sector strings. Unknown industries simply
fall back to the sector ETF, so the table can be extended over time.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd

MARKET = "SPY"

SECTOR_ETF = {
    "Technology": "XLK",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
}

INDUSTRY_ETF = {
    # Technology
    "Semiconductors": "SMH",
    "Semiconductor Equipment & Materials": "SMH",
    "Software - Application": "IGV",
    "Software - Infrastructure": "IGV",
    "Solar": "TAN",
    # Healthcare
    "Biotechnology": "XBI",
    "Medical Devices": "IHI",
    "Medical Instruments & Supplies": "IHI",
    "Drug Manufacturers - Specialty & Generic": "XPH",
    # Financial
    "Banks - Regional": "KRE",
    "Banks - Diversified": "KBE",
    "Insurance - Property & Casualty": "KIE",
    "Insurance - Life": "KIE",
    "Insurance - Diversified": "KIE",
    # Energy
    "Oil & Gas E&P": "XOP",
    "Oil & Gas Equipment & Services": "OIH",
    # Industrials
    "Aerospace & Defense": "ITA",
    "Airlines": "JETS",
    "Railroads": "IYT",
    "Trucking": "IYT",
    "Integrated Freight & Logistics": "IYT",
    # Consumer
    "Residential Construction": "XHB",
    "Specialty Retail": "XRT",
    "Apparel Retail": "XRT",
    "Department Stores": "XRT",
    # Materials
    "Gold": "GDX",
    "Steel": "SLX",
    "Copper": "COPX",
}

# Mega-caps with large weights in their own sector/industry ETF.
OVERRIDES = {
    "AAPL": "QQQ", "MSFT": "QQQ", "NVDA": "QQQ", "AMZN": "QQQ",
    "GOOGL": "QQQ", "GOOG": "QQQ", "META": "QQQ", "TSLA": "QQQ",
    "AVGO": "QQQ",
    "BRK-B": "SPY", "BRK.B": "SPY", "JPM": "SPY", "LLY": "SPY",
    "XOM": "SPY", "CVX": "SPY",
}


def get_benchmark(ticker: str, sector: str | None = None, industry: str | None = None) -> str:
    t = str(ticker).strip().upper()
    if t in OVERRIDES:
        return OVERRIDES[t]
    if industry and industry in INDUSTRY_ETF:
        return INDUSTRY_ETF[industry]
    if sector and sector in SECTOR_ETF:
        return SECTOR_ETF[sector]
    return MARKET


def build_benchmark_map(df: pd.DataFrame, ticker_col: str = "Ticker",
                        sector_col: str = "Sector", industry_col: str = "Industry") -> dict:
    """{ticker: benchmark ETF} from a finviz screener DataFrame."""
    out = {}
    for _, r in df.iterrows():
        t = str(r[ticker_col]).strip().upper()
        out[t] = get_benchmark(t, r.get(sector_col), r.get(industry_col))
    return out


def benchmark_symbols(bench_map: dict) -> list[str]:
    """All ETFs needed, including SPY (used for the sector-vs-market score)."""
    return sorted(set(bench_map.values()) | {MARKET})


# --------------------------------------------------------------------------
# Fetching sector/industry from Finviz, with a JSON cache
# --------------------------------------------------------------------------
DEFAULT_CACHE = "sector_cache.json"
CACHE_MAX_AGE_DAYS = 30
FINVIZ_CHUNK = 60   # tickers per screener request (URL stays short)


def _to_finviz(t: str) -> str:
    return t.replace(".", "-")          # BRK.B (Alpaca) -> BRK-B (Finviz)


def _load_cache(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(path: str, cache: dict) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def fetch_sector_industry(tickers, cache_path: str | None = DEFAULT_CACHE,
                          max_age_days: int = CACHE_MAX_AGE_DAYS,
                          sleep_sec: float = 0.5, refresh: bool = False) -> dict:
    """{ticker: {"sector": ..., "industry": ..., "company": ...}} for the given
    tickers. Uses the cache where fresh; fetches the rest from Finviz.
    Tickers Finviz doesn't know (e.g. ETFs, typos) are omitted from the result.
    """
    from finvizfinance.screener.overview import Overview

    tickers = sorted({str(t).strip().upper() for t in tickers if t and str(t).strip()})
    cache = {} if refresh else _load_cache(cache_path)
    cutoff = datetime.now() - timedelta(days=max_age_days)

    def fresh(t):
        e = cache.get(t)
        return e is not None and datetime.fromisoformat(e["updated"]) >= cutoff

    todo = [t for t in tickers if not fresh(t)]
    if todo:
        back = {_to_finviz(t): t for t in todo}
        for i in range(0, len(todo), FINVIZ_CHUNK):
            chunk = [_to_finviz(t) for t in todo[i:i + FINVIZ_CHUNK]]
            try:
                sc = Overview()
                sc.set_filter(ticker=",".join(chunk))
                df = sc.screener_view(verbose=0, sleep_sec=sleep_sec)
            except Exception as e:
                print(f"Finviz lookup failed for {len(chunk)} tickers ({e}); "
                      f"using cached/SPY fallback for them.")
                continue
            if df is None or df.empty:
                continue
            now = datetime.now().isoformat(timespec="seconds")
            for _, r in df.iterrows():
                t = back.get(str(r["Ticker"]).upper(), str(r["Ticker"]).upper())
                cache[t] = {"sector": r.get("Sector"), "industry": r.get("Industry"),
                            "company": r.get("Company"), "updated": now}
            if i + FINVIZ_CHUNK < len(todo):
                time.sleep(sleep_sec)
        _save_cache(cache_path, cache)

    return {t: cache[t] for t in tickers if t in cache}


def get_benchmark_map(tickers, cache_path: str | None = DEFAULT_CACHE,
                      verbose: bool = True, **kwargs) -> dict:
    """{ticker: benchmark ETF} for a plain ticker list. Tickers without
    sector data fall back to SPY (overrides still apply)."""
    tickers = sorted({str(t).strip().upper() for t in tickers if t and str(t).strip()})
    info = fetch_sector_industry(tickers, cache_path=cache_path, **kwargs)
    out, unknown = {}, []
    for t in tickers:
        i = info.get(t, {})
        if not i and t not in OVERRIDES:
            unknown.append(t)
        out[t] = get_benchmark(t, i.get("sector"), i.get("industry"))
    if verbose and unknown:
        print(f"No sector data for {', '.join(unknown)}; compared with {MARKET}.")
    return out
