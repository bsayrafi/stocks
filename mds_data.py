"""
mds_data.py
===========
Data layer for the Multi-Day Swing strategy. One normalized format everywhere:

  daily    : DatetimeIndex (tz-naive, normalized dates), columns Open High Low Close Volume
  intraday : DatetimeIndex (tz-aware America/New_York, bar START time),
             regular session only (09:30 <= t < 16:00), same columns

Two sources:
  - yfinance : near real-time, consolidated volume, 15m history ~60 days.
               Used by the live scanner.
  - Alpaca   : SIP (all exchanges) with a 16-minute buffer for the free plan,
               years of 15m history. Used by the backtest (parquet cache,
               resumable, timeout-protected -- same pattern as trend_data_pipeline).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd

from mds_config import MDS_CONFIG as CFG

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
ET = CFG["MARKET_TZ"]


# --------------------------------------------------------------------------- helpers

def _hhmm(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET)


def _normalize(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Title-case OHLCV columns, drop empty rows, sort, de-duplicate."""
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    if not set(OHLCV).issubset(df.columns):
        return None
    df = df[OHLCV].astype(float)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) else None


def to_daily_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
    df = _normalize(df)
    if df is None:
        return None
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(ET).tz_localize(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    return df[~df.index.duplicated(keep="last")]


def to_session_intraday(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """ET index, regular-session bars only."""
    df = _normalize(df)
    if df is None:
        return None
    idx = df.index
    idx = idx.tz_localize("UTC").tz_convert(ET) if idx.tz is None else idx.tz_convert(ET)
    df.index = idx
    mins = idx.hour * 60 + idx.minute
    keep = (mins >= _hhmm(CFG["SESSION_OPEN"])) & (mins < _hhmm(CFG["SESSION_CLOSE"]))
    df = df[keep]
    return df if len(df) else None


def drop_partial_daily(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """While the session is open today's daily bar is still forming -> drop it,
    so the daily setup is always computed on COMPLETED bars."""
    if df is None or df.empty:
        return df
    now = now_et()
    today = pd.Timestamp(now.date())
    if df.index[-1] == today and now.hour * 60 + now.minute < _hhmm(CFG["SESSION_CLOSE"]) + 15:
        return df.iloc[:-1]
    return df


def drop_forming_bar(df: pd.DataFrame | None, minutes: int = 15) -> pd.DataFrame | None:
    """Drop the last intraday bar if it has not closed yet."""
    if df is None or df.empty:
        return df
    if df.index[-1] + pd.Timedelta(minutes=minutes) > now_et():
        return df.iloc[:-1]
    return df


def run_with_timeout(fn, *args, timeout=45, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout), None
        except FutureTimeoutError:
            return None, f"timed out after {timeout}s"
        except Exception as e:  # noqa: BLE001
            return None, str(e)


# --------------------------------------------------------------------------- yfinance

def _yf_split(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame | None]:
    out = {}
    if raw is None or raw.empty:
        return {t: None for t in tickers}
    if not isinstance(raw.columns, pd.MultiIndex):
        return {tickers[0]: raw}
    lvl0 = set(raw.columns.get_level_values(0))
    for t in tickers:
        if t in lvl0:
            out[t] = raw[t]
        else:
            try:
                out[t] = raw.xs(t, axis=1, level=1)
            except KeyError:
                out[t] = None
    return out


def yf_daily(tickers: list[str], days: int | None = None, chunk: int = 100) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    days = days or CFG["DAILY_HISTORY_DAYS"]
    start = (now_et() - pd.Timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        raw = yf.download(part, start=start, interval="1d", auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
        for t, df in _yf_split(raw, part).items():
            df = drop_partial_daily(to_daily_index(df))
            if df is not None and len(df) > 60:
                out[t] = df
    return out


def yf_intraday(tickers: list[str], days: int | None = None, interval: str = "15m",
                chunk: int = 100) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    days = min(days or CFG["INTRADAY_HISTORY_DAYS"], 59)
    out: dict[str, pd.DataFrame] = {}
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        raw = yf.download(part, period=f"{days}d", interval=interval, auto_adjust=True,
                          prepost=False, group_by="ticker", threads=True, progress=False)
        for t, df in _yf_split(raw, part).items():
            df = drop_forming_bar(to_session_intraday(df))
            if df is not None and len(df):
                out[t] = df
    return out


# --------------------------------------------------------------------------- Alpaca

def _alpaca_client():
    from alpaca.data.historical import StockHistoricalDataClient
    key = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set (.env or environment)")
    return StockHistoricalDataClient(key, sec)


def _alpaca_timeframe(kind: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    return TimeFrame.Day if kind == "1d" else TimeFrame(15, TimeFrameUnit.Minute)


def alpaca_bars(symbols: list[str], kind: str, start: dt.datetime, end: dt.datetime | None = None,
                client=None) -> dict[str, pd.DataFrame]:
    """kind = '1d' or '15m'. SIP feed, split/dividend adjusted."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import DataFeed, Adjustment
    client = client or _alpaca_client()
    end = end or (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=CFG["ALPACA_SIP_BUFFER_MIN"]))
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=_alpaca_timeframe(kind),
                           start=start, end=end, adjustment=Adjustment.ALL, feed=DataFeed.SIP)
    bars = client.get_stock_bars(req).df
    out = {}
    if bars is None or bars.empty:
        return out
    for sym in symbols:
        try:
            df = bars.xs(sym, level=0) if isinstance(bars.index, pd.MultiIndex) else bars
        except KeyError:
            continue
        df = to_daily_index(df) if kind == "1d" else to_session_intraday(df)
        if df is not None:
            out[sym] = df
    return out


def alpaca_daily(tickers: list[str], days: int | None = None, chunk: int = 50) -> dict[str, pd.DataFrame]:
    days = days or CFG["DAILY_HISTORY_DAYS"]
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(days * 1.5))
    client = _alpaca_client()
    out = {}
    for i in range(0, len(tickers), chunk):
        res, err = run_with_timeout(alpaca_bars, tickers[i:i + chunk], "1d", start, None, client)
        if err:
            print(f"alpaca daily chunk {i}: {err}")
            continue
        out.update({k: drop_partial_daily(v) for k, v in res.items()})
    return out


def alpaca_intraday(tickers: list[str], days: int | None = None, chunk: int = 20) -> dict[str, pd.DataFrame]:
    days = days or CFG["INTRADAY_HISTORY_DAYS"]
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(days * 1.5))
    client = _alpaca_client()
    out = {}
    for i in range(0, len(tickers), chunk):
        res, err = run_with_timeout(alpaca_bars, tickers[i:i + chunk], "15m", start, None, client, timeout=120)
        if err:
            print(f"alpaca 15m chunk {i}: {err}")
            continue
        out.update(res)
    return out


# --------------------------------------------------------------------------- scanner entry points

def load_daily(tickers: list[str], source: str | None = None) -> dict[str, pd.DataFrame]:
    source = source or CFG["SCANNER_SOURCE"]
    return alpaca_daily(tickers) if source == "alpaca" else yf_daily(tickers)


def load_intraday(tickers: list[str], source: str | None = None, days: int | None = None) -> dict[str, pd.DataFrame]:
    source = source or CFG["SCANNER_SOURCE"]
    return alpaca_intraday(tickers, days) if source == "alpaca" else yf_intraday(tickers, days)


# --------------------------------------------------------------------------- backtest cache

def _cache_path(cache_dir: str, kind: str, sym: str) -> str:
    return os.path.join(cache_dir, kind, f"{sym}.parquet")


def build_cache(symbols: list[str], daily_years: float = 5, intraday_years: float = 3,
                cache_dir: str | None = None, refresh: bool = False, verbose: bool = True) -> None:
    """Fetch Alpaca SIP daily + 15m bars once per symbol into parquet files.
    Resumable: symbols already cached are skipped unless refresh=True."""
    cache_dir = cache_dir or os.path.join(CFG["CACHE_DIR"], "backtest")
    for kind in ("1d", "15m"):
        os.makedirs(os.path.join(cache_dir, kind), exist_ok=True)
    client = _alpaca_client()
    now = dt.datetime.now(dt.timezone.utc)
    spans = {"1d": now - dt.timedelta(days=int(365 * daily_years)),
             "15m": now - dt.timedelta(days=int(365 * intraday_years))}
    t0 = time.time()
    for n, sym in enumerate(symbols, 1):
        for kind, start in spans.items():
            path = _cache_path(cache_dir, kind, sym)
            if os.path.exists(path) and not refresh:
                continue
            res, err = run_with_timeout(alpaca_bars, [sym], kind, start, None, client, timeout=90)
            if err or not res or sym not in res:
                if verbose:
                    print(f"  [{sym}] {kind}: {err or 'no data'}")
                continue
            res[sym].to_parquet(path)
        if verbose and n % 10 == 0:
            print(f"  cached {n}/{len(symbols)} symbols ({time.time() - t0:.0f}s)")


def load_cache(symbols: list[str] | None = None, cache_dir: str | None = None
               ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    cache_dir = cache_dir or os.path.join(CFG["CACHE_DIR"], "backtest")
    daily, intra = {}, {}
    ddir = os.path.join(cache_dir, "1d")
    if symbols is None:
        symbols = sorted(f[:-8] for f in os.listdir(ddir) if f.endswith(".parquet")) if os.path.isdir(ddir) else []
    for sym in symbols:
        p1, p2 = _cache_path(cache_dir, "1d", sym), _cache_path(cache_dir, "15m", sym)
        if os.path.exists(p1):
            daily[sym] = pd.read_parquet(p1)
        if os.path.exists(p2):
            intra[sym] = pd.read_parquet(p2)
    return daily, intra


# --------------------------------------------------------------------------- universes

def finviz_universe(size: int = 2, extra_filters: dict | None = None) -> pd.DataFrame:
    """Finviz screen with MDS_CONFIG['FINVIZ_FILTERS'] + the market-cap bucket
    (same buckets as run_intraday_vwap / tickers.get_tickers). Returns
    DataFrame[Ticker, Sector, Industry, ...] -- the Sector column saves a
    yfinance .info call per ticker for the sector-rotation gate."""
    from finvizfinance.screener.overview import Overview
    filters = dict(CFG["FINVIZ_FILTERS"])
    filters["Market Cap."] = CFG["MARKET_CAP_OPTIONS"][size]
    if extra_filters:
        filters.update(extra_filters)
    scr = Overview()
    scr.set_filter(filters_dict=filters)
    try:
        df = scr.screener_view(sleep_sec=0.2)
    except Exception as e:  # noqa: BLE001
        print(f"finviz fast fetch failed ({e}); retrying at 1s/page")
        df = scr.screener_view(sleep_sec=1)
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Sector"])
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    return df.reset_index(drop=True)


def sp500_pit_universe(years: float = 3) -> tuple[list[str], dict[str, str]]:
    """Survivorship-free S&P 500 (every name that was a member at any point in
    the window) + GICS sector for the names still in the index."""
    from survivorship_free_universe import load_survivorship_free_universe, _load_current_sector_map
    gics = ["Information Technology", "Financials", "Health Care", "Consumer Discretionary",
            "Consumer Staples", "Energy", "Industrials", "Materials", "Real Estate", "Utilities",
            "Communication Services"]
    tickers, _ = load_survivorship_free_universe(sectors=gics, years=max(1, int(np.ceil(years))), verbose=True)
    smap = {k.replace(".", "-"): v for k, v in _load_current_sector_map().items()}
    return tickers, smap
