"""
 Multi-Timeframe Buy Signal Screener
------------------------------------------------------
Fetches hourly price data, builds 4h and daily bars from it, and combines
five indicator families into a single buy signal:

  - Stochastic Oscillator : momentum / overbought-oversold, per timeframe
  - Moving Average filter  : trend confirmation (price vs. MA), per timeframe
  - Candlestick patterns   : structural reversal/continuation confirmation,
                              per timeframe (not just oscillator math)
  - MACD                   : momentum crossover confirmation, on the 1h entry TF
  - Volume confirmation    : today's volume vs. its rolling average, on 1h
  - ATR                    : not a signal input, used only to size a stop

Signal logic:
  HARD requirements (all must pass):
    - Daily & 4h both trending up (price > MA) with bullish stochastic
    - A bullish candlestick pattern confirmed on daily OR 4h
      (structural confirmation, not just indicator math)

  SOFT conditions (need >= SCORE_THRESHOLD out of 5):
    - 1h not overbought
    - 1h MACD bullish
    - volume confirmed
    - near support
    - 1h bullish candlestick pattern present (precise entry timing)

  BUY only if hard requirements pass AND soft score threshold is met.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import html
import os
import requests
import datetime as dt
try:
    import finnhub  # pip install finnhub-python
except ImportError:
    finnhub = None
import constants

# Load a local .env file (VS Code / your own machine) into the environment, so
# NTFY_TOPIC, APCA_*, FINNHUB_API_KEY... are found there too. Variables that are
# already set (e.g. GitHub Actions secrets) are NOT overridden.
# Needs: pip install python-dotenv  (optional - skipped if not installed)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))   # .env in the folder you run from (or a parent)
    load_dotenv()                            # .env next to this file (or a parent)
except ImportError:
    pass

# API keys live in constants.CONFIG (kept out of this file):
#   CONFIG = {"ALPACA_HEADERS": {"APCA-API-KEY-ID": ..., "APCA-API-SECRET-KEY": ...},
#             "FINNHUB_API_KEY": ...}
_SECRETS = getattr(constants, "CONFIG", {}) or {}
from datetime import datetime, timedelta, timezone
from event_catalysts import *
from event_catalysts import _get_sp500_membership
from event_catalysts import fetch_apewisdom_table, get_earnings_and_ratings
import time
import functools
import pickle
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------- profiling
# Records how long each network call / step takes per ticker, and main()
# prints a summary at the end. Turn off with CONFIGH["PROFILE"] = False.
PROFILE: dict = {}   # step name -> list of (ticker, seconds)


def _prof_add(step: str, ticker: str, seconds: float) -> None:
    PROFILE.setdefault(step, []).append((ticker, seconds))


def _profiled(step: str):
    """Decorator: time every call of the function under `step`. The ticker is
    taken from the first argument (a ticker string, or a report dict)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                first = args[0] if args else kwargs.get("ticker", "")
                ticker = (first.get("ticker", "") if isinstance(first, dict)
                          else "ALL" if isinstance(first, (list, tuple)) else str(first))
                _prof_add(step, ticker, time.perf_counter() - t0)
        return wrapper
    return deco


# functions imported from event_catalysts are wrapped here so they're timed too
get_event_catalysts = _profiled("event catalysts (event_catalysts.py)")(get_event_catalysts)
_get_sp500_membership = _profiled("S&P 500 list (once per run)")(_get_sp500_membership)
fetch_apewisdom_table = _profiled("ApeWisdom ranking (once per run)")(fetch_apewisdom_table)
get_earnings_and_ratings = _profiled("earnings date + upgrades (yf)")(get_earnings_and_ratings)


# ---------------------------------------------------------------- daily disk cache
# Data that changes at most once a day (yfinance .info, analyst data, earnings
# date + rating actions, S&P 500 list) is saved to CACHE_DIR, one file per
# ticker per day. A second run the same day reads it from disk instead of the
# network. Files from earlier days are deleted automatically.
_cache_lock = threading.Lock()


def _cache_path(name: str, key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return os.path.join(CONFIGH.get("CACHE_DIR", "cache"), f"{dt.date.today():%Y%m%d}_{name}_{safe}.pkl")


def daily_cached(name: str, key: str, fn, is_valid=bool):
    """Return today's cached value for (name, key), or call fn(), cache the
    result if is_valid(result), and return it. Failures are never cached."""
    if not CONFIGH.get("CACHE_ENABLED", True):
        return fn()
    path = _cache_path(name, key)
    try:
        with open(path, "rb") as f:
            value = pickle.load(f)
        _prof_add("  cache hits (read from disk)", key, 0.0)
        return value
    except (OSError, pickle.PickleError, EOFError):
        pass
    value = fn()
    if is_valid(value):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(value, f)
            os.replace(tmp, path)   # atomic, so parallel workers never read half a file
        except OSError:
            pass
    return value


def ttl_cached(name: str, key: str, fn, ttl_minutes: float, is_valid=lambda v: v is not None):
    """Like daily_cached, but a cached value is only reused while it is younger
    than ttl_minutes (e.g. news: 60). The fetch time is stored INSIDE the cache
    file, so the age stays correct even when the folder is copied or restored
    (e.g. GitHub Actions' cache), which can reset file modification times."""
    if not CONFIGH.get("CACHE_ENABLED", True) or not ttl_minutes:
        return fn()
    path = _cache_path(name, key)
    try:
        with open(path, "rb") as f:
            saved_at, value = pickle.load(f)
        if time.time() - saved_at < ttl_minutes * 60:
            _prof_add("  cache hits (read from disk)", key, 0.0)
            return value
    except (OSError, pickle.PickleError, EOFError, TypeError, ValueError):
        pass
    value = fn()
    if is_valid(value):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump((time.time(), value), f)
            os.replace(tmp, path)
        except OSError:
            pass
    return value


def prune_cache() -> None:
    """Delete cache files from previous days."""
    d = CONFIGH.get("CACHE_DIR", "cache")
    today = f"{dt.date.today():%Y%m%d}_"
    try:
        for fn in os.listdir(d):
            if fn.endswith(".pkl") and not fn.startswith(today):
                try:
                    os.remove(os.path.join(d, fn))
                except OSError:
                    pass
    except OSError:
        pass


# ---------------------------------------------------------------- Finnhub rate limit
class _RateLimiter:
    """Thread-safe: allow at most `per_minute` calls in any 60-second window."""
    def __init__(self, per_minute: int):
        self.per_minute = max(1, int(per_minute))
        self.calls = deque()
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                if len(self.calls) < self.per_minute:
                    self.calls.append(now)
                    return
                sleep_for = 60 - (now - self.calls[0]) + 0.05
            time.sleep(sleep_for)


_finnhub_limiter = None  # created on first use from CONFIGH["FINNHUB_MAX_PER_MIN"]
_finnhub_limiter_lock = threading.Lock()


def _finnhub_wait() -> None:
    global _finnhub_limiter
    with _finnhub_limiter_lock:
        if _finnhub_limiter is None:
            _finnhub_limiter = _RateLimiter(CONFIGH.get("FINNHUB_MAX_PER_MIN", 55))
    _finnhub_limiter.wait()


def print_profile_summary(wall_seconds: float, n_tickers: int) -> None:
    """Print where the time went, biggest step first."""
    if not PROFILE:
        return
    rows = []
    for step, calls in PROFILE.items():
        total = sum(sec for _, sec in calls)
        slow_t, slow_s = max(calls, key=lambda c: c[1])
        rows.append((step, total, len(calls), total / len(calls), slow_t, slow_s))
    rows.sort(key=lambda r: -r[1])
    work = sum(r[1] for r in rows if not r[0].startswith("  ") and r[0] != "whole ticker (run_screen)")

    print(f"\n=== Timing summary: {n_tickers} tickers, {wall_seconds:.1f}s wall time "
          f"({wall_seconds / max(n_tickers, 1):.1f}s per ticker) ===")
    print(f"total work {work:.1f}s across {CONFIGH.get('MAX_WORKERS', 1)} parallel workers "
          f"(steps overlap, so work > wall time)")
    print(f"{'step':<40}{'total s':>9}{'% work':>8}{'calls':>7}{'avg s':>8}   slowest")
    for step, total, n, avg, slow_t, slow_s in rows:
        if step == "whole ticker (run_screen)":
            continue
        print(f"{step:<40}{total:>9.1f}{total / max(work, 1e-9) * 100:>7.1f}%{n:>7}{avg:>8.2f}"
              f"   {slow_t} {slow_s:.1f}s")
    per_ticker = sorted(PROFILE.get("whole ticker (run_screen)", []), key=lambda c: -c[1])[:5]
    if per_ticker:
        print("slowest tickers: " + ", ".join(f"{t} {sec:.1f}s" for t, sec in per_ticker))
    print("(sub-steps marked '  analyst:' are included in 'analyst data')\n")


CONFIGH = {
    "TICKER": "ORCL",
    "PERIOD": "200d",              # max for 1h interval on yfinance is 730d, 60d is plenty here
    "INTERVAL": "1h",

    # Stochastic
    "STOCH_K_PERIOD": 14,
    "STOCH_D_PERIOD": 3,
    "STOCH_SMOOTH": 3,
    "OVERSOLD": 20,
    "OVERBOUGHT": 80,

    # Moving average trend filter (applied per timeframe)
    "MA_PERIOD": 50,
    "MA_TYPE": "ema",              # "ema" or "sma"

    # MACD (applied on the 1h entry timeframe)
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,

    # Volume confirmation (applied on 1h)
    "VOLUME_MA_PERIOD": 20,
    "VOLUME_MULTIPLIER": 1.0,      # current volume must be >= MULTIPLIER * its rolling avg

    # ATR (for stop sizing only, not part of the signal)
    "ATR_PERIOD": 14,
    "ATR_STOP_MULTIPLIER": 2.0,

    # Candlestick pattern detection
    "CANDLE_BODY_RATIO": 0.3,      # body must be <= this fraction of range to count as "small body"
    "CANDLE_WICK_RATIO": 2.0,      # wick must be >= this multiple of body to count as "long wick"
    "CANDLE_BIG_BODY_RATIO": 0.5,  # body must be >= this fraction of range to count as "big body"

    # Support proximity (used by the signal logic) - the level itself is computed
    # dynamically each run via swing_support_resistance() on the 1h timeframe,
    # not hardcoded.
    "SUPPORT_BUFFER_PCT": 0.5,     # % distance from support still counted as "near"

    # Support/resistance swing lookback, per timeframe
    "SR_LOOKBACK": {"1D": 10, "4h": 20, "1h": 20},

    "SCORE_THRESHOLD": 3,          # out of 5 soft conditions

    # HTML report price chart (display only, not part of the signal)
    "NTFY_ENABLED": True,          # push the report(s) to your phone via ntfy (needs NTFY_TOPIC)
    "NTFY_FILES": ["up"],          # which files to send: any of "up", "down", "csv"
    "NTFY_MAX_MB": 15,             # ntfy.sh attachment limit; bigger files are announced without the file
    "CSV_REPORT": True,            # also write reports/<name>_signal_report_<timestamp>.csv (1 row per ticker)
    "PROFILE": True,               # print a timing breakdown at the end of main()
    "MAX_WORKERS": 4,              # tickers screened in parallel (1 = one at a time)
    "CACHE_ENABLED": True,         # cache once-a-day data (.info, analyst, earnings...) on disk
    "CACHE_DIR": "cache",
    "FINNHUB_MAX_PER_MIN": 55,     # Finnhub free tier allows 60 calls/minute
    "CATALYST_NEWS_DAYS": 14,      # headlines scanned for buyback/guidance keywords
    "HHHL_DAYS": 20,               # window for the higher-high/higher-low day count (completed daily bars)
    "DAILY_CHART_DAYS": 50,        # daily candles in the second (daily) chart
    # Daily support/resistance = industry-standard swing levels (see swing_sr_levels)
    "SR_DAYS": 60,                 # daily bars searched for swing support/resistance (~3 months; max ~130)
    "NEAR_SR_ATR": 0.5,            # "Near S1/S2" = price within this many daily ATRs of the level
    "SR_PIVOT": 2,                 # a swing low/high must be the lowest/highest of this many days on each side
    "SR_MERGE_ATR": 0.5,           # swing points within this many daily ATRs are merged into one level
    "TARGET_MIN_R": 2.0,           # targets below this reward:risk are flagged; fallback target when no resistance
    "AVWAP_ANCHOR": "low",         # anchored VWAP starts at the daily chart's lowest low ("low") or highest high ("high")
    "DAILY_CHART_OPEN": False,     # daily chart collapsed by default (its own show/hide link)
    "CHART_OPEN": False,           # chart collapsed by default (each card has a show/hide link)
    "CHART_DAYS": 7,               # number of recent trading days of 1h candles to plot
    "CHART_MAS": [("ema", 9), ("ema", 50)],  # MA overlays drawn on the chart
    "POC_BINS": 50,                # price buckets for the volume profile / POC
    "MARKET_TZ": "America/New_York",
    "MARKET_OPEN": "09:30",        # regular session, local to MARKET_TZ
    "MARKET_CLOSE": "16:00",

    # Linear regression channel on the chart (display only, not part of the signal)
    "LRC_ENABLED": True,
    "LRC_LENGTH": None,            # bars to fit (1h bars); None = the whole chart window
    "LRC_DEV": 2.0,                # channel half-width, in std devs of the residuals
    "LRC_MIN_R2_UP": 0.5,          # rising channels with R-squared below this go to the _down report
    "LRC_SOURCE": "Close",         # price column to fit: "Close", "High", "Low", "Open"

    # Pre-market price from Alpaca (display only). Keys come from the `headers`
    # passed to get_pre_market_price(), or the APCA_API_KEY_ID /
    # APCA_API_SECRET_KEY environment variables.
    "PREMARKET_ENABLED": True,
    "PREMARKET_START": "04:00",    # pre-market session start, MARKET_TZ
    "ALPACA_FEED": "auto",         # "auto" = pick SIP or IEX (see get_best_pre_market_price),
                                   # "sip" = all exchanges, delayed; "iex" = live but IEX-only
    "ALPACA_SIP_DELAY_MIN": 16,    # free plan: SIP must be >=15 min old; 0 if you pay for live SIP
    "PREMARKET_MOVE_PCT": 0.3,     # auto: use live IEX if price moved >= this % since the SIP snapshot
    "PREMARKET_IEX_MAX_GAP_PCT": 0.5,  # auto: distrust IEX if it differs from SIP by > this % at the same time
    "ALPACA_HEADERS": _SECRETS.get("ALPACA_HEADERS"),   # from constants.CONFIG

    # Company news (Finnhub) in the Event Catalysts section, collapsed by default.
    # Key: here, or the FINNHUB_API_KEY environment variable.
    "FINNHUB_API_KEY": _SECRETS.get("FINNHUB_API_KEY"),  # from constants.CONFIG
    "NEWS_DAYS": 0,                # 0 = today only, 1 = today + yesterday, ...
    "NEWS_MAX": 25,                # max articles shown per ticker (newest first)
    "NEWS_TZ": "America/New_York", # time zone for the article times shown
    "NEWS_FOR": "up",              # Finnhub news for: "up" = only tickers shown in the _up report,
                                   # "all" = every ticker shown, "none" = no news at all
    "NEWS_CACHE_MIN": 60,          # reuse a ticker's news for this many minutes (0 = always fetch)
}


# ---------------------------------------------------------------- data
# 2. Update fetch_hourly_data to flatten MultiIndex columns from yfinance
@_profiled("yfinance 1h download")
def fetch_hourly_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    return df


@_profiled("yfinance 1h batch download (all tickers)")
def prefetch_hourly_data(tickers: list, period: str, interval: str) -> dict:
    """Download 1h bars for ALL tickers in one yf.download call (yfinance
    fetches them concurrently). Returns {ticker: DataFrame}; tickers with no
    data are left out, so run_screen falls back to a single download / error.
    Done up front because yf.download is not safe to call from several threads."""
    if not tickers:
        return {}
    data = yf.download(list(tickers), period=period, interval=interval, auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        df = df.dropna(how="all")
        if not df.empty:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            out[t] = df
    return out


def prefetch_pre_market(tickers: list, cfg: dict) -> dict | None:
    """Pre-market prices for ALL tickers in a few batched Alpaca requests
    (instead of 1-2 per ticker, which would hit Alpaca's 200 requests/minute
    free limit on big lists). Returns {ticker: details} or None if disabled."""
    if not cfg.get("PREMARKET_ENABLED", True):
        return None
    feed = cfg.get("ALPACA_FEED", "auto")
    kw = dict(headers=cfg.get("ALPACA_HEADERS"), tz=cfg.get("MARKET_TZ", "America/New_York"),
              start_hhmm=cfg.get("PREMARKET_START", "04:00"), open_hhmm=cfg.get("MARKET_OPEN", "09:30"))
    if feed == "auto":
        return get_best_pre_market_prices(
            tickers, sip_delay_minutes=cfg.get("ALPACA_SIP_DELAY_MIN", 16),
            move_pct=cfg.get("PREMARKET_MOVE_PCT", 0.3),
            max_iex_gap_pct=cfg.get("PREMARKET_IEX_MAX_GAP_PCT", 0.5), **kw)
    return get_pre_market_prices(
        tickers, feed=feed, delay_minutes=cfg.get("ALPACA_SIP_DELAY_MIN", 16) if feed == "sip" else 0, **kw)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


# ---------------------------------------------------------------- indicators

def stochastic_oscillator(df: pd.DataFrame, k_period: int, d_period: int, smooth: int) -> pd.DataFrame:
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    raw_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    k = raw_k.rolling(smooth).mean()   # slow %K
    d = k.rolling(d_period).mean()     # %D
    return pd.DataFrame({"%K": k, "%D": d}, index=df.index)


def moving_average(df: pd.DataFrame, period: int, ma_type: str) -> pd.Series:
    if ma_type == "ema":
        return df["Close"].ewm(span=period, adjust=False).mean()
    return df["Close"].rolling(period).mean()


def macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist}, index=df.index)


def average_true_range(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


# ---------------------------------------------------------------- candlestick patterns
# Each function takes the OHLC dataframe and looks at the LAST candle(s).
# Returns True if that specific bullish pattern is present right now.

def is_bullish_engulfing(df: pd.DataFrame, cfg: dict) -> bool:
    prev, curr = df.iloc[-2], df.iloc[-1]
    prev_bearish = prev["Close"] < prev["Open"]
    curr_bullish = curr["Close"] > curr["Open"]
    engulfs = curr["Open"] <= prev["Close"] and curr["Close"] >= prev["Open"]
    return bool(prev_bearish and curr_bullish and engulfs)


def is_bullish_harami(df: pd.DataFrame, cfg: dict) -> bool:
    prev, curr = df.iloc[-2], df.iloc[-1]
    prev_bearish = prev["Close"] < prev["Open"]
    curr_bullish = curr["Close"] > curr["Open"]
    inside = curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"]
    return bool(prev_bearish and curr_bullish and inside)


def is_piercing_line(df: pd.DataFrame, cfg: dict) -> bool:
    prev, curr = df.iloc[-2], df.iloc[-1]
    prev_bearish = prev["Close"] < prev["Open"]
    curr_bullish = curr["Close"] > curr["Open"]
    midpoint = (prev["Open"] + prev["Close"]) / 2
    opens_below_prev_close = curr["Open"] < prev["Close"]
    closes_above_mid = midpoint < curr["Close"] < prev["Open"]
    return bool(prev_bearish and curr_bullish and opens_below_prev_close and closes_above_mid)


def is_hammer(df: pd.DataFrame, cfg: dict) -> bool:
    row = df.iloc[-1]
    rng = row["High"] - row["Low"]
    if rng == 0:
        return False
    body = abs(row["Close"] - row["Open"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    small_body = body <= cfg["CANDLE_BODY_RATIO"] * rng
    long_lower_wick = lower_wick >= cfg["CANDLE_WICK_RATIO"] * max(body, rng * 0.05)
    small_upper_wick = upper_wick <= cfg["CANDLE_BODY_RATIO"] * rng
    return bool(small_body and long_lower_wick and small_upper_wick)


def is_morning_star(df: pd.DataFrame, cfg: dict) -> bool:
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    c1_range = c1["High"] - c1["Low"]
    c3_range = c3["High"] - c3["Low"]
    if c1_range == 0 or c3_range == 0:
        return False
    c1_body = abs(c1["Close"] - c1["Open"])
    c2_body = abs(c2["Close"] - c2["Open"])
    c3_body = abs(c3["Close"] - c3["Open"])

    c1_bearish_big = c1["Close"] < c1["Open"] and c1_body >= cfg["CANDLE_BIG_BODY_RATIO"] * c1_range
    c2_small_body = c2_body < cfg["CANDLE_BODY_RATIO"] * max(c1_body, 0.01)
    c2_gaps_down = max(c2["Open"], c2["Close"]) < c1["Close"]
    c3_bullish_big = c3["Close"] > c3["Open"] and c3_body >= cfg["CANDLE_BIG_BODY_RATIO"] * c3_range
    c3_closes_into_c1_body = c3["Close"] > (c1["Open"] + c1["Close"]) / 2

    return bool(c1_bearish_big and c2_small_body and c2_gaps_down and c3_bullish_big and c3_closes_into_c1_body)


# Registry so new patterns can be added without touching the detection loop
BULLISH_PATTERNS = {
    "bullish_engulfing": is_bullish_engulfing,
    "bullish_harami": is_bullish_harami,
    "piercing_line": is_piercing_line,
    "hammer": is_hammer,
    "morning_star": is_morning_star,
}


def detect_candlestick_patterns(df: pd.DataFrame, cfg: dict) -> dict:
    if len(df) < 3:
        return {"patterns_detected": [], "bullish_pattern": False}

    detected = [name for name, fn in BULLISH_PATTERNS.items() if fn(df, cfg)]
    return {"patterns_detected": detected, "bullish_pattern": len(detected) > 0}


# ---------------------------------------------------------------- evaluation

def evaluate_timeframe(df: pd.DataFrame, stoch: pd.DataFrame, ma: pd.Series, cfg: dict) -> dict:
    latest_close = float(df["Close"].iloc[-1])
    latest_k = stoch["%K"].iloc[-1]
    prev_k = stoch["%K"].iloc[-2]
    latest_d = stoch["%D"].iloc[-1]
    latest_ma = ma.iloc[-1]

    rising = bool(latest_k > prev_k)
    oversold_bounce = bool((stoch["%K"].iloc[-5:].min() <= cfg["OVERSOLD"]) and rising)
    overbought_now = bool(latest_k >= cfg["OVERBOUGHT"])
    trend_up = bool(latest_close > latest_ma)
    # %K crossing/holding above %D is a classic stochastic confirmation signal,
    # shown alongside %K/%D but not currently part of the hard/soft signal logic.
    k_above_d = bool(latest_k > latest_d)

    candles = detect_candlestick_patterns(df, cfg)

    return {
        "close": round(latest_close, 2),
        "%K": round(latest_k, 2),
        "%D": round(latest_d, 2),
        "k_above_d": k_above_d,
        "MA": round(latest_ma, 2),
        "trend_up": trend_up,
        "rising": rising,
        "oversold_bounce": oversold_bounce,
        "overbought": overbought_now,
        "stoch_bullish": bool(oversold_bounce or rising),
        "patterns_detected": candles["patterns_detected"],
        "bullish_pattern": candles["bullish_pattern"],
    }


def volume_confirmed(df: pd.DataFrame, period: int, multiplier: float) -> bool:
    vol_ma = df["Volume"].rolling(period).mean()
    return float(df["Volume"].iloc[-1]) >= multiplier * float(vol_ma.iloc[-1])


def macd_bullish(macd_df: pd.DataFrame) -> bool:
    latest = macd_df.iloc[-1]
    prev = macd_df.iloc[-2]
    crossed_up = (prev["macd"] <= prev["signal"]) and (latest["macd"] > latest["signal"])
    above_and_rising = (latest["macd"] > latest["signal"]) and (latest["hist"] > prev["hist"])
    return bool(crossed_up or above_and_rising)


def near_support(price: float, support: float, buffer_pct: float) -> bool:
    return abs(price - support) / support * 100 <= buffer_pct


def suggested_stop(price: float, atr_value: float, multiplier: float) -> float:
    return round(price - multiplier * atr_value, 2)


def swing_support_resistance(df: pd.DataFrame, lookback: int) -> dict:
    """Nearest support/resistance = the lowest low / highest high over the lookback window."""
    recent = df.iloc[-lookback:]
    return {
        "support": round(float(recent["Low"].min()), 2),
        "resistance": round(float(recent["High"].max()), 2),
    }


def day_range(hourly_df: pd.DataFrame) -> dict:
    """Today's high/low/range, built from the 1h bars for the most recent calendar date."""
    latest_date = hourly_df.index[-1].date()
    todays_bars = hourly_df[hourly_df.index.date == latest_date]
    if todays_bars.empty:
        todays_bars = hourly_df.iloc[-7:]  # fallback: roughly one trading day of hourly bars

    high = float(todays_bars["High"].max())
    low = float(todays_bars["Low"].min())
    return {"day_high": round(high, 2), "day_low": round(low, 2), "day_range": round(high - low, 2)}


def chart_window(hourly_df: pd.DataFrame, overlays: dict, days: int) -> dict:
    """Slice the last `days` trading days of 1h bars, plus the matching values of
    each overlay series (MAs computed on the full history, so already warmed up)."""
    dates = pd.Index(hourly_df.index.date).unique()
    start = dates[-days] if len(dates) >= days else dates[0]
    mask = hourly_df.index.date >= start
    return {
        "ohlcv": hourly_df.loc[mask],
        "overlays": {name: series.loc[mask] for name, series in overlays.items()},
    }


def volume_poc(df: pd.DataFrame, bins: int = 50) -> float:
    """Point of Control: the price bucket with the most traded volume over `df`.
    Each bar's volume is spread evenly across the buckets its Low-High range covers."""
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    if hi <= lo:
        return round(float(df["Close"].iloc[-1]), 2)
    edges = np.linspace(lo, hi, bins + 1)
    low = df["Low"].to_numpy(dtype=float)[:, None]
    high = df["High"].to_numpy(dtype=float)[:, None]
    vol = df["Volume"].to_numpy(dtype=float)

    overlap = np.clip(np.minimum(high, edges[1:]) - np.maximum(low, edges[:-1]), 0, None)
    rng = (high - low)
    share = np.divide(overlap, rng, out=np.zeros_like(overlap), where=rng > 0)
    profile = (share * vol[:, None]).sum(axis=0)

    # zero-range bars: put all their volume in the single bucket they sit in
    flat = (rng[:, 0] == 0)
    if flat.any():
        idx = np.clip(np.searchsorted(edges, low[flat, 0], side="right") - 1, 0, bins - 1)
        np.add.at(profile, idx, vol[flat])

    k = int(profile.argmax())
    return round(float((edges[k] + edges[k + 1]) / 2), 2)


ALPACA_DATA_URL = "https://data.alpaca.markets/v2"


def _alpaca_headers(headers: dict | None) -> dict | None:
    """Explicit headers -> constants.CONFIG -> APCA_* environment variables."""
    if headers is None:
        headers = _SECRETS.get("ALPACA_HEADERS")
    if headers is None:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if key and secret:
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    return headers


def _alpaca_premarket_bars_multi(tickers: list, headers: dict | None, feed: str, delay_minutes: int = 0,
                                 tz: str = "America/New_York", start_hhmm: str = "04:00",
                                 open_hhmm: str = "09:30", timeout: float = 15.0,
                                 chunk: int = 100) -> dict | None:
    """1-minute bars for TODAY's pre-market session for MANY tickers at once, on
    one Alpaca feed, from start_hhmm up to min(now - delay_minutes, open_hhmm).
    Uses Alpaca's multi-symbol endpoint: one request per `chunk` tickers (plus
    extra pages when there are more than 10,000 bars), instead of one per ticker.
    Returns {ticker: [{"t": Timestamp (exchange tz), "c": close, "v": volume}, ...]}
    (a ticker with no pre-market trades gets []), or None on error / missing keys."""
    headers = _alpaca_headers(headers)
    if headers is None:
        return None
    now = pd.Timestamp.now(tz=tz)
    sh, sm = map(int, start_hhmm.split(":"))
    oh, om = map(int, open_hhmm.split(":"))
    start = now.normalize() + pd.Timedelta(hours=sh, minutes=sm)
    latest_allowed = (now - pd.Timedelta(minutes=delay_minutes)).floor("min")
    end = min(latest_allowed, now.normalize() + pd.Timedelta(hours=oh, minutes=om))
    out = {t: [] for t in tickers}
    if end <= start:
        return out  # today's pre-market hasn't started yet (or is still inside the delay)

    # Yahoo writes share classes as BRK-B, Alpaca as BRK.B
    to_alpaca = {t: t.replace("-", ".") for t in tickers}
    from_alpaca = {v: k for k, v in to_alpaca.items()}
    symbols = list(to_alpaca.values())
    try:
        for i in range(0, len(symbols), chunk):
            params = {"symbols": ",".join(symbols[i:i + chunk]), "timeframe": "1Min",
                      "start": start.isoformat(), "end": end.isoformat(), "feed": feed,
                      "adjustment": "raw", "limit": 10000}
            while True:
                resp = requests.get(f"{ALPACA_DATA_URL}/stocks/bars", headers=headers,
                                    params=params, timeout=timeout)
                resp.raise_for_status()
                body = resp.json()
                for sym, bars in (body.get("bars") or {}).items():
                    t = from_alpaca.get(sym, sym)
                    out.setdefault(t, []).extend(
                        {"t": pd.Timestamp(b["t"]).tz_convert(tz), "c": float(b["c"]), "v": float(b.get("v", 0))}
                        for b in bars)
                token = body.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
    except (requests.RequestException, ValueError) as e:
        print(f"  pre-market bars unavailable ({feed}, batch): {e}")
        return None
    for t in out:
        out[t].sort(key=lambda b: b["t"])
    return out


def _alpaca_premarket_bars(ticker: str, headers: dict | None, feed: str, delay_minutes: int = 0,
                           tz: str = "America/New_York", start_hhmm: str = "04:00",
                           open_hhmm: str = "09:30", timeout: float = 10.0) -> list | None:
    """Single-ticker version of _alpaca_premarket_bars_multi: the bars list, []
    if no trades yet, None on error / missing keys."""
    bars = _alpaca_premarket_bars_multi([ticker], headers, feed, delay_minutes, tz,
                                        start_hhmm, open_hhmm, timeout)
    return None if bars is None else bars.get(ticker, [])


@_profiled("pre-market (Alpaca, 1 feed)")
def get_pre_market_price(ticker: str, headers: dict | None = None, feed: str = "sip",
                         delay_minutes: int = 16, tz: str = "America/New_York",
                         start_hhmm: str = "04:00", open_hhmm: str = "09:30",
                         timeout: float = 10.0, with_time: bool = False):
    """Latest pre-market price for `ticker` from ONE Alpaca feed (today's session).

    feed="sip": all exchanges; on the free plan it must be >=15 min old, so
    delay_minutes=16 keeps the query out of the restricted window.
    feed="iex": live (pass delay_minutes=0) but only IEX-exchange trades.

    Returns the price (float) or None. with_time=True returns (price, "HH:MM").
    See get_best_pre_market_price() to combine both feeds automatically, and
    get_pre_market_prices() / get_best_pre_market_prices() for many tickers at once.
    """
    bars = _alpaca_premarket_bars(ticker, headers, feed, delay_minutes, tz, start_hhmm, open_hhmm, timeout)
    if not bars:
        return (None, None) if with_time else None
    price = round(bars[-1]["c"], 2)
    return (price, bars[-1]["t"].strftime("%H:%M")) if with_time else price


def _choose_pre_market(sip: list, iex: list, move_pct: float = 0.3, max_iex_gap_pct: float = 0.5) -> dict:
    """Decide between delayed SIP bars and live IEX bars (see get_best_pre_market_price)."""
    sip, iex = sip or [], iex or []
    d = {"price": None, "time": None, "source": None, "reason": "no pre-market data",
         "sip_price": None, "sip_time": None, "iex_price": None, "iex_time": None, "move_pct": None}
    if sip:
        d["sip_price"], d["sip_time"] = round(sip[-1]["c"], 2), sip[-1]["t"]
    if iex:
        d["iex_price"], d["iex_time"] = round(iex[-1]["c"], 2), iex[-1]["t"]

    def pick(src, reason):
        d["source"], d["reason"] = src, reason
        d["price"], d["time"] = d[f"{src}_price"], d[f"{src}_time"]

    if sip and iex:
        d["move_pct"] = round((d["iex_price"] / d["sip_price"] - 1) * 100, 2)
        # what IEX said at (or just before) the SIP bar's time - a like-for-like check
        iex_then = [b for b in iex if b["t"] <= d["sip_time"]]
        gap = abs(iex_then[-1]["c"] / d["sip_price"] - 1) * 100 if iex_then else None

        if d["iex_time"] <= d["sip_time"]:
            pick("sip", "no IEX trade newer than the SIP snapshot")
        elif gap is not None and gap > max_iex_gap_pct:
            pick("sip", f"IEX off by {gap:.2f}% vs SIP at the same time (thin IEX trading)")
        elif abs(d["move_pct"]) >= move_pct:
            pick("iex", f"moved {d['move_pct']:+.2f}% since the SIP snapshot")
        else:
            pick("sip", f"only {d['move_pct']:+.2f}% move since the SIP snapshot")
    elif sip:
        pick("sip", "no IEX pre-market trades")
    elif iex:
        pick("iex", "no SIP data yet (within the 15-min delay)")

    for k in ("time", "sip_time", "iex_time"):
        if d[k] is not None:
            d[k] = d[k].strftime("%H:%M")
    return d


@_profiled("pre-market (Alpaca auto: 2 calls)")
def get_best_pre_market_price(ticker: str, headers: dict | None = None, sip_delay_minutes: int = 16,
                              move_pct: float = 0.3, max_iex_gap_pct: float = 0.5,
                              tz: str = "America/New_York", start_hhmm: str = "04:00",
                              open_hhmm: str = "09:30", timeout: float = 10.0,
                              with_details: bool = False):
    """Pick between delayed SIP (complete, ~16 min old) and live IEX (fresh,
    but only IEX-exchange trades) for the pre-market price.

    Decision, in order:
      1. Only one feed has data            -> use that one.
      2. IEX has no trade newer than SIP   -> SIP (IEX adds nothing fresher).
      3. IEX disagreed with SIP at SIP's own timestamp by more than
         max_iex_gap_pct                   -> SIP (IEX too thin/unreliable today).
      4. Price moved >= move_pct from the SIP price to the latest IEX price
                                           -> IEX (the market has moved since the
                                              SIP snapshot, so the live price matters).
      5. Otherwise (little movement)       -> SIP (full-market price, still accurate).

    Returns the price (float) or None. with_details=True returns a dict:
    {"price", "time", "source" ("sip"/"iex"), "reason", "sip_price", "sip_time",
     "iex_price", "iex_time", "move_pct"} (price None if neither feed has data).
    For many tickers use get_best_pre_market_prices() - same logic, 2 batched requests.
    """
    kw = dict(tz=tz, start_hhmm=start_hhmm, open_hhmm=open_hhmm, timeout=timeout)
    sip = _alpaca_premarket_bars(ticker, headers, "sip", sip_delay_minutes, **kw)
    iex = _alpaca_premarket_bars(ticker, headers, "iex", 0, **kw)
    d = _choose_pre_market(sip, iex, move_pct, max_iex_gap_pct)
    return d if with_details else d["price"]


@_profiled("pre-market (Alpaca batch, all tickers)")
def get_best_pre_market_prices(tickers: list, headers: dict | None = None, sip_delay_minutes: int = 16,
                               move_pct: float = 0.3, max_iex_gap_pct: float = 0.5,
                               tz: str = "America/New_York", start_hhmm: str = "04:00",
                               open_hhmm: str = "09:30") -> dict:
    """get_best_pre_market_price(..., with_details=True) for MANY tickers, using
    one batched SIP request and one batched IEX request (per 100 tickers).
    Returns {ticker: details dict}; {} if both feeds failed."""
    kw = dict(tz=tz, start_hhmm=start_hhmm, open_hhmm=open_hhmm)
    sip = _alpaca_premarket_bars_multi(tickers, headers, "sip", sip_delay_minutes, **kw)
    iex = _alpaca_premarket_bars_multi(tickers, headers, "iex", 0, **kw)
    if sip is None and iex is None:
        return {}
    return {t: _choose_pre_market((sip or {}).get(t), (iex or {}).get(t), move_pct, max_iex_gap_pct)
            for t in tickers}


@_profiled("pre-market (Alpaca batch, all tickers)")
def get_pre_market_prices(tickers: list, headers: dict | None = None, feed: str = "sip",
                          delay_minutes: int = 16, tz: str = "America/New_York",
                          start_hhmm: str = "04:00", open_hhmm: str = "09:30") -> dict:
    """Latest pre-market price for MANY tickers from ONE feed, batched.
    Returns {ticker: {"price", "time", "source", "reason"}}; {} on failure."""
    bars = _alpaca_premarket_bars_multi(tickers, headers, feed, delay_minutes, tz, start_hhmm, open_hhmm)
    if bars is None:
        return {}
    out = {}
    for t in tickers:
        b = bars.get(t) or []
        out[t] = {"price": round(b[-1]["c"], 2) if b else None,
                  "time": b[-1]["t"].strftime("%H:%M") if b else None,
                  "source": feed if b else None, "reason": "fixed feed (ALPACA_FEED)"}
    return out


@_profiled("news (Finnhub)")
def fetch_company_news(ticker: str, api_key: str | None = None, days: int = 0,
                       max_items: int | None = 25, tz: str = "America/New_York") -> list | None:
    """Company news from Finnhub for today (and the previous `days` days).
    Returns a list of {"time": "YYYY-MM-DD HH:MM", "date": date, "headline": str,
    "url": str}, newest first, duplicates (same headline) removed, at most
    max_items (None = no limit). Returns [] if there's no news, and None if the
    finnhub package or API key is missing or the request fails.
    Calls are rate-limited to CONFIGH["FINNHUB_MAX_PER_MIN"] across threads."""
    api_key = api_key or _SECRETS.get("FINNHUB_API_KEY") or os.environ.get("FINNHUB_API_KEY")
    if finnhub is None or not api_key:
        return None
    today = dt.date.today()
    try:
        _finnhub_wait()
        client = finnhub.Client(api_key=api_key)
        news = client.company_news(ticker, _from=str(today - dt.timedelta(days=days)), to=str(today)) or []
    except Exception as e:
        print(f"  [{ticker}] news unavailable: {e}")
        return None

    items, seen = [], set()
    for n in sorted(news, key=lambda n: n.get("datetime", 0), reverse=True):
        headline, url = (n.get("headline") or "").strip(), (n.get("url") or "").strip()
        if not headline or headline in seen:
            continue
        seen.add(headline)
        when = pd.Timestamp(n.get("datetime", 0), unit="s", tz="UTC").tz_convert(tz)
        items.append({"time": when.strftime("%Y-%m-%d %H:%M"), "date": when.date(),
                      "headline": headline, "url": url})
        if max_items and len(items) >= max_items:
            break
    return items


def swing_sr_levels(daily_window: pd.DataFrame, price: float, atr: float | None, stop: float | None,
                    pivot: int = 2, merge_atr: float = 0.5, min_r: float = 2.0) -> dict:
    """Support and resistance the way most traders draw them: from SWING POINTS.
      - swing low  = a day whose low is the lowest of `pivot` days on each side
                     (buyers stepped in, price turned up)
      - swing high = a day whose high is the highest of `pivot` days on each side
    Swing points within merge_atr x daily ATR of each other are merged into one
    level (price at the members' average); more touches = a stronger level.
    Supports are levels below price (S1 = nearest), resistances above (T1 = nearest).
    Targets: T1/T2 = the two nearest resistances, each with its reward:risk vs
    the stop. If there's no resistance above (price at new highs), the target
    falls back to min_r x risk.
    Returns {"supports": [...], "resistances": [...], "target": {...}} where each
    level is {"price", "touches", "last_date", "dist_pct", "r"}."""
    out = {"supports": [], "resistances": [], "target": None}
    if daily_window is None or len(daily_window) < 2 * pivot + 1 or not price:
        return out
    lows = daily_window["Low"].to_numpy(dtype=float)
    highs = daily_window["High"].to_numpy(dtype=float)
    dates = [t.date().isoformat() for t in daily_window.index]
    n = len(lows)
    pts_low = [(lows[i], dates[i]) for i in range(pivot, n - pivot)
               if lows[i] == lows[i - pivot:i + pivot + 1].min()]
    pts_high = [(highs[i], dates[i]) for i in range(pivot, n - pivot)
                if highs[i] == highs[i - pivot:i + pivot + 1].max()]

    tol = merge_atr * atr if atr else price * 0.01

    def cluster(points):
        levels = []
        for p, d in sorted(points):
            if levels and p - levels[-1]["members"][-1] <= tol:
                levels[-1]["members"].append(p)
                levels[-1]["dates"].append(d)
            else:
                levels.append({"members": [p], "dates": [d]})
        return [{"price": round(float(np.mean(l["members"])), 2), "touches": len(l["members"]),
                 "last_date": max(l["dates"])} for l in levels]

    risk = (price - stop) if stop is not None and price > stop else None

    def enrich(lvl):
        lvl["dist_pct"] = round((lvl["price"] / price - 1) * 100, 2)
        lvl["r"] = round((lvl["price"] - price) / risk, 2) if risk and lvl["price"] > price else None
        return lvl

    # swing highs AND lows both count as levels (old resistance can become support and vice versa)
    all_levels = cluster(pts_low + pts_high)
    out["supports"] = [enrich(l) for l in sorted((l for l in all_levels if l["price"] < price),
                                                 key=lambda l: -l["price"])]
    out["resistances"] = [enrich(l) for l in sorted((l for l in all_levels if l["price"] > price),
                                                    key=lambda l: l["price"])]
    if out["resistances"]:
        t1 = out["resistances"][0]
        out["target"] = {"price": t1["price"], "r": t1["r"], "source": "T1 - nearest resistance",
                         "below_min_r": t1["r"] is not None and t1["r"] < min_r}
    elif risk:
        out["target"] = {"price": round(price + min_r * risk, 2), "r": min_r,
                         "source": f"{min_r:g}R - no resistance above", "below_min_r": False}
    return out


def anchored_vwap(hourly: pd.DataFrame, daily_window: pd.DataFrame, anchor: str = "low",
                  tz: str = "America/New_York") -> dict | None:
    """Anchored VWAP: the volume-weighted average price of every trade since an
    anchor day - here the day with the lowest low ("low") or highest high
    ("high") in `daily_window`. From a low it's the average cost of everyone who
    bought since the bottom; price holding above it = those buyers are in profit.
    Computed from the hourly bars (typical price x volume, cumulative from the
    anchor day's first bar) and sampled at each day's last bar.
    Returns {"series": value per daily bar (NaN before the anchor), "anchor_ts",
    "anchor_date", "anchor_price", "anchor_type", "value"} or None."""
    if daily_window is None or daily_window.empty or hourly is None or hourly.empty:
        return None
    anchor_ts = daily_window["Low"].idxmin() if anchor == "low" else daily_window["High"].idxmax()
    local = lambda ix: ix.tz_convert(tz) if ix.tz is not None else ix
    anchor_date = local(pd.DatetimeIndex([anchor_ts]))[0].date()
    hdates = np.asarray(local(hourly.index).date)
    seg = hourly[hdates >= anchor_date]
    if seg.empty or float(seg["Volume"].sum()) <= 0:
        return None
    typical = (seg["High"] + seg["Low"] + seg["Close"]) / 3
    cum = (typical * seg["Volume"]).cumsum() / seg["Volume"].cumsum().replace(0, np.nan)
    per_day = cum.groupby(np.asarray(local(seg.index).date)).last()
    ddates = local(daily_window.index).date
    series = pd.Series([per_day.get(d, np.nan) for d in ddates], index=daily_window.index, dtype=float)
    value = series.dropna()
    return {
        "series": series,
        "anchor_ts": anchor_ts,
        "anchor_date": anchor_date,
        "anchor_price": float(daily_window.loc[anchor_ts, "Low" if anchor == "low" else "High"]),
        "anchor_type": anchor,
        "value": round(float(value.iloc[-1]), 2) if not value.empty else None,
    }


def hhhl_stats(daily: pd.DataFrame, window: int = 20, tz: str = "America/New_York",
               close_hhmm: str = "16:00") -> dict:
    """Daily market-structure count. Each COMPLETED daily bar is compared with
    the bar before it:
        "up"    = higher high AND higher low
        "down"  = lower high AND lower low
        "mixed" = anything else (inside day, outside day, equal high/low)
    Today's bar is left out while the session is still open (its high/low can
    still change). Returns counts over the last `window` days, the current run
    of same-type days and the run before it, and {date: type} for chart markers."""
    d = daily.dropna(subset=["High", "Low"])
    in_progress = False
    if len(d):
        now = pd.Timestamp.now(tz=tz)
        last = d.index[-1]
        last_date = (last.tz_convert(tz) if last.tzinfo is not None else last).date()
        ch, cm = map(int, close_hhmm.split(":"))
        if last_date == now.date() and now < now.normalize() + pd.Timedelta(hours=ch, minutes=cm):
            in_progress = True
            d = d.iloc[:-1]

    hi, lo = d["High"].to_numpy(dtype=float), d["Low"].to_numpy(dtype=float)
    kinds = []
    for i in range(1, len(d)):
        if hi[i] > hi[i - 1] and lo[i] > lo[i - 1]:
            kinds.append("up")
        elif hi[i] < hi[i - 1] and lo[i] < lo[i - 1]:
            kinds.append("down")
        else:
            kinds.append("mixed")
    idx = d.index[1:]
    dates = [(t.tz_convert(tz) if t.tzinfo is not None else t).date() for t in idx]

    # runs of same-type days, most recent first: [("up", 3), ("mixed", 2), ...]
    runs = []
    for k in reversed(kinds):
        if runs and runs[-1][0] == k:
            runs[-1] = (k, runs[-1][1] + 1)
        else:
            runs.append((k, 1))

    recent = kinds[-window:] if window else kinds
    up, down = recent.count("up"), recent.count("down")
    return {
        "window": len(recent),
        "up_days": up, "down_days": down, "mixed_days": recent.count("mixed"),
        "net": up - down,
        "current_streak": runs[0] if runs else (None, 0),
        "previous_streak": runs[1] if len(runs) > 1 else (None, 0),
        "today_in_progress": in_progress,
        "by_date": dict(zip(dates, kinds)),
    }


def linear_regression_channel(df: pd.DataFrame, length, dev: float, source: str = "Close") -> dict | None:
    """Least-squares line through the last `length` bars of `source` (x = bar
    number, so overnight gaps don't bend it), with parallel bands at +/- `dev`
    standard deviations of the residuals. Returns None if too few bars."""
    src = df[source].dropna()
    if length:
        src = src.iloc[-int(length):]
    n = len(src)
    if n < 3:
        return None
    x = np.arange(n, dtype=float)
    yv = src.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, yv, 1)
    fitted = intercept + slope * x
    resid = yv - fitted
    sd = float(resid.std())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0

    mid = pd.Series(fitted, index=src.index)
    upper, lower = mid + dev * sd, mid - dev * sd
    last_mid, last_up, last_lo = float(mid.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])
    last_px = float(df["Close"].iloc[-1])
    width = last_up - last_lo
    bars_per_day = n / max(len(pd.Index(src.index.date).unique()), 1)
    return {
        "mid": mid, "upper": upper, "lower": lower,
        "bars": n, "dev": dev,
        "slope_per_bar": float(slope),
        "slope_pct_per_day": round(float(slope) * bars_per_day / last_mid * 100, 2) if last_mid else 0.0,
        "r2": round(r2, 2),
        "last_mid": round(last_mid, 2), "last_upper": round(last_up, 2), "last_lower": round(last_lo, 2),
        # where the current price sits in the channel: 0% = lower band, 100% = upper band
        "position_pct": round((last_px - last_lo) / width * 100, 0) if width > 0 else 50.0,
    }


def session_vwap(df: pd.DataFrame, tz: str, open_hhmm: str, close_hhmm: str) -> pd.Series:
    """Intraday VWAP that resets at each regular-session open. Uses typical price
    (H+L+C)/3 x volume. Bars outside regular hours (pre/after-market, if present)
    are excluded and get NaN, matching how most platforms draw session VWAP."""
    idx = df.index.tz_convert(tz) if df.index.tz is not None else df.index
    mins = np.asarray(idx.hour * 60 + idx.minute)
    oh, om = map(int, open_hhmm.split(":"))
    ch, cm = map(int, close_hhmm.split(":"))
    in_session = pd.Series((mins >= oh * 60 + om) & (mins < ch * 60 + cm), index=df.index)

    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = (typical * df["Volume"]).where(in_session)
    vol = df["Volume"].where(in_session)
    day = np.asarray(idx.date)
    cum_pv = pv.groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum()
    vwap = (cum_pv / cum_v.replace(0, np.nan)).where(in_session)
    return vwap


# ---------------------------------------------------------------- fundamentals
# Informational only - NOT part of the hard/soft buy-signal logic. Pulled from
# yfinance's .info, which is sometimes incomplete or slow, so every field is
# fetched defensively and missing data shows as "N/A" rather than crashing the run.

def _fmt_billions(v) -> str:
    v = float(v)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v) / 1e9:,.2f}B"


FUNDAMENTAL_FIELDS = [
    # (label,                      info_key,                       formatter)
    ("Name",                       "longName",                     lambda v: str(v)),
    ("Sector",                     "sector",                        lambda v: str(v)),
    ("Industry",                   "industry",                      lambda v: str(v)),
    ("Market Cap",                 "marketCap",                     _fmt_billions),
    ("Trailing P/E",               "trailingPE",                    lambda v: f"{v:.2f}"),
    ("Forward P/E",                "forwardPE",                     lambda v: f"{v:.2f}"),
    ("PEG Ratio",                  "pegRatio",                      lambda v: f"{v:.2f}"),
    ("Price/Book",                 "priceToBook",                   lambda v: f"{v:.2f}"),
    ("Price/Sales (TTM)",          "priceToSalesTrailing12Months",  lambda v: f"{v:.2f}"),
    ("EPS (TTM)",                  "trailingEps",                   lambda v: f"{v:.2f}"),
    ("Dividend Yield",             "dividendYield",                 lambda v: f"{v * 100:.2f}%"),
    ("Profit Margin",              "profitMargins",                 lambda v: f"{v * 100:.2f}%"),
    ("Operating Margin",           "operatingMargins",              lambda v: f"{v * 100:.2f}%"),
    ("Return on Equity",           "returnOnEquity",                lambda v: f"{v * 100:.2f}%"),
    ("Revenue Growth (YoY)",       "revenueGrowth",                 lambda v: f"{v * 100:.2f}%"),
    ("Free Cash Flow",             "freeCashflow",                  _fmt_billions),
    ("Debt/Equity",                "debtToEquity",                  lambda v: f"{v:.2f}"),
    ("Current Ratio",              "currentRatio",                  lambda v: f"{v:.2f}"),
    ("Institutional Ownership",    "heldPercentInstitutions",       lambda v: f"{v * 100:.2f}%"),
    ("Short % of Float",           "shortPercentOfFloat",           lambda v: f"{v * 100:.2f}%"),
    ("Beta",                       "beta",                          lambda v: f"{v:.2f}"),
    ("52-Week High",               "fiftyTwoWeekHigh",              lambda v: f"{v:.2f}"),
    ("52-Week Low",                "fiftyTwoWeekLow",               lambda v: f"{v:.2f}"),
    ("Analyst Target (Mean)",      "targetMeanPrice",               lambda v: f"{v:.2f}"),
]


@_profiled("yf .info (shared, once per ticker)")
def fetch_info(ticker: str, tkr: "yf.Ticker | None" = None) -> dict:
    """yfinance .info for a ticker ({} on failure). Fetched ONCE per ticker and
    shared by fundamentals, analyst data and the earnings date."""
    try:
        return (tkr if tkr is not None else yf.Ticker(ticker)).info or {}
    except Exception:
        return {}


def fetch_fundamentals(ticker: str, info: dict | None = None) -> dict:
    """Pull a handful of fundamental data points for a ticker. Returns a dict of
    label -> formatted string. Any field yfinance doesn't have is shown as "N/A";
    if there's no .info at all (network hiccup, delisted ticker, etc.) an empty
    dict is returned so the technical screen can still run.
    Pass `info` (from fetch_info) to avoid downloading it again."""
    if info is None:
        info = fetch_info(ticker)
    if not info:
        return {}

    fundamentals = {}
    for label, key, fmt in FUNDAMENTAL_FIELDS:
        raw = info.get(key)
        if raw is None:
            fundamentals[label] = "N/A"
            continue
        try:
            fundamentals[label] = fmt(raw)
        except (TypeError, ValueError):
            fundamentals[label] = "N/A"
    return fundamentals


# ---------------------------------------------------------------- analyst data
# Also informational only - not part of the buy/sell signal logic. Two pieces:
#   1. Recommendation counts (strong buy / buy / hold / sell / strong sell) for
#      the most recent period, plus the consensus key/mean and analyst count.
#   2. EPS estimate revision trend - current-quarter consensus EPS now vs. what
#      it was 30 days ago, and whether analysts are revising it up or down.
# Analyst coverage varies a lot by ticker (some have none), so every piece is
# fetched independently and defensively - a missing piece never blocks the rest.

@_profiled("analyst data (yf, 3 calls)")
def fetch_analyst_data(ticker: str, tkr: "yf.Ticker | None" = None, info: dict | None = None) -> dict:
    """Analyst consensus + EPS revisions. Pass the shared `tkr` and `info` to
    avoid creating another yf.Ticker and downloading .info a second time."""
    data = {
        "recommendation_key": "N/A",
        "recommendation_mean": "N/A",
        "num_analyst_opinions": "N/A",
        "recommendation_counts": {},
        "eps_current": "N/A",
        "eps_30d_ago": "N/A",
        "eps_improving": None,
        "eps_num_analysts": "N/A",
    }

    if tkr is None:
        try:
            tkr = yf.Ticker(ticker)
        except Exception:
            return data

    # --- Consensus key ("buy", "hold", ...), mean score, and analyst count
    try:
        if info is None:
            info = fetch_info(ticker, tkr)
        if info.get("recommendationKey") is not None:
            data["recommendation_key"] = str(info["recommendationKey"]).replace("_", " ").title()
        if info.get("recommendationMean") is not None:
            data["recommendation_mean"] = f"{float(info['recommendationMean']):.2f}"
        if info.get("numberOfAnalystOpinions") is not None:
            data["num_analyst_opinions"] = str(int(info["numberOfAnalystOpinions"]))
    except Exception:
        pass


    # --- Buy/hold/sell breakdown, most recent period ("0m" = current month)
    t0 = time.perf_counter()
    try:
        rec = tkr.recommendations
        if rec is not None and not rec.empty:
            latest = rec.iloc[0]
            data["recommendation_counts"] = {
                "Strong Buy": int(latest.get("strongBuy", 0) or 0),
                "Buy": int(latest.get("buy", 0) or 0),
                "Hold": int(latest.get("hold", 0) or 0),
                "Sell": int(latest.get("sell", 0) or 0),
                "Strong Sell": int(latest.get("strongSell", 0) or 0),
            }
    except Exception:
        pass

    _prof_add("  analyst: recommendations", ticker, time.perf_counter() - t0)

    # --- EPS estimate revision trend, current quarter ("0q")
    t0 = time.perf_counter()
    try:
        trend = tkr.eps_trend
        if trend is not None and not trend.empty and "0q" in trend.index:
            row = trend.loc["0q"]
            current = row.get("current")
            ago_30d = row.get("30daysAgo")
            if current is not None and pd.notna(current):
                data["eps_current"] = f"{float(current):.2f}"
            if ago_30d is not None and pd.notna(ago_30d):
                data["eps_30d_ago"] = f"{float(ago_30d):.2f}"
            if data["eps_current"] != "N/A" and data["eps_30d_ago"] != "N/A":
                data["eps_improving"] = float(data["eps_current"]) >= float(data["eps_30d_ago"])
    except Exception:
        pass

    _prof_add("  analyst: eps_trend", ticker, time.perf_counter() - t0)

    # --- Number of analysts contributing to the current-quarter EPS estimate
    t0 = time.perf_counter()
    try:
        est = tkr.earnings_estimate
        if est is not None and not est.empty and "0q" in est.index:
            n = est.loc["0q"].get("numberOfAnalysts")
            if n is not None and pd.notna(n):
                data["eps_num_analysts"] = str(int(n))
    except Exception:
        pass
    _prof_add("  analyst: earnings_estimate", ticker, time.perf_counter() - t0)

    return data


# ---------------------------------------------------------------- signal

def generate_signal(daily: dict, h4: dict, h1: dict, h1_macd_ok: bool, vol_ok: bool,
                     price: float, support_level: float, cfg: dict) -> dict:
    # ---- Hard requirement #1: both higher timeframes trending up + momentum bullish.
    # Broken out per sub-condition so a WAIT signal can show exactly which
    # timeframe/check failed instead of just a single pass/fail bool.
    hard_requirements_detail = {
        "daily_trend_up": daily["trend_up"],
        "daily_stoch_bullish": daily["stoch_bullish"],
        "4h_trend_up": h4["trend_up"],
        "4h_stoch_bullish": h4["stoch_bullish"],
    }
    higher_tf_pass = all(hard_requirements_detail.values())

    # ---- Hard requirement #2: a real candlestick structure confirming the
    # reversal on daily OR 4h. Broken out so we can see *which* timeframe(s)
    # confirmed and *which specific pattern(s)* fired.
    structural_confirmation_detail = {
        "daily_bullish_pattern": daily["bullish_pattern"],
        "daily_patterns": daily["patterns_detected"],
        "4h_bullish_pattern": h4["bullish_pattern"],
        "4h_patterns": h4["patterns_detected"],
    }
    structural_confirmation = daily["bullish_pattern"] or h4["bullish_pattern"]

    hard_pass = higher_tf_pass and structural_confirmation

    support_ok = near_support(price, support_level, cfg["SUPPORT_BUFFER_PCT"])
    h1_not_stretched = not h1["overbought"]

    soft_conditions = {
        "1h_not_overbought": h1_not_stretched,
        "1h_macd_bullish": h1_macd_ok,
        "volume_confirmed": vol_ok,
        "near_support": support_ok,
        "1h_bullish_pattern": h1["bullish_pattern"],
    }
    soft_score = sum(soft_conditions.values())

    signal = "BUY" if (hard_pass and soft_score >= cfg["SCORE_THRESHOLD"]) else "WAIT"

    return {
        "signal": signal,
        "hard_requirements_met": hard_pass,
        "hard_requirements_detail": hard_requirements_detail,
        "structural_confirmation": structural_confirmation,
        "structural_confirmation_detail": structural_confirmation_detail,
        "soft_score": f"{soft_score}/5",
        **soft_conditions,
    }


# ---------------------------------------------------------------- main

def run_screen(ticker: str, cfg: dict, sp500_members: dict | None = None,
               hourly: pd.DataFrame | None = None, social_table: dict | None = None,
               pre_market_batch: dict | None = None) -> dict:
    """Fetch data and compute everything needed for one ticker's report.
    Pure computation - no printing - so the same result can feed both the
    console output and the HTML report."""
    cfg = dict(cfg)
    cfg["TICKER"] = ticker
    if hourly is None or hourly.empty:   # not prefetched -> download just this one
        hourly = fetch_hourly_data(cfg["TICKER"], cfg["PERIOD"], cfg["INTERVAL"])

    tf_data = {
        "1h": hourly,
        "4h": resample_ohlc(hourly, "4h"),
        "1D": resample_ohlc(hourly, "1D"),
    }

    stoch = {tf: stochastic_oscillator(df, cfg["STOCH_K_PERIOD"], cfg["STOCH_D_PERIOD"], cfg["STOCH_SMOOTH"])
              for tf, df in tf_data.items()}
    ma = {tf: moving_average(df, cfg["MA_PERIOD"], cfg["MA_TYPE"]) for tf, df in tf_data.items()}

    results = {tf: evaluate_timeframe(tf_data[tf], stoch[tf], ma[tf], cfg) for tf in tf_data}

    macd_1h = macd(tf_data["1h"], cfg["MACD_FAST"], cfg["MACD_SLOW"], cfg["MACD_SIGNAL"])
    h1_macd_ok = macd_bullish(macd_1h)

    vol_ok = volume_confirmed(tf_data["1h"], cfg["VOLUME_MA_PERIOD"], cfg["VOLUME_MULTIPLIER"])

    atr_1h = average_true_range(tf_data["1h"], cfg["ATR_PERIOD"])
    current_price = results["1h"]["close"]
    stop = suggested_stop(current_price, float(atr_1h.iloc[-1]), cfg["ATR_STOP_MULTIPLIER"])

    # 1. Calculate Daily ATR instead of 1h ATR
    atr_daily = average_true_range(tf_data["1D"], cfg["ATR_PERIOD"])
    daily_atr_val = float(atr_daily.iloc[-1])
    
    current_price = results["1h"]["close"]
    
    # Option A: Pure 2.0x Daily ATR Stop
    volatility_stop = suggested_stop(current_price, daily_atr_val, cfg["ATR_STOP_MULTIPLIER"])

    # Option B: Structural Volatility Stop (1h Support - 0.5x Daily ATR)
    sr_levels = {tf: swing_support_resistance(tf_data[tf], cfg["SR_LOOKBACK"][tf]) for tf in tf_data}
    entry_support = sr_levels["1h"]["support"]
    structural_stop = round(entry_support - (0.5 * daily_atr_val), 2)
    
    # We will pass structural_stop into the final report as the primary stop
    stop = structural_stop

    risk_per_share = current_price - stop
    take_profit_target = round(current_price + (1.5 * risk_per_share), 2)

    sr_levels = {tf: swing_support_resistance(tf_data[tf], cfg["SR_LOOKBACK"][tf]) for tf in tf_data}
    today_range = day_range(tf_data["1h"])

    # The signal's near-support check uses the 1h swing support - the same
    # timeframe the entry itself is timed against.
    entry_support = sr_levels["1h"]["support"]
    signal = generate_signal(results["1D"], results["4h"], results["1h"], h1_macd_ok, vol_ok,
                              current_price, entry_support, cfg)

    # Chart data (display only): MA overlays on 1h + 7-day volume POC
    chart_mas = {}
    for ma_type, period in cfg.get("CHART_MAS", [(cfg["MA_TYPE"], cfg["MA_PERIOD"])]):
        chart_mas[f"{ma_type.upper()}{period}"] = moving_average(tf_data["1h"], period, ma_type)
    chart = chart_window(tf_data["1h"], chart_mas, cfg.get("CHART_DAYS", 7))
    poc = volume_poc(chart["ohlcv"], cfg.get("POC_BINS", 50))

    vwap_1h = session_vwap(tf_data["1h"], cfg.get("MARKET_TZ", "America/New_York"),
                           cfg.get("MARKET_OPEN", "09:30"), cfg.get("MARKET_CLOSE", "16:00"))
    chart["vwap"] = vwap_1h.loc[chart["ohlcv"].index]

    # ---- daily structure (HH/HL) + daily chart - built from the daily bars we
    # already resampled from the hourly download, so no extra data is fetched
    hhhl = hhhl_stats(tf_data["1D"], cfg.get("HHHL_DAYS", 20), cfg.get("MARKET_TZ", "America/New_York"),
                      cfg.get("MARKET_CLOSE", "16:00"))
    chart["hhhl_by_date"] = hhhl["by_date"]
    n_daily = cfg.get("DAILY_CHART_DAYS", 20)
    daily_mas = {}
    for ma_type, period in cfg.get("CHART_MAS", [(cfg["MA_TYPE"], cfg["MA_PERIOD"])]):
        daily_mas[f"{ma_type.upper()}{period}"] = moving_average(tf_data["1D"], period, ma_type).iloc[-n_daily:]
    chart["daily"] = {"ohlcv": tf_data["1D"].iloc[-n_daily:], "overlays": daily_mas,
                      "today_in_progress": hhhl["today_in_progress"]}
    avwap = anchored_vwap(tf_data["1h"], chart["daily"]["ohlcv"], cfg.get("AVWAP_ANCHOR", "low"),
                          cfg.get("MARKET_TZ", "America/New_York"))
    chart["daily"]["avwap"] = avwap
    # levels are searched over SR_DAYS (independent of how many days the chart shows)
    day_sr = swing_sr_levels(tf_data["1D"].iloc[-cfg.get("SR_DAYS", 60):], current_price, daily_atr_val, stop,
                             pivot=cfg.get("SR_PIVOT", 2), merge_atr=cfg.get("SR_MERGE_ATR", 0.5),
                             min_r=cfg.get("TARGET_MIN_R", 2.0))
    chart["daily"]["sr"] = day_sr
    if day_sr["target"]:                     # industry-standard target replaces the old 1.5x risk target
        take_profit_target = day_sr["target"]["price"]
    vwap_valid = vwap_1h.dropna()
    vwap_now = round(float(vwap_valid.iloc[-1]), 2) if not vwap_valid.empty else None
    above_vwap = (current_price > vwap_now) if vwap_now is not None else None

    pre_market, pre_market_time, pre_market_info = None, None, None
    if cfg.get("PREMARKET_ENABLED", True) and pre_market_batch is not None:
        # fetched for all tickers at once in main() - see prefetch_pre_market()
        pre_market_info = pre_market_batch.get(cfg["TICKER"])
        if pre_market_info and pre_market_info.get("price") is not None:
            pre_market, pre_market_time = pre_market_info["price"], pre_market_info["time"]
        else:
            pre_market_info = None
    elif cfg.get("PREMARKET_ENABLED", True):
        feed = cfg.get("ALPACA_FEED", "auto")
        pm_kw = dict(tz=cfg.get("MARKET_TZ", "America/New_York"),
                     start_hhmm=cfg.get("PREMARKET_START", "04:00"), open_hhmm=cfg.get("MARKET_OPEN", "09:30"))
        if feed == "auto":
            pre_market_info = get_best_pre_market_price(
                cfg["TICKER"], headers=cfg.get("ALPACA_HEADERS"),
                sip_delay_minutes=cfg.get("ALPACA_SIP_DELAY_MIN", 16),
                move_pct=cfg.get("PREMARKET_MOVE_PCT", 0.3),
                max_iex_gap_pct=cfg.get("PREMARKET_IEX_MAX_GAP_PCT", 0.5), with_details=True, **pm_kw)
            pre_market, pre_market_time = pre_market_info["price"], pre_market_info["time"]
        else:
            pre_market, pre_market_time = get_pre_market_price(
                cfg["TICKER"], headers=cfg.get("ALPACA_HEADERS"), feed=feed,
                delay_minutes=cfg.get("ALPACA_SIP_DELAY_MIN", 16) if feed == "sip" else 0,
                with_time=True, **pm_kw)
            if pre_market is not None:
                pre_market_info = {"source": feed, "reason": "fixed feed (ALPACA_FEED)"}

    # The channel is always computed because its slope decides whether the ticker
    # goes to the _up or _down report; LRC_ENABLED only controls whether it's shown.
    lrc_fit = linear_regression_channel(chart["ohlcv"], cfg.get("LRC_LENGTH"),
                                        cfg.get("LRC_DEV", 2.0), cfg.get("LRC_SOURCE", "Close"))
    # "up" needs a rising channel AND a decent fit: a rising line through choppy
    # prices (low R-squared) isn't a real uptrend, so it goes to the _down report.
    min_r2 = cfg.get("LRC_MIN_R2_UP", 0.5)
    if lrc_fit is not None:
        if lrc_fit["slope_per_bar"] < 0:
            trend_dir, trend_reason = "down", "channel sloping down"
        elif lrc_fit["r2"] < min_r2:
            trend_dir, trend_reason = "down", f"channel rising but weak fit (R2 {lrc_fit['r2']} < {min_r2})"
        else:
            trend_dir, trend_reason = "up", f"channel rising (R2 {lrc_fit['r2']})"
    else:  # too few bars to fit - fall back to first vs last close in the window
        w = chart["ohlcv"]["Close"]
        trend_dir = "up" if float(w.iloc[-1]) >= float(w.iloc[0]) else "down"
        trend_reason = "too few bars for a channel - first vs last close"
    lrc = lrc_fit if cfg.get("LRC_ENABLED", True) else None
    chart["lrc"] = lrc

    # ---- fundamentals / analyst / catalysts: one shared yf.Ticker and one .info,
    # with the once-a-day parts cached on disk (see daily_cached)
    tk = cfg["TICKER"]
    tkr = yf.Ticker(tk)
    info = daily_cached("info", tk, lambda: fetch_info(tk, tkr))
    fundamentals = fetch_fundamentals(tk, info=info)
    analyst = daily_cached(
        "analyst", tk, lambda: fetch_analyst_data(tk, tkr=tkr, info=info),
        is_valid=lambda a: bool(info) and (a.get("recommendation_key") != "N/A" or a.get("eps_current") != "N/A"))
    earnings_ratings = daily_cached(
        "earnings_ratings", tk, lambda: get_earnings_and_ratings(tk, yf_ticker=tkr, info=info),
        is_valid=lambda er: bool(info))

    # One Finnhub call covers both the News row (last NEWS_DAYS) and the
    # buyback/guidance keyword scan (last CATALYST_NEWS_DAYS) - no Yahoo news call.
    # Finnhub's free plan allows 60 calls/minute, so news is only fetched for the
    # tickers that NEWS_FOR selects (default: shown in the _up report), and each
    # ticker's result is reused for NEWS_CACHE_MIN minutes.
    news_days = cfg.get("NEWS_DAYS", 0)
    catalyst_days = cfg.get("CATALYST_NEWS_DAYS", 14)
    news_for = cfg.get("NEWS_FOR", "up")
    will_be_shown = not all(report_filter_checks({"signal": signal, "analyst": analyst}).values())
    if news_for == "none":
        news_skipped = "news turned off (NEWS_FOR)"
    elif not will_be_shown:
        news_skipped = "ticker filtered out"
    elif news_for == "up" and trend_dir != "up":
        news_skipped = "not fetched for down-trend tickers"
    else:
        news_skipped = None

    news_all = None
    if news_skipped is None:
        fetch_days = max(news_days, catalyst_days)
        news_all = ttl_cached(
            "news", f"{tk}_{fetch_days}",
            lambda: fetch_company_news(tk, api_key=cfg.get("FINNHUB_API_KEY"), days=fetch_days,
                                       max_items=None, tz=cfg.get("NEWS_TZ", "America/New_York")),
            ttl_minutes=cfg.get("NEWS_CACHE_MIN", 60))
    # skipped -> [] so event_catalysts doesn't fall back to a Yahoo news call
    headlines = [] if news_skipped else (None if news_all is None else [(n["date"], n["headline"]) for n in news_all])

    catalysts = get_event_catalysts(tk, news_lookback_days=catalyst_days, sp500_members=sp500_members,
                                    yf_ticker=tkr, info=info, headlines=headlines,
                                    social_table=social_table, earnings_ratings=earnings_ratings).to_dict()

    news_cutoff = dt.date.today() - dt.timedelta(days=news_days)
    news = [n for n in (news_all or []) if n["date"] >= news_cutoff][:cfg.get("NEWS_MAX", 25)]
    catalysts["news_skipped"] = news_skipped

    return {
        "ticker": cfg["TICKER"],
        "cfg": cfg,
        "current_price": current_price,
        "results": results,
        "sr_levels": sr_levels,
        "today_range": today_range,
        "daily_atr": round(daily_atr_val, 2),
        "h1_macd_ok": h1_macd_ok,
        "vol_ok": vol_ok,
        "stop": stop,
        "take_profit": take_profit_target,
        "entry_support": entry_support,
        "signal": signal,
        "fundamentals": fundamentals,
        "analyst": analyst,
        "catalysts": catalysts,
        "chart": chart,
        "hhhl": {k: v for k, v in hhhl.items() if k != "by_date"},
        "day_sr": day_sr,
        "avwap": None if not avwap else {
            "value": avwap["value"], "anchor_date": avwap["anchor_date"].isoformat(),
            "anchor_type": avwap["anchor_type"], "anchor_price": round(avwap["anchor_price"], 2),
            "above": (current_price > avwap["value"]) if avwap["value"] is not None else None},
        "poc": poc,
        "vwap": vwap_now,
        "above_vwap": above_vwap,
        "lrc": lrc,
        "trend_dir": trend_dir,
        "trend_reason": trend_reason,
        "pre_market": pre_market,
        "pre_market_time": pre_market_time,
        "pre_market_info": pre_market_info,
        "news": news,
        # raw (unformatted) yfinance values behind the Fundamentals table, for the CSV
        "fund_raw": {key: info.get(key) for _, key, _ in FUNDAMENTAL_FIELDS} if info else {},
    }


def print_report(report: dict) -> None:
    """Console rendering of a run_screen() report - unchanged formatting from before."""
    cfg = report["cfg"]
    results = report["results"]
    sr_levels = report["sr_levels"]
    today_range = report["today_range"]
    signal = report["signal"]
    current_price = report["current_price"]
    entry_support = report["entry_support"]

    print(f"\n=== {cfg['TICKER']} Multi-Indicator Signal Check ===")
    print(f"Current price: {current_price:.2f}\n")
    for tf in ["1D", "4h", "1h"]:
        r = results[tf]
        sr = sr_levels[tf]
        patterns = ", ".join(r["patterns_detected"]) if r["patterns_detected"] else "none"
        print(f"[{tf}] close={r['close']} {cfg['MA_TYPE'].upper()}{cfg['MA_PERIOD']}={r['MA']} trend_up={r['trend_up']!s:<5} "
              f"%K={r['%K']:>6} %D={r['%D']:>6} K>D={r['k_above_d']!s:<5} "
              f"stoch_bullish={r['stoch_bullish']!s:<5} overbought={r['overbought']!s:<5} "
              f"candles=[{patterns}]")
        print(f"      support={sr['support']}  resistance={sr['resistance']}  "
              f"(last {cfg['SR_LOOKBACK'][tf]} bars)")

    print(f"\nToday's range: {today_range['day_low']} - {today_range['day_high']} "
          f"(range: {today_range['day_range']})")

    if report.get("poc") is not None:
        print(f"POC ({cfg.get('CHART_DAYS', 7)}d volume profile): {report['poc']}")
    if report.get("vwap") is not None:
        side = "above" if report["above_vwap"] else "below"
        print(f"Session VWAP: {report['vwap']} (price {side} VWAP)")
    if report.get("pre_market") is not None:
        info = report.get("pre_market_info") or {}
        print(f"Pre-market price: {report['pre_market']} (as of {report.get('pre_market_time')} ET, "
              f"{str(info.get('source', '')).upper()}: {info.get('reason', '')})")
    st = report.get("hhhl")
    if st:
        print(f"HH/HL days (last {st['window']} completed): {st['up_days']} up / {st['down_days']} down / "
              f"{st['mixed_days']} mixed (net {st['net']:+d}); current streak {_streak_text(st['current_streak'])}, "
              f"previous {_streak_text(st['previous_streak'])}")
    lrc = report.get("lrc")
    if lrc:
        print(f"Regression channel ({lrc['bars']} bars, {lrc['dev']}σ): {lrc['last_lower']} / "
              f"{lrc['last_mid']} / {lrc['last_upper']}  slope {lrc['slope_pct_per_day']:+.2f}%/day  "
              f"R² {lrc['r2']}  price at {lrc['position_pct']:.0f}% of channel")

    print(f"\n1h MACD bullish: {report['h1_macd_ok']}")
    print(f"1h volume confirmed (>= {cfg['VOLUME_MULTIPLIER']}x {cfg['VOLUME_MA_PERIOD']}-period avg): {report['vol_ok']}")
    print(f"ATR({cfg['ATR_PERIOD']}) suggested stop: {report['stop']} "
          f"({cfg['ATR_STOP_MULTIPLIER']}x ATR below current price)")

    # <--- ADD THIS BLOCK --->
    risk = report['current_price'] - report['stop']
    tgt = (report.get("day_sr") or {}).get("target") or {}
    print(f"Take-Profit Target: {report['take_profit']} ({tgt.get('source', '1.5x risk')}, "
          f"{tgt.get('r')}R; risking {risk:.2f} per share)")
    # <--------------------->

    print(f"\nSignal: {signal['signal']}  (hard requirements met: {signal['hard_requirements_met']}, "
          f"structural confirmation: {signal['structural_confirmation']}, soft score {signal['soft_score']})")

    print("\n  Hard requirement #1 - Daily & 4h trend/stochastic:")
    hrd = signal["hard_requirements_detail"]
    for label, key in [
        ("Daily trend up (close > MA)", "daily_trend_up"),
        ("Daily stochastic bullish", "daily_stoch_bullish"),
        ("4h trend up (close > MA)", "4h_trend_up"),
        ("4h stochastic bullish", "4h_stoch_bullish"),
    ]:
        mark = "PASS" if hrd[key] else "FAIL"
        print(f"      [{mark}] {label}: {hrd[key]}")

    print("\n  Hard requirement #2 - Structural candlestick confirmation (daily OR 4h):")
    scd = signal["structural_confirmation_detail"]
    daily_patterns = ", ".join(scd["daily_patterns"]) if scd["daily_patterns"] else "none"
    h4_patterns = ", ".join(scd["4h_patterns"]) if scd["4h_patterns"] else "none"
    print(f"      [{'PASS' if scd['daily_bullish_pattern'] else 'FAIL'}] Daily bullish pattern: "
          f"{scd['daily_bullish_pattern']}  (patterns: {daily_patterns})")
    print(f"      [{'PASS' if scd['4h_bullish_pattern'] else 'FAIL'}] 4h bullish pattern: "
          f"{scd['4h_bullish_pattern']}  (patterns: {h4_patterns})")
    print(f"      -> structural_confirmation (daily OR 4h) = {signal['structural_confirmation']}")

    print(f"\n  near_support measured against 1h support: {entry_support}")
    print("\n  Soft conditions (need >= 3/5):")
    for k, v in signal.items():
        if k not in ("signal", "hard_requirements_met", "hard_requirements_detail",
                     "structural_confirmation", "structural_confirmation_detail", "soft_score"):
            print(f"      - {k}: {v}")


    fundamentals = report.get("fundamentals") or {}
    if fundamentals:
        print("Fundamentals:")
        for label, value in fundamentals.items():
            print(f"  {label:<24} {value}")
        print()

    analyst = report.get("analyst") or {}
    if analyst:
        print("Analyst Recommendations:")
        print(f"  Consensus: {analyst['recommendation_key']}  "
              f"(mean score {analyst['recommendation_mean']}, "
              f"{analyst['num_analyst_opinions']} analysts)")
        counts = analyst.get("recommendation_counts") or {}
        if counts:
            counts_str = "  ".join(f"{label}: {n}" for label, n in counts.items())
            print(f"  {counts_str}")
        print()

        print("EPS Estimate Revisions (current quarter):")
        print(f"  Current consensus EPS: {analyst['eps_current']}   "
              f"30 days ago: {analyst['eps_30d_ago']}   "
              f"# analysts: {analyst['eps_num_analysts']}")
        if analyst["eps_improving"] is not None:
            trend_str = "IMPROVING (revised up)" if analyst["eps_improving"] else "DETERIORATING (revised down)"
            print(f"  EPS trend: {trend_str}")
        else:
            print("  EPS trend: N/A")
        print()



def screen_ticker(ticker):
    """Run the screen for a single ticker and print its console report."""
    report = run_screen(ticker, CONFIGH)
    #print_report(report)
    return report


# ---------------------------------------------------------------- HTML report

def _badge(ok: bool, true_text: str = "PASS", false_text: str = "FAIL") -> str:
    cls = "pass" if ok else "fail"
    text = true_text if ok else false_text
    return f'<span class="badge {cls}">{html.escape(str(text))}</span>'


def _four_col_table(items: list, extra_cls: str = "") -> str:
    """Render (label_html, value_html) pairs as a 4-column table
    (label | value | label | value). First half runs down the left pair, second
    half down the right pair, so it still reads top-to-bottom. Inputs must
    already be HTML-safe (so values can contain badges)."""
    if not items:
        return ""
    half = (len(items) + 1) // 2
    rows = ""
    for i in range(half):
        right = items[i + half] if i + half < len(items) else ("", "")
        rows += (f"<tr><td>{items[i][0]}</td><td>{items[i][1]}</td>"
                 f"<td>{right[0]}</td><td>{right[1]}</td></tr>")
    return f"""
      <table class="detail-table four-col {extra_cls}">
        <colgroup><col class="c-lbl"><col class="c-val"><col class="c-lbl"><col class="c-val"></colgroup>
        <tbody>{rows}
      </tbody></table>"""


def _hhmm_to_min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def render_price_chart_svg(report: dict) -> str:
    """Inline SVG candlestick chart of the last CHART_DAYS trading days (1h bars),
    with MA overlays, POC, 1h support, stop and take-profit levels, dotted
    market open/close markers, and a volume strip. Pure SVG - no JS or
    external libraries. Hover a candle for its OHLCV."""
    chart = report.get("chart")
    if not chart or chart["ohlcv"].empty:
        return ""
    df = chart["ohlcv"]
    overlays = chart.get("overlays") or {}
    cfg = report["cfg"]
    n = len(df)

    # ---- timestamps in the exchange's local time (for sessions + labels)
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert(cfg.get("MARKET_TZ", "America/New_York"))
    dates = idx.date
    mins = np.asarray(idx.hour * 60 + idx.minute)
    open_m = _hhmm_to_min(cfg.get("MARKET_OPEN", "09:30"))
    close_m = _hhmm_to_min(cfg.get("MARKET_CLOSE", "16:00"))

    # ---- geometry: a small gap between sessions so close/open lines don't overlap
    W = 1000  # ~ the card's rendered width on desktop, so text renders near 1:1
    pad_l, pad_r, pad_t = 56, 160, 10
    AXIS_W = 46  # room for right-axis tick labels; level labels sit to the right of it
    price_h, gap, vol_h, axis_h = 500, 12, 110, 22
    H = pad_t + price_h + gap + vol_h + axis_h
    plot_w = W - pad_l - pad_r
    new_day = [i == 0 or dates[i] != dates[i - 1] for i in range(n)]
    n_days = sum(new_day)
    SESSION_GAP = 0.8  # in candle widths
    step = plot_w / (n + SESSION_GAP * (n_days - 1))
    body_w = max(1.5, step * 0.62)

    left, cur = [], float(pad_l)
    for i in range(n):
        if new_day[i] and i > 0:
            cur += SESSION_GAP * step
        left.append(cur)
        cur += step

    def x(i):
        return left[i] + step / 2

    # ---- horizontal levels
    levels = [
        ("Target", report.get("take_profit"), "lvl-target"),
        ("Pre-mkt", report.get("pre_market"), "lvl-premkt"),
        ("POC", report.get("poc"), "lvl-poc"),
        ("Support", report.get("entry_support"), "lvl-support"),
        ("Stop", report.get("stop"), "lvl-stop"),
    ]
    levels = [(lbl, float(v), cls) for lbl, v, cls in levels if v is not None and pd.notna(v)]

    overlays_clean = {name: s.dropna() for name, s in overlays.items()}
    vwap_s = chart.get("vwap")
    vwap_clean = vwap_s.dropna() if vwap_s is not None else pd.Series(dtype=float)
    if not vwap_clean.empty:
        overlays_clean_for_range = list(overlays_clean.values()) + [vwap_clean]
    else:
        overlays_clean_for_range = list(overlays_clean.values())
    lrc = chart.get("lrc")
    if lrc:
        overlays_clean_for_range += [lrc["upper"], lrc["lower"]]
    candidates = [float(df["Low"].min()), float(df["High"].max())] + [v for _, v, _ in levels]
    for s in overlays_clean_for_range:
        if not s.empty:
            candidates += [float(s.min()), float(s.max())]
    lo, hi = min(candidates), max(candidates)
    pad_p = (hi - lo) * 0.04 or 1.0
    lo, hi = lo - pad_p, hi + pad_p

    def y(p):
        return pad_t + (hi - p) / (hi - lo) * price_h

    vol_top = pad_t + price_h + gap
    vol_bot = vol_top + vol_h
    vmax = float(df["Volume"].max()) or 1.0

    def vy(v):
        return vol_bot - (v / vmax) * vol_h

    parts = []

    # ---- price grid: "nice" round-number ticks (…0.5, 1, 2, 2.5, 5…), ~10 across
    # the panel (~50px apart), dotted gridlines, labels on both axes + tick marks on the right
    raw_step = (hi - lo) / 10
    mag = 10 ** np.floor(np.log10(raw_step))
    tick_step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda st: abs(np.log(st / raw_step)))
    decimals = 2 if tick_step < 1 else (1 if tick_step % 1 else 0)
    right_x = pad_l + plot_w
    p = np.ceil(lo / tick_step) * tick_step
    while p <= hi:
        yy = y(p)
        label = f"{p:.{decimals}f}"
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{right_x}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<line class="tick" x1="{right_x}" x2="{right_x + 4}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="axis" x="{pad_l - 6}" y="{yy + 3:.1f}" text-anchor="end">{label}</text>')
        parts.append(f'<text class="axis" x="{right_x + 7}" y="{yy + 3:.1f}">{label}</text>')
        p += tick_step
    parts.append(f'<line class="tick" x1="{right_x}" x2="{right_x}" y1="{pad_t}" y2="{pad_t + price_h}"/>')

    hhhl_by_date = chart.get("hhhl_by_date") or {}   # daily HH/HL type per date (triangle after the date)

    # ---- market open / close markers (dotted white) + date labels, per session
    day_starts = [i for i in range(n) if new_day[i]] + [n]
    for d in range(n_days):
        ks = list(range(day_starts[d], day_starts[d + 1]))
        starts = mins[ks]
        kind = hhhl_by_date.get(idx[ks[0]].date())
        mark = ('<tspan class="hhhl-mark up"> &#9650;</tspan>' if kind == "up" else
                '<tspan class="hhhl-mark down"> &#9660;</tspan>' if kind == "down" else "")
        parts.append(f'<text class="axis" x="{left[ks[0]] + 2:.1f}" y="{H - 6}">'
                     f'{idx[ks[0]].strftime("%a %d %b")}{mark}</text>')

        # open: left edge of first bar if it starts at/after the open,
        # else interpolate inside the bar that contains the open (pre-market data)
        if starts[0] >= open_m:
            ox = left[ks[0]]
        else:
            j = max(j for j in range(len(ks)) if starts[j] <= open_m)
            nxt = starts[j + 1] if j + 1 < len(ks) else starts[j] + 60
            ox = left[ks[j]] + step * min(1.0, (open_m - starts[j]) / max(nxt - starts[j], 1))

        # close: right edge of the last bar if it's the closing bar, else
        # interpolate (after-hours data); skip if the session hasn't reached it yet
        cx = None
        if starts[-1] < close_m:
            if close_m - starts[-1] <= 60:
                cx = left[ks[-1]] + step
        else:
            j = max(j for j in range(len(ks)) if starts[j] <= close_m) if starts[0] <= close_m else None
            if j is not None:
                nxt = starts[j + 1] if j + 1 < len(ks) else starts[j] + 60
                cx = left[ks[j]] + step * min(1.0, (close_m - starts[j]) / max(nxt - starts[j], 1))

        for xx, what in ((ox, "Market open"), (cx, "Market close")):
            if xx is not None:
                parts.append(f'<line class="session" x1="{xx:.1f}" x2="{xx:.1f}" y1="{pad_t}" y2="{vol_bot}">'
                             f'<title>{what} {idx[ks[0]].strftime("%a %d %b")}</title></line>')

    # ---- standard filled candles: green if close >= open, red otherwise
    color_cls = ["up" if c >= o else "down" for o, c in zip(df["Open"], df["Close"])]

    # ---- linear regression channel (drawn behind candles)
    if lrc:
        pos_lrc = {ts: i for i, ts in enumerate(df.index)}
        def _pts(series):
            return [(x(pos_lrc[ts]), y(float(v))) for ts, v in series.items() if ts in pos_lrc]
        up_pts, lo_pts, mid_pts = _pts(lrc["upper"]), _pts(lrc["lower"]), _pts(lrc["mid"])
        if up_pts:
            poly = " ".join(f"{a:.1f},{b:.1f}" for a, b in up_pts + lo_pts[::-1])
            parts.append(f'<polygon class="lrc-fill" points="{poly}"/>')
            for cls, pts in (("lrc-band", up_pts), ("lrc-band", lo_pts), ("lrc-mid", mid_pts)):
                parts.append(f'<polyline class="{cls}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')

    # ---- volume bars
    for i, (_, row) in enumerate(df.iterrows()):
        cls = color_cls[i]
        top = vy(float(row["Volume"]))
        parts.append(f'<rect class="vol {cls}" x="{x(i) - body_w / 2:.1f}" y="{top:.1f}" '
                     f'width="{body_w:.1f}" height="{vol_bot - top:.1f}"/>')

    # ---- candles (with native hover tooltip)
    for i, (_, row) in enumerate(df.iterrows()):
        o, h, l, c, v = (float(row[k]) for k in ("Open", "High", "Low", "Close", "Volume"))
        cls = color_cls[i]
        body_top, body_bot = y(max(o, c)), y(min(o, c))
        tip = f"{idx[i].strftime('%a %d %b %H:%M')}  O {o:.2f}  H {h:.2f}  L {l:.2f}  C {c:.2f}  Vol {v:,.0f}"
        parts.append(
            f'<g class="candle {cls}"><title>{html.escape(tip)}</title>'
            f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(h):.1f}" y2="{y(l):.1f}"/>'
            f'<rect x="{x(i) - body_w / 2:.1f}" y="{body_top:.1f}" width="{body_w:.1f}" '
            f'height="{max(body_bot - body_top, 1):.1f}"/></g>'
        )

    # ---- moving-average overlays
    pos = {ts: i for i, ts in enumerate(df.index)}
    legend_mas = []
    for k, (name, s) in enumerate(overlays_clean.items()):
        pts = " ".join(f"{x(pos[ts]):.1f},{y(float(val)):.1f}" for ts, val in s.items())
        if pts:
            parts.append(f'<polyline class="ma ma-{k % 4}" points="{pts}"><title>{html.escape(name)}</title></polyline>')
        legend_mas.append(f'<i class="sw ma-sw ma-{k % 4}"></i>{html.escape(name)}')

    # ---- session VWAP: one segment per session (it resets at each open)
    if not vwap_clean.empty:
        vdays = np.asarray((vwap_clean.index.tz_convert(cfg.get("MARKET_TZ", "America/New_York"))
                            if vwap_clean.index.tz is not None else vwap_clean.index).date)
        for dday in pd.unique(vdays):
            seg = vwap_clean[vdays == dday]
            pts = " ".join(f"{x(pos[ts]):.1f},{y(float(val)):.1f}" for ts, val in seg.items())
            if len(seg) == 1:
                xx, yy = x(pos[seg.index[0]]), y(float(seg.iloc[0]))
                pts = f"{xx - step / 2:.1f},{yy:.1f} {xx + step / 2:.1f},{yy:.1f}"
            parts.append(f'<polyline class="vwap" points="{pts}"><title>VWAP {dday}</title></polyline>')
        levels.append(("VWAP", float(vwap_clean.iloc[-1]), "lvl-vwap"))
        legend_mas.append('<i class="sw ma-sw vwap-sw"></i>VWAP')

    # ---- horizontal levels, labelled on the right (nudged apart if they collide)
    label_ys = []
    for lbl, val, cls in sorted(levels, key=lambda t: -t[1]):
        yy = y(val)
        ly = yy + 3
        while any(abs(ly - prev) < 12 for prev in label_ys):
            ly += 12
        label_ys.append(ly)
        if cls != "lvl-vwap":  # VWAP is a moving line - label only, no horizontal
            parts.append(f'<line class="lvl {cls}" x1="{pad_l}" x2="{pad_l + plot_w}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="lvl-label {cls}" x="{pad_l + plot_w + AXIS_W}" y="{ly:.1f}">{lbl} {val:.2f}</text>')

    # ---- last-price marker
    last_c = float(df["Close"].iloc[-1])
    parts.append(f'<circle class="last" cx="{x(n - 1):.1f}" cy="{y(last_c):.1f}" r="3"/>')

    lrc_legend = ' <i class="sw lrc-sw"></i>Reg. channel' if lrc else ""
    if report.get("pre_market") is not None:
        lrc_legend = ' <i class="sw premkt-sw"></i>Pre-market' + lrc_legend
    first_o = float(df["Open"].iloc[0])
    chg = (last_c / first_o - 1) * 100 if first_o else 0.0
    chg_cls = "pass" if chg >= 0 else "fail"

    return f"""
      <div class="chart-wrap">
        <div class="chart-head">
          <span>Last {n_days} trading days &middot; 1h candles</span>
          <span class="badge {chg_cls}">{chg:+.2f}%</span>
          <span class="legend">{' '.join(legend_mas)}
            <i class="sw lvl-poc"></i>POC <i class="sw lvl-support"></i>1h support
            <i class="sw lvl-stop"></i>Stop <i class="sw lvl-target"></i>Target
            <i class="sw session-sw"></i>Open/close{lrc_legend}</span>
        </div>
        <svg class="price-chart" viewBox="0 0 {W} {H}"
             role="img" aria-label="{html.escape(report['ticker'])} {n_days}-day price chart">
          {''.join(parts)}
        </svg>
      </div>"""


def render_daily_chart_svg(report: dict) -> str:
    """Inline SVG of the last DAILY_CHART_DAYS daily candles (built from the
    same hourly download - no extra data), with the MA overlays, daily
    support/resistance, stop and target, HH/HL markers (green triangle under an
    up day, red over a down day) and a volume strip."""
    daily = (report.get("chart") or {}).get("daily")
    if not daily or daily["ohlcv"].empty:
        return ""
    df = daily["ohlcv"]
    overlays = {k: v.dropna() for k, v in (daily.get("overlays") or {}).items()}
    cfg = report["cfg"]
    by_date = (report.get("chart") or {}).get("hhhl_by_date") or {}
    tz = cfg.get("MARKET_TZ", "America/New_York")
    idx = df.index.tz_convert(tz) if df.index.tz is not None else df.index
    n = len(df)
    in_progress = daily.get("today_in_progress", False)

    W = 1000
    pad_l, pad_r, pad_t = 56, 160, 16
    AXIS_W = 46
    price_h, gap, vol_h, axis_h = 320, 10, 70, 34
    H = pad_t + price_h + gap + vol_h + axis_h
    plot_w = W - pad_l - pad_r
    step = plot_w / n
    body_w = max(2.0, step * 0.6)

    def x(i):
        return pad_l + step * (i + 0.5)

    sr = (report.get("sr_levels") or {}).get("1D") or {}
    dsr = daily.get("sr") or {}
    levels = []
    for k, lvl in enumerate((dsr.get("resistances") or [])[:2]):
        rr = f" {lvl['r']:.1f}R" if lvl.get("r") is not None else ""
        levels.append((f"T{k + 1}{rr}", lvl["price"], "lvl-target" if k == 0 else "lvl-target2"))
    tgt = dsr.get("target") or {}
    if tgt and not dsr.get("resistances"):   # fallback target (no resistance above)
        levels.append((f"T {tgt['r']:g}R", tgt["price"], "lvl-target"))
    for k, lvl in enumerate((dsr.get("supports") or [])[:2]):
        levels.append((f"S{k + 1} x{lvl['touches']}", lvl["price"], "lvl-support" if k == 0 else "lvl-support2"))
    if report.get("pre_market") is not None:   # same yellow dotted line as on the 1h chart
        levels.append(("Pre-mkt", report["pre_market"], "lvl-premkt"))
    levels += [
              ("Stop", report.get("stop"), "lvl-stop")]
    levels = [(lbl, float(v), cls) for lbl, v, cls in levels if v is not None and pd.notna(v)]

    av = daily.get("avwap")
    av_s = av["series"].dropna() if av else pd.Series(dtype=float)
    cands = [float(df["Low"].min()), float(df["High"].max())] + [v for _, v, _ in levels]
    if not av_s.empty:
        cands += [float(av_s.min()), float(av_s.max())]
    for sser in overlays.values():
        if not sser.empty:
            cands += [float(sser.min()), float(sser.max())]
    lo, hi = min(cands), max(cands)
    pad_p = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad_p, hi + pad_p

    def y(p):
        return pad_t + (hi - p) / (hi - lo) * price_h

    vol_top = pad_t + price_h + gap
    vol_bot = vol_top + vol_h
    vmax = float(df["Volume"].max()) or 1.0
    parts = []

    # ---- grid (same "nice number" ticks as the hourly chart)
    raw_step = (hi - lo) / 8
    mag = 10 ** np.floor(np.log10(raw_step))
    tick_step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda st: abs(np.log(st / raw_step)))
    decimals = 2 if tick_step < 1 else (1 if tick_step % 1 else 0)
    right_x = pad_l + plot_w
    p = np.ceil(lo / tick_step) * tick_step
    while p <= hi:
        yy = y(p)
        label = f"{p:.{decimals}f}"
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{right_x}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<line class="tick" x1="{right_x}" x2="{right_x + 4}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="axis" x="{pad_l - 6}" y="{yy + 3:.1f}" text-anchor="end">{label}</text>')
        parts.append(f'<text class="axis" x="{right_x + 7}" y="{yy + 3:.1f}">{label}</text>')
        p += tick_step
    parts.append(f'<line class="tick" x1="{right_x}" x2="{right_x}" y1="{pad_t}" y2="{pad_t + price_h}"/>')

    # ---- date labels with the abbreviated day name underneath. Every bar is
    # labelled when there's room (~20 bars); with many bars, every other one.
    every = max(1, int(np.ceil(36 / step)))   # keep labels ~36px+ apart (20 bars: all, 50 bars: every 3rd)
    for i in range(n):
        if (n - 1 - i) % every == 0:
            parts.append(f'<text class="axis" x="{x(i):.1f}" y="{vol_bot + 14}" text-anchor="middle">'
                         f'{idx[i].strftime("%d %b")}</text>')
            parts.append(f'<text class="axis dayname" x="{x(i):.1f}" y="{vol_bot + 27}" text-anchor="middle">'
                         f'{idx[i].strftime("%a")}</text>')

    # ---- dotted vertical line at the start of each new week (just before Monday,
    # or the first trading day of the week when Monday is a holiday)
    for i in range(1, n):
        if idx[i].isocalendar()[:2] != idx[i - 1].isocalendar()[:2]:
            wx = x(i) - step / 2
            parts.append(f'<line class="weeksep" x1="{wx:.1f}" x2="{wx:.1f}" y1="{pad_t}" y2="{vol_bot}">'
                         f'<title>week of {idx[i].strftime("%d %b")}</title></line>')

    # ---- volume, candles, HH/HL markers
    for i, (_, row) in enumerate(df.iterrows()):
        o, h, l, c, v = (float(row[k]) for k in ("Open", "High", "Low", "Close", "Volume"))
        cls = "up" if c >= o else "down"
        partial = in_progress and i == n - 1
        top = vol_bot - (v / vmax) * vol_h
        parts.append(f'<rect class="vol {cls}" x="{x(i) - body_w / 2:.1f}" y="{top:.1f}" '
                     f'width="{body_w:.1f}" height="{vol_bot - top:.1f}"/>')
        kind = by_date.get(idx[i].date())
        tip = (f"{idx[i].strftime('%a %d %b')}{' (in progress)' if partial else ''}  O {o:.2f}  H {h:.2f}  "
               f"L {l:.2f}  C {c:.2f}  Vol {v:,.0f}"
               + (f"  | {'higher high + higher low' if kind == 'up' else 'lower high + lower low' if kind == 'down' else 'mixed'}"
                  if kind else ""))
        body_top, body_bot = y(max(o, c)), y(min(o, c))
        parts.append(
            f'<g class="candle {cls}{" partial" if partial else ""}"><title>{html.escape(tip)}</title>'
            f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(h):.1f}" y2="{y(l):.1f}"/>'
            f'<rect x="{x(i) - body_w / 2:.1f}" y="{body_top:.1f}" width="{body_w:.1f}" '
            f'height="{max(body_bot - body_top, 1):.1f}"/></g>')
        if kind == "up":
            parts.append(f'<text class="hhhl-mark up" x="{x(i):.1f}" y="{y(l) + 13:.1f}" text-anchor="middle">&#9650;</text>')
        elif kind == "down":
            parts.append(f'<text class="hhhl-mark down" x="{x(i):.1f}" y="{y(h) - 5:.1f}" text-anchor="middle">&#9660;</text>')

    # ---- MA overlays
    pos = {ts: i for i, ts in enumerate(df.index)}
    legend = []
    for k, (name, sser) in enumerate(overlays.items()):
        pts = " ".join(f"{x(pos[ts]):.1f},{y(float(val)):.1f}" for ts, val in sser.items() if ts in pos)
        if pts:
            parts.append(f'<polyline class="ma ma-{k % 4}" points="{pts}"><title>{html.escape(name)} (daily)</title></polyline>')
        legend.append(f'<i class="sw ma-sw ma-{k % 4}"></i>{html.escape(name)}')

    # ---- anchored VWAP (line from the anchor day on, dot on the anchor low/high)
    if not av_s.empty:
        pts = " ".join(f"{x(pos[ts]):.1f},{y(float(v)):.1f}" for ts, v in av_s.items() if ts in pos)
        parts.append(f'<polyline class="avwap" points="{pts}"><title>Anchored VWAP from the '
                     f'{"low" if av["anchor_type"] == "low" else "high"} of '
                     f'{pd.Timestamp(av["anchor_date"]).strftime("%d %b")}</title></polyline>')
        if av["anchor_ts"] in pos:
            parts.append(f'<circle class="avwap-anchor" cx="{x(pos[av["anchor_ts"]]):.1f}" '
                         f'cy="{y(av["anchor_price"]):.1f}" r="3.5"/>')
        levels.append(("AVWAP", float(av_s.iloc[-1]), "lvl-avwap"))
        legend.append('<i class="sw ma-sw avwap-sw"></i>Anchored VWAP')

    # ---- horizontal levels with right-side labels
    label_ys = []
    for lbl, val, cls in sorted(levels, key=lambda t: -t[1]):
        yy = y(val)
        ly = yy + 3
        while any(abs(ly - prev) < 12 for prev in label_ys):
            ly += 12
        label_ys.append(ly)
        if cls == "lvl-avwap":   # moving line - label only
            parts.append(f'<text class="lvl-label {cls}" x="{right_x + AXIS_W}" y="{ly:.1f}">{lbl} {val:.2f}</text>')
            continue
        parts.append(f'<line class="lvl {cls}" x1="{pad_l}" x2="{right_x}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="lvl-label {cls}" x="{right_x + AXIS_W}" y="{ly:.1f}">{lbl} {val:.2f}</text>')

    first_o, last_c = float(df["Open"].iloc[0]), float(df["Close"].iloc[-1])
    chg = (last_c / first_o - 1) * 100 if first_o else 0.0
    st = report.get("hhhl") or {}
    return f"""
      <div class="chart-wrap">
        <div class="chart-head">
          <span>Last {n} trading days &middot; daily candles{' (today in progress)' if in_progress else ''}</span>
          <span class="badge {'pass' if chg >= 0 else 'fail'}">{chg:+.2f}%</span>
          <span class="legend">{' '.join(legend)}
            <i class="sw lvl-support"></i>Support S1/S2 <i class="sw lvl-target"></i>Target T1/T2
            <i class="sw lvl-stop"></i>Stop{' <i class="sw premkt-sw"></i>Pre-market' if report.get("pre_market") is not None else ""}
            <span class="hhhl-mark up">&#9650;</span>HH+HL <span class="hhhl-mark down">&#9660;</span>LH+LL</span>
        </div>
        <svg class="price-chart" viewBox="0 0 {W} {H}"
             role="img" aria-label="{html.escape(report['ticker'])} daily chart">
          {''.join(parts)}
        </svg>
      </div>"""


def _streak_text(streak) -> str:
    kind, length = streak if streak else (None, 0)
    if not kind:
        return "N/A"
    return f"{length} {kind} day{'s' if length != 1 else ''}"


def render_chart_toggles(report: dict) -> str:
    """Two independent show/hide links side by side ("1h chart", "Daily chart"),
    each opening its chart below. Pure CSS (hidden checkboxes) - no JavaScript."""
    cfg = report["cfg"]
    hourly, daily = render_price_chart_svg(report), render_daily_chart_svg(report)
    if not hourly and not daily:
        return ""
    uid = "".join(c if c.isalnum() else "_" for c in report["ticker"])
    h_chk = " checked" if cfg.get("CHART_OPEN", False) else ""
    d_chk = " checked" if cfg.get("DAILY_CHART_OPEN", False) else ""
    # daily chart first (link on the left, chart on top), then the hourly chart
    links, panels, inputs = [], [], []
    if daily:
        inputs.append(f'<input type="checkbox" class="tog tog-d" id="tog-d-{uid}"{d_chk}>')
        links.append(f'<label for="tog-d-{uid}" class="lnk lnk-d"><span class="when-closed">&#9656; Show daily chart</span>'
                     f'<span class="when-open">&#9662; Hide daily chart</span></label>')
        panels.append(f'<div class="panel panel-d">{daily}</div>')
    if hourly:
        inputs.append(f'<input type="checkbox" class="tog tog-h" id="tog-h-{uid}"{h_chk}>')
        links.append(f'<label for="tog-h-{uid}" class="lnk lnk-h"><span class="when-closed">&#9656; Show 1h chart</span>'
                     f'<span class="when-open">&#9662; Hide 1h chart</span></label>')
        panels.append(f'<div class="panel panel-h">{hourly}</div>')
    return (f'\n      <div class="charts">{"".join(inputs)}'
            f'<div class="tog-links">{"".join(links)}</div>{"".join(panels)}</div>')


def _sr_text(levels, with_r: bool = False) -> str:
    """'88.20 (-5.1%, 3 touches) / 84.10 (-9.5%, 1 touch)' for the two nearest levels."""
    if not levels:
        return "none in window"
    parts = []
    for lvl in levels[:2]:
        extra = f", {lvl['r']:.1f}R" if with_r and lvl.get("r") is not None else ""
        parts.append(f"{lvl['price']} ({lvl['dist_pct']:+.1f}%, {lvl['touches']} "
                     f"touch{'es' if lvl['touches'] != 1 else ''}{extra})")
    return " / ".join(parts)


def near_level(price: float, level: float | None, atr: float | None, max_atr: float = 0.5) -> dict | None:
    """Is `price` within max_atr daily ATRs of `level`? Returns
    {"near": bool, "dist_pct": %, "dist_atr": distance in ATRs (positive = price above level)}."""
    if level is None or not price:
        return None
    dist = price - level
    dist_atr = dist / atr if atr else None
    near = abs(dist_atr) <= max_atr if dist_atr is not None else abs(dist / price) <= 0.01
    return {"near": bool(near), "dist_pct": round(dist / price * 100, 2),
            "dist_atr": round(dist_atr, 2) if dist_atr is not None else None}


def _near_sr_rows(report: dict) -> list:
    """'Near S1' / 'Near S2' rows for the Other levels table."""
    rows = []
    sups = (report.get("day_sr") or {}).get("supports") or []
    atr, max_atr = report.get("daily_atr"), report["cfg"].get("NEAR_SR_ATR", 0.5)
    for k in range(2):
        lvl = sups[k] if len(sups) > k else None
        nr = near_level(report["current_price"], lvl["price"] if lvl else None, atr, max_atr)
        if not nr:
            rows.append((f"Near S{k + 1}", "N/A (no support in window)"))
            continue
        atr_txt = f", {nr['dist_atr']:.2f} ATR" if nr["dist_atr"] is not None else ""
        rows.append((f"Near S{k + 1} ({lvl['price']})",
                     f"{_badge(nr['near'], 'YES', 'NO')} {nr['dist_pct']:.2f}% above{atr_txt}"))
    return rows


def _target_r_html(report: dict) -> str:
    tgt = (report.get("day_sr") or {}).get("target") or {}
    if tgt.get("r") is None:
        return ""
    badge = " " + _badge(False, "", f"UNDER {report['cfg'].get('TARGET_MIN_R', 2.0):g}R") if tgt.get("below_min_r") else ""
    return f" ({tgt['r']:.1f}R){badge}"


@_profiled("HTML rendering (incl. chart)")
def render_ticker_html(report: dict) -> str:
    cfg = report["cfg"]
    results = report["results"]
    sr_levels = report["sr_levels"]
    today_range = report["today_range"]
    signal = report["signal"]
    hrd = signal["hard_requirements_detail"]
    scd = signal["structural_confirmation_detail"]

    signal_cls = "buy" if signal["signal"] == "BUY" else "wait"
    vwap_badge = "" if report.get("above_vwap") is None else _badge(report["above_vwap"], "ABOVE", "BELOW")

    # pre-market source line, e.g. "IEX live - moved +0.62% since the SIP snapshot (SIP 147.35 @ 08:05)"
    pm_info = report.get("pre_market_info") or {}
    pm_src_html = ""
    if pm_info.get("source"):
        src_lbl = {"sip": "SIP (all exchanges, ~16 min delayed)", "iex": "IEX live"}.get(pm_info["source"], pm_info["source"])
        other = ""
        if pm_info.get("source") == "iex" and pm_info.get("sip_price") is not None:
            other = f" | SIP {pm_info['sip_price']:.2f} @ {pm_info['sip_time']}"
        elif pm_info.get("source") == "sip" and pm_info.get("iex_price") is not None:
            other = f" | IEX {pm_info['iex_price']:.2f} @ {pm_info['iex_time']}"
        pm_src_html = html.escape(f"{src_lbl} - {pm_info.get('reason', '')}{other}")

    # ---- "Other levels" (own 4-column section)
    price = report["current_price"]
    other_items = []

    # 52-week range (from yfinance .info via fundamentals) + where price sits in it
    fund = report.get("fundamentals") or {}
    try:
        w52_lo, w52_hi = float(fund.get("52-Week Low")), float(fund.get("52-Week High"))
        pos52 = (price - w52_lo) / (w52_hi - w52_lo) * 100 if w52_hi > w52_lo else 50.0
        other_items.append(("52-week range", f"{w52_lo:.2f} - {w52_hi:.2f} (at {pos52:.0f}%)"))
    except (TypeError, ValueError):
        other_items.append(("52-week range", "N/A"))

    # ATR range: daily ATR, how much of it today has used, and the ATR projection
    # (today's low + ATR = projected high, today's high - ATR = projected low)
    atr_d = report.get("daily_atr")
    if atr_d:
        used = today_range["day_range"] / atr_d * 100
        other_items.append((f"Daily ATR({cfg['ATR_PERIOD']})", f"{atr_d:.2f} (today used {used:.0f}%)"))
        other_items.append(("ATR projected range",
                            f"{today_range['day_high'] - atr_d:.2f} - {today_range['day_low'] + atr_d:.2f}"))

    other_items += [
        ("Pre-market price", (
            f"{report['pre_market']:.2f} @ {report.get('pre_market_time')} ET "
            f"({(report['pre_market'] / price - 1) * 100:+.2f}% vs last close)"
            + (f"<br><span class=\"muted-small\">{pm_src_html}</span>" if pm_src_html else "")
            if report.get("pre_market") is not None else "N/A")),
        *([(f"Anchored VWAP (from {report['avwap']['anchor_type']} of {report['avwap']['anchor_date']})",
             f"{report['avwap']['value']} " + ("" if report['avwap']['above'] is None
                                               else _badge(report['avwap']['above'], "ABOVE", "BELOW")))]
          if report.get("avwap") and report["avwap"].get("value") is not None else []),
        *([(f"HH/HL days (last {report['hhhl']['window']} completed)",
             f"{report['hhhl']['up_days']} up / {report['hhhl']['down_days']} down / "
             f"{report['hhhl']['mixed_days']} mixed (net {report['hhhl']['net']:+d})"),
            ("Current streak", _streak_text(report["hhhl"]["current_streak"])),
            ("Previous streak", _streak_text(report["hhhl"]["previous_streak"]))]
          if report.get("hhhl") else []),
        ("Today's range", f"{today_range['day_low']} - {today_range['day_high']} (range {today_range['day_range']})"),
        ("1h entry support", f"{report['entry_support']}"),
        ("Day support S1 / S2", _sr_text((report.get("day_sr") or {}).get("supports"))),
        ("Day resistance T1 / T2", _sr_text((report.get("day_sr") or {}).get("resistances"), with_r=True)),
        *_near_sr_rows(report),
        (f"POC ({cfg.get('CHART_DAYS', 7)}d volume profile)", f"{report.get('poc', 'N/A')}"),
        ("Session VWAP", f"{report.get('vwap') if report.get('vwap') is not None else 'N/A'} {vwap_badge}"),
    ]
    lrc = report.get("lrc")
    if lrc:
        pos = lrc["position_pct"]
        pos_txt = f"{pos:.0f}%" + (" (below)" if pos < 0 else " (above)" if pos > 100 else "")
        other_items += [
            (f"Reg. channel ({lrc['bars']} bars, {lrc['dev']}&sigma;)",
             f"{lrc['last_lower']} / {lrc['last_mid']} / {lrc['last_upper']}"),
            ("Channel slope", f"{lrc['slope_pct_per_day']:+.2f}%/day"),
            ("Channel fit (R&sup2;)", f"{lrc['r2']}"),
            ("Price in channel", pos_txt),
        ]
    other_items += [
        ("Structural stop (1h support - 0.5x daily ATR)", f"{report['stop']}"),
        (f"<strong>Take-Profit Target ({html.escape(((report.get('day_sr') or {}).get('target') or {}).get('source', '1.5x risk'))})</strong>",
         f"<strong>{report['take_profit']}</strong>" + _target_r_html(report)),
    ]
    other_levels_block = "\n      <h3>Other levels</h3>" + _four_col_table(other_items)

    fundamentals = report.get("fundamentals") or {}
    fundamentals_block = ""
    if fundamentals:
        fundamentals_block = "\n      <h3>Fundamentals</h3>" + _four_col_table(
            [(html.escape(label), html.escape(str(value))) for label, value in fundamentals.items()],
            "fundamentals-table")

    analyst = report.get("analyst") or {}
    analyst_block = ""
    if analyst:
        counts = analyst.get("recommendation_counts") or {}
        eps_improving = analyst.get("eps_improving")
        if eps_improving is None:
            eps_trend_html = '<span class="badge">N/A</span>'
        else:
            eps_trend_html = _badge(eps_improving, "IMPROVING", "DETERIORATING")

        rec_items = [
            ("Consensus", html.escape(analyst["recommendation_key"])),
            ("Mean Score", html.escape(analyst["recommendation_mean"])),
            ("# Analyst Opinions", html.escape(analyst["num_analyst_opinions"])),
        ] + [(html.escape(label), str(n)) for label, n in counts.items()]
        eps_items = [
            ("Current Consensus EPS", html.escape(analyst["eps_current"])),
            ("EPS 30 Days Ago", html.escape(analyst["eps_30d_ago"])),
            ("# Analysts (EPS)", html.escape(analyst["eps_num_analysts"])),
            ("EPS Trend", eps_trend_html),
        ]
        analyst_block = (
            "\n      <h3>Analyst Recommendations</h3>" + _four_col_table(rec_items)
            + "\n\n      <h3>EPS Estimate Revisions (current quarter)</h3>" + _four_col_table(eps_items)
        )

    news = report.get("news") or []
    if news:
        tz_lbl = {"America/New_York": "ET", "Asia/Jerusalem": "IL", "UTC": "UTC"}.get(
            cfg.get("NEWS_TZ", "America/New_York"), "")
        li = ""
        for n in news:
            url = n["url"] if n["url"].lower().startswith(("http://", "https://")) else ""
            title = html.escape(n["headline"])
            link = (f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                    if url else title)
            li += f'<li><span class="news-time">{html.escape(n["time"])} {tz_lbl}</span>{link}</li>'
        plural = "s" if len(news) != 1 else ""
        news_cell = (f'<details class="news-toggle"><summary>'
                     f'<span class="when-closed">Show {len(news)} article{plural}</span>'
                     f'<span class="when-open">Hide article{plural}</span></summary>'
                     f'<ul class="news-list">{li}</ul></details>')
    else:
        skipped = (report.get("catalysts") or {}).get("news_skipped")
        news_cell = f'<span class="muted-small">{html.escape(skipped)}</span>' if skipped else "none"
    news_days = cfg.get("NEWS_DAYS", 0)
    news_label = "today" if news_days == 0 else f"last {news_days + 1} days"
    news_row = f"<tr><td>News ({len(news)}, {news_label})</td><td>{news_cell}</td></tr>"

    catalysts = report.get("catalysts") or {}
    catalysts_block = ""
    if not catalysts and news:
        catalysts_block = f"""
      <h3>Event Catalysts</h3>
      <table class="detail-table"><tbody>{news_row}</tbody></table>"""
    if catalysts:
        earnings_date = catalysts.get("next_earnings_date") or "N/A"
        days_to_earnings = catalysts.get("days_to_earnings")
        earnings_str = earnings_date
        if days_to_earnings is not None:
            earnings_str = f"{earnings_date} ({days_to_earnings:+d}d)"
        earnings_window_html = _badge(
            catalysts.get("in_earnings_window", False), "IN WINDOW", "CLEAR"
        )

        sp500_member = catalysts.get("in_sp500")
        if sp500_member is None:
            sp500_html = '<span class="badge">N/A</span>'
        else:
            sp500_extra = ""
            if catalysts.get("sp500_added_date"):
                sp500_extra = (
                    f" (added {catalysts['sp500_added_date']}, "
                    f"{catalysts.get('days_since_index_addition')}d ago)"
                )
            sp500_html = _badge(sp500_member, "MEMBER", "NOT MEMBER") + html.escape(sp500_extra)
        recent_addition_html = ""
        if catalysts.get("recent_index_addition"):
            recent_addition_html = f"<tr><td>Recent index addition</td><td>{_badge(True, 'RECENT ADD')}</td></tr>"

        buyback_headlines = catalysts.get("buyback_headlines") or []
        guidance_headlines = catalysts.get("guidance_headlines") or []
        def _head_list(heads, limit=5):
            shown = "<br>".join(html.escape(h) for h in heads[:limit])
            more = f'<br><span class="muted-small">... and {len(heads) - limit} more</span>' if len(heads) > limit else ""
            return (shown + more) or "none"
        buyback_html = _head_list(buyback_headlines)
        guidance_html = _head_list(guidance_headlines)
        if catalysts.get("news_skipped"):   # no headlines were scanned for this ticker
            buyback_html = guidance_html = '<span class="muted-small">not checked (no news fetched)</span>'


        rating_actions = catalysts.get("rating_actions") or []
        rating_html = "<br>".join(html.escape(a) for a in rating_actions) or "none"

        social_rank = catalysts.get("social_rank", "N/A")
        if social_rank != "N/A":
            mom = catalysts.get("social_momentum_pct", "N/A")
            mom_str = f"+{mom}%" if isinstance(mom, (int, float)) and mom > 0 else f"{mom}%"
            social_html = (
                f"Rank <strong>#{social_rank}</strong> | "
                f"Mentions: {catalysts.get('social_mentions', 0)} ({mom_str} 24h) | "
                f"Upvotes: {catalysts.get('social_upvotes', 0)}"
            )
        else:
            social_html = '<span class="badge">Not in Top 50</span>'
        catalysts_block = f"""
      <h3>Event Catalysts</h3>
      <table class="detail-table"><tbody>
        <tr><td>Next earnings</td><td>{html.escape(str(earnings_str))} {earnings_window_html}</td></tr>
        <tr><td>S&amp;P 500 membership</td><td>{sp500_html}</td></tr>
        {recent_addition_html}
        <tr><td>Buyback headlines ({len(buyback_headlines)})</td><td>{buyback_html}</td></tr>
        <tr><td>Guidance headlines ({len(guidance_headlines)})</td><td>{guidance_html}</td></tr>
        <tr><td>Analyst actions ({catalysts.get('upgrades', 0)} up / {catalysts.get('downgrades', 0)} down)</td><td>{rating_html}</td></tr>
        <tr><td>Reddit Sentiment (ApeWisdom)</td><td>{social_html}</td></tr>
        {news_row}
      </tbody></table>"""


    tf_rows = ""
    for tf in ["1D", "4h", "1h"]:
        r = results[tf]
        sr = sr_levels[tf]
        patterns = ", ".join(r["patterns_detected"]) if r["patterns_detected"] else "-"
        tf_rows += f"""
        <tr>
          <td>{tf}</td>
          <td>{r['close']}</td>
          <td>{r['MA']}</td>
          <td>{_badge(r['trend_up'], 'UP', 'DOWN')}</td>
          <td>{r['%K']}</td>
          <td>{r['%D']}</td>
          <td>{_badge(r['k_above_d'], 'K>D', 'K<D')}</td>
          <td>{_badge(r['stoch_bullish'])}</td>
          <td>{_badge(not r['overbought'], 'OK', 'HOT')}</td>
          <td>{html.escape(patterns)}</td>
          <td>{sr['support']}</td>
          <td>{sr['resistance']}</td>
        </tr>"""

    hard1_rows = ""
    for label, key in [
        ("Daily trend up (close > MA)", "daily_trend_up"),
        ("Daily stochastic bullish", "daily_stoch_bullish"),
        ("4h trend up (close > MA)", "4h_trend_up"),
        ("4h stochastic bullish", "4h_stoch_bullish"),
    ]:
        hard1_rows += f"""
        <tr><td>{html.escape(label)}</td><td>{_badge(hrd[key])}</td></tr>"""

    daily_patterns = ", ".join(scd["daily_patterns"]) if scd["daily_patterns"] else "none"
    h4_patterns = ", ".join(scd["4h_patterns"]) if scd["4h_patterns"] else "none"
    hard2_rows = f"""
        <tr><td>Daily bullish pattern ({html.escape(daily_patterns)})</td><td>{_badge(scd['daily_bullish_pattern'])}</td></tr>
        <tr><td>4h bullish pattern ({html.escape(h4_patterns)})</td><td>{_badge(scd['4h_bullish_pattern'])}</td></tr>"""

    soft_labels = {
        "1h_not_overbought": "1h not overbought",
        "1h_macd_bullish": "1h MACD bullish",
        "volume_confirmed": "Volume confirmed",
        "near_support": "Near support",
        "1h_bullish_pattern": "1h bullish pattern",
    }
    soft_rows = ""
    for key, label in soft_labels.items():
        soft_rows += f"""
        <tr><td>{html.escape(label)}</td><td>{_badge(signal[key])}</td></tr>"""

    return f"""
    <section class="card">
      <div class="card-header">
        <h2>{html.escape(report['ticker'])}</h2>
        <div class="price">${report['current_price']:.2f}</div>
        <div class="signal-badge {signal_cls}">{'BUY_SIGNAL' if signal['signal'] == 'BUY' else signal['signal']}</div>
      </div>
      {render_chart_toggles(report)}

      <table class="tf-table">
        <thead>
          <tr>
            <th>TF</th><th>Close</th><th>{cfg['MA_TYPE'].upper()}{cfg['MA_PERIOD']}</th><th>Trend</th>
            <th>%K</th><th>%D</th><th>K vs D</th><th>Stoch</th><th>Overbought</th>
            <th>Candles</th><th>Support</th><th>Resistance</th>
          </tr>
        </thead>
        <tbody>{tf_rows}
        </tbody>
      </table>

      <div class="grid2">
        <div>
          <h3>Hard requirement #1 - Daily &amp; 4h trend/stochastic</h3>
          <table class="detail-table"><tbody>{hard1_rows}
          </tbody></table>

          <h3>Structural requirement #2 - Structural confirmation (daily OR 4h)</h3>
          <table class="detail-table"><tbody>{hard2_rows}
          </tbody></table>
          <p class="note">Overall hard requirements met: {_badge(signal['hard_requirements_met'])}</p>
        </div>

        <div>
          <h3>Soft conditions (need &ge; {cfg['SCORE_THRESHOLD']}/5, score {signal['soft_score']})</h3>
          <table class="detail-table"><tbody>{soft_rows}
          </tbody></table>


        </div>
      </div>
      {other_levels_block}
      {catalysts_block}
      {fundamentals_block}
      {analyst_block}
    </section>"""


def build_html_report(title: str, ticker_sections_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #0f1115;
    --card: #171a21;
    --border: #2a2f3a;
    --text: #e6e6e6;
    --muted: #9aa4b2;
    --accent: #9cdcfe;
    --pass: #2ecc71;
    --fail: #e74c3c;
    --buy: #2ecc71;
    --wait: #556070;
    --ma0: #c39bd3;  /* EMA9  */
    --ma1: #f5b041;
    --ma2: #48c9b0;
    --ma3: #85c1e9;
    --poc: #ff79c6;
    --vwap: #ffffff;
    --premkt: #ffe600;
    --avwap: #4dd0e1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text); margin: 0; padding: 32px;
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }}
  h1 {{ font-size: 20px; color: var(--accent); margin: 0 0 4px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 28px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px 24px; margin-bottom: 24px;
  }}
  .card-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
  .card-header h2 {{ margin: 0; font-size: 18px; }}
  .price {{ color: var(--muted); font-size: 15px; }}
  .signal-badge {{
    margin-left: auto; padding: 6px 14px; border-radius: 999px; font-weight: 700;
    font-size: 13px; letter-spacing: 0.04em;
  }}
  .signal-badge.buy {{ background: rgba(46,204,113,0.15); color: var(--buy); border: 1px solid var(--buy); }}
  .signal-badge.wait {{ background: rgba(85,96,112,0.25); color: var(--muted); border: 1px solid var(--border); }}

  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.03em; }}
  .tf-table {{ margin-bottom: 20px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 800px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  h3 {{ font-size: 13px; color: var(--accent); margin: 16px 0 6px; text-transform: uppercase; letter-spacing: 0.03em; }}
  .detail-table td:first-child {{ color: var(--text); }}
  .fundamentals-table {{ margin-bottom: 18px; }}
  .four-col {{ table-layout: fixed; margin-bottom: 18px; }}
  .four-col col.c-lbl {{ width: 30%; }}
  .four-col col.c-val {{ width: 20%; }}
  .four-col td:nth-child(odd) {{ color: var(--muted); }}
  .four-col td:nth-child(2) {{ border-right: 1px solid var(--border); padding-right: 16px; }}
  .four-col td:nth-child(3) {{ padding-left: 16px; }}
  .four-col td {{ overflow-wrap: anywhere; }}
  .note {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}

  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px;
    font-weight: 700; letter-spacing: 0.03em;
  }}
  .badge.pass {{ background: rgba(46,204,113,0.15); color: var(--pass); }}
  .badge.fail {{ background: rgba(231,76,60,0.15); color: var(--fail); }}

  /* ---- 7-day price chart */
  .chart-wrap {{ margin: 4px 0 20px; }}
  .charts {{ margin: 0 0 12px; }}
  .charts .tog, .charts .panel {{ display: none; }}
  .charts .tog-links {{ display: flex; gap: 22px; margin-bottom: 6px; }}
  .charts .lnk {{ cursor: pointer; color: var(--accent); font-size: 12px; user-select: none; }}
  .charts .lnk:hover {{ text-decoration: underline; }}
  .charts .lnk .when-open {{ display: none; }}
  .charts .tog-h:checked ~ .panel-h, .charts .tog-d:checked ~ .panel-d {{ display: block; }}
  .charts .tog-h:checked ~ .tog-links .lnk-h .when-open,
  .charts .tog-d:checked ~ .tog-links .lnk-d .when-open {{ display: inline; }}
  .charts .tog-h:checked ~ .tog-links .lnk-h .when-closed,
  .charts .tog-d:checked ~ .tog-links .lnk-d .when-closed {{ display: none; }}
  .hhhl-mark {{ font-size: 10px; }}
  .price-chart .axis.dayname {{ fill: var(--muted); opacity: 0.75; }}
  .price-chart .weeksep {{ stroke: #ffffff; stroke-opacity: 0.4; stroke-width: 1; stroke-dasharray: 1 3; }}
  .price-chart .avwap {{ fill: none; stroke: var(--avwap); stroke-width: 1.8; }}
  .price-chart .avwap-anchor {{ fill: var(--avwap); }}
  .price-chart .lvl-label.lvl-avwap {{ fill: var(--avwap); }}
  .sw.avwap-sw {{ border-color: var(--avwap); }}
  .hhhl-mark.up {{ fill: var(--pass); color: var(--pass); }}
  .hhhl-mark.down {{ fill: var(--fail); color: var(--fail); }}
  .price-chart .candle.partial {{ opacity: 0.55; }}
  .price-chart .lvl.lvl-res {{ stroke: var(--muted); }}
  .price-chart .lvl-label.lvl-res {{ fill: var(--muted); }}
  .sw.lvl-res {{ border-color: var(--muted); }}
  .price-chart .lvl.lvl-support2 {{ stroke: var(--accent); stroke-opacity: 0.5; stroke-dasharray: 2 4; }}
  .price-chart .lvl-label.lvl-support2 {{ fill: var(--accent); opacity: 0.7; }}
  .price-chart .lvl.lvl-target2 {{ stroke: var(--pass); stroke-opacity: 0.5; stroke-dasharray: 2 4; }}
  .price-chart .lvl-label.lvl-target2 {{ fill: var(--pass); opacity: 0.7; }}
  .chart-toggle {{ margin: 0 0 12px; }}
  .chart-toggle summary {{ list-style: none; cursor: pointer; color: var(--accent);
                           font-size: 12px; display: inline-block; margin-bottom: 6px; user-select: none; }}
  .muted-small {{ color: var(--muted); font-size: 11px; }}
  .news-toggle summary {{ cursor: pointer; color: var(--accent); user-select: none; }}
  .news-toggle summary:hover {{ text-decoration: underline; }}
  .news-list {{ list-style: none; margin: 8px 0 2px; padding: 0; }}
  .news-list li {{ padding: 4px 0; border-bottom: 1px solid var(--border); line-height: 1.4; }}
  .news-list li:last-child {{ border-bottom: none; }}
  .news-time {{ color: var(--muted); font-variant-numeric: tabular-nums; margin-right: 10px; white-space: nowrap; }}
  .news-list a {{ color: var(--text); text-decoration: none; }}
  .news-list a:hover {{ color: var(--accent); text-decoration: underline; }}
  .chart-toggle summary::-webkit-details-marker {{ display: none; }}
  .chart-toggle summary:hover {{ text-decoration: underline; }}
  .chart-toggle .when-open, .news-toggle .when-open {{ display: none; }}
  .chart-toggle[open] .when-open, .news-toggle[open] .when-open {{ display: inline; }}
  .chart-toggle[open] .when-closed, .news-toggle[open] .when-closed {{ display: none; }}
  .chart-head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
                 color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
  .chart-head .legend {{ margin-left: auto; display: flex; align-items: center; gap: 6px; }}
  .sw {{ display: inline-block; width: 14px; height: 0; border-top: 2px dashed; margin-left: 8px; }}
  .sw.ma-sw {{ border-top-style: solid; }}
  .sw.ma-0 {{ border-color: var(--ma0); }} .sw.ma-1 {{ border-color: var(--ma1); }}
  .sw.ma-2 {{ border-color: var(--ma2); }} .sw.ma-3 {{ border-color: var(--ma3); }}
  .sw.lvl-poc {{ border-color: var(--poc); }}
  .sw.session-sw {{ border-top: 2px dotted var(--text); }}
  .sw.lvl-support {{ border-color: var(--accent); }}
  .sw.lvl-stop {{ border-color: var(--fail); }}
  .sw.lvl-target {{ border-color: var(--pass); }}
  .price-chart {{ width: 100%; height: auto; display: block; }}  /* keeps aspect ratio, so text isn't stretched */
  .price-chart .grid {{ stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 1; stroke-dasharray: 1 4; }}
  .price-chart .tick {{ stroke: var(--muted); stroke-width: 1; }}
  .price-chart .axis {{ fill: var(--muted); font-size: 10px; }}
  .price-chart .candle line {{ stroke-width: 1; }}
  .price-chart .candle.up line {{ stroke: var(--pass); }}
  .price-chart .candle.down line {{ stroke: var(--fail); }}
  .price-chart .candle.up rect {{ fill: var(--pass); }}
  .price-chart .candle.down rect {{ fill: var(--fail); }}
  .price-chart .candle:hover rect {{ opacity: 0.7; }}
  .price-chart .vol.up {{ fill: rgba(46,204,113,0.35); }}
  .price-chart .vol.down {{ fill: rgba(231,76,60,0.35); }}
  .price-chart .ma {{ fill: none; stroke-width: 1.5; }}
  .price-chart .ma.ma-0 {{ stroke: var(--ma0); }} .price-chart .ma.ma-1 {{ stroke: var(--ma1); }}
  .price-chart .ma.ma-2 {{ stroke: var(--ma2); }} .price-chart .ma.ma-3 {{ stroke: var(--ma3); }}
  .price-chart .session {{ stroke: #ffffff; stroke-opacity: 0.55; stroke-width: 1; stroke-dasharray: 1 3; }}
  .price-chart .lvl.lvl-poc {{ stroke: var(--poc); }}
  .price-chart .lvl-label.lvl-poc {{ fill: var(--poc); }}
  .price-chart .vwap {{ fill: none; stroke: var(--vwap); stroke-width: 1.8; stroke-opacity: 0.9; }}
  .price-chart .lvl-label.lvl-vwap {{ fill: var(--vwap); }}
  .sw.vwap-sw {{ border-color: var(--vwap); }}
  .sw.premkt-sw {{ border-top: 2px dotted var(--premkt); }}
  .price-chart .lvl.lvl-premkt {{ stroke: var(--premkt); stroke-width: 1.5; stroke-dasharray: 2 4; }}
  .price-chart .lvl-label.lvl-premkt {{ fill: var(--premkt); }}
  .sw.lrc-sw {{ border-top: 6px solid rgba(154,164,178,0.25); height: 0; }}
  .price-chart .lrc-fill {{ fill: var(--muted); fill-opacity: 0.07; }}
  .price-chart .lrc-band {{ fill: none; stroke: var(--muted); stroke-opacity: 0.6; stroke-width: 1; }}
  .price-chart .lrc-mid {{ fill: none; stroke: var(--muted); stroke-opacity: 0.8; stroke-width: 1; stroke-dasharray: 6 4; }}
  .price-chart .lvl {{ stroke-width: 1; stroke-dasharray: 5 4; }}
  .price-chart .lvl.lvl-support {{ stroke: var(--accent); }}
  .price-chart .lvl.lvl-stop {{ stroke: var(--fail); }}
  .price-chart .lvl.lvl-target {{ stroke: var(--pass); }}
  .price-chart .lvl-label {{ font-size: 11px; font-weight: 600; }}
  .price-chart .lvl-label.lvl-support {{ fill: var(--accent); }}
  .price-chart .lvl-label.lvl-stop {{ fill: var(--fail); }}
  .price-chart .lvl-label.lvl-target {{ fill: var(--pass); }}
  .price-chart .last {{ fill: var(--text); }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="subtitle">Multi-Timeframe Buy Signal Screener</div>
{ticker_sections_html}
</body>
</html>
"""


def _snake(label: str) -> str:
    """'Price/Sales (TTM)' -> 'price_sales_ttm'"""
    out = "".join(c.lower() if c.isalnum() else "_" for c in label.replace("%", " pct "))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _ascii(text: str) -> str:
    """HTTP header values must be plain ASCII."""
    return str(text).encode("ascii", "replace").decode("ascii")


def send_ntfy_file(path: str, topic: str | None = None, server: str | None = None,
                   token: str | None = None, title: str | None = None, message: str | None = None,
                   tags: str | None = None, priority: str | None = None, timeout: float = 60) -> bool:
    """Send a file (e.g. the HTML report) as an ntfy attachment; the phone app
    shows a notification you tap to open the file.
    topic / server / token come from the arguments, else constants.CONFIG
    ("NTFY_TOPIC", "NTFY_SERVER", "NTFY_TOKEN"), else the environment variables
    of the same names. Server defaults to https://ntfy.sh.
    Files larger than CONFIGH["NTFY_MAX_MB"] are announced without the file.
    Returns True if ntfy accepted it."""
    topic = topic or _SECRETS.get("NTFY_TOPIC") or os.environ.get("NTFY_TOPIC")
    if not topic:
        print("ntfy: no NTFY_TOPIC set - skipped")
        return False
    server = (server or _SECRETS.get("NTFY_SERVER") or os.environ.get("NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
    token = token or _SECRETS.get("NTFY_TOKEN") or os.environ.get("NTFY_TOKEN")

    size_mb = os.path.getsize(path) / 1024 / 1024
    headers = {"Title": _ascii(title or os.path.basename(path))}
    if tags:
        headers["Tags"] = _ascii(tags)
    if priority:
        headers["Priority"] = _ascii(priority)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        if size_mb > CONFIGH.get("NTFY_MAX_MB", 15):
            body = (f"{message or ''}\n{os.path.basename(path)} is {size_mb:.1f} MB - too big to attach; "
                    f"get it from the GitHub Actions run.").strip()
            resp = requests.post(f"{server}/{topic}", data=body.encode("utf-8"), headers=headers, timeout=timeout)
        else:
            headers["Filename"] = _ascii(os.path.basename(path))
            if message:
                headers["Message"] = _ascii(message)
            with open(path, "rb") as f:
                resp = requests.put(f"{server}/{topic}", data=f, headers=headers, timeout=timeout)
        resp.raise_for_status()
        print(f"ntfy: sent {os.path.basename(path)} ({size_mb:.1f} MB) to {server}/<topic>")
        return True
    except requests.RequestException as e:
        print(f"ntfy: failed to send {os.path.basename(path)}: {e}")
        return False


def report_to_row(report: dict, report_group: str) -> dict:
    """Flatten one ticker's report into a single CSV row (column -> value).
    Lists of text (news, buyback/guidance headlines, rating actions) keep only
    the LATEST item plus a count. report_group is "up", "down" or "filtered"."""
    cfg = report["cfg"]
    sig = report["signal"]
    row = {
        "ticker": report["ticker"],
        "report": report_group,
        "signal": sig["signal"],
        "current_price": report["current_price"],
    }

    # --- pre-market
    pm = report.get("pre_market_info") or {}
    pre = report.get("pre_market")
    row.update({
        "pre_market": pre,
        "pre_market_time_et": report.get("pre_market_time"),
        "pre_market_source": pm.get("source"),
        "pre_market_gap_pct": round((pre / report["current_price"] - 1) * 100, 2) if pre else None,
        "pre_market_reason": pm.get("reason") if pre else None,
    })

    # --- signal breakdown
    row["hard_requirements_met"] = sig["hard_requirements_met"]
    for k, v in sig["hard_requirements_detail"].items():
        row[f"hard_{k}"] = v
    row["structural_confirmation"] = sig["structural_confirmation"]
    scd = sig["structural_confirmation_detail"]
    row["structural_daily_pattern"] = scd["daily_bullish_pattern"]
    row["structural_4h_pattern"] = scd["4h_bullish_pattern"]
    row["soft_score"] = int(str(sig["soft_score"]).split("/")[0])
    for k in ("1h_not_overbought", "1h_macd_bullish", "volume_confirmed", "near_support", "1h_bullish_pattern"):
        row[f"soft_{k}"] = sig.get(k)
    fails = report_filter_checks(report)
    row["filter_checks_failed"] = sum(fails.values())
    for k, v in fails.items():
        row[f"fails_{k}"] = v

    # --- per timeframe (d1_ = daily, h4_ = 4-hour, h1_ = 1-hour)
    ma_lbl = f"{cfg['MA_TYPE']}{cfg['MA_PERIOD']}"
    for tf, pre_tf in (("1D", "d1"), ("4h", "h4"), ("1h", "h1")):
        r, sr = report["results"][tf], report["sr_levels"][tf]
        row.update({
            f"{pre_tf}_close": r["close"], f"{pre_tf}_{ma_lbl}": r["MA"], f"{pre_tf}_trend_up": r["trend_up"],
            f"{pre_tf}_stoch_k": r["%K"], f"{pre_tf}_stoch_d": r["%D"], f"{pre_tf}_k_above_d": r["k_above_d"],
            f"{pre_tf}_stoch_bullish": r["stoch_bullish"], f"{pre_tf}_overbought": r["overbought"],
            f"{pre_tf}_patterns": ";".join(r["patterns_detected"]),
            f"{pre_tf}_support": sr["support"], f"{pre_tf}_resistance": sr["resistance"],
        })

    # --- levels
    tr = report["today_range"]
    atr = report.get("daily_atr")
    row.update({
        "today_low": tr["day_low"], "today_high": tr["day_high"], "today_range": tr["day_range"],
        "daily_atr": atr,
        "atr_used_today_pct": round(tr["day_range"] / atr * 100, 1) if atr else None,
        "atr_projected_low": round(tr["day_high"] - atr, 2) if atr else None,
        "atr_projected_high": round(tr["day_low"] + atr, 2) if atr else None,
        "entry_support_1h": report["entry_support"],
        "stop": report["stop"], "take_profit": report["take_profit"],
        "target_source": ((report.get("day_sr") or {}).get("target") or {}).get("source"),
        "target_r": ((report.get("day_sr") or {}).get("target") or {}).get("r"),
        "target_below_min_r": ((report.get("day_sr") or {}).get("target") or {}).get("below_min_r"),
        "poc": report.get("poc"), "vwap": report.get("vwap"), "above_vwap": report.get("above_vwap"),
    })
    lrc = report.get("lrc") or {}
    row.update({
        "trend_dir": report.get("trend_dir"),
        "trend_reason": report.get("trend_reason"),
        "avwap": (report.get("avwap") or {}).get("value"),
        "avwap_anchor_type": (report.get("avwap") or {}).get("anchor_type"),
        "avwap_anchor_date": (report.get("avwap") or {}).get("anchor_date"),
        "above_avwap": (report.get("avwap") or {}).get("above"),
        "hhhl_window_days": (report.get("hhhl") or {}).get("window"),
        "hhhl_up_days": (report.get("hhhl") or {}).get("up_days"),
        "hhhl_down_days": (report.get("hhhl") or {}).get("down_days"),
        "hhhl_mixed_days": (report.get("hhhl") or {}).get("mixed_days"),
        "hhhl_net": (report.get("hhhl") or {}).get("net"),
        "streak_current_type": ((report.get("hhhl") or {}).get("current_streak") or (None, None))[0],
        "streak_current_days": ((report.get("hhhl") or {}).get("current_streak") or (None, None))[1],
        "streak_previous_type": ((report.get("hhhl") or {}).get("previous_streak") or (None, None))[0],
        "streak_previous_days": ((report.get("hhhl") or {}).get("previous_streak") or (None, None))[1],
        "channel_lower": lrc.get("last_lower"), "channel_mid": lrc.get("last_mid"),
        "channel_upper": lrc.get("last_upper"), "channel_slope_pct_day": lrc.get("slope_pct_per_day"),
        "channel_r2": lrc.get("r2"), "channel_position_pct": lrc.get("position_pct"),
    })

    dsr = report.get("day_sr") or {}
    for k in range(2):
        sups = dsr.get("supports") or []
        nr = near_level(report["current_price"], sups[k]["price"] if len(sups) > k else None,
                        report.get("daily_atr"), cfg.get("NEAR_SR_ATR", 0.5))
        row[f"near_s{k + 1}"] = nr["near"] if nr else None
        row[f"near_s{k + 1}_atr"] = nr["dist_atr"] if nr else None
    for pre, key in (("s", "supports"), ("t", "resistances")):
        for k in range(2):
            lvl = (dsr.get(key) or [])[k] if len(dsr.get(key) or []) > k else {}
            row[f"day_{pre}{k + 1}"] = lvl.get("price")
            row[f"day_{pre}{k + 1}_pct"] = lvl.get("dist_pct")
            row[f"day_{pre}{k + 1}_touches"] = lvl.get("touches")
            if pre == "t":
                row[f"day_{pre}{k + 1}_r"] = lvl.get("r")

    # --- fundamentals (raw yfinance values: ratios as fractions, e.g. 0.25 = 25%)
    raw = report.get("fund_raw") or {}
    for label, key, _ in FUNDAMENTAL_FIELDS:
        row[_snake(label)] = raw.get(key)
    lo52, hi52 = raw.get("fiftyTwoWeekLow"), raw.get("fiftyTwoWeekHigh")
    row["pos_in_52w_range_pct"] = (round((report["current_price"] - lo52) / (hi52 - lo52) * 100, 1)
                                  if isinstance(lo52, (int, float)) and isinstance(hi52, (int, float)) and hi52 > lo52
                                  else None)

    # --- analyst
    an = report.get("analyst") or {}
    row.update({
        "analyst_consensus": an.get("recommendation_key"), "analyst_mean_score": an.get("recommendation_mean"),
        "analyst_opinions": an.get("num_analyst_opinions"),
    })
    for label, n in (an.get("recommendation_counts") or {}).items():
        row[f"analyst_{_snake(label)}"] = n
    row.update({
        "eps_current": an.get("eps_current"), "eps_30d_ago": an.get("eps_30d_ago"),
        "eps_improving": an.get("eps_improving"), "eps_num_analysts": an.get("eps_num_analysts"),
    })

    # --- catalysts (latest item only for text lists)
    cat = report.get("catalysts") or {}
    actions = sorted(cat.get("rating_actions") or [], reverse=True)   # strings start with the ISO date
    buy, guid = cat.get("buyback_headlines") or [], cat.get("guidance_headlines") or []
    row.update({
        "next_earnings_date": cat.get("next_earnings_date"), "days_to_earnings": cat.get("days_to_earnings"),
        "in_earnings_window": cat.get("in_earnings_window"),
        "in_sp500": cat.get("in_sp500"), "sp500_added_date": cat.get("sp500_added_date"),
        "upgrades": cat.get("upgrades"), "downgrades": cat.get("downgrades"),
        "latest_rating_action": actions[0] if actions else None,
        "buyback_headlines_count": len(buy), "latest_buyback_headline": buy[0] if buy else None,
        "guidance_headlines_count": len(guid), "latest_guidance_headline": guid[0] if guid else None,
        "reddit_rank": cat.get("social_rank"), "reddit_mentions": cat.get("social_mentions"),
        "reddit_momentum_pct_24h": cat.get("social_momentum_pct"), "reddit_upvotes": cat.get("social_upvotes"),
    })

    # --- news (latest article only)
    news = report.get("news") or []
    row.update({
        "news_count": len(news),
        "latest_news_time": news[0]["time"] if news else None,
        "latest_news_headline": news[0]["headline"] if news else None,
        "latest_news_url": news[0]["url"] if news else None,
        "news_not_fetched_reason": cat.get("news_skipped"),
        "error": None,
    })
    return row


def write_csv_report(rows: list, out_path: str) -> None:
    """One row per ticker. utf-8-sig so Excel shows non-ASCII headlines correctly."""
    if not rows:
        return
    full = next((r for r in rows if r.get("error") is None), rows[0])
    cols = list(full.keys())
    cols += [c for r in rows for c in r if c not in cols]   # any extra columns (e.g. error-only rows)
    pd.DataFrame(rows, columns=list(dict.fromkeys(cols))).to_csv(out_path, index=False, encoding="utf-8-sig")


def report_filter_checks(report: dict) -> dict:
    """The four report-inclusion checks. Each value is True if the ticker FAILS
    that check. A ticker is left out of the HTML only when it fails ALL four."""
    signal = report["signal"]
    hard_passed = sum(bool(v) for v in signal["hard_requirements_detail"].values())
    scd = signal["structural_confirmation_detail"]
    soft_score = int(str(signal["soft_score"]).split("/")[0])
    eps_improving = (report.get("analyst") or {}).get("eps_improving")
    return {
        "hard_lt_3": hard_passed < 3,                                    # fewer than 3 of 4 hard checks pass
        "no_structural": not (scd["daily_bullish_pattern"] or scd["4h_bullish_pattern"]),
        "soft_lt_2": soft_score < 2,
        "eps_deteriorating": eps_improving is False,                     # N/A does not count as failing
    }


def _skipped_card(skipped: list) -> str:
    if not skipped:
        return ""
    items = "".join(f"<li><strong>{html.escape(t)}</strong>: {html.escape(why)}</li>" for t, why in skipped)
    return (f'<section class="card"><h3>Not shown ({len(skipped)})</h3>'
            f'<ul class="note">{items}</ul></section>')


def main(tickers, fileapp):
    """Screen a list of tickers and write two HTML reports:
      <fileapp>_signal_report_<timestamp>_up.html   - regression channel sloping up
      <fileapp>_signal_report_<timestamp>_down.html - regression channel sloping down
    A ticker is left out entirely only if it fails ALL four checks in
    report_filter_checks(). Call as main(TICKERS, "name")."""
    sections = {"up": [], "down": []}
    skipped = []  # (ticker, reason) - listed at the bottom of both reports
    PROFILE.clear()
    run_t0 = time.perf_counter()
    prune_cache()
    sp500_members = daily_cached("sp500", "all", lambda: _get_sp500_membership(verbose=False))
    social_table = fetch_apewisdom_table()   # once per run (mentions change intraday, so not cached)
    pre_market_batch = prefetch_pre_market(tickers, CONFIGH)   # 2 batched Alpaca requests, not 2 per ticker

    try:
        prefetched = prefetch_hourly_data(tickers, CONFIGH["PERIOD"], CONFIGH["INTERVAL"])
    except Exception as e:
        print(f"batch download failed ({type(e).__name__}: {e}) - falling back to one download per ticker")
        prefetched = {}

    def screen_one(ticker):
        """Runs in a worker thread. Returns (status, direction_or_reason, html, csv_row)."""
        try:
            t0 = time.perf_counter()
            report = run_screen(ticker, CONFIGH, sp500_members=sp500_members,
                                hourly=prefetched.get(ticker), social_table=social_table,
                                pre_market_batch=pre_market_batch)
            elapsed = time.perf_counter() - t0
            _prof_add("whole ticker (run_screen)", ticker, elapsed)
            fetched = sum(sec for step, calls in list(PROFILE.items())
                          if not step.startswith("  ") and step != "whole ticker (run_screen)"
                          for t, sec in list(calls) if t == ticker)
            _prof_add("indicators & signal (CPU, no network)", ticker, max(elapsed - fetched, 0.0))
            #print_report(report)
            fails = report_filter_checks(report)
            if all(fails.values()):
                print(f"=== {ticker}: filtered out (fails all 4 checks) ===")
                return ("skip", "filtered out - fails all 4 checks (hard <3/4, no pattern, "
                                "soft <2, EPS deteriorating)", None, report_to_row(report, "filtered"))
            return ("ok", report["trend_dir"], render_ticker_html(report),
                    report_to_row(report, report["trend_dir"]))
        except Exception as e:
            print(f"\n=== {ticker}: skipped due to error ===")
            print(f"  {type(e).__name__}: {e}")
            return ("skip", f"error - {type(e).__name__}: {e}", None,
                    {"ticker": ticker, "report": "error", "error": f"{type(e).__name__}: {e}"})

    # executor.map keeps results in the same order as `tickers`
    with ThreadPoolExecutor(max_workers=max(1, int(CONFIGH.get("MAX_WORKERS", 4)))) as ex:
        results = list(ex.map(screen_one, tickers))

    csv_rows = [res[3] for res in results if res[3] is not None]
    for ticker, res in zip(tickers, results):
        if res[0] == "ok":
            sections[res[1]].append(res[2])
        else:
            skipped.append((ticker, res[1]))

    tz_gmt3 = timezone(timedelta(hours=3))
    #timestamp = datetime.now(tz_gmt3).strftime("%Y%m%d_%H%M")
    timestamp = constants.get_dayprefix()+"_" + constants.get_timeprefix()

    out_paths = {}
    for direction in ("up", "down"):
        out_path = os.path.join("reports", f"{fileapp}_signal_report_{timestamp}_{direction}.html")
        out_paths[direction] = out_path
        report_title = f"{fileapp}_Signal Report {timestamp}_{direction}"
        body = "\n".join(sections[direction]) or (
            f'<section class="card"><p class="note">No tickers with a {direction}-sloping channel.</p></section>')
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(build_html_report(report_title, body + _skipped_card(skipped)))
        print(f"HTML report ({direction}, {len(sections[direction])} tickers) written to {out_path}")

    if CONFIGH.get("CSV_REPORT", True):
        csv_path = os.path.join("reports", f"{fileapp}_signal_report_{timestamp}.csv")
        try:
            write_csv_report(csv_rows, csv_path)
            out_paths["csv"] = csv_path
            print(f"CSV report ({len(csv_rows)} rows) written to {csv_path}")
        except Exception as e:
            print(f"CSV report failed: {type(e).__name__}: {e}")

    if CONFIGH.get("NTFY_ENABLED", True):
        buys = {g: [r["ticker"] for r in csv_rows if r.get("report") == g and r.get("signal") == "BUY"]
                for g in ("up", "down")}
        counts = {g: sum(1 for r in csv_rows if r.get("report") == g) for g in ("up", "down")}
        for kind in CONFIGH.get("NTFY_FILES", ["up"]):
            path = out_paths.get(kind)
            if not path or not os.path.exists(path):
                continue
            if kind in buys:
                msg = f"{counts[kind]} tickers, BUY_SIGNAL: {', '.join(buys[kind]) or 'none'}"
                tags = "chart_with_upwards_trend" if kind == "up" else "chart_with_downwards_trend"
                prio = "high" if (kind == "up" and buys["up"]) else "default"
            else:
                msg, tags, prio = f"{len(csv_rows)} rows", "page_facing_up", "default"
            send_ntfy_file(path, title=f"{fileapp} {kind} report {timestamp}", message=msg,
                           tags=tags, priority=prio)

    if CONFIGH.get("PROFILE", True):
        print_profile_summary(time.perf_counter() - run_t0, len(tickers))
