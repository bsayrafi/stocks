"""
flow_common.py
Shared data access and feature engineering for the accumulation-confirmation
modules (dip_confirm.py and trend_confirm.py).

Data: Alpaca historical 30-minute bars, SIP feed, split/dividend adjusted.
The free Alpaca plan serves SIP history older than ~15 minutes, so the request
end time is capped at "now minus 16 minutes".
Only regular-session bars (09:30-16:00 ET) are used. Alpaca's 30-minute bars
start on :00 and :30, so they line up with the 09:30 open.

Credentials: pass a client, or set APCA_API_KEY_ID and APCA_API_SECRET_KEY.

IMPORTANT: the "score" these modules produce is a heuristic confidence
(0-100) that accumulation is present. It is shaped like a probability but is
NOT calibrated. It becomes a real probability only after the weights are fitted
to backtest outcomes (see the z-score columns in the output).
"""
from __future__ import annotations

import math
import os
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
BAR_MINUTES = 30
SESSION_START = time(9, 30)
SESSION_END = time(16, 0)
LAST_HOUR_START = time(15, 0)
DEFAULT_LOOKBACK_DAYS = 180   # calendar days, ~120 sessions
RVOL_SLOT_LOOKBACK = 20       # sessions used for time-of-day volume baseline
Z_CLIP = 3.0


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
def get_client(api_key: str | None = None, secret_key: str | None = None):
    from alpaca.data.historical import StockHistoricalDataClient

    api_key = api_key or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Set APCA_API_KEY_ID / APCA_API_SECRET_KEY or pass a client.")
    return StockHistoricalDataClient(api_key, secret_key)


def _resolve_end(as_of) -> datetime:
    """End of the data request. A plain date means 'after that day's close'."""
    now_limit = datetime.now(tz=NY) - timedelta(minutes=16)
    if as_of is None:
        return now_limit
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        if ts == ts.normalize():
            ts = ts + pd.Timedelta(hours=20)
        ts = ts.tz_localize(NY)
    return min(ts.to_pydatetime(), now_limit)


def normalize_tickers(tickers) -> list[str]:
    return sorted({str(t).strip().upper() for t in tickers if t and str(t).strip()})


def fetch_intraday_bars(
    tickers,
    client=None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    as_of=None,
    feed: str = "sip",
) -> pd.DataFrame:
    """Fetch 30-minute bars for all tickers in one request.

    Fetch once for dips + trends combined and pass the result to both
    score functions via `bars=` to avoid downloading twice. For backtests,
    fetch a long range once and call the score functions with different
    `as_of` dates.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tickers = normalize_tickers(tickers)
    if not tickers:
        return pd.DataFrame()
    client = client or get_client()
    end = _resolve_end(as_of)
    start = end - timedelta(days=lookback_days)
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed=DataFeed(feed),
    )
    raw = client.get_stock_bars(req).df
    return prepare_bars(raw)


def prepare_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw bars: NY time, regular session only, date and slot columns."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = raw.reset_index() if "symbol" not in raw.columns else raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df["slot"] = df["timestamp"].dt.time
    df = df[(df["slot"] >= SESSION_START) & (df["slot"] < SESSION_END)].copy()
    df["date"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    if "trade_count" not in df.columns:
        df["trade_count"] = np.nan
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def build_daily(bars_sym: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one symbol's 30-min bars into a daily feature table.

    Money flow is computed per 30-min bar and summed, which captures where
    volume actually traded within the day (more accurate than daily A/D).
    """
    b = bars_sym.sort_values("timestamp").copy()

    rng = b["high"] - b["low"]
    b["clv"] = np.where(rng > 0, ((b["close"] - b["low"]) - (b["high"] - b["close"])) / rng.where(rng > 0, 1), 0.0)
    b["mfv"] = b["clv"] * b["volume"]
    prev_close = b["close"].shift(1)
    b["up_vol"] = np.where(b["close"] > prev_close, b["volume"], 0.0)
    b["dn_vol"] = np.where(b["close"] < prev_close, b["volume"], 0.0)
    b["pv"] = b["vwap"] * b["volume"]
    b["last_hour"] = b["slot"] >= LAST_HOUR_START

    # Time-of-day relative volume: each bar vs the median of the same slot
    # over previous sessions (removes the intraday U-shaped volume pattern).
    b["slot_med"] = b.groupby("slot")["volume"].transform(
        lambda s: s.shift(1).rolling(RVOL_SLOT_LOOKBACK, min_periods=10).median()
    )
    b["rvol_bar"] = b["volume"] / b["slot_med"].where(b["slot_med"] > 0)

    g = b.groupby("date")
    d = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "trades": g["trade_count"].sum(min_count=1),
        "pv": g["pv"].sum(),
        "mfv": g["mfv"].sum(),
        "up_vol": g["up_vol"].sum(),
        "dn_vol": g["dn_vol"].sum(),
        "rvol": g["rvol_bar"].mean(),
        "n_bars": g.size(),
    })
    lh = b[b["last_hour"]].groupby("date")
    d["lh_mfv"] = lh["mfv"].sum()
    d["lh_vol"] = lh["volume"].sum()

    d = d[d["n_bars"] >= 6].copy()  # drop badly incomplete sessions (half-days have 7)

    vol = d["volume"].where(d["volume"] > 0)
    d["session_vwap"] = d["pv"] / vol
    d["mf_ratio"] = d["mfv"] / vol                                   # -1..1
    ud = d["up_vol"] + d["dn_vol"]
    d["up_vol_frac"] = d["up_vol"] / ud.where(ud > 0)                 # 0..1
    d["lh_mf"] = d["lh_mfv"] / d["lh_vol"].where(d["lh_vol"] > 0)     # -1..1
    d["close_vs_vwap"] = d["close"] / d["session_vwap"] - 1
    d["avg_trade_size"] = d["volume"] / d["trades"].where(d["trades"] > 0)
    drng = d["high"] - d["low"]
    d["clv"] = np.where(drng > 0, ((d["close"] - d["low"]) - (d["high"] - d["close"])) / drng.where(drng > 0, 1), 0.0)
    d["ret"] = d["close"].pct_change()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14, min_periods=5).mean()
    return d


def daily_tables(bars: pd.DataFrame, tickers, as_of=None) -> tuple[dict, dict]:
    """Return ({ticker: daily_df}, {ticker: skip_reason})."""
    tables, skipped = {}, {}
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of is not None else None
    if cutoff is not None and pd.Timestamp(as_of).tzinfo is not None:
        cutoff = pd.Timestamp(as_of).tz_convert(NY).tz_localize(None).normalize()
    for t in normalize_tickers(tickers):
        sub = bars[bars["symbol"] == t] if len(bars) else bars
        if len(sub) == 0:
            skipped[t] = "no data returned"
            continue
        d = build_daily(sub)
        if cutoff is not None:
            d = d[d.index <= cutoff]
        tables[t] = d
    return tables, skipped


# --------------------------------------------------------------------------
# Normalization and scoring helpers
# --------------------------------------------------------------------------
def series_z(rolled: pd.Series, window: int, baseline: int) -> float:
    """Robust z-score of the latest value of an already-rolled series against
    its own values from the `baseline` sessions before the current window."""
    rolled = rolled.astype(float)
    if len(rolled) < window + 15:
        return np.nan
    current = rolled.iloc[-1]
    base = rolled.iloc[-(baseline + window):-window].dropna()
    if pd.isna(current) or len(base) < 15:
        return np.nan
    med = base.median()
    scale = (base - med).abs().median() * 1.4826
    if not scale or scale < 1e-12:
        scale = base.std(ddof=0)
    if not scale or scale < 1e-12:
        return 0.0
    return float(np.clip((current - med) / scale, -Z_CLIP, Z_CLIP))


def window_z(series: pd.Series, window: int, baseline: int) -> float:
    """z-score of the mean over the last `window` sessions, compared with
    rolling `window`-means from the preceding baseline period."""
    rolled = series.astype(float).rolling(window, min_periods=max(2, math.ceil(window * 0.6))).mean()
    return series_z(rolled, window, baseline)


def combine(zs: dict, weights: dict, intercept: float, scale: float,
            penalty: float = 0.0) -> tuple[float, float]:
    """Weighted average of available z-scores -> logistic score in 0..1.
    Missing features drop out; at least half the total weight must be present."""
    total = sum(abs(w) for w in weights.values())
    num = den = 0.0
    for k, w in weights.items():
        z = zs.get(k)
        if z is not None and not pd.isna(z):
            num += w * z
            den += abs(w)
    if den == 0 or den < 0.5 * total:
        return np.nan, np.nan
    composite = num / den - penalty
    prob = 1.0 / (1.0 + math.exp(-(intercept + scale * composite)))
    return composite, prob


def label(score_pct: float) -> str:
    if pd.isna(score_pct):
        return "n/a"
    if score_pct >= 70:
        return "Strong"
    if score_pct >= 55:
        return "Moderate"
    if score_pct >= 40:
        return "Weak"
    return "None"


def finalize(rows: list[dict], skipped: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    df.attrs["skipped"] = skipped
    return df


def print_report(df: pd.DataFrame, title: str, display_cols: list[str]) -> None:
    print(f"\n=== {title} ===")
    if len(df) == 0:
        print("No tickers scored.")
    else:
        cols = [c for c in display_cols if c in df.columns]
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.float_format", lambda x: f"{x:,.2f}"):
            print(df[cols].to_string(index=False))
    for t, reason in df.attrs.get("skipped", {}).items():
        print(f"  skipped {t}: {reason}")
    print("Score = uncalibrated accumulation confidence (0-100), not a true probability.")
