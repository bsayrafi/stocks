"""
intraday_strategy.py

Intraday S&P 500 strategy: VWAP + EMA9/SMA9 + Volume Profile (POC/VAH/VAL),
with a SPY + sector-ETF top-down bias filter and a 10:00-11:30 ET time window.

Design assumptions (adjust as needed):
- You already have ~1yr of DAILY OHLCV for the S&P 500 universe in your screener
  (the `data` frame in your loop). That is NOT used for the intraday levels here;
  it's daily-bar data and can't produce a same-day VWAP or intraday volume profile.
- This module pulls its own INTRADAY bars (5-minute) via yfinance, separately,
  for: the stock itself, SPY (market), and the stock's sector ETF.
  yfinance keeps 5m bars for ~60 days, which is plenty for "today" + "prior session".
- "Prior session" (most recent completed trading day) is what the volume profile
  (POC/VAH/VAL) and the naked-POC scan are built from. "Today's session" (partial,
  developing) is what VWAP/EMA9/SMA9 and the live bias/regime/entries are built from.
- Only 1 share is being sized (per your note), so position sizing/risk-per-trade is
  intentionally left out — this returns levels, not order sizes.
- Everything here is descriptive/decision-support output, not automated execution.

Call `analyze_intraday(symbol, sector=..., spy_df=..., sector_df=...)` once per
symbol per loop pass. Pass in `spy_df` / `sector_df` yourself if you want to fetch
SPY and each sector ETF ONCE per loop pass and reuse them across symbols, instead
of re-downloading SPY/sector bars for every ticker (recommended — see bottom of
file for the loop-integration example).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

ET = "America/New_York"
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "5d"          # gives us "today" + a few prior sessions
EMA_PERIOD = 9
SMA_PERIOD = 9
VWAP_BAND_MULTS = (1.0, 2.0)    # std-dev bands around session VWAP
VALUE_AREA_PCT = 0.70           # standard 70% value area
PROFILE_BINS = 30               # price buckets for the volume profile histogram
NAKED_POC_LOOKBACK_SESSIONS = 5 # how many prior sessions to scan for untested POCs

TRADE_WINDOW_START = dt.time(10, 0)
TRADE_WINDOW_END = dt.time(11, 30)

# --- NEW: gap/outlier-session detection + higher-timeframe trend filter ---
GAP_PCT_THRESHOLD = 3.0      # session open vs. prior session's close, in %
RANGE_OUTLIER_MULT = 2.5     # session's High-Low vs. median of other sessions in the window
VOLUME_OUTLIER_MULT = 2.5    # session's total volume vs. median of other sessions in the window
HIGHER_TF_SMA_PERIOD = 50    # daily-bar SMA period used as the higher-timeframe trend filter

# Rough GICS-sector -> SPDR sector ETF map. Prefer passing `sector` in yourself
# from whatever sector field your screener already has; this is just a fallback.
SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Financial Services": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Cyclical": "XLY",
    "Consumer Staples": "XLP",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

# Simple in-process cache so repeated calls (every few minutes) don't
# re-download the same intraday bars needlessly within a short window.
_CACHE: dict[str, tuple[dt.datetime, pd.DataFrame]] = {}
_CACHE_TTL = dt.timedelta(minutes=2)


# --------------------------------------------------------------------------
# Data fetch
# --------------------------------------------------------------------------

def fetch_intraday(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch recent 5-minute bars for `ticker`, tz-converted to US/Eastern.
    Cached for _CACHE_TTL to avoid hammering yfinance when you call this
    every few minutes across many symbols."""
    now = dt.datetime.utcnow()
    cached = _CACHE.get(ticker)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    try:
        df = yf.download(
            ticker,
            period=INTRADAY_PERIOD,
            interval=INTRADAY_INTERVAL,
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # yfinance sometimes returns MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)

    _CACHE[ticker] = (now, df)
    return df


def split_sessions(df: pd.DataFrame) -> dict[dt.date, pd.DataFrame]:
    """Split an intraday bar DataFrame into one DataFrame per calendar (ET) date."""
    return {d: g for d, g in df.groupby(df.index.date)}


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def add_vwap_bands(session_df: pd.DataFrame) -> pd.DataFrame:
    """Session-anchored VWAP (resets at the start of `session_df`) with
    standard-deviation bands, computed on typical price."""
    df = session_df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan)

    cum_vol = vol.cumsum()
    cum_tpv = (tp * vol).cumsum()
    df["vwap"] = cum_tpv / cum_vol

    # Running (population) variance of typical price around VWAP, volume-weighted.
    cum_tp2v = ((tp ** 2) * vol).cumsum()
    variance = (cum_tp2v / cum_vol) - (df["vwap"] ** 2)
    df["vwap_std"] = np.sqrt(variance.clip(lower=0))

    for mult in VWAP_BAND_MULTS:
        df[f"vwap_up_{mult}"] = df["vwap"] + mult * df["vwap_std"]
        df[f"vwap_dn_{mult}"] = df["vwap"] - mult * df["vwap_std"]
    return df


def add_emas(session_df: pd.DataFrame) -> pd.DataFrame:
    df = session_df.copy()
    df["ema9"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["sma9"] = df["Close"].rolling(SMA_PERIOD).mean()
    return df


@dataclass
class VolumeProfile:
    poc: float
    vah: float
    val: float
    price_levels: np.ndarray
    volume_at_level: np.ndarray


def compute_volume_profile(session_df: pd.DataFrame, bins: int = PROFILE_BINS) -> Optional[VolumeProfile]:
    """Build a simple price-bucketed volume profile for one session and
    derive POC (highest-volume bucket) and the 70% value area (VAH/VAL)."""
    if session_df.empty or session_df["Volume"].sum() == 0:
        return None

    lo = session_df["Low"].min()
    hi = session_df["High"].max()
    if hi <= lo:
        return None

    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vol_at_bin = np.zeros(bins)

    # Distribute each bar's volume across the bins its High-Low range spans,
    # weighted evenly across those bins (a standard TPO/volume-profile approximation).
    for _, row in session_df.iterrows():
        b_lo, b_hi, b_vol = row["Low"], row["High"], row["Volume"]
        if b_vol == 0 or b_hi <= b_lo:
            continue
        start_idx = np.searchsorted(edges, b_lo, side="right") - 1
        end_idx = np.searchsorted(edges, b_hi, side="right") - 1
        start_idx = max(0, min(start_idx, bins - 1))
        end_idx = max(0, min(end_idx, bins - 1))
        span = end_idx - start_idx + 1
        vol_at_bin[start_idx:end_idx + 1] += b_vol / span

    if vol_at_bin.sum() == 0:
        return None

    poc_idx = int(np.argmax(vol_at_bin))
    poc = centers[poc_idx]

    # Grow the value area outward from POC, bin by bin, until >= 70% of volume.
    total_vol = vol_at_bin.sum()
    target = total_vol * VALUE_AREA_PCT
    included = {poc_idx}
    acc = vol_at_bin[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    while acc < target and (lo_idx > 0 or hi_idx < bins - 1):
        next_lo = vol_at_bin[lo_idx - 1] if lo_idx > 0 else -1
        next_hi = vol_at_bin[hi_idx + 1] if hi_idx < bins - 1 else -1
        if next_hi >= next_lo:
            hi_idx += 1
            acc += vol_at_bin[hi_idx]
            included.add(hi_idx)
        else:
            lo_idx -= 1
            acc += vol_at_bin[lo_idx]
            included.add(lo_idx)

    vah = centers[hi_idx]
    val = centers[lo_idx]

    return VolumeProfile(poc=poc, vah=vah, val=val,
                          price_levels=centers, volume_at_level=vol_at_bin)


def flag_outlier_sessions(sessions: list[pd.DataFrame]) -> list[bool]:
    """NEW. Flag sessions that gapped hard from the prior session's close, or
    whose range/volume are outliers versus the rest of the window — the
    signature of an earnings or news-driven day. `sessions` must be in
    chronological order. Used so a POC/VAH/VAL built on a gap day isn't
    quietly trusted as an ordinary reference level."""
    if len(sessions) < 2:
        return [False] * len(sessions)

    ranges = [float(s["High"].max() - s["Low"].min()) for s in sessions if not s.empty]
    volumes = [float(s["Volume"].sum()) for s in sessions if not s.empty]
    median_range = float(np.median(ranges)) if ranges else 0.0
    median_volume = float(np.median(volumes)) if volumes else 0.0

    flags = []
    prev_close = None
    for s in sessions:
        if s.empty:
            flags.append(False)
            prev_close = None
            continue
        day_open = float(s["Open"].iloc[0])
        day_close = float(s["Close"].iloc[-1])
        day_range = float(s["High"].max() - s["Low"].min())
        day_volume = float(s["Volume"].sum())

        gap_pct = abs(day_open - prev_close) / prev_close * 100 if prev_close else 0.0
        is_outlier = (
            gap_pct >= GAP_PCT_THRESHOLD
            or (median_range > 0 and day_range >= RANGE_OUTLIER_MULT * median_range)
            or (median_volume > 0 and day_volume >= VOLUME_OUTLIER_MULT * median_volume)
        )
        flags.append(bool(is_outlier))
        prev_close = day_close
    return flags


def find_naked_pocs(sessions: list[pd.DataFrame], current_price: float,
                     exclude_last: int = 1) -> list[float]:
    """POCs from prior sessions (excluding the most recent `exclude_last`,
    typically today) that price has NOT traded back through since. These act
    as strong untested magnets."""
    naked = []
    ordered = sessions[:-exclude_last] if exclude_last else sessions
    for i, sess in enumerate(ordered):
        vp = compute_volume_profile(sess)
        if vp is None:
            continue
        poc = vp.poc
        # "Untested" = no later session (including partial today) traded through it.
        later = sessions[len(ordered[:i]) + exclude_last:] if exclude_last else sessions[i + 1:]
        touched = any(
            (not s.empty) and (s["Low"].min() <= poc <= s["High"].max())
            for s in later
        )
        if not touched:
            naked.append(poc)
    return sorted(set(round(p, 2) for p in naked))


# --------------------------------------------------------------------------
# Bias / regime / entries
# --------------------------------------------------------------------------

def classify_bias(last_close: float, vwap: float, prior_poc: float) -> str:
    if last_close > vwap and last_close > prior_poc:
        return "bullish"
    if last_close < vwap and last_close < prior_poc:
        return "bearish"
    return "neutral"


def classify_regime(today_df: pd.DataFrame, vp_today: Optional[VolumeProfile]) -> str:
    """Trend day = value area is a narrow slice of the day's total range and
    price sits near one edge of it (one-sided). Rotational = value area takes
    up most of the range (balanced, POC near the middle)."""
    if vp_today is None or today_df.empty:
        return "undetermined"
    day_range = today_df["High"].max() - today_df["Low"].min()
    if day_range <= 0:
        return "undetermined"
    va_width = vp_today.vah - vp_today.val
    width_ratio = va_width / day_range

    last_close = today_df["Close"].iloc[-1]
    # Where does price sit relative to the value area? Near an edge -> trending.
    if vp_today.vah > vp_today.val:
        edge_pos = (last_close - vp_today.val) / (vp_today.vah - vp_today.val)
    else:
        edge_pos = 0.5

    if width_ratio < 0.55 and (edge_pos < 0.15 or edge_pos > 0.85):
        return "trend"
    return "rotational"


def pct_distance(price: float, level: Optional[float]) -> Optional[float]:
    if level is None or level == 0 or price is None:
        return None
    return round((price - level) / level * 100, 3)


def in_trade_window(now_et: dt.datetime) -> bool:
    t = now_et.timetz().replace(tzinfo=None)
    return TRADE_WINDOW_START <= t <= TRADE_WINDOW_END


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def _bias_from_ticker_df(df: Optional[pd.DataFrame]) -> tuple[str, Optional[float], Optional[float]]:
    """Given a raw intraday df for an index/sector ETF, return
    (bias, today_vwap, prior_poc)."""
    if df is None or df.empty:
        return "undetermined", None, None
    sessions_map = split_sessions(df)
    dates = sorted(sessions_map.keys())
    if len(dates) < 2:
        return "undetermined", None, None
    today_df = add_vwap_bands(sessions_map[dates[-1]])
    prior_vp = compute_volume_profile(sessions_map[dates[-2]])
    if today_df.empty or prior_vp is None:
        return "undetermined", None, None
    last_close = today_df["Close"].iloc[-1]
    vwap_now = today_df["vwap"].iloc[-1]
    bias = classify_bias(last_close, vwap_now, prior_vp.poc)
    return bias, vwap_now, prior_vp.poc


def analyze_intraday(
    symbol: str,
    sector: Optional[str] = None,
    now: Optional[dt.datetime] = None,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    daily_df: Optional[pd.DataFrame] = None,  # NEW
) -> dict:
    """
    Run the full intraday strategy check for one symbol and return a flat
    dict of everything useful for a trade decision (merge this straight
    into your `intraday_block`).

    Args:
        symbol: ticker to analyze.
        sector: sector name (from your own screener metadata) used to pick
                the sector ETF via SECTOR_TO_ETF, for both the `sector_etf`
                output field and (if `sector_df` isn't supplied) the fetch.
        now: current time (ET). Defaults to actual current time.
        spy_df / sector_df: pass pre-fetched intraday DataFrames (from
                fetch_intraday) to avoid re-downloading SPY/sector bars for
                every symbol in your loop. If omitted, they're fetched here.
        daily_df: NEW. Pass in the same daily OHLCV `df` you already slice
                per symbol from your 1yr screener data (`data.xs(symbol, ...)`
                in your loop). Used only for the higher-timeframe trend
                filter (a daily 50-SMA check) — if omitted, that filter is
                skipped and `higher_tf_trend` comes back "unknown".
    """
    now_et = now or dt.datetime.now(tz=pd.Timestamp.now(tz=ET).tz)
    out: dict = {"symbol": symbol, "checked_at": now_et.isoformat()}

    # --- time filter -------------------------------------------------
    out["in_time_window"] = in_trade_window(now_et)

    # --- fetch stock intraday bars ------------------------------------
    stock_raw = fetch_intraday(symbol)
    if stock_raw is None or stock_raw.empty:
        out["error"] = "no_intraday_data"
        return out

    sessions_map = split_sessions(stock_raw)
    dates = sorted(sessions_map.keys())
    if len(dates) < 2:
        out["error"] = "insufficient_session_history"
        return out

    today_raw = sessions_map[dates[-1]]
    today_df = add_emas(add_vwap_bands(today_raw))
    prior_sessions = [sessions_map[d] for d in dates[:-1]]
    prior_vp = compute_volume_profile(prior_sessions[-1])
    today_vp = compute_volume_profile(today_df)

    if today_df.empty or prior_vp is None:
        out["error"] = "insufficient_session_data"
        return out

    last_row = today_df.iloc[-1]
    current_price = float(last_row["Close"])
    vwap_now = float(last_row["vwap"])
    ema9_now = float(last_row["ema9"])
    sma9_now = float(last_row["sma9"]) if not np.isnan(last_row["sma9"]) else None
    vwap_up1 = float(last_row.get("vwap_up_1.0", np.nan))
    vwap_dn1 = float(last_row.get("vwap_dn_1.0", np.nan))
    vwap_up2 = float(last_row.get("vwap_up_2.0", np.nan))
    vwap_dn2 = float(last_row.get("vwap_dn_2.0", np.nan))

    # --- market (SPY) and sector bias ---------------------------------
    if spy_df is None:
        spy_df = fetch_intraday("SPY")
    market_bias, market_vwap, market_prior_poc = _bias_from_ticker_df(spy_df)
    out["market_bias"] = market_bias

    etf = SECTOR_TO_ETF.get(sector) if sector else None
    if sector_df is None and etf:
        sector_df = fetch_intraday(etf)
    sector_bias, sector_vwap, sector_prior_poc = _bias_from_ticker_df(sector_df)
    out["sector"] = sector
    out["sector_etf"] = etf
    out["sector_bias"] = sector_bias

    # --- stock-level bias & regime -------------------------------------
    stock_bias = classify_bias(current_price, vwap_now, prior_vp.poc)
    out["stock_bias"] = stock_bias
    regime = classify_regime(today_df, today_vp)
    out["regime"] = regime

    # --- NEW: gap / outlier session check ---------------------------------
    # Flags whether the session behind prior_poc/vah/val (or today's own
    # session) was itself a gap/news-style outlier day, so a POC computed
    # from a crash/gap session isn't quietly trusted as an ordinary level.
    all_sessions_ordered = [sessions_map[d] for d in dates]
    session_outlier_flags = flag_outlier_sessions(all_sessions_ordered)
    prior_session_is_outlier = session_outlier_flags[-2] if len(session_outlier_flags) >= 2 else False
    today_session_is_outlier = session_outlier_flags[-1] if len(session_outlier_flags) >= 1 else False
    out["prior_session_is_outlier"] = prior_session_is_outlier
    out["today_session_is_outlier"] = today_session_is_outlier

    # --- NEW: higher-timeframe trend filter (needs daily_df) ---------------
    # A session-level bounce can look "bullish" while the stock is still in a
    # multi-week downtrend on a higher timeframe. If daily_df is supplied,
    # check price against a daily SMA to catch that; otherwise this is
    # skipped ("unknown") rather than blocking anything.
    daily_sma = None
    higher_tf_trend = "unknown"
    if daily_df is not None and "Close" in daily_df and len(daily_df) >= HIGHER_TF_SMA_PERIOD:
        daily_sma = float(daily_df["Close"].rolling(HIGHER_TF_SMA_PERIOD).mean().iloc[-1])
        higher_tf_trend = "above_sma" if current_price > daily_sma else "below_sma"
    out["daily_sma50"] = round(daily_sma, 4) if daily_sma is not None else None
    out["higher_tf_trend"] = higher_tf_trend

    # combined directional bias: only "confirmed" when SPY, sector, and stock agree
    if stock_bias == "bullish" and market_bias in ("bullish", "neutral") and sector_bias in ("bullish", "neutral"):
        combined_bias = "bullish"
    elif stock_bias == "bearish" and market_bias in ("bearish", "neutral") and sector_bias in ("bearish", "neutral"):
        combined_bias = "bearish"
    else:
        combined_bias = "conflicted"

    # --- NEW: downgrade to "caution" if the agreement above rests on a
    # gap-distorted reference level, or fights the higher-timeframe trend.
    # entry_trend/entry_fade only fire on "bullish"/"bearish", so this is
    # what actually suppresses false positives like the QTWO case.
    bias_notes = []
    if prior_session_is_outlier:
        bias_notes.append("prior session gapped or had a volume/range outlier — POC/VAH/VAL reference may be unreliable")
    if today_session_is_outlier:
        bias_notes.append("today's session is itself a gap/volume outlier — treat levels built from it cautiously")
    if higher_tf_trend == "below_sma" and stock_bias == "bullish":
        bias_notes.append(f"price is below the daily {HIGHER_TF_SMA_PERIOD}-SMA — bullish session bias conflicts with the higher-timeframe trend")
    if higher_tf_trend == "above_sma" and stock_bias == "bearish":
        bias_notes.append(f"price is above the daily {HIGHER_TF_SMA_PERIOD}-SMA — bearish session bias conflicts with the higher-timeframe trend")

    if bias_notes and combined_bias in ("bullish", "bearish"):
        combined_bias = "caution"
    out["combined_bias"] = combined_bias
    out["bias_notes"] = bias_notes

    # --- naked POCs ------------------------------------------------------
    naked_pocs = find_naked_pocs(prior_sessions + [today_df], current_price)
    out["naked_pocs"] = naked_pocs
    nearest_naked_poc = min(naked_pocs, key=lambda p: abs(p - current_price)) if naked_pocs else None
    out["nearest_naked_poc"] = nearest_naked_poc

    # --- levels / key reference prices ------------------------------------
    out["current_price"] = current_price
    out["vwap"] = round(vwap_now, 4)
    out["ema9"] = round(ema9_now, 4)
    out["sma9"] = round(sma9_now, 4) if sma9_now is not None else None
    out["vwap_band_up_1std"] = round(vwap_up1, 4) if not np.isnan(vwap_up1) else None
    out["vwap_band_dn_1std"] = round(vwap_dn1, 4) if not np.isnan(vwap_dn1) else None
    out["vwap_band_up_2std"] = round(vwap_up2, 4) if not np.isnan(vwap_up2) else None
    out["vwap_band_dn_2std"] = round(vwap_dn2, 4) if not np.isnan(vwap_dn2) else None
    out["prior_poc"] = round(prior_vp.poc, 4)
    out["prior_vah"] = round(prior_vp.vah, 4)
    out["prior_val"] = round(prior_vp.val, 4)
    if today_vp is not None:
        out["today_poc"] = round(today_vp.poc, 4)
        out["today_vah"] = round(today_vp.vah, 4)
        out["today_val"] = round(today_vp.val, 4)
    else:
        out["today_poc"] = out["today_vah"] = out["today_val"] = None

    # --- distances to key levels (%) --------------------------------------
    out["dist_to_vwap_pct"] = pct_distance(current_price, vwap_now)
    out["dist_to_ema9_pct"] = pct_distance(current_price, ema9_now)
    out["dist_to_prior_poc_pct"] = pct_distance(current_price, prior_vp.poc)
    out["dist_to_prior_vah_pct"] = pct_distance(current_price, prior_vp.vah)
    out["dist_to_prior_val_pct"] = pct_distance(current_price, prior_vp.val)
    out["dist_to_nearest_naked_poc_pct"] = pct_distance(current_price, nearest_naked_poc)

    # --- entry: trend leg -------------------------------------------------
    # Pullback-to-EMA9 continuation in the direction of combined_bias.
    # "Setup present" = last few bars pulled back to/through EMA9 while price
    # is still holding the right side of VWAP. "Triggered" = price has since
    # actually broken the trigger level (the prior bar's high/low), which is
    # the actual entry signal — setup_present alone is just "on watch."
    entry_trend = None
    lookback = today_df.tail(6)
    if combined_bias == "bullish" and regime in ("trend", "undetermined"):
        pulled_back = (lookback["Low"] <= lookback["ema9"]).any()
        held_vwap = current_price > vwap_now
        trigger_level = float(lookback["High"].iloc[-2]) if len(lookback) >= 2 else None
        setup_present = bool(pulled_back and held_vwap)
        triggered = bool(setup_present and trigger_level is not None and current_price > trigger_level)
        entry_trend = {
            "direction": "long",
            "setup_present": setup_present,
            "triggered": triggered,
            "trigger_price": trigger_level,
            "trigger_condition": "break above prior bar high after EMA9 pullback",
        }
    elif combined_bias == "bearish" and regime in ("trend", "undetermined"):
        pulled_back = (lookback["High"] >= lookback["ema9"]).any()
        held_vwap = current_price < vwap_now
        trigger_level = float(lookback["Low"].iloc[-2]) if len(lookback) >= 2 else None
        setup_present = bool(pulled_back and held_vwap)
        triggered = bool(setup_present and trigger_level is not None and current_price < trigger_level)
        entry_trend = {
            "direction": "short",
            "setup_present": setup_present,
            "triggered": triggered,
            "trigger_price": trigger_level,
            "trigger_condition": "break below prior bar low after EMA9 pullback",
        }
    out["entry_trend"] = entry_trend

    # --- entry: fade leg ----------------------------------------------------
    # Price stretched beyond a VWAP band near an untested POC/value-area edge.
    # "Setup present" = the band was touched/exceeded within the recent lookback
    # (looks back a few bars rather than only the current one, so we can still
    # see the setup after price has started to pull back). "Triggered" = price
    # has since pulled back from that stretch's extreme by at least
    # required_rejection_pct — an actual (partial) rejection, not just a touch.
    # This is a coarser proxy for "reversal candle confirmed" than a real
    # candlestick-pattern check would be; treat it as a first-pass filter.
    entry_fade = None
    lookback_fade = today_df.tail(6)
    extreme_high = float(lookback_fade["High"].max())
    extreme_low = float(lookback_fade["Low"].min())

    if not np.isnan(vwap_up2) and extreme_high >= vwap_up2:
        target_level = nearest_naked_poc or prior_vp.poc
        required_rejection_pct = round(abs(extreme_high - vwap_now) / extreme_high * 0.25 * 100, 3)
        pulled_back_pct = round((extreme_high - current_price) / extreme_high * 100, 3)
        triggered = bool(current_price < extreme_high and pulled_back_pct >= required_rejection_pct)
        entry_fade = {
            "direction": "short",
            "setup_present": True,
            "triggered": triggered,
            "stretched_beyond": "vwap_+2std",
            "extreme_price": round(extreme_high, 4),
            "pulled_back_pct": pulled_back_pct,
            "fade_target": target_level,
            "required_rejection_pct": required_rejection_pct,
            "trigger_condition": "reversal candle / failed new high at or above +2 std VWAP band",
        }
    elif not np.isnan(vwap_dn2) and extreme_low <= vwap_dn2:
        target_level = nearest_naked_poc or prior_vp.poc
        required_rejection_pct = round(abs(extreme_low - vwap_now) / extreme_low * 0.25 * 100, 3)
        pulled_back_pct = round((current_price - extreme_low) / extreme_low * 100, 3)
        triggered = bool(current_price > extreme_low and pulled_back_pct >= required_rejection_pct)
        entry_fade = {
            "direction": "long",
            "setup_present": True,
            "triggered": triggered,
            "stretched_beyond": "vwap_-2std",
            "extreme_price": round(extreme_low, 4),
            "pulled_back_pct": pulled_back_pct,
            "fade_target": target_level,
            "required_rejection_pct": required_rejection_pct,
            "trigger_condition": "reversal candle / failed new low at or below -2 std VWAP band",
        }
    out["entry_fade"] = entry_fade

    # --- stops & target -------------------------------------------------
    stop_trend = None
    if entry_trend and entry_trend["direction"] == "long":
        stop_trend = round(min(ema9_now, prior_vp.poc) * 0.999, 4)
    elif entry_trend and entry_trend["direction"] == "short":
        stop_trend = round(max(ema9_now, prior_vp.poc) * 1.001, 4)
    out["stop_trend"] = stop_trend

    stop_fade = None
    if entry_fade and entry_fade["direction"] == "short":
        stop_fade = round(current_price * 1.002, 4)  # just beyond the stretch extreme
    elif entry_fade and entry_fade["direction"] == "long":
        stop_fade = round(current_price * 0.998, 4)
    out["stop_fade"] = stop_fade

    target = None
    if entry_trend and entry_trend["direction"] == "long":
        candidates = [lvl for lvl in (prior_vp.vah, nearest_naked_poc) if lvl and lvl > current_price]
        target = min(candidates) if candidates else None
    elif entry_trend and entry_trend["direction"] == "short":
        candidates = [lvl for lvl in (prior_vp.val, nearest_naked_poc) if lvl and lvl < current_price]
        target = max(candidates) if candidates else None
    elif entry_fade:
        target = entry_fade["fade_target"]
    out["target"] = round(target, 4) if target else None

    # --- misc / notes -----------------------------------------------------
    out["prior_session_date"] = str(dates[-2])
    out["today_session_date"] = str(dates[-1])
    out["bars_today"] = len(today_df)
    out["volume_profile_today"] = (
        {"levels": today_vp.price_levels.round(4).tolist(),
         "volume": today_vp.volume_at_level.round(0).tolist()}
        if today_vp is not None else None
    )

    return out


# --------------------------------------------------------------------------
# Loop integration example (not executed on import)
# --------------------------------------------------------------------------
"""
from intraday_strategy import analyze_intraday, fetch_intraday

# Fetch SPY once per pass through the whole shortlist, reuse for every symbol.
spy_df = fetch_intraday("SPY")
sector_cache: dict[str, pd.DataFrame] = {}

for symbol in sp500_tickers:
    try:
        df = data.xs(symbol, level=1, axis=1).dropna()
        if len(df) < 200:
            continue
        ...
        sector = company_block.get("sector")   # however your screener stores it
        etf = SECTOR_TO_ETF.get(sector)
        if etf and etf not in sector_cache:
            sector_cache[etf] = fetch_intraday(etf)

        intraday_block = analyze_intraday(
            symbol,
            sector=sector,
            spy_df=spy_df,
            sector_df=sector_cache.get(etf),
            daily_df=df,  # NEW: your existing 1yr daily df, for the higher-timeframe filter
        )

        results2.append({
            " Ticker ": symbol,
            **company_block,
            **intraday_block,
        })
    except Exception as e:
        continue
"""
