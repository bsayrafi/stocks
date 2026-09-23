"""
Swing-trade buy-zone confirmation.

Two required core signals (support holding + structural entry type) and four
bonus signals. Data failures are reported as NoData, never as Rejected.
"""

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _flatten_columns(df):
    """yfinance nests as ('Close', 'TICK'); field name is on level 0."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    return df


def _trim_incomplete_tail(df):
    """
    Drop trailing placeholder/partial bars.

    yfinance appends a row for a session that has not settled; its OHLCV are
    NaN. Left in place it makes every downstream comparison evaluate False and
    ATR NaN. Keyed off the data, not the date, because a timezone-shifted index
    can stamp the phantom bar with yesterday's date.
    """
    while len(df) and df.iloc[-1][OHLCV].isna().any():
        df = df.iloc[:-1]
    return df


def validate_ohlcv(df, min_rows):
    """Return (ok, reason). Run after trimming, never before."""
    if df is None or df.empty:
        return False, "empty frame"
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        return False, f"missing columns: {missing}"
    if len(df) < min_rows:
        return False, f"{len(df)} bars, need {min_rows}"
    if df[OHLCV].iloc[-1].isna().any():
        return False, "NaN in last bar"
    if (df["Close"] <= 0).any():
        return False, "non-positive close"
    return True, "ok"


def safe_download(ticker, retries=3, delay=1.5, min_rows=30, **kwargs):
    """Fetch, flatten, trim. Returns whatever it got; caller validates."""
    last_df = pd.DataFrame()
    for attempt in range(retries):
        try:
            df = yf.download(ticker, auto_adjust=True, progress=False, **kwargs)
            df = _trim_incomplete_tail(_flatten_columns(df))
            if not df.empty and len(df) >= min_rows:
                return df
            last_df = df
            log.warning("%s: short frame (%d bars) on attempt %d",
                        ticker, len(df), attempt + 1)
        except Exception as exc:
            log.warning("%s: %s on attempt %d — %s",
                        ticker, type(exc).__name__, attempt + 1, exc)
        if attempt < retries - 1:
            time.sleep(delay)
    return last_df


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


# ---------------------------------------------------------------------------
# Buy-zone confirmation
# ---------------------------------------------------------------------------

def check_buy_zone_confirmation(tickers, lookback_days=60, fetch_period="1y",
                                swing_window=5,
                                bonus_confirmed_threshold=2, bonus_near_miss_threshold=1,
                                atr_period=14, atr_stop_multiplier=0.5, reward_risk_ratio=1.5,
                                break_lookback=10, pullback_window=15, pullback_tolerance_pct=2.0,
                                bounce_tolerance_pct=3.0, ma_touch_tolerance_pct=1.0,
                                min_risk_pct=0.5):
    """
    fetch_period governs history pulled (needs >= 56 bars for SMA50 + its prior
    reference); lookback_days governs how many recent BARS the swing-structure
    detection looks at. These were previously the same knob, which capped the
    frame at ~41 bars and made SMA50 — and therefore the MA Bounce entry type —
    permanently unavailable.
    """
    results = []
    near_misses = []
    no_data = []

    # SMA50 needs 50 bars; sma50_prior reads 6 bars back from the end of it.
    min_rows = max(60, atr_period + 1, swing_window * 2 + 1)

    for ticker in tickers:
        try:
            df = safe_download(ticker, period=fetch_period, interval="1d",
                               min_rows=min_rows)
            ok, reason = validate_ohlcv(df, min_rows)
            if not ok:
                log.warning("%s: NoData — %s", ticker, reason)
                no_data.append({"Ticker": ticker, "Reason": reason})
                results.append({"Ticker": ticker, "Tier": "NoData",
                                "Setup_Status": f"NoData ({reason})"})
                continue

            df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["SMA50"] = df["Close"].rolling(50, min_periods=50).mean()
            df["VolAvg20"] = df["Volume"].rolling(20, min_periods=20).mean()
            df["ATR"] = compute_atr(df, period=atr_period)

            # Structure detection window: last N bars. Indicators above are
            # computed on full history so their warm-up is already absorbed.
            win = df.iloc[-lookback_days:]

            close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            ema21 = float(df["EMA21"].iloc[-1])
            sma50 = df["SMA50"].iloc[-1]
            sma50_prior = df["SMA50"].iloc[-6] if len(df) >= 56 else np.nan
            volume = float(df["Volume"].iloc[-1])
            vol_avg = df["VolAvg20"].iloc[-1]
            atr_val = df["ATR"].iloc[-1]

            # Support: most recent local swing low
            low_idx = argrelextrema(win["Low"].values, np.less_equal, order=swing_window)[0]
            recent_low = float(win["Low"].iloc[low_idx[-1]]) if len(low_idx) \
                else float(win["Low"].iloc[-20:].min())
            support_holding = close > recent_low * 0.995  # allow tiny undershoot

            # Resistance: most recent local swing high, excluding last 2 sessions
            high_idx = argrelextrema(win["High"].iloc[:-2].values,
                                     np.greater_equal, order=swing_window)[0]
            resistance = float(win["High"].iloc[high_idx[-1]]) if len(high_idx) \
                else float(win["High"].iloc[-20:-2].max())

            # ============================================================
            # ENTRY TYPE DETECTION — checked in priority order
            # ============================================================
            entry_type = "None"
            stop_anchor = recent_low

            # --- Type 1: Fresh Breakout
            was_below_resistance = (df["Close"].iloc[-break_lookback:-2] <= resistance).any()
            fresh_breakout = bool(was_below_resistance
                                  and close > resistance
                                  and prev_close > resistance)

            # --- Type 2: Pullback/Retest
            broke_out_earlier = (df["Close"].iloc[-pullback_window:-5] > resistance).any()
            pullback_low_window = df["Low"].iloc[-5:]
            pulled_back_near_level = (
                pullback_low_window <= resistance * (1 + pullback_tolerance_pct / 100)
            ).any()
            pullback_entry = bool(broke_out_earlier
                                  and pulled_back_near_level
                                  and close > resistance)

            # --- Type 3: Support Bounce
            low_tolerance = recent_low * (1 + bounce_tolerance_pct / 100)
            support_touches = int((df["Low"].iloc[-20:] <= low_tolerance).sum())
            bounce_confirmation = close > prev_close and close <= low_tolerance
            support_bounce = bool(support_touches >= 2 and bounce_confirmation)

            # --- Type 4: MA Bounce
            sma50_rising = bool(pd.notna(sma50) and pd.notna(sma50_prior)
                                and sma50 > sma50_prior)
            trend_up = bool(pd.notna(sma50) and close > sma50 and sma50_rising)
            touched_ema = bool((df["Low"].iloc[-3:] <= ema21 * (1 + ma_touch_tolerance_pct / 100)).any())
            ma_bounce_today = close > ema21 and close > prev_close
            ma_bounce = bool(trend_up and touched_ema and ma_bounce_today)

            if fresh_breakout:
                entry_type = "Fresh Breakout"
                stop_anchor = recent_low
            elif pullback_entry:
                entry_type = "Pullback"
                stop_anchor = float(pullback_low_window.min())
            elif support_bounce:
                entry_type = "Support Bounce"
                stop_anchor = recent_low
            elif ma_bounce:
                entry_type = "MA Bounce"
                stop_anchor = float(df["Low"].iloc[-3:].min())

            structural_signal = entry_type != "None"

            ema_reclaim = bool(close > ema21 and df["Close"].iloc[-5:-1].min() < ema21)
            sma_reclaim = bool(pd.notna(sma50) and close > sma50
                               and df["Close"].iloc[-5:-1].min() < sma50)
            volume_confirmation = bool(volume > 1.5 * vol_avg) if pd.notna(vol_avg) else False

            lows = win["Low"]
            swing_lows = []
            for i in range(swing_window, len(lows) - swing_window):
                window = lows.iloc[i - swing_window:i + swing_window + 1]
                if lows.iloc[i] == window.min():
                    swing_lows.append(float(lows.iloc[i]))
            higher_lows = bool(len(swing_lows) >= 3
                               and swing_lows[-1] > swing_lows[-2] > swing_lows[-3])

            # --- Required (structural) vs Bonus (confidence boosters) ---
            required = {
                "Support holding": support_holding,
                "Structural signal (any entry type)": structural_signal,
            }
            bonus = {
                "EMA reclaim": ema_reclaim,
                "SMA reclaim": sma_reclaim,
                "Volume confirmation": volume_confirmation,
                "Higher lows": higher_lows,
            }
            checklist = {**required, **bonus}

            core_score = sum(required.values())
            bonus_score = sum(bonus.values())
            total_score = sum(checklist.values())

            if core_score == 2 and bonus_score >= bonus_confirmed_threshold:
                tier = "Confirmed"
            elif core_score == 2 and bonus_score >= bonus_near_miss_threshold:
                tier = "Near Miss"
            else:
                tier = "Rejected"

            # --- Stop loss / take profit (structural anchor + ATR) ---
            stop_loss = None
            take_profit = None
            risk_per_share = None
            effective_multiplier = atr_stop_multiplier

            if pd.notna(atr_val) and pd.notna(stop_anchor):
                stop_loss = round(stop_anchor - (atr_stop_multiplier * atr_val), 2)
                risk_per_share = round(close - stop_loss, 2)

                if risk_per_share <= 0:
                    setup_status = "Invalid (stop above price)"
                elif (risk_per_share / close) * 100 < min_risk_pct:
                    effective_multiplier = atr_stop_multiplier * 2
                    stop_loss = round(stop_anchor - (effective_multiplier * atr_val), 2)
                    risk_per_share = round(close - stop_loss, 2)
                    setup_status = "Valid (widened ATR cushion)"
                else:
                    setup_status = "Valid"

                take_profit = round(close + (reward_risk_ratio * risk_per_share), 2)
            else:
                setup_status = "Invalid (ATR unavailable)"

            results.append({
                "Ticker": ticker,
                **checklist,
                "Entry_Type": entry_type,
                "Core_Score": f"{core_score}/2",
                "Bonus_Score": f"{bonus_score}/4",
                "Total_Score": f"{total_score}/6",
                "Tier": tier,
                "Entry_Price": round(close, 2),
                "Stop_Loss": stop_loss,
                "Take_Profit": take_profit,
                "Risk_Per_Share": risk_per_share,
                "ATR_Multiplier_Used": effective_multiplier,
                "ATR": round(float(atr_val), 2) if pd.notna(atr_val) else None,
                "Setup_Status": setup_status,
                "Bars_Used": len(df),
            })

            if tier == "Near Miss":
                missing = [k for k, v in bonus.items() if not v]
                near_misses.append({"Ticker": ticker,
                                    "Missing_Bonus": ", ".join(missing),
                                    "Bonus_Score": f"{bonus_score}/4",
                                    "Entry_Type": entry_type})

        except Exception as exc:
            log.exception("%s: unhandled %s", ticker, type(exc).__name__)
            results.append({"Ticker": ticker, "Tier": "Error",
                            "Setup_Status": f"Error ({type(exc).__name__}: {exc})"})

    full_df = pd.DataFrame(results)
    near_miss_df = pd.DataFrame(near_misses)
    no_data_df = pd.DataFrame(no_data)

    if {"Tier", "Setup_Status"}.issubset(full_df.columns):
        confirmed_df = full_df[
            (full_df["Tier"] == "Confirmed")
            & (full_df["Setup_Status"].str.startswith("Valid", na=False))
        ]
    else:
        confirmed_df = pd.DataFrame()

    return {"all": full_df, "confirmed": confirmed_df,
            "near_misses": near_miss_df, "no_data": no_data_df}


# ---------------------------------------------------------------------------
# Intraday entry timing
# ---------------------------------------------------------------------------

def find_intraday_entry(ticker, entry_type="None", interval="1h", period="5d"):
    """
    Lightweight entry-timing check for swing trades (few days to few weeks hold).
    Goal: avoid buying into an extended spike, not precision-time the entry.
    Extension tolerance flexes by entry_type since different patterns carry
    different amounts of expected intraday movement.
    """
    df = safe_download(ticker, period=period, interval=interval, min_rows=10)
    ok, reason = validate_ohlcv(df, 10)
    if not ok:
        return {"Ticker": ticker, "Error": f"NoData ({reason})"}

    # VWAP anchored per session, not across the whole fetch window.
    day = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) \
        else pd.Index(df.index)
    pv = (df["Close"] * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum()
    df["VWAP"] = pv / vv
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()

    close = float(df["Close"].iloc[-1])
    vwap = float(df["VWAP"].iloc[-1])
    ema9 = float(df["EMA9"].iloc[-1])

    # Today's low from the actual session, not a fixed bar count that would
    # straddle the session boundary near the open.
    today_low = float(df.loc[day == day[-1], "Low"].min())
    extension_pct = ((close - today_low) / today_low) * 100

    # Momentum patterns (breakout/pullback) can legitimately run further before
    # it's "too extended". Mean-reversion patterns (support/MA bounce) should
    # still look calm — a big intraday pop there is a red flag, not confirmation.
    extension_limits = {
        "Fresh Breakout": 5.0,
        "Pullback": 4.0,
        "Support Bounce": 2.5,
        "MA Bounce": 2.5,
        "None": 3.0,
    }
    max_extension_pct = extension_limits.get(entry_type, 3.0)

    signals = {
        "At or below VWAP": close <= vwap * 1.005,
        "Not extended from today's low": extension_pct <= max_extension_pct,
        "EMA9 reclaim (not first candle of spike)":
            bool(close > ema9 and df["Close"].iloc[-3:-1].min() < ema9),
    }

    return {
        "Ticker": ticker,
        "Entry_Type": entry_type,
        "Close": round(close, 2),
        "Extension_%": round(extension_pct, 2),
        "Max_Extension_Allowed_%": max_extension_pct,
        **signals,
        "Entry_Score": f"{sum(signals.values())}/3",
    }
