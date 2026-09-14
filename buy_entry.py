import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import time

def safe_download(ticker, retries=3, delay=1.5, **kwargs):
    last_df = pd.DataFrame()
    for attempt in range(retries):
        try:
            df = yf.download(ticker, auto_adjust=True, progress=False, **kwargs)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= 30:
                return df
            last_df = df
        except Exception:
            pass  # swallow transient errors (timeouts, connection resets, etc.) and retry
        if attempt < retries - 1:
            time.sleep(delay)
    return last_df  # best available attempt, even if still empty/short
   
def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

def check_buy_zone_confirmation(tickers, lookback_days=60, swing_window=5,
                                 bonus_confirmed_threshold=2, bonus_near_miss_threshold=1,
                                 atr_period=14, atr_stop_multiplier=0.5, reward_risk_ratio=1.5,
                                 break_lookback=10, pullback_window=15, pullback_tolerance_pct=2.0,
                                 bounce_tolerance_pct=3.0, ma_touch_tolerance_pct=1.0):
    results = []
    near_misses = []

    for ticker in tickers:
        try:
            df = safe_download(ticker, period=f"{lookback_days}d", interval="1d")            
            if df.empty or len(df) < 30:
                results.append({"Ticker": ticker, "Error": "Insufficient data"})
                continue

            df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["SMA50"] = df["Close"].rolling(50).mean() if len(df) >= 50 else None
            df["VolAvg20"] = df["Volume"].rolling(20).mean()
            df["ATR"] = compute_atr(df, period=atr_period)

            close = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2]
            ema21 = df["EMA21"].iloc[-1]
            sma50 = df["SMA50"].iloc[-1] if df["SMA50"] is not None else None
            sma50_prior = df["SMA50"].iloc[-6] if df["SMA50"] is not None and len(df) >= 56 else None
            volume = df["Volume"].iloc[-1]
            vol_avg = df["VolAvg20"].iloc[-1]
            atr_val = df["ATR"].iloc[-1]

            # Support: local swing lows via scipy
            low_idx = argrelextrema(df["Low"].values, np.less_equal, order=swing_window)[0]
            recent_low = df["Low"].iloc[low_idx[-1]] if len(low_idx) else df["Low"].iloc[-20:].min()
            support_holding = close > recent_low * 0.995  # allow tiny undershoot / noise

            # Resistance: local swing highs via scipy (excluding last 2 days)
            high_idx = argrelextrema(df["High"].iloc[:-2].values, np.greater_equal, order=swing_window)[0]
            resistance = df["High"].iloc[high_idx[-1]] if len(high_idx) else df["High"].iloc[-20:-2].max()

            # ============================================================
            # ENTRY TYPE DETECTION — checked in priority order
            # ============================================================
            entry_type = "None"
            stop_anchor = recent_low  # default anchor, overridden per type below

            # --- Type 1: Fresh Breakout — broke resistance recently, still holding above it
            was_below_resistance = (df["Close"].iloc[-break_lookback:-2] <= resistance).any()
            fresh_breakout = was_below_resistance and (close > resistance) and (prev_close > resistance)

            # --- Type 2: Pullback/Retest — broke out earlier, dipped back near the level, reclaiming
            broke_out_earlier = (df["Close"].iloc[-pullback_window:-5] > resistance).any()
            pullback_low_window = df["Low"].iloc[-5:]
            pulled_back_near_level = (pullback_low_window <= resistance * (1 + pullback_tolerance_pct / 100)).any()
            pullback_entry = broke_out_earlier and pulled_back_near_level and (close > resistance)

            # --- Type 3: Support Bounce — range bottom, tested multiple times, bouncing now
            low_tolerance = recent_low * (1 + bounce_tolerance_pct / 100)
            support_touches = (df["Low"].iloc[-20:] <= low_tolerance).sum()
            bounce_confirmation = close > prev_close and close <= low_tolerance
            support_bounce = support_touches >= 2 and bounce_confirmation

            # --- Type 4: MA Bounce — trend continuation, pulled back to rising MA, bouncing off it
            sma50_rising = (sma50 is not None and sma50_prior is not None and sma50 > sma50_prior)
            trend_up = (sma50 is not None) and (close > sma50) and sma50_rising
            touched_ema = (df["Low"].iloc[-3:] <= ema21 * (1 + ma_touch_tolerance_pct / 100)).any()
            ma_bounce_today = close > ema21 and close > prev_close
            ma_bounce = trend_up and touched_ema and ma_bounce_today

            # --- Assign entry type by priority, and set the stop anchor accordingly ---
            if fresh_breakout:
                entry_type = "Fresh Breakout"
                stop_anchor = recent_low
            elif pullback_entry:
                entry_type = "Pullback"
                stop_anchor = pullback_low_window.min()
            elif support_bounce:
                entry_type = "Support Bounce"
                stop_anchor = recent_low
            elif ma_bounce:
                entry_type = "MA Bounce"
                stop_anchor = df["Low"].iloc[-3:].min()  # the actual retest low near the MA

            structural_signal = entry_type != "None"

            ema_reclaim = close > ema21 and df["Close"].iloc[-5:-1].min() < ema21
            sma_reclaim = (sma50 is not None) and close > sma50 and df["Close"].iloc[-5:-1].min() < sma50
            volume_confirmation = volume > 1.5 * vol_avg if pd.notna(vol_avg) else False

            lows = df["Low"]
            swing_lows = []
            for i in range(swing_window, len(lows) - swing_window):
                window = lows.iloc[i - swing_window:i + swing_window + 1]
                if lows.iloc[i] == window.min():
                    swing_lows.append(lows.iloc[i])
            higher_lows = len(swing_lows) >= 3 and swing_lows[-1] > swing_lows[-2] > swing_lows[-3]

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
            setup_status = "N/A"
            min_risk_pct = 0.5
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
                setup_status = "Invalid (missing ATR/support)"

            row = {
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
                "ATR": round(atr_val, 2) if pd.notna(atr_val) else None,
                "Setup_Status": setup_status,
            }
            results.append(row)

            if tier == "Near Miss":
                missing = [k for k, v in bonus.items() if not v]
                near_misses.append({"Ticker": ticker, "Missing_Bonus": ", ".join(missing),
                                     "Bonus_Score": f"{bonus_score}/4", "Entry_Type": entry_type})

        except Exception as e:
            results.append({"Ticker": ticker, "Error": str(e)})

    full_df = pd.DataFrame(results)
    near_miss_df = pd.DataFrame(near_misses)
    confirmed_df = full_df[
        (full_df.get("Tier") == "Confirmed") & (full_df["Setup_Status"].str.startswith("Valid"))
    ] if "Tier" in full_df.columns else pd.DataFrame()

    return {"all": full_df, "confirmed": confirmed_df, "near_misses": near_miss_df}
    



def find_intraday_entry(ticker, entry_type="None", interval="1h", period="5d"):
    """
    Lightweight entry-timing check for swing trades (few days to few weeks hold).
    Goal: avoid buying into an extended spike, not precision-time the entry.
    Extension tolerance flexes by entry_type since different patterns carry
    different amounts of expected intraday movement.
    """
    df = safe_download(ticker, period=period, interval=interval)
    if df.empty:
        return {"Ticker": ticker, "Error": "No intraday data"}

    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()

    close = df["Close"].iloc[-1]
    vwap = df["VWAP"].iloc[-1]
    ema9 = df["EMA9"].iloc[-1]

    bars_per_day = 7 if interval == "1h" else 26  # rough session length for extension window
    today_low = df["Low"].iloc[-bars_per_day:].min()
    extension_pct = ((close - today_low) / today_low) * 100

    # Momentum patterns (breakout/pullback) can legitimately run further before it's "too extended".
    # Mean-reversion patterns (support bounce/MA bounce) should still look calm — a big intraday pop
    # on a bounce setup is a red flag, not confirmation.
    extension_limits = {
        "Fresh Breakout": 5.0,
        "Pullback": 4.0,
        "Support Bounce": 2.5,
        "MA Bounce": 2.5,
        "None": 3.0,  # fallback default
    }
    max_extension_pct = extension_limits.get(entry_type, 3.0)

    signals = {
        "At or below VWAP": close <= vwap * 1.005,
        "Not extended from today's low": extension_pct <= max_extension_pct,
        "EMA9 reclaim (not first candle of spike)": close > ema9 and df["Close"].iloc[-3:-1].min() < ema9,
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