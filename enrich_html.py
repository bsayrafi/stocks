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
import html
import os
import constants
from datetime import datetime, timedelta, timezone
from event_catalysts import *
from event_catalysts import _get_sp500_membership


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
}


# ---------------------------------------------------------------- data
# 2. Update fetch_hourly_data to flatten MultiIndex columns from yfinance
def fetch_hourly_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    return df


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


def fetch_fundamentals(ticker: str) -> dict:
    """Pull a handful of fundamental data points for a ticker. Returns a dict of
    label -> formatted string. Any field yfinance doesn't have is shown as "N/A";
    if the whole fetch fails (network hiccup, delisted ticker, etc.) an empty
    dict is returned so the technical screen can still run."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
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

def fetch_analyst_data(ticker: str) -> dict:
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

    try:
        tkr = yf.Ticker(ticker)
    except Exception:
        return data

    # --- Consensus key ("buy", "hold", ...), mean score, and analyst count
    try:
        info = tkr.info or {}
        if info.get("recommendationKey") is not None:
            data["recommendation_key"] = str(info["recommendationKey"]).replace("_", " ").title()
        if info.get("recommendationMean") is not None:
            data["recommendation_mean"] = f"{float(info['recommendationMean']):.2f}"
        if info.get("numberOfAnalystOpinions") is not None:
            data["num_analyst_opinions"] = str(int(info["numberOfAnalystOpinions"]))
    except Exception:
        pass

    # --- Buy/hold/sell breakdown, most recent period ("0m" = current month)
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

    # --- EPS estimate revision trend, current quarter ("0q")
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

    # --- Number of analysts contributing to the current-quarter EPS estimate
    try:
        est = tkr.earnings_estimate
        if est is not None and not est.empty and "0q" in est.index:
            n = est.loc["0q"].get("numberOfAnalysts")
            if n is not None and pd.notna(n):
                data["eps_num_analysts"] = str(int(n))
    except Exception:
        pass

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

def run_screen(ticker: str, cfg: dict, sp500_members: dict | None = None) -> dict:
    """Fetch data and compute everything needed for one ticker's report.
    Pure computation - no printing - so the same result can feed both the
    console output and the HTML report."""
    cfg = dict(cfg)
    cfg["TICKER"] = ticker
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

    fundamentals = fetch_fundamentals(cfg["TICKER"])
    analyst = fetch_analyst_data(cfg["TICKER"])
    #catalysts = catalyst_snapshot(cfg["TICKER"], sp500_members=sp500_members)
    catalysts = get_event_catalysts(cfg["TICKER"], sp500_members=sp500_members).to_dict()

    return {
        "ticker": cfg["TICKER"],
        "cfg": cfg,
        "current_price": current_price,
        "results": results,
        "sr_levels": sr_levels,
        "today_range": today_range,
        "h1_macd_ok": h1_macd_ok,
        "vol_ok": vol_ok,
        "stop": stop,
        "take_profit": take_profit_target,
        "entry_support": entry_support,
        "signal": signal,
        "fundamentals": fundamentals,
        "analyst": analyst,
        "catalysts": catalysts,

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
        print(f"[{tf}] close={r['close']} MA{cfg['MA_PERIOD']}={r['MA']} trend_up={r['trend_up']!s:<5} "
              f"%K={r['%K']:>6} %D={r['%D']:>6} K>D={r['k_above_d']!s:<5} "
              f"stoch_bullish={r['stoch_bullish']!s:<5} overbought={r['overbought']!s:<5} "
              f"candles=[{patterns}]")
        print(f"      support={sr['support']}  resistance={sr['resistance']}  "
              f"(last {cfg['SR_LOOKBACK'][tf]} bars)")

    print(f"\nToday's range: {today_range['day_low']} - {today_range['day_high']} "
          f"(range: {today_range['day_range']})")

    print(f"\n1h MACD bullish: {report['h1_macd_ok']}")
    print(f"1h volume confirmed (>= {cfg['VOLUME_MULTIPLIER']}x {cfg['VOLUME_MA_PERIOD']}-period avg): {report['vol_ok']}")
    print(f"ATR({cfg['ATR_PERIOD']}) suggested stop: {report['stop']} "
          f"({cfg['ATR_STOP_MULTIPLIER']}x ATR below current price)")

    # <--- ADD THIS BLOCK --->
    risk = report['current_price'] - report['stop']
    print(f"Take-Profit Target (1.5x R:R): {report['take_profit_target']} "
          f"(Risking {risk:.2f} per share)")
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
    report = run_screen(ticker, CONFIG)
    #print_report(report)
    return report


# ---------------------------------------------------------------- HTML report

def _badge(ok: bool, true_text: str = "PASS", false_text: str = "FAIL") -> str:
    cls = "pass" if ok else "fail"
    text = true_text if ok else false_text
    return f'<span class="badge {cls}">{html.escape(str(text))}</span>'


def render_ticker_html(report: dict) -> str:
    cfg = report["cfg"]
    results = report["results"]
    sr_levels = report["sr_levels"]
    today_range = report["today_range"]
    signal = report["signal"]
    hrd = signal["hard_requirements_detail"]
    scd = signal["structural_confirmation_detail"]

    signal_cls = "buy" if signal["signal"] == "BUY" else "wait"

    fundamentals = report.get("fundamentals") or {}
    fundamentals_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(value))}</td></tr>"
        for label, value in fundamentals.items()
    )
    fundamentals_block = ""
    if fundamentals_rows:
        fundamentals_block = f"""
      <h3>Fundamentals</h3>
      <table class="detail-table fundamentals-table"><tbody>{fundamentals_rows}
      </tbody></table>"""

    analyst = report.get("analyst") or {}
    analyst_block = ""
    if analyst:
        counts = analyst.get("recommendation_counts") or {}
        counts_rows = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{n}</td></tr>" for label, n in counts.items()
        )
        eps_improving = analyst.get("eps_improving")
        if eps_improving is None:
            eps_trend_html = '<span class="badge">N/A</span>'
        else:
            eps_trend_html = _badge(eps_improving, "IMPROVING", "DETERIORATING")

        analyst_block = f"""
      <h3>Analyst Recommendations</h3>
      <table class="detail-table"><tbody>
        <tr><td>Consensus</td><td>{html.escape(analyst['recommendation_key'])}</td></tr>
        <tr><td>Mean Score</td><td>{html.escape(analyst['recommendation_mean'])}</td></tr>
        <tr><td># Analyst Opinions</td><td>{html.escape(analyst['num_analyst_opinions'])}</td></tr>
        {counts_rows}
      </tbody></table>

      <h3>EPS Estimate Revisions (current quarter)</h3>
      <table class="detail-table"><tbody>
        <tr><td>Current Consensus EPS</td><td>{html.escape(analyst['eps_current'])}</td></tr>
        <tr><td>EPS 30 Days Ago</td><td>{html.escape(analyst['eps_30d_ago'])}</td></tr>
        <tr><td># Analysts (EPS)</td><td>{html.escape(analyst['eps_num_analysts'])}</td></tr>
        <tr><td>EPS Trend</td><td>{eps_trend_html}</td></tr>
      </tbody></table>"""

    catalysts = report.get("catalysts") or {}
    catalysts_block = ""
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
        buyback_html = "<br>".join(html.escape(h) for h in buyback_headlines) or "none"
        guidance_html = "<br>".join(html.escape(h) for h in guidance_headlines) or "none"

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
        <div class="signal-badge {signal_cls}">{signal['signal']}</div>
      </div>

      <table class="tf-table">
        <thead>
          <tr>
            <th>TF</th><th>Close</th><th>MA{cfg['MA_PERIOD']}</th><th>Trend</th>
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

          <h3>Other levels</h3>
          <table class="detail-table"><tbody>
            <tr><td>Today's range</td><td>{today_range['day_low']} - {today_range['day_high']} (range {today_range['day_range']})</td></tr>
            <tr><td>1h entry support</td><td>{report['entry_support']}</td></tr>
            <tr><td>ATR({cfg['ATR_PERIOD']}) Suggested stop</td><td>{report['stop']} ({cfg['ATR_STOP_MULTIPLIER']}x ATR)</td></tr>
            <tr><td><strong>Take-Profit Target (1.5x)</strong></td><td><strong>{report['take_profit']}</strong></td></tr>
          </tbody></table>

        </div>
      </div>
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
  .fundamentals-table td:first-child {{ color: var(--muted); width: 45%; }}
  .note {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}

  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px;
    font-weight: 700; letter-spacing: 0.03em;
  }}
  .badge.pass {{ background: rgba(46,204,113,0.15); color: var(--pass); }}
  .badge.fail {{ background: rgba(231,76,60,0.15); color: var(--fail); }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="subtitle">Multi-Timeframe Buy Signal Screener</div>
{ticker_sections_html}
</body>
</html>
"""


def main(tickers, fileapp):
    """Screen a list of tickers: print each to console and write one HTML
    report covering all of them. Call as main(TICKERS) or main(["ORCL", ...])."""
    sections_html = []
    sp500_members = _get_sp500_membership(verbose=False)

    for ticker in tickers:
        try:
            report = run_screen(ticker, CONFIGH, sp500_members=sp500_members)
            #print_report(report)
            sections_html.append(render_ticker_html(report))
        except Exception as e:
            print(f"\n=== {ticker}: skipped due to error ===")
            print(f"  {type(e).__name__}: {e}")
            sections_html.append(
                f'<section class="card"><h2>{html.escape(ticker)}</h2>'
                f'<p class="note">Skipped: {html.escape(type(e).__name__)}: {html.escape(str(e))}</p></section>'
            )

    tz_gmt3 = timezone(timedelta(hours=3))
    #timestamp = datetime.now(tz_gmt3).strftime("%Y%m%d_%H%M")
    timestamp = constants.get_dayprefix()+"_" + constants.get_timeprefix()
    out_path = os.path.join("reports", f"{fileapp}_signal_report_{timestamp}.html")

    report_title = f"{fileapp}_Signal Report {timestamp}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_html_report(report_title, "\n".join(sections_html)))

    print(f"HTML report written to {out_path}")


