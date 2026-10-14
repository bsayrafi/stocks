"""
Indicators, candle patterns, levels and session helpers - pure pandas / numpy (no network, no HTML).
Split out of enrich_html_v2.py, which re-exports every name (enrich_html_v2.X still works).
"""

import pandas as pd
import numpy as np
from setups_v2.config import CONFIGH

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV bars from 1h bars. "4h" is anchored to the regular session: two bars per day, MARKET_OPEN to
    MARKET_OPEN + 4h and the rest of the session (09:30-13:30 and 13:30-16:00 ET, the split TradingView uses).
    A plain resample("4h") counts 4-hour blocks from midnight of the FIRST day in the data, so its bars shift with
    daylight saving time and differ between the live scan and the backtest (different start dates)."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if str(rule).lower() == "4h" and isinstance(df.index, pd.DatetimeIndex) and len(df):
        tz = CONFIGH.get("MARKET_TZ", "America/New_York")
        idx = df.index.tz_convert(tz) if df.index.tz is not None else df.index
        open_min = int(_hhmm_td(CONFIGH.get("MARKET_OPEN", "09:30")).total_seconds() // 60)
        mins = idx.hour * 60 + idx.minute
        start = np.where(mins >= open_min + 240, open_min + 240, open_min)      # which half of the session
        key = idx.normalize() + pd.to_timedelta(start, unit="m")
        return df.groupby(key).agg(agg).dropna()
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


# ---------------------------------------------------------------- extended hours
# Close (last completed regular session) -> after-hours (16:00-20:00 that day)
# -> overnight (20:00 to 04:00 of the next session, Blue Ocean ATS) -> pre-market
# (04:00-09:30 of the next session). Session dates come from the price data, so
# weekends and market holidays are handled.

def _hhmm_td(hhmm: str) -> pd.Timedelta:
    h, m = map(int, hhmm.split(":"))
    return pd.Timedelta(hours=h, minutes=m)


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


def prev_session(daily: pd.DataFrame, in_progress: bool):
    """O/H/L/C of the last COMPLETED regular session: yesterday while today's bar is still forming
    (market open), otherwise the latest daily bar (pre-market / after-hours / overnight)."""
    d = daily.dropna(subset=["Close"])
    pos = -2 if in_progress else -1
    if len(d) < abs(pos):
        return None
    r = d.iloc[pos]
    before = d.iloc[pos - 1]["Close"] if len(d) > abs(pos) else None      # close of the session before it
    return {"date": d.index[pos].date().isoformat(), "open": round(float(r["Open"]), 2),
            "high": round(float(r["High"]), 2), "low": round(float(r["Low"]), 2),
            "close": round(float(r["Close"]), 2),
            "prior_close": None if before is None else round(float(before), 2)}
