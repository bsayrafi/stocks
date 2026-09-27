"""
trend_data_pipeline.py
=========================
Long-only trend-following pipeline. Two layers, deliberately separated:

  DAILY (the trend gate — "is a real trend happening at all"):
    Multi-horizon (20/60/120/250d) volatility-adjusted time-series momentum
    (Moskowitz/Ooi/Pedersen style), confirmed by OBV slope agreeing with
    price. Computed once per day per symbol, then SHIFTED by one row so
    the value attached to trading date D reflects what was known as of
    D-1's close — never peeks at the still-forming day.

  HOURLY (the entry trigger — "when exactly to get in"):
    Causal swing-low/swing-high detection (a pivot is only confirmed once
    K bars afterward exist — you can't know a low was a swing low until
    the market proves it). A trend-continuation entry fires when a new
    CONFIRMED swing low sits higher than the prior one (a genuine rising-
    lows structure, not a moving-average proxy for one) AND price then
    breaks back above the swing high that preceded it, gated by ADX
    (real trend strength, not chop) and relative volume (real
    participation, not a low-conviction move).

  EXIT: no fixed target. A chandelier trailing stop
    (max(initial_stop, highest_close_since_entry - TRAIL_ATR_MULT*ATR))
    only ever moves up; the trade exits on the first bar that trades
    through it, or at a hold-window cap. Since there's no fixed R target,
    the label is the REALIZED R-multiple (continuous), not a binary
    win/loss — this is a regression problem, not classification.

Data feed: uses SIP (full consolidated volume) with a safety buffer on
the end timestamp, since Alpaca's free tier only restricts the LAST 15
MINUTES of SIP data, not historical depth — confirmed empirically
(IEX captured ~3.8% of AAPL's true SIP volume in a direct comparison).

Setup: same venv as the FVG project; needs macro_calendar.py and
fvg_data_pipeline_hourly.py (for load_tickers_by_sector) in the same folder.

Run:
    python3 trend_data_pipeline.py --sectors Technology "Health Care" --years 2 --out data/trend_dataset.parquet
"""

import os
import time
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from macro_calendar import load_calendar, nearest_event, DEFAULT_CALENDAR_PATH
from fvg_data_pipeline_hourly import load_tickers_by_sector

load_dotenv()

REQUEST_TIMEOUT_SEC = 30  # hard cap per network call — a single hung request can never freeze the whole run again


def run_with_timeout(fn, *args, timeout=REQUEST_TIMEOUT_SEC, **kwargs):
    """Runs fn in a worker thread with a hard timeout. Returns (result, None)
    on success, or (None, error_message) on timeout/exception — never hangs
    the caller, no matter what the underlying call does."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout), None
        except FutureTimeoutError:
            return None, f"timed out after {timeout}s"
        except Exception as e:
            return None, str(e)

CONFIG = {
    "HOURLY_LOOKBACK_YEARS": 2,
    "DAILY_LOOKBACK_YEARS": 8,          # generous — momentum/regime history, cheap on the free tier (since 2016)
    "SIP_BUFFER_MIN": 30,               # keep every request's `end` this far before "now" to clear the 15-min SIP restriction
    "MARKET_TZ": "America/New_York",
    "FEED": DataFeed.SIP,

    # -- daily trend gate --
    "MOMENTUM_HORIZONS": [20, 60, 120, 250],
    "DAILY_MOMENTUM_MIN_SCORE": 0.0,    # composite score must clear this to count as "trending"
    "OBV_SLOPE_WINDOW": 20,

    # -- hourly swing structure --
    "SWING_K": 5,                       # bars required on each side to confirm a pivot
    "ADX_PERIOD": 14,
    "ADX_MIN": 22,
    "REL_VOLUME_MIN": 1.3,              # breakout bar's volume vs its own 20-bar average
    "VOL_AVG_WINDOW": 20,
    "ATR_PERIOD": 14,
    "ATR_PCTILE_WINDOW": 100,

    # -- trade simulation --
    "STOP_CUSHION_ATR": 0.25,           # initial stop = swing low - this * ATR
    "TRAIL_ATR_MULT": 2.0,              # chandelier trail distance from highest close since entry
    "MAX_HOLD_BARS": 60,                # ~9 trading days at ~6.5 bars/day

    "REQUEST_SLEEP": 0.3,
    "SECTORS": ["Technology"],
}


# ---------------------------------------------------------------------------
# Data fetch — SIP by default, with a safety buffer so we clear the free
# tier's last-15-minutes restriction rather than falling back to IEX.
# ---------------------------------------------------------------------------

def fetch_bars(client, symbol, timeframe, years, cfg=CONFIG):
    end = datetime.now(timezone.utc) - timedelta(minutes=cfg["SIP_BUFFER_MIN"])
    start = end - timedelta(days=365 * years)
    req = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end,
        adjustment="all", feed=cfg["FEED"],
    )
    try:
        bars = client.get_stock_bars(req).df
    except Exception as e:
        print(f"[{symbol}] fetch failed: {e}")
        return None
    if bars is None or bars.empty:
        return None
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    bars = bars.rename(columns=str.lower).sort_index()
    return bars[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Shared indicators (ATR, EMA, RSI, ADX) — causal, no lookahead
# ---------------------------------------------------------------------------

def add_indicators(df, cfg=CONFIG):
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(cfg["ATR_PERIOD"]).mean()
    df["atr_pctile"] = df["atr"].rolling(cfg["ATR_PCTILE_WINDOW"]).rank(pct=True)

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["vol_avg20"] = df["volume"].rolling(cfg["VOL_AVG_WINDOW"]).mean()

    # ADX (Wilder) — real trend-strength filter, separate from direction
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_wilder = tr.ewm(alpha=1 / cfg["ADX_PERIOD"], adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / cfg["ADX_PERIOD"], adjust=False).mean() / atr_wilder
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / cfg["ADX_PERIOD"], adjust=False).mean() / atr_wilder
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / cfg["ADX_PERIOD"], adjust=False).mean()

    return df


# ---------------------------------------------------------------------------
# Daily trend gate: multi-horizon vol-adjusted momentum + OBV confirmation,
# shifted by one row so date D reflects only what was known as of D-1 close.
# ---------------------------------------------------------------------------

def build_daily_trend_context(daily_bars, cfg=CONFIG):
    df = daily_bars.copy()
    daily_ret = df["close"].pct_change()

    scores = []
    for N in cfg["MOMENTUM_HORIZONS"]:
        ret_N = df["close"].pct_change(N)
        vol_N = daily_ret.rolling(N).std() * np.sqrt(252)
        scores.append(ret_N / vol_N.replace(0, np.nan))
    composite_momentum = pd.concat(scores, axis=1).mean(axis=1)

    obv = (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()

    def _slope(y):
        x = np.arange(len(y))
        if np.any(np.isnan(y)):
            return np.nan
        return np.polyfit(x, y, 1)[0]
    obv_slope = obv.rolling(cfg["OBV_SLOPE_WINDOW"]).apply(_slope, raw=True)
    # normalize slope by OBV's own rolling scale so it's comparable across symbols/price levels
    obv_scale = obv.rolling(cfg["OBV_SLOPE_WINDOW"]).std().replace(0, np.nan)
    obv_slope_norm = obv_slope / obv_scale

    tz = cfg["MARKET_TZ"]
    idx = df.index
    dates = idx.tz_convert(tz).date if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz).date

    ctx = pd.DataFrame(index=dates)
    ctx["composite_momentum"] = composite_momentum.values
    ctx["obv_slope_norm"] = obv_slope_norm.values
    ctx["daily_trend_confirmed"] = (
        (ctx["composite_momentum"] > cfg["DAILY_MOMENTUM_MIN_SCORE"]) & (ctx["obv_slope_norm"] > 0)
    )
    # shift: date D gets the value computed as of D-1's completed close
    ctx = ctx.shift(1)
    ctx = ctx[~ctx.index.duplicated(keep="last")]
    return ctx


# ---------------------------------------------------------------------------
# Causal swing-point detection
# ---------------------------------------------------------------------------

def detect_swing_points(df, k):
    """A bar at position i is a confirmed pivot once k bars exist on both
    sides. The caller must only trust is_low[i]/is_high[i] once the scan
    has reached position i+k — that's what makes this causal."""
    lows, highs = df["low"].values, df["high"].values
    n = len(df)
    is_low = np.zeros(n, dtype=bool)
    is_high = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        window_low = lows[i - k:i + k + 1]
        if lows[i] <= window_low.min():
            is_low[i] = True
        window_high = highs[i - k:i + k + 1]
        if highs[i] >= window_high.max():
            is_high[i] = True
    return is_low, is_high


# ---------------------------------------------------------------------------
# Trend-continuation entry detection: rising confirmed swing lows +
# breakout above the swing high between them, gated by ADX and volume.
# ---------------------------------------------------------------------------

def find_trend_entries(df, daily_ctx, symbol, calendar, cfg=CONFIG):
    k = cfg["SWING_K"]
    is_low, is_high = detect_swing_points(df, k)
    n = len(df)
    close, high, low = df["close"].values, df["high"].values, df["low"].values

    tz = cfg["MARKET_TZ"]
    idx = df.index
    local_ts = idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz)
    local_dates = local_ts.date
    local_hours = local_ts.hour
    local_dow = local_ts.dayofweek

    last_two_lows = []          # [(pos, price), ...] most recent confirmed swing lows
    pending_structure_high = None
    awaiting_breakout = False
    breakout_level = None
    breakout_source_low = None
    consecutive_higher_lows = 0

    entries = []

    for j in range(n):
        conf_pos = j - k
        if conf_pos >= 0:
            if is_low[conf_pos]:
                price = low[conf_pos]
                if last_two_lows and price > last_two_lows[-1][1] and pending_structure_high is not None:
                    awaiting_breakout = True
                    breakout_level = pending_structure_high[1]
                    breakout_source_low = (conf_pos, price)
                    consecutive_higher_lows += 1
                elif last_two_lows and price <= last_two_lows[-1][1]:
                    consecutive_higher_lows = 0  # structure broke, reset
                last_two_lows.append((conf_pos, price))
                if len(last_two_lows) > 2:
                    last_two_lows.pop(0)
                pending_structure_high = None
            if is_high[conf_pos]:
                price = high[conf_pos]
                if pending_structure_high is None or price > pending_structure_high[1]:
                    pending_structure_high = (conf_pos, price)

        if awaiting_breakout and close[j] > breakout_level:
            atr_j = df["atr"].iloc[j]
            adx_j = df["adx"].iloc[j]
            vol_avg_j = df["vol_avg20"].iloc[j]
            if pd.isna(atr_j) or atr_j <= 0 or pd.isna(adx_j) or pd.isna(vol_avg_j) or vol_avg_j <= 0:
                awaiting_breakout = False
                continue

            rel_volume = df["volume"].iloc[j] / vol_avg_j
            trade_date = local_dates[j]
            htf = daily_ctx.loc[trade_date] if trade_date in daily_ctx.index else None
            daily_ok = (htf is not None) and bool(htf.get("daily_trend_confirmed", False)) \
                if htf is not None and not pd.isna(htf.get("daily_trend_confirmed", np.nan)) else False

            passes_filters = (adx_j >= cfg["ADX_MIN"]) and (rel_volume >= cfg["REL_VOLUME_MIN"]) and daily_ok

            if passes_filters:
                near_macro = 0
                if calendar is not None:
                    ev, _ = nearest_event(idx[j], calendar)
                    near_macro = 1 if ev is not None else 0

                swing_low_idx, swing_low_price = breakout_source_low
                trend_steepness_atr = (swing_low_price - last_two_lows[0][1]) / atr_j if len(last_two_lows) == 2 else np.nan

                entries.append({
                    "symbol": symbol,
                    "entry_idx": j,
                    "entry_time": idx[j],
                    "entry_price": close[j],
                    "swing_low_idx": swing_low_idx,
                    "swing_low_price": swing_low_price,
                    "breakout_level": breakout_level,
                    "breakout_strength_atr": (close[j] - breakout_level) / atr_j,
                    "adx": adx_j,
                    "rel_volume": rel_volume,
                    "trend_steepness_atr": trend_steepness_atr,
                    "consecutive_higher_lows": consecutive_higher_lows,
                    "rsi14": df["rsi14"].iloc[j],
                    "atr_pctile": df["atr_pctile"].iloc[j],
                    "atr_at_entry": atr_j,
                    "daily_composite_momentum": htf["composite_momentum"] if htf is not None else np.nan,
                    "daily_obv_slope_norm": htf["obv_slope_norm"] if htf is not None else np.nan,
                    "hour_of_day": local_hours[j],
                    "day_of_week": local_dow[j],
                    "near_macro_event": near_macro,
                })
            awaiting_breakout = False  # consume signal either way

    return pd.DataFrame(entries)


# ---------------------------------------------------------------------------
# Trailing-stop trade simulation -> realized R-multiple (regression label)
# ---------------------------------------------------------------------------

def simulate_trailing_trades(df, entries, cfg=CONFIG):
    if entries.empty:
        return entries
    close, low = df["close"].values, df["low"].values
    atr = df["atr"].values
    n = len(df)

    realized_r, exit_reasons, bars_held = [], [], []
    for _, row in entries.iterrows():
        i = int(row["entry_idx"])
        entry_price = row["entry_price"]
        initial_stop = row["swing_low_price"] - cfg["STOP_CUSHION_ATR"] * row["atr_at_entry"]
        initial_risk = entry_price - initial_stop
        if initial_risk <= 0:
            realized_r.append(np.nan); exit_reasons.append("invalid"); bars_held.append(np.nan)
            continue

        stop = initial_stop
        highest_close = entry_price
        end = min(i + 1 + cfg["MAX_HOLD_BARS"], n)
        exit_price, exit_idx, reason = None, None, "time_exit"

        for j in range(i + 1, end):
            c = close[j]
            if c > highest_close:
                highest_close = c
            atr_j = atr[j]
            if not pd.isna(atr_j):
                stop = max(stop, highest_close - cfg["TRAIL_ATR_MULT"] * atr_j)
            if low[j] <= stop:
                exit_price, exit_idx, reason = stop, j, "trailing_stop"
                break

        if exit_price is None:
            exit_idx = end - 1
            exit_price = close[exit_idx]

        realized_r.append((exit_price - entry_price) / initial_risk)
        exit_reasons.append(reason)
        bars_held.append(exit_idx - i)

    entries = entries.copy()
    entries["realized_r"] = realized_r
    entries["exit_reason"] = exit_reasons
    entries["bars_held"] = bars_held
    return entries


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_dataset(tickers, api_key, api_secret, cfg=CONFIG, cache_dir="cache/trend_run", verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    client = StockHistoricalDataClient(api_key, api_secret)
    try:
        calendar = load_calendar(DEFAULT_CALENDAR_PATH)
    except FileNotFoundError:
        calendar = None
        print(f"Warning: {DEFAULT_CALENDAR_PATH} not found — near_macro_event will be 0 for all rows.")

    for n, symbol in enumerate(tickers, 1):
        cache_path = os.path.join(cache_dir, f"{symbol}.parquet")
        if os.path.exists(cache_path):
            if verbose:
                print(f"[{n}/{len(tickers)}] {symbol} — already cached, skipping")
            continue

        daily_raw, err = run_with_timeout(fetch_bars, client, symbol, TimeFrame.Day, cfg["DAILY_LOOKBACK_YEARS"], cfg)
        if err:
            print(f"[{n}/{len(tickers)}] {symbol} — daily fetch failed ({err}), skipping")
            continue
        time.sleep(cfg["REQUEST_SLEEP"])
        if daily_raw is None or len(daily_raw) < 260:
            pd.DataFrame().to_parquet(cache_path)  # empty marker so we don't retry a symbol with genuinely too little history
            continue
        daily_ctx = build_daily_trend_context(daily_raw, cfg)

        hourly_raw, err = run_with_timeout(fetch_bars, client, symbol, TimeFrame.Hour, cfg["HOURLY_LOOKBACK_YEARS"], cfg)
        if err:
            print(f"[{n}/{len(tickers)}] {symbol} — hourly fetch failed ({err}), skipping")
            continue
        time.sleep(cfg["REQUEST_SLEEP"])
        if hourly_raw is None or len(hourly_raw) < 250:
            pd.DataFrame().to_parquet(cache_path)
            continue
        hourly_ind = add_indicators(hourly_raw, cfg)

        entries = find_trend_entries(hourly_ind, daily_ctx, symbol, calendar, cfg)
        if entries.empty:
            pd.DataFrame().to_parquet(cache_path)
            if verbose:
                print(f"[{n}/{len(tickers)}] {symbol} — 0 entries")
            continue
        labeled = simulate_trailing_trades(hourly_ind, entries, cfg)
        labeled.to_parquet(cache_path)
        if verbose:
            print(f"[{n}/{len(tickers)}] {symbol} — {len(labeled)} entries")

    # assemble final dataset from whatever's in the cache dir (this run's results + any prior resumed ones)
    cached_files = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
    all_rows = [pd.read_parquet(f) for f in cached_files]
    all_rows = [df for df in all_rows if not df.empty]

    dataset = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not dataset.empty:
        dataset["stop_cushion_atr"] = cfg["STOP_CUSHION_ATR"]
        dataset["trail_atr_mult"] = cfg["TRAIL_ATR_MULT"]
        dataset["max_hold_bars"] = cfg["MAX_HOLD_BARS"]
    if verbose:
        print(f"\nDone. {len(dataset)} total entries across {len(tickers)} tickers.")
        if not dataset.empty:
            valid = dataset.dropna(subset=["realized_r"])
            print(f"Usable (valid risk) entries: {len(valid)}")
            print(f"Mean realized R: {valid['realized_r'].mean():.3f}  |  "
                  f"Median: {valid['realized_r'].median():.3f}  |  "
                  f"Win rate (R>0): {(valid['realized_r'] > 0).mean():.3f}")
            print(valid["exit_reason"].value_counts())
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build the trend-following dataset from Alpaca bars.")
    parser.add_argument("--sectors", nargs="*", default=CONFIG["SECTORS"],
                         help="GICS sectors to pull the universe from (space-separated, quote multi-word ones).")
    parser.add_argument("--tickers", nargs="*", default=None, help="Explicit tickers — overrides --sectors.")
    parser.add_argument("--years", type=int, default=CONFIG["HOURLY_LOOKBACK_YEARS"], help="Years of hourly history.")
    parser.add_argument("--out", default="data/trend_dataset.parquet")
    parser.add_argument("--cache-dir", default="cache/trend_run",
                         help="Per-symbol checkpoint directory — lets an interrupted run resume instead of restarting.")
    args = parser.parse_args()

    api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise SystemExit("Alpaca keys not found: set APCA_API_KEY_ID / APCA_API_SECRET_KEY in your .env file.")

    cfg = dict(CONFIG)
    cfg["HOURLY_LOOKBACK_YEARS"] = args.years

    if args.tickers:
        tickers = args.tickers
    else:
        tickers = []
        for sector in args.sectors:
            tickers.extend(load_tickers_by_sector(sector))
        tickers = sorted(set(tickers))
        print(f"Universe: {len(tickers)} tickers across sectors {args.sectors}")

    dataset = build_dataset(tickers, api_key, api_secret, cfg, cache_dir=args.cache_dir)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dataset.to_parquet(args.out, index=False)
    print(f"Wrote {len(dataset)} rows to {args.out}")


if __name__ == "__main__":
    main()
