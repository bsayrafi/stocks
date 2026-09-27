"""
fvg_data_pipeline_hourly.py
=============================
Hourly-timeframe FVG detection with higher-timeframe (daily) trend
confluence. Separate from fvg_data_pipeline.py (daily) — that file is
untouched; this is a parallel pipeline for the lower-timeframe variant.

Key differences from the daily pipeline:
  - FVGs are detected on hourly bars, but every FVG also carries daily-
    timeframe trend context as of the prior COMPLETED daily bar (causal —
    never peeks at the still-forming day).
  - FVGs whose 3-candle window spans a session boundary (different trading
    dates) are discarded — those are overnight/weekend gaps, not the
    displacement-candle pattern the concept describes.
  - MAX_HOLD_BARS is recalibrated for hourly bars (~6.5 bars/trading day),
    not reused from the daily config.
  - Ticker universe is filtered by GICS sector via --sector.

Setup: same venv/requirements as fvg_data_pipeline.py.

Run:
    python3 fvg_data_pipeline_hourly.py --sector Technology --years 2 --out data/fvg_hourly_tech.parquet
    python3 fvg_data_pipeline_hourly.py --sector Technology --tickers NVDA MSFT --years 1 --out data/fvg_hourly_test.parquet
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from fvg_data_pipeline import add_indicators, label_outcomes
from macro_calendar import load_calendar, nearest_event, DEFAULT_CALENDAR_PATH

load_dotenv()

CONFIG = {
    "LOOKBACK_YEARS": 2,
    "DAILY_CONTEXT_BUFFER_YEARS": 1,     # extra daily history fetched for EMA200 warm-up
    "ATR_PERIOD": 14,
    "MAX_HOLD_BARS": 20,                 # ~3 trading days at ~6.5 hourly bars/day
    "TARGET_R_MULTIPLE": 1.0,            # horizon sweep: AUC peaks at 1.0R (0.645) vs 1.5R+ (~0.60); pattern predicts fast moves better than sustained ones
    "STOP_ATR_MULT": 1.0,
    "MIN_GAP_ATR_MULT": 0.10,
    "SWING_LOOKBACK": 10,
    "VOL_AVG_WINDOW": 20,
    "ATR_PCTILE_WINDOW": 100,
    "CLUSTER_WINDOW_BARS": 30,           # ~5 trading days, for the recent-gap-clustering feature
    "REQUEST_SLEEP": 0.3,
    "FEED": DataFeed.IEX,
    "MARKET_TZ": "America/New_York",
    "SECTOR": "Technology",
}

SECTOR_ALIASES = {
    "tech": "Information Technology",
    "technology": "Information Technology",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "financials": "Financials",
    "financial": "Financials",
    "energy": "Energy",
    "industrials": "Industrials",
    "materials": "Materials",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "communication services": "Communication Services",
}


# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------

def load_tickers_by_sector(sector, verbose=True):
    target = SECTOR_ALIASES.get(sector.strip().lower(), sector)
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]

    symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    sector_col = next((c for c in df.columns if "sector" in c.lower()), None)
    if sector_col is None:
        raise RuntimeError(f"No sector column found in ticker source; columns were {list(df.columns)}")

    mask = df[sector_col].astype(str).str.contains(target, case=False, na=False)
    tickers = df.loc[mask, symbol_col].str.replace(".", "-", regex=False).tolist()
    if verbose:
        print(f"Sector '{sector}' -> matched '{target}': {len(tickers)} tickers")
    return tickers


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_bars(client, symbol, timeframe, years, cfg=CONFIG):
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment="all",
        feed=cfg["FEED"],
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
# Daily higher-timeframe context (causal — shifted to the prior completed day)
# ---------------------------------------------------------------------------

def build_daily_context(daily_bars, cfg=CONFIG):
    """Compute daily trend/momentum context, then shift by one row so that
    the value attached to trading date D is what was KNOWN as of D's open
    (i.e. derived from D-1's completed daily bar). Never leaks the forming day."""
    ind = add_indicators(daily_bars, cfg)
    tz = cfg["MARKET_TZ"]
    idx = ind.index
    dates = idx.tz_convert(tz).date if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz).date

    htf = pd.DataFrame(index=dates)
    htf["htf_price_vs_ema20"] = (ind["close"] - ind["ema20"]).values / ind["atr"].values
    htf["htf_price_vs_ema50"] = (ind["close"] - ind["ema50"]).values / ind["atr"].values
    htf["htf_price_vs_ema200"] = (ind["close"] - ind["ema200"]).values / ind["atr"].values
    htf["htf_trend_slope20_atr"] = ind["trend_slope20"].values / ind["atr"].values
    htf["htf_rsi14"] = ind["rsi14"].values
    htf["htf_atr_pctile"] = ind["atr_pctile"].values
    close, ema20, ema50 = ind["close"].values, ind["ema20"].values, ind["ema50"].values
    htf["htf_trend_dir"] = np.where(
        (close > ema20) & (ema20 > ema50), 1,
        np.where((close < ema20) & (ema20 < ema50), -1, 0)
    )
    # shift by one row: date D now holds the indicator values computed as of D-1's close
    htf = htf.shift(1)
    # collapse to one row per calendar date (defensive, in case of dupes)
    htf = htf[~htf.index.duplicated(keep="last")]
    return htf


# ---------------------------------------------------------------------------
# FVG detection on hourly bars, with session-boundary filtering + HTF join
# ---------------------------------------------------------------------------

def detect_fvgs_hourly(df, daily_ctx, symbol, cfg=CONFIG, calendar=None):
    tz = cfg["MARKET_TZ"]
    idx = df.index
    local_ts = idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz)
    local_dates = local_ts.date
    local_hours = local_ts.hour
    local_dow = local_ts.dayofweek

    rows = []
    n = len(df)
    recent_positions = []

    for i in range(1, n - 1):
        # session-boundary guard: all three candles must belong to the same trading day
        if not (local_dates[i - 1] == local_dates[i] == local_dates[i + 1]):
            continue

        c0, c1, c2 = df.iloc[i - 1], df.iloc[i], df.iloc[i + 1]
        atr = c2["atr"]
        if pd.isna(atr) or atr <= 0:
            continue

        direction = None
        if c2["low"] > c0["high"]:
            direction = 1
            gap_low, gap_high = c0["high"], c2["low"]
        elif c2["high"] < c0["low"]:
            direction = 0
            gap_low, gap_high = c2["high"], c0["low"]
        if direction is None:
            continue

        gap_size = gap_high - gap_low
        if gap_size < cfg["MIN_GAP_ATR_MULT"] * atr:
            continue

        trade_date = local_dates[i + 1]
        if trade_date not in daily_ctx.index:
            continue
        htf_row = daily_ctx.loc[trade_date]
        if isinstance(htf_row, pd.DataFrame):  # guard against any residual dupes
            htf_row = htf_row.iloc[-1]
        if htf_row.isna().any():
            continue  # no completed prior-day context yet (e.g. start of history)

        # clustering: FVGs formed within the trailing CLUSTER_WINDOW_BARS bars
        recent_positions = [p for p in recent_positions if (i + 1 - p) <= cfg["CLUSTER_WINDOW_BARS"]]
        recent_gap_count = len(recent_positions)
        recent_positions.append(i + 1)

        vol_ratio = c1["volume"] / c2["vol_avg20"] if c2["vol_avg20"] else np.nan
        dist_to_swing_high = (c2["swing_high"] - c2["close"]) / atr if not pd.isna(c2["swing_high"]) else np.nan
        dist_to_swing_low = (c2["close"] - c2["swing_low"]) / atr if not pd.isna(c2["swing_low"]) else np.nan

        htf_confluence = htf_row["htf_trend_dir"] * (1 if direction == 1 else -1)

        near_macro_event = 0
        if calendar is not None:
            event_name, _ = nearest_event(idx[i + 1], calendar)
            near_macro_event = 1 if event_name is not None else 0

        feat = {
            "symbol": symbol,
            "formation_time": idx[i + 1],
            "formation_idx": i + 1,
            "direction": direction,
            "gap_size_pct": gap_size / c2["close"],
            "gap_size_atr": gap_size / atr,
            "displacement_body_atr": abs(c1["close"] - c1["open"]) / atr,
            "displacement_range_atr": (c1["high"] - c1["low"]) / atr,
            "displacement_vol_ratio": vol_ratio,
            "price_vs_ema20": (c2["close"] - c2["ema20"]) / atr,
            "price_vs_ema50": (c2["close"] - c2["ema50"]) / atr,
            "price_vs_ema200": (c2["close"] - c2["ema200"]) / atr if not pd.isna(c2["ema200"]) else np.nan,
            "ema20_slope": c2["ema20_slope"],
            "trend_slope20_atr": c2["trend_slope20"] / atr if not pd.isna(c2["trend_slope20"]) else np.nan,
            "rsi14": c2["rsi14"],
            "atr_pctile": c2["atr_pctile"],
            "dist_to_swing_high_atr": dist_to_swing_high,
            "dist_to_swing_low_atr": dist_to_swing_low,
            "recent_gap_count": recent_gap_count,
            "hour_of_day": local_hours[i + 1],
            "hours_until_close": 16 - local_hours[i + 1],  # NYSE/Nasdaq close at 16:00 ET
            "day_of_week": local_dow[i + 1],
            "htf_price_vs_ema20": htf_row["htf_price_vs_ema20"],
            "htf_price_vs_ema50": htf_row["htf_price_vs_ema50"],
            "htf_price_vs_ema200": htf_row["htf_price_vs_ema200"],
            "htf_trend_slope20_atr": htf_row["htf_trend_slope20_atr"],
            "htf_rsi14": htf_row["htf_rsi14"],
            "htf_atr_pctile": htf_row["htf_atr_pctile"],
            "htf_trend_dir": htf_row["htf_trend_dir"],
            "htf_confluence": htf_confluence,
            "near_macro_event": near_macro_event,
            "_gap_low": gap_low,
            "_gap_high": gap_high,
            "_atr_at_formation": atr,
        }
        rows.append(feat)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_dataset(tickers, api_key, api_secret, cfg=CONFIG, verbose=True):
    client = StockHistoricalDataClient(api_key, api_secret)
    try:
        calendar = load_calendar(DEFAULT_CALENDAR_PATH)
        if verbose:
            print(f"Loaded macro calendar: {len(calendar)} events from {DEFAULT_CALENDAR_PATH}")
    except FileNotFoundError:
        calendar = None
        print(f"Warning: {DEFAULT_CALENDAR_PATH} not found — near_macro_event will be 0 for all rows.")
    all_rows = []
    for n, symbol in enumerate(tickers, 1):
        daily_raw = fetch_bars(client, symbol, TimeFrame.Day,
                                cfg["LOOKBACK_YEARS"] + cfg["DAILY_CONTEXT_BUFFER_YEARS"], cfg)
        time.sleep(cfg["REQUEST_SLEEP"])
        if daily_raw is None or len(daily_raw) < 250:
            continue
        daily_ctx = build_daily_context(daily_raw, cfg)

        hourly_raw = fetch_bars(client, symbol, TimeFrame.Hour, cfg["LOOKBACK_YEARS"], cfg)
        time.sleep(cfg["REQUEST_SLEEP"])
        if hourly_raw is None or len(hourly_raw) < 250:
            continue
        hourly_ind = add_indicators(hourly_raw, cfg)

        fvgs = detect_fvgs_hourly(hourly_ind, daily_ctx, symbol, cfg, calendar=calendar)
        if fvgs.empty:
            continue
        labeled = label_outcomes(hourly_ind, fvgs, cfg)
        all_rows.append(labeled)
        if verbose and n % 10 == 0:
            print(f"[{n}/{len(tickers)}] {symbol} — {len(labeled)} FVGs, running total {sum(len(x) for x in all_rows)}")

    dataset = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not dataset.empty:
        # stamp the exact labeling config used, so a saved model always knows
        # its own trade parameters regardless of the pipeline's current defaults
        dataset["target_r"] = cfg["TARGET_R_MULTIPLE"]
        dataset["max_hold_bars"] = cfg["MAX_HOLD_BARS"]
        dataset["stop_atr_mult"] = cfg["STOP_ATR_MULT"]
    if verbose:
        print(f"\nDone. {len(dataset)} total FVGs across {len(tickers)} tickers.")
        if not dataset.empty:
            print(dataset["outcome"].value_counts())
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build the hourly FVG dataset with daily confluence, from Alpaca bars.")
    parser.add_argument("--sector", default=CONFIG["SECTOR"],
                         help="GICS sector to filter the S&P 500 universe by (e.g. Technology, Health Care, Financials).")
    parser.add_argument("--tickers", nargs="*", default=None,
                         help="Explicit tickers — overrides --sector if given.")
    parser.add_argument("--years", type=int, default=CONFIG["LOOKBACK_YEARS"], help="Years of hourly history to pull.")
    parser.add_argument("--target-r", type=float, default=CONFIG["TARGET_R_MULTIPLE"], help="Target as a multiple of risk (R).")
    parser.add_argument("--max-hold", type=int, default=CONFIG["MAX_HOLD_BARS"], help="Max bars to hold before giving up on the trade.")
    parser.add_argument("--out", default="data/fvg_hourly_dataset.parquet", help="Output parquet path.")
    args = parser.parse_args()

    api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise SystemExit("Alpaca keys not found: set APCA_API_KEY_ID / APCA_API_SECRET_KEY in your .env file.")

    cfg = dict(CONFIG)
    cfg["LOOKBACK_YEARS"] = args.years
    cfg["SECTOR"] = args.sector
    cfg["TARGET_R_MULTIPLE"] = args.target_r
    cfg["MAX_HOLD_BARS"] = args.max_hold

    tickers = args.tickers if args.tickers else load_tickers_by_sector(args.sector)
    dataset = build_dataset(tickers, api_key, api_secret, cfg)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dataset.to_parquet(args.out, index=False)
    print(f"Wrote {len(dataset)} rows to {args.out}")


if __name__ == "__main__":
    main()
