"""
fvg_data_pipeline.py
=====================
Builds a large labeled dataset of Fair Value Gaps (FVGs) across a stock
universe using Alpaca historical bars, engineers pre-formation features
(no lookahead), and labels each FVG by its forward trade outcome.

Setup (VS Code, local venv):
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt
    cp .env.example .env   # fill in ALPACA_API_KEY / ALPACA_SECRET_KEY

Run:
    python fvg_data_pipeline.py --out data/fvg_dataset.parquet
    python fvg_data_pipeline.py --tickers AAPL MSFT NVDA --years 3 --out data/fvg_small.parquet

Output: one row per detected FVG, with engineered features + binary label,
written to a local parquet file.
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

load_dotenv()

CONFIG = {
    "TIMEFRAME": TimeFrame.Day,          # TimeFrame.Day / TimeFrame.Hour / etc.
    "LOOKBACK_YEARS": 6,
    "ATR_PERIOD": 14,
    "MAX_HOLD_BARS": 10,                 # bars allowed to reach target/stop after fill
    "TARGET_R_MULTIPLE": 1.5,
    "STOP_ATR_MULT": 1.0,                # fixed-ATR stop distance from entry (decoupled from gap width)
    "MIN_GAP_ATR_MULT": 0.10,            # discard microscopic/noise gaps
    "SWING_LOOKBACK": 10,                # bars for nearest swing high/low
    "VOL_AVG_WINDOW": 20,
    "ATR_PCTILE_WINDOW": 100,
    "MAX_WORKERS": 10,
    "REQUEST_SLEEP": 0.3,                # throttle between symbol requests
    "FEED": DataFeed.IEX,                # free-tier accounts can't query SIP; switch to DataFeed.SIP if you upgrade
}


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_bars(client, symbol, cfg=CONFIG):
    """Fetch daily OHLCV bars for one symbol from Alpaca."""
    end = datetime.utcnow()
    start = end - timedelta(days=365 * cfg["LOOKBACK_YEARS"])
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=cfg["TIMEFRAME"],
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
# Indicators (computed causally — only past/current bar data)
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
    df["ema20_slope"] = df["ema20"].pct_change(5)

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["vol_avg20"] = df["volume"].rolling(cfg["VOL_AVG_WINDOW"]).mean()

    # linear-regression slope of close over trailing 20 bars (trend strength)
    def _slope(y):
        x = np.arange(len(y))
        if np.any(np.isnan(y)):
            return np.nan
        return np.polyfit(x, y, 1)[0]
    df["trend_slope20"] = df["close"].rolling(20).apply(_slope, raw=True)

    # rolling swing high/low over SWING_LOOKBACK bars (excludes current bar)
    df["swing_high"] = df["high"].shift(1).rolling(cfg["SWING_LOOKBACK"]).max()
    df["swing_low"] = df["low"].shift(1).rolling(cfg["SWING_LOOKBACK"]).min()

    return df


# ---------------------------------------------------------------------------
# FVG detection + feature extraction (strictly causal at formation index i)
# ---------------------------------------------------------------------------

def detect_fvgs(df, symbol, cfg=CONFIG):
    """
    3-candle FVG: candle i-1, i (displacement candle), i+1.
    Bullish: low[i+1] > high[i-1]  -> gap = (high[i-1], low[i+1])
    Bearish: high[i+1] < low[i-1]  -> gap = (high[i+1], low[i-1])
    Formation is only confirmed once candle i+1 closes, so all features
    use data available up to and including bar i+1 — no future leakage.
    """
    rows = []
    n = len(df)
    recent_gap_count = 0
    gap_history = []  # timestamps of recent FVGs for clustering feature

    for i in range(1, n - 1):
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

        ts = df.index[i + 1]
        # clustering: FVGs formed in the trailing 10 bars
        gap_history = [t for t in gap_history if (ts - t).days <= 20]
        recent_gap_count = len(gap_history)
        gap_history.append(ts)

        vol_ratio = c1["volume"] / c2["vol_avg20"] if c2["vol_avg20"] else np.nan
        dist_to_swing_high = (c2["swing_high"] - c2["close"]) / atr if not pd.isna(c2["swing_high"]) else np.nan
        dist_to_swing_low = (c2["close"] - c2["swing_low"]) / atr if not pd.isna(c2["swing_low"]) else np.nan

        feat = {
            "symbol": symbol,
            "formation_time": ts,
            "formation_idx": i + 1,
            "direction": direction,  # 1 = bullish, 0 = bearish
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
            "recent_gap_count_20d": recent_gap_count,
            "day_of_week": ts.dayofweek,
            "_gap_low": gap_low,
            "_gap_high": gap_high,
            "_atr_at_formation": atr,
        }
        rows.append(feat)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outcome labeling — simulated forward on OHLC, causal (post-formation only)
# ---------------------------------------------------------------------------

def label_outcomes(df, fvgs, cfg=CONFIG):
    """
    For each FVG: wait for price to retrace into the gap (fill), then
    track whether target (TARGET_R_MULTIPLE * risk) or stop is hit first
    within MAX_HOLD_BARS. Unfilled or unresolved trades are labeled
    separately so they can be excluded from training.
    """
    labels, outcomes = [], []
    highs, lows = df["high"].values, df["low"].values
    n = len(df)

    for _, row in fvgs.iterrows():
        i = row["formation_idx"]
        gap_low, gap_high = row["_gap_low"], row["_gap_high"]
        atr = row["_atr_at_formation"]
        direction = row["direction"]
        entry = (gap_low + gap_high) / 2.0  # 50% gap-fill entry (standard convention)

        if direction == 1:
            stop = entry - cfg["STOP_ATR_MULT"] * atr
        else:
            stop = entry + cfg["STOP_ATR_MULT"] * atr
        risk = abs(entry - stop)
        if risk <= 0:
            labels.append(np.nan); outcomes.append("invalid"); continue
        target = entry + cfg["TARGET_R_MULTIPLE"] * risk if direction == 1 else entry - cfg["TARGET_R_MULTIPLE"] * risk

        filled = False
        outcome = "no_fill"
        label = np.nan
        end = min(i + 1 + cfg["MAX_HOLD_BARS"], n)
        for j in range(i + 1, end):
            if not filled:
                if direction == 1 and lows[j] <= gap_high:
                    filled = True
                elif direction == 0 and highs[j] >= gap_low:
                    filled = True
                if not filled:
                    continue
            # once filled, check stop/target same bar and subsequent bars
            if direction == 1:
                hit_stop = lows[j] <= stop
                hit_target = highs[j] >= target
            else:
                hit_stop = highs[j] >= stop
                hit_target = lows[j] <= target
            if hit_stop and hit_target:
                # ambiguous same-bar resolution — assume stop hit first (conservative)
                outcome, label = "stop", 0
                break
            elif hit_target:
                outcome, label = "target", 1
                break
            elif hit_stop:
                outcome, label = "stop", 0
                break
        else:
            if filled:
                outcome = "unresolved"

        labels.append(label)
        outcomes.append(outcome)

    fvgs = fvgs.copy()
    fvgs["label"] = labels
    fvgs["outcome"] = outcomes
    return fvgs.drop(columns=["_gap_low", "_gap_high", "_atr_at_formation"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_dataset(tickers, api_key, api_secret, cfg=CONFIG, verbose=True):
    client = StockHistoricalDataClient(api_key, api_secret)
    all_rows = []
    for n, symbol in enumerate(tickers, 1):
        bars = fetch_bars(client, symbol, cfg)
        time.sleep(cfg["REQUEST_SLEEP"])
        if bars is None or len(bars) < 250:
            continue
        bars = add_indicators(bars, cfg)
        fvgs = detect_fvgs(bars, symbol, cfg)
        if fvgs.empty:
            continue
        labeled = label_outcomes(bars, fvgs, cfg)
        all_rows.append(labeled)
        if verbose and n % 25 == 0:
            print(f"[{n}/{len(tickers)}] {symbol} — {len(labeled)} FVGs, running total {sum(len(x) for x in all_rows)}")

    dataset = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if verbose:
        print(f"\nDone. {len(dataset)} total FVGs across {len(tickers)} tickers.")
        if not dataset.empty:
            print(dataset["outcome"].value_counts())
    return dataset


def load_sp500_tickers():
    """Pulls current S&P 500 membership from the GitHub-hosted CSV used elsewhere in the screener."""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    try:
        return pd.read_csv(url)["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        print(f"S&P 500 list fetch failed ({e}); falling back to Wikipedia")
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


def main():
    parser = argparse.ArgumentParser(description="Build the FVG training dataset from Alpaca bars.")
    parser.add_argument("--tickers", nargs="*", default=None,
                         help="Space-separated tickers. Omit to use the full S&P 500.")
    parser.add_argument("--years", type=int, default=CONFIG["LOOKBACK_YEARS"],
                         help="Years of daily history to pull per symbol.")
    parser.add_argument("--out", default="data/fvg_dataset.parquet", help="Output parquet path.")
    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise SystemExit("Set ALPACA_API_KEY / ALPACA_SECRET_KEY in your .env file.")

    cfg = dict(CONFIG)
    cfg["LOOKBACK_YEARS"] = args.years

    tickers = args.tickers if args.tickers else load_sp500_tickers()
    dataset = build_dataset(tickers, api_key, api_secret, cfg)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dataset.to_parquet(args.out, index=False)
    print(f"Wrote {len(dataset)} rows to {args.out}")


if __name__ == "__main__":
    main()
