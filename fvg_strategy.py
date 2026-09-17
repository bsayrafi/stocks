"""
Fair Value Gap (FVG) Day Trading Strategy — Backtester
========================================================

Strategy logic
--------------
1. Detect 3-candle Fair Value Gaps (FVGs):
     - Bullish FVG: low of candle[i]   > high of candle[i-2]   (gap left by the impulse candle i-1)
     - Bearish FVG: high of candle[i]  < low  of candle[i-2]

2. Track each FVG as an active zone until price fully fills it (or it expires
   after `max_bars_active` bars).

3. Entry: when price retraces INTO an active, unfilled FVG zone, enter in the
   direction of the gap (i.e. treat the FVG as support in an uptrend / resistance
   in a downtrend).

4. Risk management:
     - Stop loss: just beyond the far edge of the FVG.
     - Take profit: fixed reward multiple of risk (default 2R).

5. Filters:
     - Minimum gap size (as a multiple of ATR) to avoid trading noise.
     - Optional trend filter using an EMA (only take bullish FVGs above the EMA,
       bearish FVGs below it).

This is a long/short intraday strategy meant to run on lower-timeframe bars
(e.g. 1-5 min). It's a template for you to adapt and validate — it is NOT
financial advice, and results here are for educational/research purposes only.

Usage
-----
    python fvg_strategy.py --csv your_data.csv
    python fvg_strategy.py --demo      # runs on generated synthetic data

CSV format expected: columns [timestamp, open, high, low, close, volume]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class FVG:
    idx: int            # bar index where the FVG was formed (candle i)
    direction: str       # 'bull' or 'bear'
    top: float
    bottom: float
    filled: bool = False
    expired: bool = False


@dataclass
class Trade:
    entry_idx: int
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    exit_idx: int = None
    exit_price: float = None
    result: str = None   # 'win', 'loss', 'open'
    r_multiple: float = None


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ema(series: pd.Series, period: int = 50) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# --------------------------------------------------------------------------
# FVG detection
# --------------------------------------------------------------------------

def detect_fvgs(df: pd.DataFrame, min_gap_atr_mult: float = 0.1) -> list:
    """Scan the dataframe for 3-candle fair value gaps."""
    fvgs = []
    a = df["atr"].values
    high, low = df["high"].values, df["low"].values

    for i in range(2, len(df)):
        gap_atr = a[i] if not np.isnan(a[i]) else 0
        min_gap = min_gap_atr_mult * gap_atr

        # Bullish FVG: candle i's low above candle i-2's high
        if low[i] - high[i - 2] > min_gap:
            fvgs.append(FVG(idx=i, direction="bull", top=low[i], bottom=high[i - 2]))

        # Bearish FVG: candle i's high below candle i-2's low
        elif low[i - 2] - high[i] > min_gap:
            fvgs.append(FVG(idx=i, direction="bear", top=low[i - 2], bottom=high[i]))

    return fvgs


# --------------------------------------------------------------------------
# Backtest engine
# --------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    reward_risk: float = 2.0,
    max_bars_active: int = 40,
    use_trend_filter: bool = True,
    trend_ema_period: int = 50,
    min_gap_atr_mult: float = 0.1,
    max_concurrent_trades: int = 1,
):
    df = df.copy().reset_index(drop=True)
    df["atr"] = atr(df)
    df["ema"] = ema(df["close"], trend_ema_period)

    fvgs = detect_fvgs(df, min_gap_atr_mult=min_gap_atr_mult)

    trades = []
    open_trades = []
    active_fvgs = []  # fvgs not yet filled/expired/traded

    fvg_by_start = {}
    for f in fvgs:
        fvg_by_start.setdefault(f.idx, []).append(f)

    for i in range(len(df)):
        bar = df.iloc[i]

        # Register new FVGs formed at this bar
        if i in fvg_by_start:
            active_fvgs.extend(fvg_by_start[i])

        # --- Manage open trades: check stop / target hit ---
        still_open = []
        for tr in open_trades:
            hit_stop = (bar["low"] <= tr.stop_price) if tr.direction == "bull" else (bar["high"] >= tr.stop_price)
            hit_target = (bar["high"] >= tr.target_price) if tr.direction == "bull" else (bar["low"] <= tr.target_price)

            if hit_stop and hit_target:
                # Conservative: assume stop hit first if both in same bar
                tr.exit_idx, tr.exit_price, tr.result = i, tr.stop_price, "loss"
                tr.r_multiple = -1.0
                trades.append(tr)
            elif hit_stop:
                tr.exit_idx, tr.exit_price, tr.result = i, tr.stop_price, "loss"
                tr.r_multiple = -1.0
                trades.append(tr)
            elif hit_target:
                tr.exit_idx, tr.exit_price, tr.result = i, tr.target_price, "win"
                tr.r_multiple = reward_risk
                trades.append(tr)
            else:
                still_open.append(tr)
        open_trades = still_open

        # --- Check active FVGs for fill / expiry / entry trigger ---
        still_active = []
        for f in active_fvgs:
            if i <= f.idx:
                still_active.append(f)
                continue

            bars_since = i - f.idx
            if bars_since > max_bars_active:
                f.expired = True
                continue  # drop it

            fully_filled = (bar["low"] <= f.bottom) if f.direction == "bull" else (bar["high"] >= f.top)
            price_in_zone = (bar["low"] <= f.top and bar["high"] >= f.bottom)

            if len(open_trades) < max_concurrent_trades and price_in_zone and not f.filled:
                trend_ok = True
                if use_trend_filter and not np.isnan(bar["ema"]):
                    trend_ok = (bar["close"] > bar["ema"]) if f.direction == "bull" else (bar["close"] < bar["ema"])

                if trend_ok:
                    entry_price = bar["close"]
                    if f.direction == "bull":
                        stop_price = f.bottom
                        risk = entry_price - stop_price
                        target_price = entry_price + reward_risk * risk
                    else:
                        stop_price = f.top
                        risk = stop_price - entry_price
                        target_price = entry_price - reward_risk * risk

                    if risk > 0:
                        trade = Trade(
                            entry_idx=i, direction=f.direction, entry_price=entry_price,
                            stop_price=stop_price, target_price=target_price,
                        )
                        open_trades.append(trade)
                        f.filled = True  # one trade per FVG

            if fully_filled:
                continue  # drop from active list (fully filled, no further trades)
            still_active.append(f)

        active_fvgs = still_active

    # Close any trades still open at the end (mark to last close)
    for tr in open_trades:
        tr.exit_idx = len(df) - 1
        tr.exit_price = df.iloc[-1]["close"]
        pnl = (tr.exit_price - tr.entry_price) if tr.direction == "bull" else (tr.entry_price - tr.exit_price)
        risk = abs(tr.entry_price - tr.stop_price)
        tr.r_multiple = pnl / risk if risk else 0
        tr.result = "open"
        trades.append(tr)

    return trades, fvgs


# --------------------------------------------------------------------------
# Performance reporting
# --------------------------------------------------------------------------

def summarize(trades: list) -> dict:
    if not trades:
        return {"num_trades": 0}

    r_values = np.array([t.r_multiple for t in trades])
    wins = r_values[r_values > 0]
    losses = r_values[r_values <= 0]

    return {
        "num_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "avg_R": round(r_values.mean(), 3),
        "total_R": round(r_values.sum(), 2),
        "avg_win_R": round(wins.mean(), 3) if len(wins) else 0,
        "avg_loss_R": round(losses.mean(), 3) if len(losses) else 0,
        "profit_factor": round(wins.sum() / abs(losses.sum()), 2) if losses.sum() != 0 else float("inf"),
        "max_drawdown_R": round(_max_drawdown(r_values), 2),
    }


def _max_drawdown(r_values: np.ndarray) -> float:
    equity = np.cumsum(r_values)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    return drawdown.min() if len(drawdown) else 0.0


# --------------------------------------------------------------------------
# Real data sources
# --------------------------------------------------------------------------

def fetch_yfinance(symbol: str, interval: str = "5m", period: str = None,
                    start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Yahoo Finance (free, no API key).

    Intraday limits enforced by Yahoo: 1m data -> max 7 days back,
    other sub-daily intervals (2m/5m/15m/30m/60m/90m) -> max ~60 days back.
    For anything longer, use daily bars ('1d') or switch to Alpaca.

    pip install yfinance
    """
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. Run: pip install yfinance")

    kwargs = {"interval": interval}
    if period:
        kwargs["period"] = period          # e.g. "5d", "60d", "1mo"
    elif start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        # sensible default given Yahoo's intraday history limits
        kwargs["period"] = "7d" if interval == "1m" else "60d"

    raw = yf.download(symbol, progress=False, auto_adjust=False, **kwargs)
    if raw.empty:
        sys.exit(f"No data returned for {symbol} (interval={interval}). "
                  f"Check the symbol and that the interval/period is within Yahoo's limits.")

    # yfinance can return a MultiIndex on columns for single-symbol downloads
    # depending on version; flatten it if so.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
    df = raw.rename(columns={
        ts_col: "timestamp", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def fetch_alpaca(symbol: str, timeframe: str = "5Min", start: str = None,
                  end: str = None, limit: int = 10000) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca's Market Data API.

    Requires a free Alpaca account and API keys set as environment variables:
        export APCA_API_KEY_ID=your_key_id
        export APCA_API_SECRET_KEY=your_secret_key

    pip install alpaca-py

    timeframe examples: "1Min", "5Min", "15Min", "1Hour", "1Day"
    start/end: ISO date strings, e.g. "2024-01-01"
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        sys.exit("alpaca-py is not installed. Run: pip install alpaca-py")

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables first.")

    client = StockHistoricalDataClient(key, secret)

    unit_map = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour, "Day": TimeFrameUnit.Day}
    amount = int("".join(filter(str.isdigit, timeframe)) or 1)
    unit_str = "".join(filter(str.isalpha, timeframe))
    unit = unit_map.get(unit_str, TimeFrameUnit.Minute)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount, unit),
        start=start,
        end=end,
        limit=limit,
    )
    bars = client.get_stock_bars(request).df

    if bars.empty:
        sys.exit(f"No data returned for {symbol}. Check symbol, date range, and API keys.")

    bars = bars.reset_index()
    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol]

    df = bars.rename(columns={"timestamp": "timestamp"})[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]
    return df


def fetch_data(source: str, symbol: str, **kwargs) -> pd.DataFrame:
    if source == "yfinance":
        return fetch_yfinance(symbol, **kwargs)
    elif source == "alpaca":
        return fetch_alpaca(symbol, **kwargs)
    else:
        raise ValueError(f"Unknown source: {source}")


# --------------------------------------------------------------------------
# Synthetic demo data (random walk with a mild trend, for smoke-testing)
# --------------------------------------------------------------------------

def make_demo_data(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2024-01-01 09:30", periods=n_bars, freq="5min")

    drift = 0.00005
    vol = 0.0015
    returns = rng.normal(drift, vol, n_bars)
    close = 100 * np.exp(np.cumsum(returns))

    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, n_bars))
    volume = rng.integers(1000, 10000, n_bars)

    return pd.DataFrame({
        "timestamp": dt, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FVG day trading strategy backtester")
    parser.add_argument("--csv", type=str, help="Path to OHLCV CSV file")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic demo data")
    parser.add_argument("--source", choices=["yfinance", "alpaca"], help="Fetch real data from this source")
    parser.add_argument("--symbol", type=str, help="Ticker symbol, e.g. AAPL, SPY")
    parser.add_argument("--interval", type=str, default="5m",
                         help="Bar size. yfinance: 1m/5m/15m/30m/60m/1d. alpaca: 1Min/5Min/15Min/1Hour/1Day")
    parser.add_argument("--period", type=str, default=None, help="yfinance lookback, e.g. 5d, 60d, 1mo")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--save-csv", type=str, default=None, help="Save fetched data to this CSV path")
    parser.add_argument("--reward-risk", type=float, default=2.0)
    parser.add_argument("--max-bars-active", type=int, default=40)
    parser.add_argument("--no-trend-filter", action="store_true")
    parser.add_argument("--trend-ema", type=int, default=50)
    parser.add_argument("--min-gap-atr", type=float, default=0.1)
    args = parser.parse_args()

    if args.source:
        if not args.symbol:
            sys.exit("--symbol is required when using --source")
        print(f"Fetching {args.symbol} from {args.source} (interval={args.interval})...\n")
        if args.source == "yfinance":
            df = fetch_data(args.source, args.symbol, interval=args.interval,
                             period=args.period, start=args.start, end=args.end)
        else:  # alpaca
            df = fetch_data(args.source, args.symbol, timeframe=args.interval,
                             start=args.start, end=args.end)
        if args.save_csv:
            df.to_csv(args.save_csv, index=False)
            print(f"Saved fetched data to {args.save_csv}\n")
    elif args.csv:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    else:
        print("No --csv/--source provided, running on synthetic demo data...\n")
        df = make_demo_data()

    trades, fvgs = backtest(
        df,
        reward_risk=args.reward_risk,
        max_bars_active=args.max_bars_active,
        use_trend_filter=not args.no_trend_filter,
        trend_ema_period=args.trend_ema,
        min_gap_atr_mult=args.min_gap_atr,
    )

    print(f"FVGs detected: {len(fvgs)}")
    stats = summarize(trades)
    print("\n--- Backtest results ---")
    for k, v in stats.items():
        print(f"{k:16s}: {v}")


if __name__ == "__main__":
    main()
