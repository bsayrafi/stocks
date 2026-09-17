"""
FVG Strategy v2 — Rejection Candle + First-Hour Session Filter + HTF Confluence
==================================================================================

The plain "touch the zone" entry (v1) failed at scale on SPY/MSFT/NVDA
(win rates 33-37% vs a 40% breakeven at RR=1.5, on 700+ trades each — a real,
statistically meaningful shortfall, not noise). This version adds three
confirmation layers, each targeting a specific weakness of the naive version:

1. REJECTION CANDLE: instead of entering on any bar that merely overlaps the
   FVG zone, require the bar to show actual rejection — a wick into the zone
   with the candle closing back out in the trade direction. This filters for
   bars where the zone visibly acted as support/resistance, not just bars
   that happened to trade through it.

2. SESSION-TIME FILTER: restrict entries to the first hour of the regular
   session (09:30-10:30 America/New_York by default). Day-trading edges are
   often concentrated in high-participation windows; averaging across a full
   session can dilute or hide a real intraday-specific effect.

3. HIGHER-TIMEFRAME CONFLUENCE: only take a lower-timeframe FVG if price is
   also inside an active, same-direction FVG zone on a higher timeframe
   (1-hour by default). This is meant to filter for zones with structural
   significance across timeframes, not ones that only exist on the noisiest
   chart.

None of these are guaranteed to fix anything — they're a specific, testable
hypothesis about why the naive version failed. Validate the same way as
before: parameter sweep on TRAIN only (sweep.py logic can be pointed at this
engine), then out-of-sample and multi-symbol checks before trusting it.

Usage
-----
    python fvg_strategy_v2.py --demo
    python fvg_strategy_v2.py --source yfinance --symbol SPY --interval 15m --period 60d
    python fvg_strategy_v2.py --source alpaca --symbol SPY --interval 15Min --start 2023-01-01 --end 2024-06-01
"""

import argparse
import sys
import numpy as np
import pandas as pd

from fvg_strategy import (
    FVG, Trade, atr, ema, detect_fvgs, summarize, fetch_data, make_demo_data,
)


# --------------------------------------------------------------------------
# 1. Rejection candle
# --------------------------------------------------------------------------

def is_rejection_candle(bar, direction: str, min_wick_ratio: float = 0.3) -> bool:
    """
    True if this candle shows a rejection wick in the trade direction:
    for a bullish setup, a lower wick that's a meaningful fraction of the
    bar's range, AND the candle closes green (close > open). Mirror for bearish.
    """
    rng = bar["high"] - bar["low"]
    if rng <= 0:
        return False

    if direction == "bull":
        lower_wick = min(bar["open"], bar["close"]) - bar["low"]
        return (lower_wick / rng) >= min_wick_ratio and bar["close"] > bar["open"]
    else:
        upper_wick = bar["high"] - max(bar["open"], bar["close"])
        return (upper_wick / rng) >= min_wick_ratio and bar["close"] < bar["open"]


# --------------------------------------------------------------------------
# 2. Session-time filter
# --------------------------------------------------------------------------

def add_session_time(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """
    Adds a 'session_time' column (datetime.time, in the given exchange
    timezone) to the dataframe. If timestamps are tz-aware, converts them.
    If naive, assumes they are already in exchange-local time (true for
    yfinance intraday data and for the synthetic demo data).
    """
    df = df.copy()
    ts = df["timestamp"]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(tz)
    df["session_time"] = ts.dt.time
    return df


def in_session_window(t, start_str: str = "09:30", window_minutes: int = 60) -> bool:
    start_h, start_m = (int(x) for x in start_str.split(":"))
    start_minutes = start_h * 60 + start_m
    t_minutes = t.hour * 60 + t.minute
    return start_minutes <= t_minutes < start_minutes + window_minutes


# --------------------------------------------------------------------------
# 3. Higher-timeframe confluence
# --------------------------------------------------------------------------

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.set_index("timestamp")
    out = d.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return out.reset_index()


def compute_htf_active_zones(df: pd.DataFrame, htf_rule: str,
                              min_gap_atr_mult: float = 0.2,
                              max_bars_active: int = 20) -> list:
    """
    Resamples to the higher timeframe, detects FVGs there, and simulates
    forward to find each one's active window: [start_ts, end_ts) during
    which it is unfilled. Returns a list of dicts:
        {direction, top, bottom, start_ts, end_ts}
    end_ts is None if the zone is still active through the end of the data.
    """
    htf = resample_ohlc(df, htf_rule)
    if len(htf) < 5:
        return []  # not enough higher-timeframe bars to do anything useful

    htf["atr"] = atr(htf)
    fvgs = detect_fvgs(htf, min_gap_atr_mult=min_gap_atr_mult)

    zones = []
    for f in fvgs:
        start_ts = htf["timestamp"].iloc[f.idx]
        end_ts = None
        for j in range(f.idx + 1, min(f.idx + 1 + max_bars_active, len(htf))):
            bar = htf.iloc[j]
            fully_filled = (bar["low"] <= f.bottom) if f.direction == "bull" else (bar["high"] >= f.top)
            if fully_filled:
                end_ts = htf["timestamp"].iloc[j]
                break
        if end_ts is None and f.idx + 1 + max_bars_active < len(htf):
            end_ts = htf["timestamp"].iloc[f.idx + max_bars_active]
        zones.append({
            "direction": f.direction, "top": f.top, "bottom": f.bottom,
            "start_ts": start_ts, "end_ts": end_ts,
        })
    return zones


def htf_confluence_ok(ts, price: float, direction: str, htf_zones: list) -> bool:
    for z in htf_zones:
        if z["direction"] != direction:
            continue
        if ts <= z["start_ts"]:
            continue
        if z["end_ts"] is not None and ts > z["end_ts"]:
            continue
        if z["bottom"] <= price <= z["top"]:
            return True
    return False


# --------------------------------------------------------------------------
# Backtest engine v2
# --------------------------------------------------------------------------

def backtest_v2(
    df: pd.DataFrame,
    reward_risk: float = 1.5,
    max_bars_active: int = 40,
    use_trend_filter: bool = False,
    trend_ema_period: int = 50,
    min_gap_atr_mult: float = 0.2,
    max_concurrent_trades: int = 1,
    use_rejection: bool = True,
    min_wick_ratio: float = 0.3,
    use_session_filter: bool = True,
    session_start: str = "09:30",
    session_minutes: int = 60,
    tz: str = "America/New_York",
    use_htf_confluence: bool = True,
    htf_interval: str = "1h",
    htf_min_gap_atr: float = 0.2,
    htf_max_bars_active: int = 20,
    min_confluence: int = None,
    friction_R: float = 0.0,
):
    df = df.copy().reset_index(drop=True)
    df["atr"] = atr(df)
    df["ema"] = ema(df["close"], trend_ema_period)

    if use_session_filter:
        df = add_session_time(df, tz=tz)

    htf_zones = []
    if use_htf_confluence:
        htf_zones = compute_htf_active_zones(
            df, htf_interval, min_gap_atr_mult=htf_min_gap_atr, max_bars_active=htf_max_bars_active
        )

    fvgs = detect_fvgs(df, min_gap_atr_mult=min_gap_atr_mult)

    trades = []
    open_trades = []
    active_fvgs = []

    # Funnel counters: how many zone-touch candidates survive each filter stage,
    # applied independently (not cumulatively) so you can see which single
    # filter is the bottleneck, not just the combined effect.
    funnel = {
        "zone_touch_candidates": 0,
        "passed_trend": 0,
        "passed_rejection": 0,
        "passed_session": 0,
        "passed_htf_confluence": 0,
        "passed_all": 0,
    }

    fvg_by_start = {}
    for f in fvgs:
        fvg_by_start.setdefault(f.idx, []).append(f)

    for i in range(len(df)):
        bar = df.iloc[i]

        if i in fvg_by_start:
            active_fvgs.extend(fvg_by_start[i])

        # --- Manage open trades ---
        still_open = []
        for tr in open_trades:
            hit_stop = (bar["low"] <= tr.stop_price) if tr.direction == "bull" else (bar["high"] >= tr.stop_price)
            hit_target = (bar["high"] >= tr.target_price) if tr.direction == "bull" else (bar["low"] <= tr.target_price)

            if hit_stop:
                tr.exit_idx, tr.exit_price, tr.result = i, tr.stop_price, "loss"
                tr.r_multiple = -1.0 - friction_R
                trades.append(tr)
            elif hit_target:
                tr.exit_idx, tr.exit_price, tr.result = i, tr.target_price, "win"
                tr.r_multiple = reward_risk - friction_R
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
                continue  # expired, drop

            fully_filled = (bar["low"] <= f.bottom) if f.direction == "bull" else (bar["high"] >= f.top)
            price_in_zone = (bar["low"] <= f.top and bar["high"] >= f.bottom)

            if len(open_trades) < max_concurrent_trades and price_in_zone and not f.filled:
                funnel["zone_touch_candidates"] += 1
                checks_passed = True

                trend_ok = True
                if use_trend_filter and not np.isnan(bar["ema"]):
                    trend_ok = (bar["close"] > bar["ema"]) if f.direction == "bull" else (bar["close"] < bar["ema"])
                if trend_ok:
                    funnel["passed_trend"] += 1
                checks_passed = checks_passed and trend_ok

                rejection_ok = is_rejection_candle(bar, f.direction, min_wick_ratio) if use_rejection else True
                if rejection_ok:
                    funnel["passed_rejection"] += 1

                session_ok = in_session_window(bar["session_time"], session_start, session_minutes) if use_session_filter else True
                if session_ok:
                    funnel["passed_session"] += 1

                htf_ok = htf_confluence_ok(bar["timestamp"], bar["close"], f.direction, htf_zones) if use_htf_confluence else True
                if htf_ok:
                    funnel["passed_htf_confluence"] += 1

                # Confluence scoring: instead of requiring ALL enabled filters (strict AND),
                # require at least `min_confluence` of the enabled ones to agree. This matters
                # because the funnel can show each filter passing a healthy % individually while
                # their intersection is far smaller than independence would predict — i.e. they're
                # flagging different subsets of candidates, not compounding one coherent signal.
                active_filter_results = []
                if use_rejection:
                    active_filter_results.append(rejection_ok)
                if use_session_filter:
                    active_filter_results.append(session_ok)
                if use_htf_confluence:
                    active_filter_results.append(htf_ok)

                score = sum(active_filter_results)
                required = min_confluence if min_confluence is not None else len(active_filter_results)
                confluence_ok = score >= required

                checks_passed = checks_passed and confluence_ok
                if checks_passed:
                    funnel["passed_all"] += 1
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
                        f.filled = True

            if fully_filled:
                continue
            still_active.append(f)

        active_fvgs = still_active

    for tr in open_trades:
        tr.exit_idx = len(df) - 1
        tr.exit_price = df.iloc[-1]["close"]
        pnl = (tr.exit_price - tr.entry_price) if tr.direction == "bull" else (tr.entry_price - tr.exit_price)
        risk = abs(tr.entry_price - tr.stop_price)
        tr.r_multiple = (pnl / risk if risk else 0) - friction_R
        tr.result = "open"
        trades.append(tr)

    return trades, fvgs, htf_zones, funnel


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FVG v2: rejection candle + session filter + HTF confluence")
    parser.add_argument("--csv", type=str)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--source", choices=["yfinance", "alpaca"])
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--period", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)

    parser.add_argument("--reward-risk", type=float, default=1.5)
    parser.add_argument("--min-gap-atr", type=float, default=0.2)
    parser.add_argument("--max-bars-active", type=int, default=40)
    parser.add_argument("--use-trend-filter", action="store_true")

    parser.add_argument("--no-rejection", action="store_true", help="Disable rejection-candle requirement")
    parser.add_argument("--min-wick-ratio", type=float, default=0.3)

    parser.add_argument("--no-session-filter", action="store_true", help="Disable session-time filter")
    parser.add_argument("--session-start", type=str, default="09:30")
    parser.add_argument("--session-minutes", type=int, default=60)
    parser.add_argument("--tz", type=str, default="America/New_York")

    parser.add_argument("--no-htf-confluence", action="store_true", help="Disable higher-timeframe confluence")
    parser.add_argument("--htf-interval", type=str, default="1h")
    parser.add_argument("--htf-min-gap-atr", type=float, default=0.2)
    parser.add_argument("--htf-max-bars-active", type=int, default=20)

    parser.add_argument("--min-confluence", type=int, default=None,
                         help="Require at least N of {rejection, session, htf} to pass, instead of "
                              "requiring ALL enabled ones (strict AND). Default: require all (old behavior).")

    parser.add_argument("--friction-R", type=float, default=0.0)
    args = parser.parse_args()

    if args.source:
        if not args.symbol:
            sys.exit("--symbol is required when using --source")
        print(f"Fetching {args.symbol} from {args.source} (interval={args.interval})...\n")
        if args.source == "yfinance":
            df = fetch_data(args.source, args.symbol, interval=args.interval,
                             period=args.period, start=args.start, end=args.end)
        else:
            df = fetch_data(args.source, args.symbol, timeframe=args.interval,
                             start=args.start, end=args.end)
    elif args.csv:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    else:
        print("No --csv/--source provided, running on synthetic demo data...\n")
        df = make_demo_data()

    trades, fvgs, htf_zones, funnel = backtest_v2(
        df,
        reward_risk=args.reward_risk,
        max_bars_active=args.max_bars_active,
        use_trend_filter=args.use_trend_filter,
        min_gap_atr_mult=args.min_gap_atr,
        use_rejection=not args.no_rejection,
        min_wick_ratio=args.min_wick_ratio,
        use_session_filter=not args.no_session_filter,
        session_start=args.session_start,
        session_minutes=args.session_minutes,
        tz=args.tz,
        use_htf_confluence=not args.no_htf_confluence,
        htf_interval=args.htf_interval,
        htf_min_gap_atr=args.htf_min_gap_atr,
        htf_max_bars_active=args.htf_max_bars_active,
        min_confluence=args.min_confluence,
        friction_R=args.friction_R,
    )

    print(f"LTF FVGs detected: {len(fvgs)}")
    print(f"HTF ({args.htf_interval}) FVG zones detected: {len(htf_zones)}")
    print(f"Filters active: rejection={not args.no_rejection}, "
          f"session_filter={not args.no_session_filter} ({args.session_start}, {args.session_minutes}min), "
          f"htf_confluence={not args.no_htf_confluence} ({args.htf_interval}), "
          f"trend_filter={args.use_trend_filter}")
    n_active = sum([not args.no_rejection, not args.no_session_filter, not args.no_htf_confluence])
    req = args.min_confluence if args.min_confluence is not None else n_active
    print(f"Confluence mode: require {req} of {n_active} enabled filters to pass "
          f"({'strict AND' if req == n_active else 'partial/OR-style'})")

    print("\n--- Filter funnel (each row = % of zone-touch candidates that pass THAT filter alone) ---")
    total = funnel["zone_touch_candidates"]
    print(f"{'zone_touch_candidates':28s}: {total}")
    for k in ["passed_trend", "passed_rejection", "passed_session", "passed_htf_confluence"]:
        pct = (funnel[k] / total * 100) if total else 0
        print(f"{k:28s}: {funnel[k]:5d}  ({pct:.1f}% individually)")
    print(f"{'passed_all (final trades)':28s}: {funnel['passed_all']}")

    stats = summarize(trades)
    print("\n--- Backtest results ---")
    for k, v in stats.items():
        print(f"{k:16s}: {v}")


if __name__ == "__main__":
    main()
