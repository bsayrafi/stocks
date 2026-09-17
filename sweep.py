"""
FVG Strategy Parameter Sweep — with train/test split
======================================================

Why this exists
----------------
Manually trying parameter combinations and keeping whichever one looks best
on a single dataset is a fast way to "discover" an edge that's actually just
noise (overfitting / data-snooping bias). This script guards against that by:

  1. Splitting your data chronologically into a TRAIN period and a TEST period
     (default 60/40 split — train is the earlier portion, test is the later,
     never shuffled, since shuffling would leak future information backwards).
  2. Grid-searching all parameter combinations ONLY on the train period.
  3. Taking the single best combination (by total_R, subject to a minimum
     trade count so we don't pick a fluke 3-trade sample) and running it
     ONCE on the untouched test period.
  4. Reporting both side by side. A real edge should survive the trip to
     test data reasonably intact. A large drop-off (or sign flip) means the
     "edge" found in training was likely overfitting.

This is still a single train/test split, not a full walk-forward analysis —
treat it as a first filter, not final proof. If it survives this, the next
step is testing on other symbols/periods entirely.

Usage
-----
    python sweep.py --source yfinance --symbol SPY --interval 15m --period 60d
    python sweep.py --csv your_data.csv --train-frac 0.6

Grid ranges are configurable via comma-separated lists:
    python sweep.py --demo --rr-grid 1.5,2,2.5 --min-gap-grid 0.1,0.15,0.2,0.3 \\
                     --max-bars-grid 20,40,60 --trend-grid on,off
"""

import argparse
import itertools
import sys
import pandas as pd

from fvg_strategy import (
    backtest, summarize, fetch_data, make_demo_data,
)


def parse_float_list(s: str) -> list:
    return [float(x) for x in s.split(",")]


def parse_int_list(s: str) -> list:
    return [int(x) for x in s.split(",")]


def run_grid(df: pd.DataFrame, rr_grid, min_gap_grid, max_bars_grid, trend_grid,
             max_concurrent_trades: int = 1):
    """Run backtest for every combination in the grid. Returns a list of dict rows."""
    results = []
    trend_options = []
    if "on" in trend_grid:
        trend_options.append(True)
    if "off" in trend_grid:
        trend_options.append(False)

    combos = list(itertools.product(rr_grid, min_gap_grid, max_bars_grid, trend_options))
    for rr, min_gap, max_bars, use_trend in combos:
        trades, _ = backtest(
            df,
            reward_risk=rr,
            max_bars_active=max_bars,
            use_trend_filter=use_trend,
            min_gap_atr_mult=min_gap,
            max_concurrent_trades=max_concurrent_trades,
        )
        stats = summarize(trades)
        stats.update({
            "reward_risk": rr,
            "min_gap_atr": min_gap,
            "max_bars_active": max_bars,
            "trend_filter": use_trend,
        })
        results.append(stats)
    return results


def print_table(rows: list, sort_key: str = "total_R", top_n: int = 15):
    valid = [r for r in rows if r.get("num_trades", 0) > 0]
    valid.sort(key=lambda r: r.get(sort_key, -999), reverse=True)

    cols = ["reward_risk", "min_gap_atr", "max_bars_active", "trend_filter",
            "num_trades", "win_rate", "profit_factor", "total_R", "max_drawdown_R"]
    header = " | ".join(f"{c:>15s}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in valid[:top_n]:
        print(" | ".join(f"{str(r.get(c, '')):>15s}" for c in cols))
    return valid


def main():
    parser = argparse.ArgumentParser(description="FVG strategy parameter sweep with train/test split")
    parser.add_argument("--csv", type=str)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--source", choices=["yfinance", "alpaca"])
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--period", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)

    parser.add_argument("--train-frac", type=float, default=0.6,
                         help="Fraction of (chronologically ordered) data used for training. Default 0.6")
    parser.add_argument("--min-trades", type=int, default=20,
                         help="Minimum trades in TRAIN required for a param combo to be considered. Default 20")

    parser.add_argument("--rr-grid", type=str, default="1.5,2.0,2.5,3.0")
    parser.add_argument("--min-gap-grid", type=str, default="0.1,0.15,0.2,0.3,0.5")
    parser.add_argument("--max-bars-grid", type=str, default="20,40,60")
    parser.add_argument("--trend-grid", type=str, default="on,off",
                         help="Comma list from {on, off}")
    parser.add_argument("--sort-by", type=str, default="total_R",
                         choices=["total_R", "profit_factor", "avg_R"])
    args = parser.parse_args()

    # --- Load data (same logic as fvg_strategy.py) ---
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

    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Chronological train/test split (no shuffling — this is time series) ---
    split_idx = int(len(df) * args.train_frac)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Total bars: {len(df)}  |  Train: {len(train_df)} bars "
          f"({train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]})")
    print(f"Test:  {len(test_df)} bars "
          f"({test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]})\n")

    # --- Grid search on TRAIN only ---
    rr_grid = parse_float_list(args.rr_grid)
    min_gap_grid = parse_float_list(args.min_gap_grid)
    max_bars_grid = parse_int_list(args.max_bars_grid)
    trend_grid = args.trend_grid.split(",")

    print(f"Running grid search on TRAIN data "
          f"({len(rr_grid) * len(min_gap_grid) * len(max_bars_grid) * len(trend_grid)} combinations)...\n")
    train_results = run_grid(train_df, rr_grid, min_gap_grid, max_bars_grid, trend_grid)

    qualifying = [r for r in train_results if r.get("num_trades", 0) >= args.min_trades]
    if not qualifying:
        print(f"No parameter combination reached --min-trades {args.min_trades} on the train set.")
        print("Try a longer period, a lower --min-trades, or wider grid ranges.")
        return

    print("=== TOP TRAIN RESULTS (ranked by {}) ===".format(args.sort_by))
    ranked = print_table(qualifying, sort_key=args.sort_by)
    best = ranked[0]

    print(f"\nBest combo on TRAIN: reward_risk={best['reward_risk']}, "
          f"min_gap_atr={best['min_gap_atr']}, max_bars_active={best['max_bars_active']}, "
          f"trend_filter={best['trend_filter']}")

    # --- Validate best combo on TEST (untouched, single run) ---
    test_trades, _ = backtest(
        test_df,
        reward_risk=best["reward_risk"],
        max_bars_active=best["max_bars_active"],
        use_trend_filter=best["trend_filter"],
        min_gap_atr_mult=best["min_gap_atr"],
    )
    test_stats = summarize(test_trades)

    print("\n=== OUT-OF-SAMPLE (TEST) RESULT for the best TRAIN combo ===")
    for k in ["num_trades", "win_rate", "avg_R", "total_R", "avg_win_R",
              "avg_loss_R", "profit_factor", "max_drawdown_R"]:
        train_val = str(best.get(k, "n/a"))
        test_val = str(test_stats.get(k, "n/a"))
        print(f"{k:16s}: train={train_val:<10}  test={test_val}")

    print("\n--- Interpretation guide ---")
    if test_stats.get("num_trades", 0) < max(5, args.min_trades // 4):
        print("Very few test trades — not enough out-of-sample data to draw a conclusion either way.")
    elif test_stats.get("total_R", 0) > 0 and test_stats.get("profit_factor", 0) >= 1.0:
        print("Edge held up out-of-sample — promising, but still just one split/one symbol. "
              "Test on other symbols and periods before trusting it.")
    else:
        print("Edge did NOT hold up out-of-sample — this is the classic signature of overfitting "
              "the train period rather than finding a real, generalizable pattern.")


if __name__ == "__main__":
    main()
