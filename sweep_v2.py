"""
FVG Strategy v2 — Parameter Sweep with Train/Test Split
===========================================================

Same discipline as sweep.py (chronological train/test split, grid search
ONLY on train, single untouched validation run on test) but pointed at the
v2 engine (fvg_strategy_v2.backtest_v2), so the new knobs — min_wick_ratio,
session_minutes, htf_interval, min_confluence — get the same overfitting
protection as reward_risk/min_gap_atr did in v1.

Usage
-----
    python sweep_v2.py --source yfinance --symbol SPY --interval 15m --period 60d
    python sweep_v2.py --demo --min-trades 10   # smoke test / small samples

Grid ranges are configurable via comma-separated lists. Defaults are kept
modest to avoid a combinatorial explosion — widen deliberately, not by habit,
since each extra grid dimension multiplies runtime and multiplies the number
of "shots on goal" at finding something that merely looks good by chance.
"""

import argparse
import itertools
import sys
import pandas as pd

from fvg_strategy import summarize, fetch_data, make_demo_data
from fvg_strategy_v2 import backtest_v2


def parse_float_list(s: str) -> list:
    return [float(x) for x in s.split(",")]


def parse_int_list(s: str) -> list:
    return [int(x) for x in s.split(",")]


def parse_str_list(s: str) -> list:
    return [x.strip() for x in s.split(",")]


def run_grid_v2(df: pd.DataFrame, grids: dict, fixed: dict) -> list:
    """
    grids: dict of lists to itertools.product over, keys matching backtest_v2
           kwarg names (reward_risk, min_gap_atr, max_bars_active,
           min_wick_ratio, session_minutes, htf_interval, min_confluence).
    fixed: dict of kwargs held constant across the whole sweep.
    """
    keys = list(grids.keys())
    combos = list(itertools.product(*[grids[k] for k in keys]))

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        kwargs = {**fixed, **params}
        trades, _, _, _ = backtest_v2(df, **kwargs)
        stats = summarize(trades)
        stats.update(params)
        results.append(stats)
    return results


def print_table(rows: list, param_cols: list, sort_key: str = "total_R", top_n: int = 15):
    valid = [r for r in rows if r.get("num_trades", 0) > 0]
    valid.sort(key=lambda r: r.get(sort_key, -999), reverse=True)

    stat_cols = ["num_trades", "win_rate", "profit_factor", "total_R", "max_drawdown_R"]
    cols = param_cols + stat_cols
    header = " | ".join(f"{c:>13s}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in valid[:top_n]:
        print(" | ".join(f"{str(r.get(c, '')):>13s}" for c in cols))
    return valid


def main():
    parser = argparse.ArgumentParser(description="v2 FVG strategy sweep with train/test split")
    parser.add_argument("--csv", type=str)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--source", choices=["yfinance", "alpaca"])
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--period", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)

    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--min-trades", type=int, default=20,
                         help="Minimum trades in TRAIN for a combo to be considered")
    parser.add_argument("--sort-by", type=str, default="total_R",
                         choices=["total_R", "profit_factor", "avg_R"])

    # Grid dimensions
    parser.add_argument("--rr-grid", type=str, default="1.5,2.0")
    parser.add_argument("--min-gap-grid", type=str, default="0.1,0.2")
    parser.add_argument("--max-bars-grid", type=str, default="40")
    parser.add_argument("--min-wick-grid", type=str, default="0.2,0.3,0.4")
    parser.add_argument("--session-minutes-grid", type=str, default="60,120")
    parser.add_argument("--htf-interval-grid", type=str, default="1h,4h")
    parser.add_argument("--min-confluence-grid", type=str, default="2,3")

    # Fixed (not swept) settings
    parser.add_argument("--session-start", type=str, default="09:30")
    parser.add_argument("--tz", type=str, default="America/New_York")
    parser.add_argument("--htf-min-gap-atr", type=float, default=0.2)
    parser.add_argument("--htf-max-bars-active", type=int, default=20)
    parser.add_argument("--use-trend-filter", action="store_true")
    parser.add_argument("--friction-R", type=float, default=0.0)
    args = parser.parse_args()

    # --- Load data ---
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

    # --- Chronological train/test split ---
    split_idx = int(len(df) * args.train_frac)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Total bars: {len(df)}  |  Train: {len(train_df)} bars "
          f"({train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]})")
    print(f"Test:  {len(test_df)} bars "
          f"({test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]})\n")

    grids = {
        "reward_risk": parse_float_list(args.rr_grid),
        "min_gap_atr_mult": parse_float_list(args.min_gap_grid),
        "max_bars_active": parse_int_list(args.max_bars_grid),
        "min_wick_ratio": parse_float_list(args.min_wick_grid),
        "session_minutes": parse_int_list(args.session_minutes_grid),
        "htf_interval": parse_str_list(args.htf_interval_grid),
        "min_confluence": parse_int_list(args.min_confluence_grid),
    }
    fixed = {
        "use_trend_filter": args.use_trend_filter,
        "use_rejection": True,
        "use_session_filter": True,
        "use_htf_confluence": True,
        "session_start": args.session_start,
        "tz": args.tz,
        "htf_min_gap_atr": args.htf_min_gap_atr,
        "htf_max_bars_active": args.htf_max_bars_active,
        "friction_R": args.friction_R,
    }

    n_combos = 1
    for v in grids.values():
        n_combos *= len(v)
    print(f"Running grid search on TRAIN data ({n_combos} combinations)... this may take a moment.\n")

    train_results = run_grid_v2(train_df, grids, fixed)
    qualifying = [r for r in train_results if r.get("num_trades", 0) >= args.min_trades]

    if not qualifying:
        best_n = max((r.get("num_trades", 0) for r in train_results), default=0)
        print(f"No combination reached --min-trades {args.min_trades} on TRAIN "
              f"(best found: {best_n} trades).")
        print("Try a longer history, a lower --min-trades, or a looser grid "
              "(e.g. add --min-confluence-grid 1,2,3, or widen --session-minutes-grid).")
        return

    param_cols = list(grids.keys())
    print(f"=== TOP TRAIN RESULTS (ranked by {args.sort_by}, min {args.min_trades} trades) ===")
    ranked = print_table(qualifying, param_cols, sort_key=args.sort_by)
    best = ranked[0]
    best_params = {k: best[k] for k in grids.keys()}

    print(f"\nBest combo on TRAIN: {best_params}")

    # --- Validate on TEST, untouched, single run ---
    test_kwargs = {**fixed, **best_params}
    test_trades, _, _, _ = backtest_v2(test_df, **test_kwargs)
    test_stats = summarize(test_trades)

    print("\n=== OUT-OF-SAMPLE (TEST) RESULT for the best TRAIN combo ===")
    for k in ["num_trades", "win_rate", "avg_R", "total_R", "avg_win_R",
              "avg_loss_R", "profit_factor", "max_drawdown_R"]:
        train_val = str(best.get(k, "n/a"))
        test_val = str(test_stats.get(k, "n/a"))
        print(f"{k:16s}: train={train_val:<10}  test={test_val}")

    print("\n--- Interpretation guide ---")
    if test_stats.get("num_trades", 0) < max(5, args.min_trades // 4):
        print("Very few test trades — not enough out-of-sample data to draw a conclusion. "
              "This is common with v2's stacked filters on a short history; pulling a longer "
              "period (e.g. via Alpaca) matters more than further tuning right now.")
    elif test_stats.get("total_R", 0) > 0 and test_stats.get("profit_factor", 0) >= 1.0:
        print("Edge held up out-of-sample. Still just one split/one symbol — validate on "
              "other symbols (validate.py) with these exact fixed parameters before trusting it.")
    else:
        print("Edge did NOT hold up out-of-sample — signature of overfitting to the train period.")


if __name__ == "__main__":
    main()
