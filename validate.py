"""
FVG Strategy — Multi-Symbol Validation with Friction Haircut
================================================================

This script does NOT tune any parameters. It takes the fixed parameter set
you already selected via sweep.py's train/test split, and checks whether
it holds up:

  1. On a longer/fresh history for a symbol (e.g. pull more SPY bars via
     Alpaca than Yahoo's 60-day cap allows), and
  2. On symbols it was never tuned on at all (e.g. MSFT, NVDA) — this is a
     much stronger generalization test than re-slicing the same symbol's
     data, since a real edge in an approach like FVG retracement should
     show up across similar liquid instruments, not just one.

It also applies a friction haircut (default -0.05R per trade) to every
closed trade, as a rough stand-in for spread + commissions. Any edge that
doesn't survive a small friction haircut isn't tradeable in practice,
however good it looks frictionless.

IMPORTANT: because no tuning happens here, it's safe to point this at any
date range or symbol without worrying about data-snooping bias — that
concern only applies to the parameter *selection* step (sweep.py), not to
using an already-fixed parameter set here.

Usage
-----
    # Multi-symbol validation, fixed params, via Alpaca (needs API keys set):
    python validate.py --source alpaca --symbols SPY,MSFT,NVDA \\
        --interval 15Min --start 2023-01-01 --end 2024-06-01 \\
        --reward-risk 1.5 --min-gap-atr 0.2 --max-bars-active 40 \\
        --friction-R 0.05

    # Same, via Yahoo Finance (shorter history, no API keys needed):
    python validate.py --source yfinance --symbols SPY,MSFT,NVDA \\
        --interval 15m --period 60d \\
        --reward-risk 1.5 --min-gap-atr 0.2 --max-bars-active 40

    # Smoke-test on synthetic data (no network needed):
    python validate.py --demo --symbols SPY,MSFT,NVDA
"""

import argparse
import copy
import sys
import pandas as pd

from fvg_strategy import backtest, summarize, fetch_data, make_demo_data


def apply_friction(trades: list, friction_R: float) -> list:
    """Subtract a fixed R-cost per trade to approximate spread + commissions."""
    adjusted = []
    for tr in trades:
        tr2 = copy.copy(tr)
        if tr2.r_multiple is not None:
            tr2.r_multiple -= friction_R
        adjusted.append(tr2)
    return adjusted


def run_symbol(df: pd.DataFrame, params: dict, friction_R: float) -> dict:
    trades, fvgs = backtest(
        df,
        reward_risk=params["reward_risk"],
        max_bars_active=params["max_bars_active"],
        use_trend_filter=params["use_trend_filter"],
        min_gap_atr_mult=params["min_gap_atr"],
    )
    raw_stats = summarize(trades)
    adjusted_trades = apply_friction(trades, friction_R)
    adj_stats = summarize(adjusted_trades)
    return {"fvgs_detected": len(fvgs), "raw": raw_stats, "adjusted": adj_stats}


def print_comparison(all_results: dict, friction_R: float):
    cols = ["num_trades", "win_rate", "profit_factor", "total_R", "avg_R", "max_drawdown_R"]

    print("\n=== RAW (frictionless) results ===")
    header = f"{'symbol':>10} | " + " | ".join(f"{c:>14s}" for c in cols)
    print(header)
    print("-" * len(header))
    for symbol, res in all_results.items():
        row = res["raw"]
        print(f"{symbol:>10} | " + " | ".join(f"{str(row.get(c, '')):>14s}" for c in cols))

    print(f"\n=== FRICTION-ADJUSTED results (-{friction_R}R per trade) ===")
    print(header)
    print("-" * len(header))
    for symbol, res in all_results.items():
        row = res["adjusted"]
        print(f"{symbol:>10} | " + " | ".join(f"{str(row.get(c, '')):>14s}" for c in cols))


def main():
    parser = argparse.ArgumentParser(description="Multi-symbol, friction-adjusted validation of fixed FVG params")
    parser.add_argument("--symbols", type=str, default="SPY,MSFT,NVDA",
                         help="Comma-separated list of tickers")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data per symbol (smoke test)")
    parser.add_argument("--source", choices=["yfinance", "alpaca"])
    parser.add_argument("--interval", type=str, default="15m",
                         help="yfinance: 15m etc. alpaca: 15Min etc.")
    parser.add_argument("--period", type=str, default=None, help="yfinance lookback, e.g. 60d")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)

    # Fixed strategy params — no grid, no tuning happens in this script
    parser.add_argument("--reward-risk", type=float, default=1.5)
    parser.add_argument("--min-gap-atr", type=float, default=0.2)
    parser.add_argument("--max-bars-active", type=int, default=40)
    parser.add_argument("--use-trend-filter", action="store_true",
                         help="Off by default, matching the best combo found in the sweep")
    parser.add_argument("--friction-R", type=float, default=0.05,
                         help="R-cost subtracted per trade to approximate spread+commissions. Default 0.05")
    args = parser.parse_args()

    params = {
        "reward_risk": args.reward_risk,
        "min_gap_atr": args.min_gap_atr,
        "max_bars_active": args.max_bars_active,
        "use_trend_filter": args.use_trend_filter,
    }
    print(f"Fixed parameters (not tuned here): {params}")
    print(f"Friction haircut: -{args.friction_R}R per trade\n")

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    all_results = {}

    for symbol in symbols:
        print(f"--- Loading {symbol} ---")
        if args.demo:
            # Different seed per symbol so the synthetic series differ, for smoke-testing only
            df = make_demo_data(seed=hash(symbol) % 10000)
        elif args.source == "yfinance":
            df = fetch_data("yfinance", symbol, interval=args.interval,
                             period=args.period, start=args.start, end=args.end)
        elif args.source == "alpaca":
            df = fetch_data("alpaca", symbol, timeframe=args.interval,
                             start=args.start, end=args.end)
        else:
            sys.exit("Specify --source yfinance|alpaca, or use --demo")

        result = run_symbol(df, params, args.friction_R)
        all_results[symbol] = result
        print(f"  {len(df)} bars, {result['fvgs_detected']} FVGs detected, "
              f"{result['raw']['num_trades']} trades\n")

    print_comparison(all_results, args.friction_R)

    print("\n--- Interpretation guide ---")
    surviving = [s for s, r in all_results.items()
                 if r["adjusted"].get("total_R", -999) > 0
                 and r["adjusted"].get("num_trades", 0) >= 10]
    print(f"Symbols with a positive, friction-adjusted edge (>=10 trades): "
          f"{surviving if surviving else 'none'}")
    if len(surviving) == len(symbols):
        print("Edge held across ALL tested symbols after friction — the strongest signal so far "
              "that this reflects a real, generalizable pattern rather than a fluke.")
    elif surviving:
        print("Edge held on some symbols but not others — worth checking whether the symbols "
              "that failed have structurally different volatility/liquidity, or whether this "
              "is still within the range of noise given trade counts.")
    else:
        print("Edge did not survive friction on any symbol at meaningful trade counts — "
              "the frictionless result was likely not economically tradeable.")


if __name__ == "__main__":
    main()
