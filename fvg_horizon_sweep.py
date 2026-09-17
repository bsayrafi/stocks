"""
fvg_horizon_sweep.py
======================
Sweeps target R-multiple and hold-window combinations to find which
horizon the hourly FVG pattern actually predicts best. Fetches raw bars
and detects (unlabeled) FVGs ONCE per symbol, caches them locally, then
relabels + quick-trains for each grid point without hitting Alpaca again.

Two phases:
  1. cache  — fetch bars, detect FVGs (no labels yet), save to --cache-dir.
              Only needs to run once per ticker universe / lookback window.
  2. sweep  — for each (target_r, max_hold) pair: relabel from cache,
              quick single-split train/eval (no 5-fold CV — this is a
              coarse comparison tool, not a final model), report AUC.

Usage:
    python3 fvg_horizon_sweep.py cache --sector Technology --years 2 --cache-dir cache/tech_2y
    python3 fvg_horizon_sweep.py sweep --cache-dir cache/tech_2y --out sweep_results.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from sklearn.metrics import roc_auc_score

from fvg_data_pipeline import add_indicators, label_outcomes
from fvg_data_pipeline_hourly import (
    CONFIG as BASE_CONFIG,
    fetch_bars,
    build_daily_context,
    detect_fvgs_hourly,
    load_tickers_by_sector,
)
from fvg_train_model_hourly import FEATURE_COLS, build_model, time_based_split

load_dotenv()

TARGET_R_GRID = [1.0, 1.5, 2.0, 2.5]
MAX_HOLD_GRID = [10, 20, 30, 40]


# ---------------------------------------------------------------------------
# Phase 1: cache raw bars + unlabeled FVGs (one-time fetch per universe)
# ---------------------------------------------------------------------------

def build_cache(tickers, api_key, api_secret, cache_dir, cfg=BASE_CONFIG, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    client = StockHistoricalDataClient(api_key, api_secret)
    n_cached = 0

    for n, symbol in enumerate(tickers, 1):
        bars_path = os.path.join(cache_dir, f"{symbol}_bars.parquet")
        fvgs_path = os.path.join(cache_dir, f"{symbol}_fvgs.parquet")
        if os.path.exists(bars_path) and os.path.exists(fvgs_path):
            n_cached += 1
            continue  # resumable — skip symbols already cached

        daily_raw = fetch_bars(client, symbol, TimeFrame.Day,
                                cfg["LOOKBACK_YEARS"] + cfg["DAILY_CONTEXT_BUFFER_YEARS"], cfg)
        if daily_raw is None or len(daily_raw) < 250:
            continue
        daily_ctx = build_daily_context(daily_raw, cfg)

        hourly_raw = fetch_bars(client, symbol, TimeFrame.Hour, cfg["LOOKBACK_YEARS"], cfg)
        if hourly_raw is None or len(hourly_raw) < 250:
            continue
        hourly_ind = add_indicators(hourly_raw, cfg)

        fvgs = detect_fvgs_hourly(hourly_ind, daily_ctx, symbol, cfg)
        if fvgs.empty:
            continue

        # cache only what label_outcomes needs to relabel later: OHLC + the
        # unlabeled FVG rows (still carrying _gap_low/_gap_high/_atr_at_formation)
        hourly_ind[["open", "high", "low", "close", "volume"]].to_parquet(bars_path)
        fvgs.to_parquet(fvgs_path)
        n_cached += 1
        if verbose:
            print(f"[{n}/{len(tickers)}] cached {symbol} — {len(fvgs)} unlabeled FVGs")

    print(f"\nCache complete: {n_cached}/{len(tickers)} symbols in {cache_dir}")


# ---------------------------------------------------------------------------
# Phase 2: relabel from cache + quick eval, for each grid point
# ---------------------------------------------------------------------------

def relabel_all(cache_dir, cfg):
    """Reload cached bars/FVGs and relabel with the given target/hold config —
    no network calls, pure local recomputation."""
    fvg_files = sorted(glob.glob(os.path.join(cache_dir, "*_fvgs.parquet")))
    all_rows = []
    for fvgs_path in fvg_files:
        symbol = os.path.basename(fvgs_path).replace("_fvgs.parquet", "")
        bars_path = os.path.join(cache_dir, f"{symbol}_bars.parquet")
        bars = pd.read_parquet(bars_path)
        fvgs = pd.read_parquet(fvgs_path)
        labeled = label_outcomes(bars, fvgs, cfg)
        all_rows.append(labeled)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def quick_eval(dataset):
    """Single time-based train/test split, no CV — fast comparison across
    the grid, not a final model. Uses the non-htf feature set."""
    df = dataset.dropna(subset=["label"]).copy()
    if len(df) < 500:
        return None  # too few usable rows to trust a split at this grid point
    df = df.sort_values("formation_time").reset_index(drop=True)
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median(numeric_only=True))
    y = df["label"].astype(int)

    X_train, y_train, X_test, y_test, _ = time_based_split(df, X, y, test_fraction=0.2)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None

    model = build_model(y_train)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "n_usable": len(df),
        "label_rate": round(y.mean(), 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_auc": round(roc_auc_score(y_test, proba), 3),
    }


def sweep(cache_dir, cfg=BASE_CONFIG, target_grid=TARGET_R_GRID, hold_grid=MAX_HOLD_GRID):
    results = []
    for target_r in target_grid:
        for max_hold in hold_grid:
            run_cfg = dict(cfg)
            run_cfg["TARGET_R_MULTIPLE"] = target_r
            run_cfg["MAX_HOLD_BARS"] = max_hold
            dataset = relabel_all(cache_dir, run_cfg)
            metrics = quick_eval(dataset)
            row = {"target_r": target_r, "max_hold_bars": max_hold}
            if metrics is None:
                row.update({"n_usable": len(dataset) if not dataset.empty else 0,
                             "label_rate": None, "n_train": None, "n_test": None, "test_auc": None})
                print(f"target_r={target_r:.1f}  max_hold={max_hold:2d}  -> too few usable rows, skipped")
            else:
                row.update(metrics)
                print(f"target_r={target_r:.1f}  max_hold={max_hold:2d}  -> "
                      f"n={metrics['n_usable']:6d}  label_rate={metrics['label_rate']:.3f}  "
                      f"test_auc={metrics['test_auc']:.3f}")
            results.append(row)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Sweep FVG target R-multiple / hold-window combinations.")
    sub = parser.add_subparsers(dest="phase", required=True)

    p_cache = sub.add_parser("cache", help="Fetch + detect FVGs once, cache locally.")
    p_cache.add_argument("--sector", default=BASE_CONFIG["SECTOR"])
    p_cache.add_argument("--tickers", nargs="*", default=None)
    p_cache.add_argument("--years", type=int, default=BASE_CONFIG["LOOKBACK_YEARS"])
    p_cache.add_argument("--cache-dir", required=True)

    p_sweep = sub.add_parser("sweep", help="Relabel from cache across the grid and report AUC.")
    p_sweep.add_argument("--cache-dir", required=True)
    p_sweep.add_argument("--out", default="sweep_results.csv")
    p_sweep.add_argument("--targets", nargs="*", type=float, default=TARGET_R_GRID)
    p_sweep.add_argument("--holds", nargs="*", type=int, default=MAX_HOLD_GRID)

    args = parser.parse_args()

    if args.phase == "cache":
        api_key = os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            raise SystemExit("Set ALPACA_API_KEY / ALPACA_SECRET_KEY in your .env file.")
        cfg = dict(BASE_CONFIG)
        cfg["LOOKBACK_YEARS"] = args.years
        tickers = args.tickers if args.tickers else load_tickers_by_sector(args.sector)
        build_cache(tickers, api_key, api_secret, args.cache_dir, cfg)

    elif args.phase == "sweep":
        results = sweep(args.cache_dir, target_grid=args.targets, hold_grid=args.holds)
        results.to_csv(args.out, index=False)
        print(f"\nSaved sweep results to {args.out}")
        valid = results.dropna(subset=["test_auc"])
        if not valid.empty:
            best = valid.loc[valid["test_auc"].idxmax()]
            print(f"\nBest: target_r={best['target_r']}, max_hold_bars={int(best['max_hold_bars'])}, "
                  f"test_auc={best['test_auc']}")


if __name__ == "__main__":
    main()
