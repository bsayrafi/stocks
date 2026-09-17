"""
analyze_fill_delay.py
========================
Checks whether "how long after formation did the FVG actually get filled"
predicts win rate — i.e. whether fresher fills (less time-since-formation
used up) really do better, or whether that's just a plausible-sounding
guess. Reuses the cache from fvg_horizon_sweep.py (bars + unlabeled FVGs),
so no new Alpaca calls needed.

Usage:
    python3 analyze_fill_delay.py --cache-dir cache/tech_2y --target-r 1.0 --max-hold 20
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

from fvg_data_pipeline_hourly import CONFIG as BASE_CONFIG


def label_with_fill_delay(df, fvgs, cfg):
    """Same walk-forward logic as fvg_data_pipeline.label_outcomes, but
    additionally records bars_to_fill (bars from formation to the gap
    being retraced into) and whether the stop/target resolution happened
    on that SAME bar — a same-bar fill+resolve means price never really
    traded through the midpoint entry a live trader would have gotten,
    which would artificially inflate win rate for delayed fills if sharp
    snapback bars are more common after a longer pullback."""
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    labels, outcomes, fill_delays, same_bar_flags = [], [], [], []

    for _, row in fvgs.iterrows():
        i = int(row["formation_idx"])
        gap_low, gap_high, atr = row["_gap_low"], row["_gap_high"], row["_atr_at_formation"]
        direction = row["direction"]
        entry = (gap_low + gap_high) / 2.0

        if direction == 1:
            stop = entry - cfg["STOP_ATR_MULT"] * atr
        else:
            stop = entry + cfg["STOP_ATR_MULT"] * atr
        risk = abs(entry - stop)
        if risk <= 0:
            labels.append(np.nan); outcomes.append("invalid"); fill_delays.append(np.nan); same_bar_flags.append(np.nan)
            continue
        target = entry + cfg["TARGET_R_MULTIPLE"] * risk if direction == 1 else entry - cfg["TARGET_R_MULTIPLE"] * risk

        filled, fill_bar = False, None
        outcome, label, same_bar = "no_fill", np.nan, np.nan
        end = min(i + 1 + cfg["MAX_HOLD_BARS"], n)
        for j in range(i + 1, end):
            if not filled:
                if direction == 1 and lows[j] <= gap_high:
                    filled, fill_bar = True, j - i
                elif direction == 0 and highs[j] >= gap_low:
                    filled, fill_bar = True, j - i
                if not filled:
                    continue
            if direction == 1:
                hit_stop, hit_target = lows[j] <= stop, highs[j] >= target
            else:
                hit_stop, hit_target = highs[j] >= stop, lows[j] <= target
            if hit_stop:
                outcome, label = "stop", 0
                same_bar = ((j - i) == fill_bar)
                break
            if hit_target:
                outcome, label = "target", 1
                same_bar = ((j - i) == fill_bar)
                break
        else:
            if filled:
                outcome = "unresolved"

        labels.append(label)
        outcomes.append(outcome)
        fill_delays.append(fill_bar)
        same_bar_flags.append(same_bar)

    out = fvgs.copy()
    out["label"] = labels
    out["outcome"] = outcomes
    out["bars_to_fill"] = fill_delays
    out["same_bar_resolution"] = same_bar_flags
    return out.drop(columns=["_gap_low", "_gap_high", "_atr_at_formation"])


def load_and_label(cache_dir, cfg):
    fvg_files = sorted(glob.glob(os.path.join(cache_dir, "*_fvgs.parquet")))
    all_rows = []
    for fvgs_path in fvg_files:
        symbol = os.path.basename(fvgs_path).replace("_fvgs.parquet", "")
        bars = pd.read_parquet(os.path.join(cache_dir, f"{symbol}_bars.parquet"))
        fvgs = pd.read_parquet(fvgs_path)
        all_rows.append(label_with_fill_delay(bars, fvgs, cfg))
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-r", type=float, default=BASE_CONFIG["TARGET_R_MULTIPLE"])
    parser.add_argument("--max-hold", type=int, default=BASE_CONFIG["MAX_HOLD_BARS"])
    args = parser.parse_args()

    cfg = dict(BASE_CONFIG)
    cfg["TARGET_R_MULTIPLE"] = args.target_r
    cfg["MAX_HOLD_BARS"] = args.max_hold

    df = load_and_label(args.cache_dir, cfg)
    print(f"Total FVGs: {len(df)}")
    print(df["outcome"].value_counts())

    filled = df.dropna(subset=["label"]).copy()  # excludes no_fill and unresolved
    print(f"\nFilled + resolved FVGs usable for this analysis: {len(filled)}")

    # uniform-width buckets, as requested, instead of granular-then-coarse
    bins = [0.5, 5.5, 10.5, 15.5, args.max_hold + 0.5]
    labels = ["1-5", "6-10", "11-15", f"16-{args.max_hold}"]
    filled["fill_bucket"] = pd.cut(filled["bars_to_fill"], bins=bins, labels=labels)

    summary = filled.groupby("fill_bucket", observed=True).agg(
        n=("label", "size"),
        win_rate=("label", "mean"),
        pct_same_bar_resolution=("same_bar_resolution", "mean"),
    ).round(3)
    print("\nWin rate + same-bar-resolution rate by bars-to-fill:")
    print(summary)
    print("\npct_same_bar_resolution = fraction of trades where the stop/target hit on the SAME bar as the fill —")
    print("high values here mean the price never really traded through the entry; the label may be overstating")
    print("what a live trader entering at the midpoint would have actually experienced.")

    overall = filled["label"].mean()
    print(f"\nOverall win rate (all filled+resolved FVGs): {overall:.3f}")


if __name__ == "__main__":
    main()
