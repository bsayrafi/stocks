"""
analyze_fill_decomposition.py
================================
Separates two changes that got bundled together in analyze_realistic_fill.py:
  1. FILL DEFINITION — "loose" (touch the near edge of the gap, the
     original production behavior) vs "strict" (must actually reach the
     midpoint entry price before counting as filled).
  2. TIE-BREAK ORDERING — "naive" (if both stop and target are touched in
     the same bar, assume stop — the original production behavior) vs
     "realistic" (simulate which was actually touched first using the
     OHLC intrabar path heuristic).

Runs all 4 combinations on the same cached data so each factor's isolated
effect on win rate and quick AUC is visible, instead of one entangled number.
loose+naive should exactly reproduce the original production numbers —
that's the sanity check this script is built around.

Usage:
    python3 analyze_fill_decomposition.py --cache-dir cache/tech_2y --target-r 1.0 --max-hold 20
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fvg_data_pipeline_hourly import CONFIG as BASE_CONFIG
from fvg_train_model_hourly import FEATURE_COLS, build_model, time_based_split
from analyze_realistic_fill import bar_path_segments, threshold_position


def label_outcomes_configurable(df, fvgs, cfg, strict_fill, realistic_order):
    opens, highs, lows, closes = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    labels, outcomes, fill_delays, same_bar_flags = [], [], [], []

    for _, row in fvgs.iterrows():
        i = int(row["formation_idx"])
        gap_low, gap_high, atr = row["_gap_low"], row["_gap_high"], row["_atr_at_formation"]
        direction = row["direction"]
        entry = (gap_low + gap_high) / 2.0
        fill_trigger = entry if strict_fill else (gap_high if direction == 1 else gap_low)

        if direction == 1:
            stop = entry - cfg["STOP_ATR_MULT"] * atr
        else:
            stop = entry + cfg["STOP_ATR_MULT"] * atr
        risk = abs(entry - stop)
        if risk <= 0:
            labels.append(np.nan); outcomes.append("invalid"); fill_delays.append(np.nan); same_bar_flags.append(np.nan)
            continue
        target = entry + cfg["TARGET_R_MULTIPLE"] * risk if direction == 1 else entry - cfg["TARGET_R_MULTIPLE"] * risk

        filled, fill_bar, resolved = False, None, False
        outcome, label, same_bar = "no_fill", np.nan, np.nan
        end = min(i + 1 + cfg["MAX_HOLD_BARS"], n)

        for j in range(i + 1, end):
            if realistic_order:
                segs = bar_path_segments(opens[j], highs[j], lows[j], closes[j])
                fill_pos = threshold_position(segs, fill_trigger)
                stop_pos = threshold_position(segs, stop)
                target_pos = threshold_position(segs, target)
            else:
                if direction == 1:
                    fill_touch, stop_touch, target_touch = lows[j] <= fill_trigger, lows[j] <= stop, highs[j] >= target
                else:
                    fill_touch, stop_touch, target_touch = highs[j] >= fill_trigger, highs[j] >= stop, lows[j] <= target

            if not filled:
                if realistic_order:
                    if fill_pos is None:
                        continue
                    filled, fill_bar = True, j - i
                    cands = []
                    if stop_pos is not None and stop_pos >= fill_pos:
                        cands.append(("stop", stop_pos))
                    if target_pos is not None and target_pos >= fill_pos:
                        cands.append(("target", target_pos))
                    if cands:
                        cands.sort(key=lambda x: x[1])
                        outcome, label = cands[0][0], (1 if cands[0][0] == "target" else 0)
                        same_bar, resolved = True, True
                        break
                    continue
                else:
                    if not fill_touch:
                        continue
                    filled, fill_bar = True, j - i
                    if stop_touch:  # matches original: ties assumed stop
                        outcome, label = "stop", 0
                    elif target_touch:
                        outcome, label = "target", 1
                    else:
                        continue
                    same_bar, resolved = True, True
                    break
            else:
                if realistic_order:
                    cands = []
                    if stop_pos is not None:
                        cands.append(("stop", stop_pos))
                    if target_pos is not None:
                        cands.append(("target", target_pos))
                    if cands:
                        cands.sort(key=lambda x: x[1])
                        outcome, label = cands[0][0], (1 if cands[0][0] == "target" else 0)
                        same_bar, resolved = False, True
                        break
                else:
                    if stop_touch:
                        outcome, label = "stop", 0
                        same_bar, resolved = False, True
                        break
                    elif target_touch:
                        outcome, label = "target", 1
                        same_bar, resolved = False, True
                        break

        if not resolved and filled:
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


def load_cache(cache_dir):
    fvg_files = sorted(glob.glob(os.path.join(cache_dir, "*_fvgs.parquet")))
    pairs = []
    for fvgs_path in fvg_files:
        symbol = os.path.basename(fvgs_path).replace("_fvgs.parquet", "")
        bars = pd.read_parquet(os.path.join(cache_dir, f"{symbol}_bars.parquet"))
        fvgs = pd.read_parquet(fvgs_path)
        pairs.append((bars, fvgs))
    return pairs


def quick_auc(dataset):
    df = dataset.dropna(subset=["label"]).copy()
    if len(df) < 500:
        return None
    df = df.sort_values("formation_time").reset_index(drop=True)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(df[cols].median(numeric_only=True))
    y = df["label"].astype(int)
    X_train, y_train, X_test, y_test, _ = time_based_split(df, X, y, test_fraction=0.2)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None
    model = build_model(y_train)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return round(roc_auc_score(y_test, proba), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-r", type=float, default=BASE_CONFIG["TARGET_R_MULTIPLE"])
    parser.add_argument("--max-hold", type=int, default=BASE_CONFIG["MAX_HOLD_BARS"])
    args = parser.parse_args()

    cfg = dict(BASE_CONFIG)
    cfg["TARGET_R_MULTIPLE"] = args.target_r
    cfg["MAX_HOLD_BARS"] = args.max_hold

    pairs = load_cache(args.cache_dir)

    results = []
    for strict_fill in [False, True]:
        for realistic_order in [False, True]:
            rows = [label_outcomes_configurable(bars, fvgs.copy(), cfg, strict_fill, realistic_order)
                    for bars, fvgs in pairs]
            dataset = pd.concat(rows, ignore_index=True)
            filled = dataset.dropna(subset=["label"])
            auc = quick_auc(dataset)
            label_name = f"{'strict' if strict_fill else 'loose'}_fill + {'realistic' if realistic_order else 'naive'}_order"
            results.append({
                "config": label_name,
                "n_resolved": len(filled),
                "win_rate": round(filled["label"].mean(), 3) if len(filled) else None,
                "test_auc": auc,
            })
            print(f"{label_name:35s}  n={len(filled):6d}  win_rate={results[-1]['win_rate']}  test_auc={auc}")

    print("\n(loose_fill + naive_order should match the original production numbers: "
          "n=35190, win_rate=0.585, test_auc=0.645 — sanity check.)")


if __name__ == "__main__":
    main()
