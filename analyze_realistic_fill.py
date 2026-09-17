"""
analyze_realistic_fill.py
============================
The original label_outcomes() couldn't tell, within a single bar, whether
price reached the entry BEFORE or AFTER it reached stop/target — it just
assumed worst-case (stop wins) if both happened in the same bar. That
undersold same-bar resolutions.

This uses a standard OHLC-only intrabar path heuristic (up-close bars
assumed to move open->low->high->close; down-close bars open->high->low
->close) to determine the actual order events happened in, then relabels
the whole dataset and compares win rate / same-bar-resolution rate /
quick AUC against the original approach — so you can see exactly how much
the fix moves the headline numbers, not just take it on faith.

Usage:
    python3 analyze_realistic_fill.py --cache-dir cache/tech_2y --target-r 1.0 --max-hold 20
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fvg_data_pipeline import label_outcomes as label_outcomes_original
from fvg_data_pipeline_hourly import CONFIG as BASE_CONFIG
from fvg_train_model_hourly import FEATURE_COLS, build_model, time_based_split


# ---------------------------------------------------------------------------
# Intrabar path heuristic
# ---------------------------------------------------------------------------

def bar_path_segments(open_, high, low, close):
    """Three monotonic segments approximating the path price took within
    the bar, in chronological order."""
    if close >= open_:
        return [(open_, low), (low, high), (high, close)]  # dip then rally
    return [(open_, high), (high, low), (low, close)]        # rally then drop


def threshold_position(segments, threshold):
    """Where in the bar's path (as a continuous 0-3 position) a threshold
    was first crossed, or None if never reached this bar."""
    for idx, (a, b) in enumerate(segments):
        lo, hi = (a, b) if a <= b else (b, a)
        if lo <= threshold <= hi:
            frac = 0.0 if a == b else abs(threshold - a) / abs(b - a)
            return idx + frac
    return None


# ---------------------------------------------------------------------------
# Realistic labeling
# ---------------------------------------------------------------------------

def label_outcomes_realistic(df, fvgs, cfg):
    opens, highs, lows, closes = df["open"].values, df["high"].values, df["low"].values, df["close"].values
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
        resolved = False

        for j in range(i + 1, end):
            segs = bar_path_segments(opens[j], highs[j], lows[j], closes[j])
            entry_pos = threshold_position(segs, entry)
            stop_pos = threshold_position(segs, stop)
            target_pos = threshold_position(segs, target)

            if not filled:
                if entry_pos is None:
                    continue
                filled, fill_bar = True, j - i
                candidates = []
                if stop_pos is not None and stop_pos >= entry_pos:
                    candidates.append(("stop", stop_pos))
                if target_pos is not None and target_pos >= entry_pos:
                    candidates.append(("target", target_pos))
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    outcome, label = candidates[0][0], (1 if candidates[0][0] == "target" else 0)
                    same_bar, resolved = True, True
                    break
                continue
            else:
                candidates = []
                if stop_pos is not None:
                    candidates.append(("stop", stop_pos))
                if target_pos is not None:
                    candidates.append(("target", target_pos))
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    outcome, label = candidates[0][0], (1 if candidates[0][0] == "target" else 0)
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


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

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

    original_rows, realistic_rows = [], []
    for bars, fvgs in pairs:
        original_rows.append(label_outcomes_original(bars, fvgs.copy(), cfg))
        realistic_rows.append(label_outcomes_realistic(bars, fvgs.copy(), cfg))

    original = pd.concat(original_rows, ignore_index=True)
    realistic = pd.concat(realistic_rows, ignore_index=True)

    for name, df in [("ORIGINAL", original), ("REALISTIC", realistic)]:
        filled = df.dropna(subset=["label"])
        print(f"\n=== {name} ===")
        print(df["outcome"].value_counts())
        print(f"Win rate (filled+resolved): {filled['label'].mean():.3f}  (n={len(filled)})")
        if "same_bar_resolution" in filled.columns:
            sb = filled["same_bar_resolution"].dropna()
            if len(sb) > 0:
                print(f"Same-bar resolution rate: {sb.mean():.3f}")
        auc = quick_auc(df)
        print(f"Quick single-split test AUC: {auc}")

    print("\n=== Direct comparison (rows where original and realistic disagree) ===")
    merged = original[["symbol", "formation_time", "label"]].rename(columns={"label": "label_original"}).merge(
        realistic[["symbol", "formation_time", "label"]].rename(columns={"label": "label_realistic"}),
        on=["symbol", "formation_time"]
    )
    both_resolved = merged.dropna(subset=["label_original", "label_realistic"])
    disagree = both_resolved[both_resolved["label_original"] != both_resolved["label_realistic"]]
    print(f"{len(disagree)} / {len(both_resolved)} resolved FVGs got a DIFFERENT label "
          f"({len(disagree) / len(both_resolved):.1%}) once same-bar ordering was simulated realistically.")


if __name__ == "__main__":
    main()
