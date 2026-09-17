"""
fvg_train_model.py
====================
Trains an ML model to predict FVG success probability ("quality score")
from the dataset produced by fvg_data_pipeline.py.

Usage (VS Code, after activating the venv from fvg_data_pipeline.py):
    python fvg_train_model.py --data data/fvg_dataset.parquet --out models/fvg_quality_model.joblib

Or from another script:
    from fvg_train_model import train, score_fvg
    model, feature_cols, metrics = train('data/fvg_dataset.parquet')
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

FEATURE_COLS = [
    "direction", "gap_size_pct", "gap_size_atr",
    "displacement_body_atr", "displacement_range_atr", "displacement_vol_ratio",
    "price_vs_ema20", "price_vs_ema50", "price_vs_ema200", "ema20_slope",
    "trend_slope20_atr", "rsi14", "atr_pctile",
    "dist_to_swing_high_atr", "dist_to_swing_low_atr",
    "recent_gap_count_20d", "day_of_week",
]

# in fvg_train_model.py, temporarily:
FEATURE_COLS = [c for c in FEATURE_COLS if c not in ("gap_size_atr", "gap_size_pct")]

TRAIN_CFG = {
    "TEST_FRACTION": 0.2,        # most recent 20% of formation dates held out
    "N_CV_SPLITS": 5,
    "RANDOM_STATE": 42,
}


def prep_dataset(df):
    """Drop unresolved/invalid rows, sort chronologically for a leak-free split."""
    df = df.dropna(subset=["label"]).copy()
    df = df.sort_values("formation_time").reset_index(drop=True)
    X = df[FEATURE_COLS].copy()
    y = df["label"].astype(int)
    # median-impute remaining NaNs (e.g. price_vs_ema200 for young tickers)
    X = X.fillna(X.median(numeric_only=True))
    return df, X, y


def time_based_split(df, X, y, test_fraction=TRAIN_CFG["TEST_FRACTION"]):
    """Chronological split by formation_time — never random, to avoid leaking
    future price behavior into training (many symbols overlap in time)."""
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff_date = df["formation_time"].iloc[cutoff_idx]
    train_mask = df["formation_time"] < cutoff_date
    return (X[train_mask], y[train_mask], X[~train_mask], y[~train_mask], cutoff_date)


def build_model(y_train, cfg=TRAIN_CFG):
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=cfg["RANDOM_STATE"],
            n_jobs=-1,
        )
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=cfg["RANDOM_STATE"],
    )


def cross_validate(X_train, y_train, cfg=TRAIN_CFG):
    """Walk-forward CV within the training window to sanity-check stability
    before touching the held-out test set."""
    tscv = TimeSeriesSplit(n_splits=cfg["N_CV_SPLITS"])
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        model = build_model(y_train.iloc[tr_idx], cfg)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y_train.iloc[val_idx], preds)
        aucs.append(auc)
        print(f"  fold {fold}: AUC={auc:.3f}  (train={len(tr_idx)}, val={len(val_idx)})")
    print(f"CV mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    return aucs


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "avg_precision": average_precision_score(y_test, proba),
        "base_rate": y_test.mean(),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "report": classification_report(y_test, preds, output_dict=True),
    }
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}  |  base rate: {metrics['base_rate']:.3f}  |  "
          f"avg precision: {metrics['avg_precision']:.3f}")
    print(classification_report(y_test, preds))
    return metrics, proba


def feature_importance(model, feature_cols=FEATURE_COLS):
    if HAS_XGB:
        imp = model.feature_importances_
    else:
        imp = model.feature_importances_
    return pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


def best_threshold(y_test, proba, min_precision=0.55):
    """Pick the probability cutoff maximizing recall subject to a minimum
    precision — useful for turning the score into an actionable signal."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    candidates = [(p, r, t) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds) if p >= min_precision]
    if not candidates:
        return 0.5
    candidates.sort(key=lambda x: -x[1])  # highest recall among those meeting precision floor
    return candidates[0][2]


def train(dataset_path, cfg=TRAIN_CFG, save_path="models/fvg_quality_model.joblib"):
    raw = pd.read_parquet(dataset_path)
    df, X, y = prep_dataset(raw)
    print(f"Usable FVGs: {len(df)}  (label=1 rate: {y.mean():.3f})")
    print(df["outcome"].value_counts())

    X_train, y_train, X_test, y_test, cutoff = time_based_split(df, X, y, cfg["TEST_FRACTION"])
    print(f"\nTrain: {len(X_train)} FVGs (< {cutoff.date()})  |  Test: {len(X_test)} FVGs (>= {cutoff.date()})")

    print("\n-- Cross-validation on training window --")
    cross_validate(X_train, y_train, cfg)

    model = build_model(y_train, cfg)
    model.fit(X_train, y_train)

    print("\n-- Held-out test evaluation --")
    metrics, proba = evaluate(model, X_test, y_test)

    print("\n-- Feature importance --")
    fi = feature_importance(model)
    print(fi.to_string(index=False))

    thresh = best_threshold(y_test, proba, min_precision=0.55)
    print(f"\nSuggested probability threshold for min 55% precision: {thresh:.3f}")

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLS, "threshold": thresh}, save_path)
    print(f"Saved model to {save_path}")

    metrics["feature_importance"] = fi
    metrics["threshold"] = thresh
    return model, FEATURE_COLS, metrics


def score_fvg(model, feature_dict, feature_cols=FEATURE_COLS):
    """Score a single live FVG. feature_dict must contain the same keys
    produced by fvg_data_pipeline.detect_fvgs (minus the _gap_*/formation_* meta cols).
    Returns a 0-100 quality score."""
    x = pd.DataFrame([{c: feature_dict.get(c, np.nan) for c in feature_cols}])
    proba = model.predict_proba(x)[0, 1]
    return round(proba * 100, 1)


def main():
    parser = argparse.ArgumentParser(description="Train the FVG quality-scoring model.")
    parser.add_argument("--data", default="data/fvg_dataset.parquet", help="Path to the dataset parquet.")
    parser.add_argument("--out", default="models/fvg_quality_model.joblib", help="Path to save the trained model.")
    args = parser.parse_args()
    train(args.data, save_path=args.out)


if __name__ == "__main__":
    main()
