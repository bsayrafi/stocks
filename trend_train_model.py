"""
trend_train_model.py
=======================
Trains a REGRESSION model (predicting realized R-multiple) on the
trend-following dataset — not classification, since a trailing stop
produces a continuous outcome (small loss, scratch, huge winner), not a
clean win/lose. Evaluation is framed around that: mean/median realized R
overall and by predicted-score decile (does the model's top decile
actually capture better trades), not AUC/precision/recall.

Usage:
    python3 trend_train_model.py --data data/trend_dataset.parquet --out models/trend_model.joblib
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False

FEATURE_COLS = [
    "breakout_strength_atr", "adx", "rel_volume", "trend_steepness_atr",
    "consecutive_higher_lows", "rsi14", "atr_pctile",
    "daily_composite_momentum", "daily_obv_slope_norm",
    "hour_of_day", "day_of_week", "near_macro_event",
]

TRAIN_CFG = {"TEST_FRACTION": 0.2, "N_CV_SPLITS": 5, "RANDOM_STATE": 42}


def prep_dataset(df):
    df = df.dropna(subset=["realized_r"]).copy()
    df = df.sort_values("entry_time").reset_index(drop=True)
    X = df[FEATURE_COLS].copy()
    y = df["realized_r"].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    return df, X, y


def time_based_split(df, X, y, test_fraction=TRAIN_CFG["TEST_FRACTION"]):
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff = df["entry_time"].iloc[cutoff_idx]
    train_mask = df["entry_time"] < cutoff
    return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask], cutoff


def build_model(cfg=TRAIN_CFG):
    if HAS_XGB:
        return XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=2.0, random_state=cfg["RANDOM_STATE"], n_jobs=-1,
        )
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=cfg["RANDOM_STATE"],
    )


def cross_validate(X_train, y_train, cfg=TRAIN_CFG):
    tscv = TimeSeriesSplit(n_splits=cfg["N_CV_SPLITS"])
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        model = build_model(cfg)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = model.predict(X_train.iloc[val_idx])
        r2 = r2_score(y_train.iloc[val_idx], preds)
        mae = mean_absolute_error(y_train.iloc[val_idx], preds)
        scores.append(r2)
        print(f"  fold {fold}: R²={r2:.3f}  MAE={mae:.3f}R  (train={len(tr_idx)}, val={len(val_idx)})")
    print(f"CV mean R²: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    return scores


def evaluate(model, X_test, y_test, n_deciles=5):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"Test R²: {r2:.3f}  |  MAE: {mae:.3f}R  |  "
          f"actual mean R: {y_test.mean():.3f}  |  actual median R: {y_test.median():.3f}")

    # the metric that actually matters: does the model's own predicted score
    # rank trades usefully — does the top bucket really realize more R?
    df = pd.DataFrame({"pred": preds, "actual": y_test.values})
    df["bucket"] = pd.qcut(df["pred"], n_deciles, labels=False, duplicates="drop")
    bucket_summary = df.groupby("bucket")["actual"].agg(["mean", "median", "count"]).round(3)
    print(f"\nActual realized R by predicted-score bucket (0=lowest predicted, {n_deciles - 1}=highest):")
    print(bucket_summary)
    return {"r2": r2, "mae": mae, "bucket_summary": bucket_summary}


def feature_importance(model, feature_cols=FEATURE_COLS):
    imp = model.feature_importances_
    return pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values(
        "importance", ascending=False).reset_index(drop=True)


def train(dataset_path, cfg=TRAIN_CFG, save_path="models/trend_model.joblib"):
    raw = pd.read_parquet(dataset_path)

    if {"stop_cushion_atr", "trail_atr_mult", "max_hold_bars"}.issubset(raw.columns):
        label_cfg = {
            "STOP_CUSHION_ATR": float(raw["stop_cushion_atr"].iloc[0]),
            "TRAIL_ATR_MULT": float(raw["trail_atr_mult"].iloc[0]),
            "MAX_HOLD_BARS": int(raw["max_hold_bars"].iloc[0]),
        }
    else:
        print("Warning: dataset has no stamped label config — using hardcoded defaults, verify these match.")
        label_cfg = {"STOP_CUSHION_ATR": 0.25, "TRAIL_ATR_MULT": 2.0, "MAX_HOLD_BARS": 60}

    df, X, y = prep_dataset(raw)
    print(f"Usable trades: {len(df)}  |  mean R: {y.mean():.3f}  |  median R: {y.median():.3f}  |  "
          f"win rate (R>0): {(y > 0).mean():.3f}")
    print(df["exit_reason"].value_counts())

    X_train, y_train, X_test, y_test, cutoff = time_based_split(df, X, y, cfg["TEST_FRACTION"])
    print(f"\nTrain: {len(X_train)} trades (< {cutoff})  |  Test: {len(X_test)} trades (>= {cutoff})")

    print("\n-- Cross-validation on training window --")
    cross_validate(X_train, y_train, cfg)

    model = build_model(cfg)
    model.fit(X_train, y_train)

    print("\n-- Held-out test evaluation --")
    metrics = evaluate(model, X_test, y_test)

    print("\n-- Feature importance --")
    fi = feature_importance(model)
    print(fi.to_string(index=False))

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLS, "label_cfg": label_cfg}, save_path)
    print(f"\nSaved model to {save_path}")

    metrics["feature_importance"] = fi
    return model, FEATURE_COLS, metrics


def main():
    parser = argparse.ArgumentParser(description="Train the trend-following R-multiple regression model.")
    parser.add_argument("--data", default="data/trend_dataset.parquet")
    parser.add_argument("--out", default="models/trend_model.joblib")
    args = parser.parse_args()
    train(args.data, save_path=args.out)


if __name__ == "__main__":
    main()
