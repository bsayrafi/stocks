"""
fvg_live_signal.py
=====================
Run this on a schedule (cron / Task Scheduler) shortly after each hourly
candle closes. Each run:
  1. Fetches recent hourly bars for the ticker universe.
  2. Detects FVGs using the exact same logic as training (via
     detect_fvgs_hourly), so features are guaranteed consistent with what
     the model was trained on.
  3. Scores every FVG still within its hold window with the trained model.
  4. Reports each one's current status (WATCHING / IN_TRADE / TARGET_HIT /
     STOPPED / EXPIRED) and quality score, and flags which ones clear the
     model's threshold.
  5. Appends everything to a CSV log so you have a running record.

This is a SIGNAL GENERATOR, not an execution system — it does not place
any orders. Review its output and decide/execute manually (or wire it
into your own order-placement code separately, deliberately, once you've
watched it enough to trust it).

Setup: same venv as the rest of this project.

One-time (or whenever you retrain):
    Make sure models/fvg_quality_model_hourly.joblib exists.

Run manually:
    python3 fvg_live_signal.py --sector Technology

Schedule with cron (macOS/Linux) — runs 5 min after each hourly candle,
Mon-Fri during market hours (9am-4pm ET; adjust to your machine's local tz):
    5 9-16 * * 1-5 cd /path/to/stocks && .venv/bin/python3 fvg_live_signal.py --sector Technology >> logs/fvg_signal.log 2>&1
"""

import os
import argparse
import requests
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame

from fvg_data_pipeline import add_indicators
from fvg_data_pipeline_hourly import (
    CONFIG as PIPE_CONFIG,
    fetch_bars,
    detect_fvgs_hourly,
    load_tickers_by_sector,
)
from fvg_train_model_hourly import FEATURE_COLS
from macro_calendar import load_calendar, DEFAULT_CALENDAR_PATH

load_dotenv()

DEFAULT_MODEL_PATH = "models/fvg_quality_model_hourly.joblib"
DEFAULT_LOG_PATH = "logs/fvg_signals.csv"
LOOKBACK_DAYS = 120  # generous buffer for 250+ bar indicator warm-up


# ---------------------------------------------------------------------------
# Dummy daily context — htf_* features aren't in FEATURE_COLS anymore, so we
# don't need a real daily fetch. detect_fvgs_hourly still expects a daily_ctx
# argument (for full train/serve code-path parity), so hand it one that never
# filters anything out: every date maps to neutral (0.0) values.
# ---------------------------------------------------------------------------

def build_dummy_daily_ctx():
    dates = pd.date_range(end=pd.Timestamp.today() + pd.Timedelta(days=30), periods=365 * 8).date
    cols = ["htf_price_vs_ema20", "htf_price_vs_ema50", "htf_price_vs_ema200",
            "htf_trend_slope20_atr", "htf_rsi14", "htf_atr_pctile", "htf_trend_dir"]
    return pd.DataFrame(0.0, index=dates, columns=cols)


# ---------------------------------------------------------------------------
# Live status of an FVG as of the most recently fetched bar
# ---------------------------------------------------------------------------

def compute_trade_plan(gap_low, gap_high, atr, direction, cfg):
    entry = (gap_low + gap_high) / 2.0
    if direction == 1:
        stop = entry - cfg["STOP_ATR_MULT"] * atr
    else:
        stop = entry + cfg["STOP_ATR_MULT"] * atr
    risk = abs(entry - stop)
    target = entry + cfg["TARGET_R_MULTIPLE"] * risk if direction == 1 else entry - cfg["TARGET_R_MULTIPLE"] * risk
    return entry, stop, target, risk


def check_midpoint_touch(df, formation_idx, entry, direction):
    """Whether price has actually traded AT the midpoint entry (not just
    the near edge of the gap) at any point since formation — resolves the
    ambiguity where IN_TRADE only confirms a near-edge touch."""
    lows = df["low"].values[formation_idx + 1:]
    highs = df["high"].values[formation_idx + 1:]
    if len(lows) == 0:
        return False
    return bool((lows <= entry).any()) if direction == 1 else bool((highs >= entry).any())


def compute_status(df, formation_idx, gap_low, gap_high, entry, stop, target, direction, cfg):
    """Mirrors fvg_data_pipeline.label_outcomes' walk-forward logic, but
    stops at 'now' (end of available data) instead of requiring a final
    resolution — this is what makes it usable live, mid-trade."""
    n = len(df)
    highs, lows = df["high"].values, df["low"].values
    end = min(formation_idx + 1 + cfg["MAX_HOLD_BARS"], n)
    filled = False
    for j in range(formation_idx + 1, end):
        if not filled:
            if direction == 1 and lows[j] <= gap_high:
                filled = True
            elif direction == 0 and highs[j] >= gap_low:
                filled = True
            if not filled:
                continue
        if direction == 1:
            hit_stop, hit_target = lows[j] <= stop, highs[j] >= target
        else:
            hit_stop, hit_target = highs[j] >= stop, lows[j] <= target
        if hit_stop:
            return "STOPPED"
        if hit_target:
            return "TARGET_HIT"

    bars_elapsed = (n - 1) - formation_idx
    if bars_elapsed >= cfg["MAX_HOLD_BARS"]:
        return "EXPIRED"
    return "IN_TRADE" if filled else "WATCHING"


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def scan(tickers, api_key, api_secret, model_bundle, cfg=PIPE_CONFIG, verbose=True):
    model = model_bundle["model"]
    features = model_bundle["features"]
    label_cfg = model_bundle["label_cfg"]

    try:
        calendar = load_calendar(DEFAULT_CALENDAR_PATH)
    except FileNotFoundError:
        calendar = None
        if verbose:
            print(f"Warning: {DEFAULT_CALENDAR_PATH} not found — macro-event flagging disabled.")

    client = StockHistoricalDataClient(api_key, api_secret)
    dummy_ctx = build_dummy_daily_ctx()
    results = []

    for symbol in tickers:
        bars = fetch_bars(client, symbol, TimeFrame.Hour, LOOKBACK_DAYS / 365, cfg)
        if bars is None or len(bars) < 250:
            if verbose:
                print(f"[{symbol}] insufficient history, skipping")
            continue
        ind = add_indicators(bars, cfg)
        last_close = round(float(ind["close"].iloc[-1]), 2)
        fvgs = detect_fvgs_hourly(ind, dummy_ctx, symbol, cfg, calendar=calendar)
        if fvgs.empty:
            continue

        # keep only FVGs still within their hold window (formed within the
        # last MAX_HOLD_BARS bars of the fetched data) — older ones are stale
        last_idx = len(ind) - 1
        active = fvgs[fvgs["formation_idx"] >= last_idx - label_cfg["MAX_HOLD_BARS"]].copy()
        if active.empty:
            continue

        for _, row in active.iterrows():
            entry, stop, target, risk = compute_trade_plan(
                row["_gap_low"], row["_gap_high"], row["_atr_at_formation"], row["direction"], label_cfg
            )
            status = compute_status(
                ind, int(row["formation_idx"]), row["_gap_low"], row["_gap_high"],
                entry, stop, target, row["direction"], label_cfg
            )
            midpoint_touched = check_midpoint_touch(ind, int(row["formation_idx"]), entry, row["direction"])
            x = pd.DataFrame([{c: row.get(c, np.nan) for c in features}]).fillna(0)
            score = round(model.predict_proba(x)[0, 1] * 100, 1)

            results.append({
                "symbol": symbol,
                "direction": "BULL" if row["direction"] == 1 else "BEAR",
                "formation_time": row["formation_time"],
                "status": status,
                "quality_score": score,
                "macro_event": bool(row.get("near_macro_event", 0)),
                "midpoint_touched": midpoint_touched,
                "gap_low": round(row["_gap_low"], 2),
                "gap_high": round(row["_gap_high"], 2),
                "last_close": last_close,
                "entry": round(entry, 2),
                "stop": round(stop, 2),
                "target": round(target, 2),
                "risk_per_share": round(risk, 2),
                "bars_since_formation": last_idx - int(row["formation_idx"]),
            })

    return pd.DataFrame(results)


def send_ntfy_notification(title, message):
    """Best-effort push notification via ntfy.sh. Reads the topic from the
    NTFY_TOPIC environment variable (set locally in .env, or as a GitHub
    Actions secret when run in CI). Never raises — a failed notification
    shouldn't break the scan."""
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC not set — skipping notification.")
        return
    try:
        resp = requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "default", "Tags": "chart_with_upwards_trend"},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"Notification sent to ntfy.sh/{topic} (status {resp.status_code}).")
    except Exception as e:
        print(f"Notification failed (non-fatal): {e}")


def main():
    parser = argparse.ArgumentParser(description="Scan for live FVG setups and score them.")
    parser.add_argument("--sector", default=PIPE_CONFIG["SECTOR"])
    parser.add_argument("--tickers", nargs="*", default=None, help="Explicit tickers — overrides --sector.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--min-score", type=float, default=None,
                         help="Override the model's saved threshold (0-100 scale).")
    parser.add_argument("--show-all", action="store_true",
                         help="Also print resolved/expired setups, not just actionable ones.")
    parser.add_argument("--exclude-macro", action="store_true",
                         help="Exclude setups formed near a known macro event (FOMC, etc.) from the actionable list.")
    parser.add_argument("--no-notify", action="store_true",
                         help="Disable the ntfy.sh push notification for newly actionable setups.")
    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise SystemExit("Set ALPACA_API_KEY / ALPACA_SECRET_KEY in your .env file.")

    model_bundle = joblib.load(args.model)
    min_score = args.min_score if args.min_score is not None else model_bundle["threshold"] * 100

    tickers = args.tickers if args.tickers else load_tickers_by_sector(args.sector, verbose=False)
    print(f"Scanning {len(tickers)} tickers  |  model threshold: {min_score:.1f}  |  "
          f"label config: {model_bundle['label_cfg']}")

    df = scan(tickers, api_key, api_secret, model_bundle)

    if df.empty:
        print("No active FVGs found this run.")
        return

    df = df.sort_values("quality_score", ascending=False).reset_index(drop=True)
    df["actionable"] = (df["quality_score"] >= min_score) & (df["status"].isin(["WATCHING", "IN_TRADE"]))
    if args.exclude_macro:
        df["actionable"] = df["actionable"] & (~df["macro_event"])

    n_macro = int(df["macro_event"].sum())
    if n_macro > 0:
        print(f"NOTE: {n_macro} setup(s) formed near a known macro event (FOMC, etc.) — "
              f"these move on a different mechanism than the model was trained to recognize. "
              f"{'Excluded from actionable count.' if args.exclude_macro else 'Included below; rerun with --exclude-macro to drop them.'}\n")

    # log everything (append, dedup by symbol+formation_time so reruns don't duplicate)
    log_dir = os.path.dirname(args.log)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    df_to_log = df.copy()
    df_to_log["run_time"] = pd.Timestamp.now(tz="America/New_York")
    if os.path.exists(args.log):
        existing = pd.read_csv(args.log, parse_dates=["formation_time"])
        seen = set(zip(existing["symbol"], existing["formation_time"].astype(str), existing["status"]))
        n_before = len(df_to_log)
        df_to_log = df_to_log[~df_to_log.apply(
            lambda r: (r["symbol"], str(r["formation_time"]), r["status"]) in seen, axis=1)]
        n_new = len(df_to_log)
        if n_new > 0:
            df_to_log.to_csv(args.log, mode="a", header=False, index=False)
        print(f"Logged {n_new} new row(s) to {args.log} "
              f"({n_before - n_new} already recorded from a prior run, unchanged).")
    else:
        df_to_log.to_csv(args.log, index=False)
        print(f"Logged {len(df_to_log)} row(s) to new file {args.log}.")
        n_new = len(df_to_log)

    if not args.no_notify:
        new_actionable = df_to_log[df_to_log["actionable"] & df_to_log["status"].isin(["WATCHING", "IN_TRADE"])]
        if not new_actionable.empty:
            top = new_actionable.sort_values("quality_score", ascending=False).head(5)
            lines = [f"{r.symbol} {r.direction} {r.status} (score {r.quality_score})" for r in top.itertuples()]
            msg = "; ".join(lines)
            if len(new_actionable) > 5:
                msg += f"  (+{len(new_actionable) - 5} more)"
            send_ntfy_notification(f"{len(new_actionable)} new FVG setup(s)", msg)

    shown = df if args.show_all else df[df["status"].isin(["WATCHING", "IN_TRADE"])]
    if shown.empty:
        print("No currently active (WATCHING/IN_TRADE) setups. Use --show-all to see resolved/expired ones too.")
    else:
        cols = ["symbol", "direction", "status", "quality_score", "actionable", "macro_event",
                "gap_low", "gap_high", "last_close", "entry", "stop", "target", "midpoint_touched",
                "bars_since_formation", "formation_time"]
        print(shown[cols].to_string(index=False))

    n_actionable = int(df["actionable"].sum())
    print(f"\n{n_actionable} actionable setup(s) (score >= {min_score:.1f}, status WATCHING or IN_TRADE).")


if __name__ == "__main__":
    main()
