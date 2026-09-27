"""
trend_confirm.py
Scores trend candidates for signs the uptrend is backed by real buying.

Question answered: "This quality stock is trending up. Are large buyers behind it?"

Evidence (each measured against the stock's OWN history):
  - Positive intraday money flow over the last 20 sessions
  - More volume on up-moves than down-moves
  - Pullback days on lighter volume than advance days
  - Consistency: most days show positive money flow (not one big day)
  - Closing above the session VWAP
  - Buying into the last hour of trading
  - Rising average trade size
Penalty: distribution days (down closes on above-normal volume).

Usage:
    from trend_confirm import score_trends
    df = score_trends(["NVDA", "LITE"])
    df = score_trends(trend_list, bars=bars)
    df = score_trends(trend_list, bars=bars, as_of="2026-06-15")  # backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import flow_common as fc

DEFAULT_WEIGHTS = {
    "mf_z": 1.0,            # money flow, 20 sessions
    "up_vol_z": 1.0,        # up-move vs down-move volume, 20 sessions
    "pullback_z": -0.8,     # down-day rvol / up-day rvol (lower is better)
    "consistency_z": 0.7,   # share of days with positive money flow
    "vwap_z": 0.5,          # close vs session VWAP, 10 sessions
    "last_hour_z": 0.5,     # final-hour money flow, 10 sessions
    "trade_size_z": 0.3,    # average trade size, 10 sessions
}
INTERCEPT = -0.5
SCALE = 2.0
WINDOW = 20
SHORT = 10
BASELINE = 60

DISPLAY_COLS = ["rank", "ticker", "score", "signal", "ret_20d_pct", "money_flow_20d",
                "up_vol_pct", "pullback_vol_ratio", "mf_pos_days_pct",
                "vs_vwap_10d_pct", "dist_days", "ext_atr", "flags"]


def _pullback_series(d: pd.DataFrame, window: int) -> pd.Series:
    mp = max(3, window // 4)
    dn = d["rvol"].where(d["ret"] < 0).rolling(window, min_periods=mp).mean()
    up = d["rvol"].where(d["ret"] > 0).rolling(window, min_periods=mp).mean()
    return np.log(dn / up)


def _score_one(t: str, d: pd.DataFrame, weights: dict) -> dict:
    if len(d) < BASELINE + WINDOW + 10:
        raise ValueError(f"insufficient history ({len(d)} sessions)")
    d = d.copy()
    last = d.iloc[-1]
    win = d.tail(WINDOW)
    pull = _pullback_series(d, WINDOW)

    zs = {
        "mf_z": fc.window_z(d["mf_ratio"], WINDOW, BASELINE),
        "up_vol_z": fc.window_z(d["up_vol_frac"], WINDOW, BASELINE),
        "pullback_z": fc.series_z(pull, WINDOW, BASELINE),
        "consistency_z": fc.window_z((d["mf_ratio"] > 0).astype(float), WINDOW, BASELINE),
        "vwap_z": fc.window_z(d["close_vs_vwap"], SHORT, BASELINE),
        "last_hour_z": fc.window_z(d["lh_mf"], SHORT, BASELINE),
        "trade_size_z": fc.window_z(np.log(d["avg_trade_size"]), SHORT, BASELINE),
    }

    dist = win[(win["ret"] < -0.002) & (win["rvol"] > 1.2)]
    penalty = 0.15 * max(0, len(dist) - 2)
    composite, prob = fc.combine(zs, weights, INTERCEPT, SCALE, penalty)

    sma20 = win["close"].mean()
    ext_atr = (last["close"] - sma20) / last["atr"] if last["atr"] > 0 else np.nan
    flags = []
    if len(dist) >= 4:
        flags.append(f"{len(dist)} distribution days in 20")
    if not pd.isna(ext_atr) and ext_atr > 3:
        flags.append("extended >3 ATR above 20d avg")
    if last["rvol"] > 1.5 and last["clv"] < -0.5:
        flags.append("heavy selling into close today")

    score = round(prob * 100, 1) if not pd.isna(prob) else np.nan
    row = {
        "ticker": t,
        "score": score,
        "signal": fc.label(score),
        "as_of": d.index[-1].date(),
        "close": round(float(last["close"]), 2),
        "ret_20d_pct": round((last["close"] / d["close"].iloc[-WINDOW - 1] - 1) * 100, 2),
        "money_flow_20d": round(float(win["mf_ratio"].mean()), 3),
        "up_vol_pct": round(float(win["up_vol_frac"].mean()) * 100, 1),
        "pullback_vol_ratio": round(float(np.exp(pull.iloc[-1])), 2) if not pd.isna(pull.iloc[-1]) else np.nan,
        "mf_pos_days_pct": round(float((win["mf_ratio"] > 0).mean()) * 100, 1),
        "vs_vwap_10d_pct": round(float(d["close_vs_vwap"].tail(SHORT).mean()) * 100, 2),
        "dist_days": len(dist),
        "ext_atr": round(float(ext_atr), 2) if not pd.isna(ext_atr) else np.nan,
        "composite": round(composite, 3) if not pd.isna(composite) else np.nan,
        "flags": "; ".join(flags),
    }
    row.update({k: round(v, 2) if not pd.isna(v) else np.nan for k, v in zs.items()})
    return row


def score_trends(tickers, bars: pd.DataFrame | None = None, client=None, as_of=None,
                 weights: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    """Score and rank trend candidates. Returns a DataFrame sorted by score."""
    weights = weights or DEFAULT_WEIGHTS
    if bars is None:
        bars = fc.fetch_intraday_bars(tickers, client=client, as_of=as_of)
    tables, skipped = fc.daily_tables(bars, tickers, as_of=as_of)
    rows = []
    for t, d in tables.items():
        try:
            rows.append(_score_one(t, d, weights))
        except Exception as e:
            skipped[t] = str(e)
    df = fc.finalize(rows, skipped)
    if verbose:
        fc.print_report(df, "Trend candidates: buying behind the move", DISPLAY_COLS)
    return df


if __name__ == "__main__":
    import sys
    score_trends(sys.argv[1:] or ["NVDA", "LITE"])
