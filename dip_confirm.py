"""
dip_confirm.py
Scores dip candidates for signs that large buyers are absorbing the selling.

Question answered: "This quality stock has pulled back. Is someone big buying it?"

Evidence of absorption (each measured against the stock's OWN history):
  - Intraday money flow holding up while price falls
  - High-volume days closing strong rather than at the lows
  - More volume on up-moves than down-moves inside the session
  - Recent closes pulling away from the daily lows
  - Buying into the last hour of trading
  - Closing above the session VWAP
  - Rising average trade size (larger participants)
Penalty: heavy-volume days that close near the low (distribution).

Usage:
    from dip_confirm import score_dips
    df = score_dips(["MU", "ORCL", "GM"])          # fetches data itself
    df = score_dips(dip_list, bars=bars)            # reuse pre-fetched bars
    df = score_dips(dip_list, bars=bars, as_of="2026-06-15")   # backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import flow_common as fc

# Weights are starting points, not fitted values. Refit after backtesting.
DEFAULT_WEIGHTS = {
    "mf_z": 1.0,            # intraday money flow over the dip
    "vol_confirm_z": 1.0,   # volume-weighted close strength over the dip
    "up_vol_z": 0.6,        # up-move vs down-move volume over the dip
    "clv_recent_z": 0.7,    # where the last 3 closes sit in the daily range
    "last_hour_z": 0.6,     # money flow in the final hour, last 3 days
    "vwap_z": 0.5,          # close vs session VWAP, last 3 days
    "trade_size_z": 0.3,    # average trade size, last 5 days
}
INTERCEPT = -0.5            # neutral evidence -> ~38
SCALE = 2.0
PEAK_LOOKBACK = 20          # sessions to search for the pre-dip high
MIN_WINDOW, MAX_WINDOW = 3, 10
RECENT = 3
BASELINE = 60

DISPLAY_COLS = ["rank", "ticker", "score", "signal", "dip_pct", "dip_atr",
                "days_since_peak", "money_flow", "up_vol_pct", "close_pos_3d",
                "rvol_3d", "last_hr_mf", "vs_vwap_3d_pct", "flags"]


def _score_one(t: str, d: pd.DataFrame, weights: dict) -> dict:
    if len(d) < BASELINE + MAX_WINDOW + 10:
        raise ValueError(f"insufficient history ({len(d)} sessions)")
    d = d.copy()
    d["vol_confirm"] = d["clv"] * d["rvol"]          # big volume + strong close = +
    last = d.iloc[-1]

    recent = d.tail(PEAK_LOOKBACK)
    peak_date = recent["close"].idxmax()
    peak_close = recent.loc[peak_date, "close"]
    days_since_peak = int((d.index > peak_date).sum())
    w = int(np.clip(days_since_peak, MIN_WINDOW, MAX_WINDOW))
    win = d.tail(w)

    zs = {
        "mf_z": fc.window_z(d["mf_ratio"], w, BASELINE),
        "vol_confirm_z": fc.window_z(d["vol_confirm"], w, BASELINE),
        "up_vol_z": fc.window_z(d["up_vol_frac"], w, BASELINE),
        "clv_recent_z": fc.window_z(d["clv"], RECENT, BASELINE),
        "last_hour_z": fc.window_z(d["lh_mf"], RECENT, BASELINE),
        "vwap_z": fc.window_z(d["close_vs_vwap"], RECENT, BASELINE),
        "trade_size_z": fc.window_z(np.log(d["avg_trade_size"]), 5, BASELINE),
    }

    heavy = win[(win["ret"] < 0) & (win["rvol"] > 1.5) & (win["clv"] < -0.3)]
    penalty = 0.2 * min(len(heavy), 3)
    composite, prob = fc.combine(zs, weights, INTERCEPT, SCALE, penalty)

    flags = []
    if days_since_peak == 0:
        flags.append("no dip: at 20d high")
    if last["rvol"] > 1.5 and last["clv"] < -0.5:
        flags.append("heavy selling into close today")
    if len(heavy) >= 2:
        flags.append(f"{len(heavy)} distribution days in dip")

    score = round(prob * 100, 1) if not pd.isna(prob) else np.nan
    row = {
        "ticker": t,
        "score": score,
        "signal": fc.label(score),
        "as_of": d.index[-1].date(),
        "close": round(float(last["close"]), 2),
        "dip_pct": round((last["close"] / peak_close - 1) * 100, 2),
        "dip_atr": round((peak_close - last["close"]) / last["atr"], 2) if last["atr"] > 0 else np.nan,
        "days_since_peak": days_since_peak,
        "money_flow": round(float(win["mf_ratio"].mean()), 3),
        "up_vol_pct": round(float(win["up_vol_frac"].mean()) * 100, 1),
        "close_pos_3d": round(float(d["clv"].tail(RECENT).mean()), 2),
        "rvol_3d": round(float(d["rvol"].tail(RECENT).mean()), 2),
        "last_hr_mf": round(float(d["lh_mf"].tail(RECENT).mean()), 3),
        "vs_vwap_3d_pct": round(float(d["close_vs_vwap"].tail(RECENT).mean()) * 100, 2),
        "composite": round(composite, 3) if not pd.isna(composite) else np.nan,
        "flags": "; ".join(flags),
    }
    row.update({k: round(v, 2) if not pd.isna(v) else np.nan for k, v in zs.items()})
    return row


def score_dips(tickers, bars: pd.DataFrame | None = None, client=None, as_of=None,
               weights: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    """Score and rank dip candidates. Returns a DataFrame sorted by score.

    Columns ending in _z are the per-feature z-scores (useful for fitting
    weights in a backtest). df.attrs["skipped"] lists tickers not scored.
    """
    weights = weights or DEFAULT_WEIGHTS
    if bars is None:
        bars = fc.fetch_intraday_bars(tickers, client=client, as_of=as_of)
    tables, skipped = fc.daily_tables(bars, tickers, as_of=as_of)
    rows = []
    for t, d in tables.items():
        try:
            rows.append(_score_one(t, d, weights))
        except Exception as e:  # keep scoring the rest
            skipped[t] = str(e)
    df = fc.finalize(rows, skipped)
    if verbose:
        fc.print_report(df, "Dip candidates: absorption / accumulation", DISPLAY_COLS)
    return df


if __name__ == "__main__":
    import sys
    score_dips(sys.argv[1:] or ["MU", "ORCL", "GM"])
