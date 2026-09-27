"""
dip_confirm.py
Scores dip candidates for signs that large buyers are absorbing the selling,
relative to each stock's sector/industry benchmark.

Question answered: "This quality stock has pulled back. Is someone big buying
it, more than they're buying its sector?"

Evidence of absorption (each measured against the stock's OWN history, then
compared with the benchmark ETF's deviation from ITS own history over the
same window):
  - Intraday money flow holding up while price falls
  - High-volume days closing strong rather than at the lows
  - More volume on up-moves than down-moves inside the session
  - Recent closes pulling away from the daily lows
  - Buying into the last hour of trading
  - Closing above the session VWAP
  - Rising average trade size (larger participants)
Penalty: heavy-volume days that close near the low (distribution).

Scores per ticker (0-100, uncalibrated; ~38 = normal activity):
  score         sqrt(abs_score * rel_score)  (used for ranking)
  abs_score     stock vs its own history
  rel_score     stock vs its benchmark ETF (accumulation beyond the sector)
  sector_score  benchmark ETF vs SPY over the same window

Usage:
    from benchmarks import build_benchmark_map
    from dip_confirm import score_dips
    bench_map = build_benchmark_map(finviz_df)
    df = score_dips(dip_list, bars=bars, benchmarks=bench_map)
    df = score_dips(dip_list, bars=bars, benchmarks=bench_map, as_of="2026-06-15")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import flow_common as fc
from benchmarks import MARKET, benchmark_symbols

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
MIN_SESSIONS = BASELINE + MAX_WINDOW + 10

DISPLAY_COLS = ["rank", "ticker", "benchmark", "score", "signal", "abs_score",
                "rel_score", "sector_score", "dip_pct", "bench_dip_pct", "dip_atr", "days_since_peak",
                "money_flow", "up_vol_pct", "close_pos_3d", "rvol_3d", "last_hr_mf", "flags"]


def _prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["vol_confirm"] = d["clv"] * d["rvol"]      # big volume + strong close = +
    return d


def _dip_window(d: pd.DataFrame):
    recent = d.tail(PEAK_LOOKBACK)
    peak_date = recent["close"].idxmax()
    days_since_peak = int((d.index > peak_date).sum())
    w = int(np.clip(days_since_peak, MIN_WINDOW, MAX_WINDOW))
    return peak_date, days_since_peak, w


def _features(d: pd.DataFrame, w: int) -> dict:
    """Feature z-scores of one symbol against its own history, dip window w."""
    return {
        "mf_z": fc.window_z(d["mf_ratio"], w, BASELINE),
        "vol_confirm_z": fc.window_z(d["vol_confirm"], w, BASELINE),
        "up_vol_z": fc.window_z(d["up_vol_frac"], w, BASELINE),
        "clv_recent_z": fc.window_z(d["clv"], RECENT, BASELINE),
        "last_hour_z": fc.window_z(d["lh_mf"], RECENT, BASELINE),
        "vwap_z": fc.window_z(d["close_vs_vwap"], RECENT, BASELINE),
        "trade_size_z": fc.window_z(np.log(d["avg_trade_size"]), 5, BASELINE),
    }


def score_dips(tickers, bars: pd.DataFrame | None = None, benchmarks: dict | None = None,
               client=None, as_of=None, weights: dict | None = None,
               verbose: bool = True, daily: dict | None = None) -> pd.DataFrame:
    """Score and rank dip candidates. Returns a DataFrame sorted by `score`.

    benchmarks: {ticker: ETF} from benchmarks.build_benchmark_map(). Tickers
    not in it are compared with SPY. Benchmark ETFs missing from `bars` are
    fetched automatically.

    daily: precomputed {symbol: daily table} from fc.build_all_daily(), used by
    the backtest so tables aren't rebuilt for every historical date.
    """
    weights = weights or DEFAULT_WEIGHTS
    tickers = fc.normalize_tickers(tickers)
    bmap = {t: (benchmarks or {}).get(t, MARKET) for t in tickers}
    etfs = benchmark_symbols(bmap)

    if daily is None:
        if bars is None:
            bars = fc.fetch_intraday_bars(tickers + etfs, client=client, as_of=as_of)
        else:
            try:
                bars = fc.ensure_symbols(bars, etfs, client=client, as_of=as_of)
            except Exception as e:
                print(f"Could not fetch benchmark ETFs ({e}); continuing with available data.")

    tables, missing = fc.get_tables(bars, tickers + etfs, as_of=as_of, daily=daily)
    etf_d = {e: _prep(d) for e, d in tables.items()
             if e in etfs and d is not None and len(d) >= MIN_SESSIONS}
    cache: dict = {}

    def etf_features(e, w):
        if e not in etf_d:
            return None
        if (e, w) not in cache:
            cache[(e, w)] = _features(etf_d[e], w)
        return cache[(e, w)]

    rows, skipped = [], {t: r for t, r in missing.items() if t in tickers}
    for t in tickers:
        if t not in tables:
            continue
        try:
            d = tables[t]
            if len(d) < MIN_SESSIONS:
                raise ValueError(f"insufficient history ({len(d)} sessions)")
            rows.append(_score_one(t, _prep(d), bmap[t], etf_features, etf_d, weights))
        except Exception as e:
            skipped[t] = str(e)
    df = fc.finalize(rows, skipped)
    if verbose:
        fc.print_report(df, "Dip candidates: absorption / accumulation (vs benchmark)", DISPLAY_COLS)
    return df


def _score_one(t, d, bench, etf_features, etf_d, weights) -> dict:
    last = d.iloc[-1]
    peak_date, days_since_peak, w = _dip_window(d)
    peak_close = d.loc[peak_date, "close"]
    win = d.tail(w)

    own = _features(d, w)
    bench_zs = etf_features(bench, w)
    rel = fc.relative_zs(own, bench_zs)
    own = fc.clip_zs(own)
    sector = None
    if bench != MARKET and bench_zs:
        sector = fc.relative_zs(bench_zs, etf_features(MARKET, w), etf_vs_etf=True)

    # rvol is NaN on quarterly expiration days, so they never count here
    heavy = win[(win["ret"] < 0) & (win["rvol"] > 1.5) & (win["clv"] < -0.3)]
    penalty = 0.2 * min(len(heavy), 3)
    composite, p_rel = fc.combine(rel, weights, INTERCEPT, SCALE, penalty)
    abs_comp, p_abs = fc.combine(own, weights, INTERCEPT, SCALE, penalty)
    _, p_sec = fc.combine(sector, weights, INTERCEPT, SCALE) if sector else (np.nan, np.nan)

    flags = []
    if pd.isna(p_rel):
        flags.append(f"benchmark {bench} unavailable: absolute score used")
        p_rel, composite = p_abs, abs_comp
    if days_since_peak == 0:
        flags.append("no dip: at 20d high")
    if last["rvol"] > 1.5 and last["clv"] < -0.5:
        flags.append("heavy selling into close today")
    if len(heavy) >= 2:
        flags.append(f"{len(heavy)} distribution days in dip")

    # How much the benchmark fell over the same sessions (sector-wide dip?)
    bench_dip = np.nan
    bd = etf_d.get(bench)
    if bd is not None and days_since_peak > 0:
        bd = bd[bd.index <= d.index[-1]]
        if len(bd) > days_since_peak:
            bench_dip = (bd["close"].iloc[-1] / bd["close"].iloc[-days_since_peak - 1] - 1) * 100

    abs_score = fc.to_score(p_abs)
    rel_score = fc.to_score(p_rel)
    score = fc.blend(abs_score, rel_score)
    row = {
        "ticker": t,
        "benchmark": bench,
        "score": score,
        "signal": fc.label(score),
        "abs_score": abs_score,
        "rel_score": rel_score,
        "sector_score": fc.to_score(p_sec),
        "as_of": d.index[-1].date(),
        "close": round(float(last["close"]), 2),
        "dip_pct": round((last["close"] / peak_close - 1) * 100, 2),
        "bench_dip_pct": round(bench_dip, 2) if not pd.isna(bench_dip) else np.nan,
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
    row.update(fc.round_zs(rel))
    row.update(fc.round_zs(own, "own_"))
    return row


if __name__ == "__main__":
    import sys
    score_dips(sys.argv[1:] or ["MU", "ORCL", "GM"])
