"""
trend_confirm.py
Scores trend candidates for signs the uptrend is backed by real buying,
relative to each stock's sector/industry benchmark.

Question answered: "This quality stock is trending up. Are large buyers
behind it, more than behind its sector?"

Evidence (each measured against the stock's OWN history, then compared with
the benchmark ETF's deviation from ITS own history):
  - Positive intraday money flow over the last 20 sessions
  - More volume on up-moves than down-moves
  - Pullback days on lighter volume than advance days
  - Consistency: most days show positive money flow (not one big day)
  - Closing above the session VWAP
  - Buying into the last hour of trading
  - Rising average trade size
Penalty: distribution days (down closes on above-normal volume).

Scores per ticker (0-100, uncalibrated; ~38 = normal activity):
  score         sqrt(abs_score * rel_score)  (used for ranking)
  abs_score     stock vs its own history
  rel_score     stock vs its benchmark ETF (accumulation beyond the sector)
  sector_score  benchmark ETF vs SPY (sector rotation)

Usage:
    from benchmarks import build_benchmark_map
    from trend_confirm import score_trends
    bench_map = build_benchmark_map(finviz_df)
    df = score_trends(trend_list, bars=bars, benchmarks=bench_map)
    df = score_trends(trend_list, bars=bars, benchmarks=bench_map, as_of="2026-06-15")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import flow_common as fc
from benchmarks import MARKET, benchmark_symbols

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
MIN_SESSIONS = BASELINE + WINDOW + 10

DISPLAY_COLS = ["rank", "ticker", "benchmark", "score", "signal", "abs_score",
                "rel_score", "sector_score", "ret_20d_pct", "rel_ret_20d_pct", "money_flow_20d",
                "up_vol_pct", "pullback_vol_ratio", "mf_pos_days_pct", "dist_days",
                "ext_atr", "flags"]


def _pullback_series(d: pd.DataFrame, window: int) -> pd.Series:
    mp = max(3, window // 4)
    dn = d["rvol"].where(d["ret"] < 0).rolling(window, min_periods=mp).mean()
    up = d["rvol"].where(d["ret"] > 0).rolling(window, min_periods=mp).mean()
    return np.log(dn / up)


def _features(d: pd.DataFrame) -> dict:
    """Feature z-scores of one symbol against its own history."""
    return {
        "mf_z": fc.window_z(d["mf_ratio"], WINDOW, BASELINE),
        "up_vol_z": fc.window_z(d["up_vol_frac"], WINDOW, BASELINE),
        "pullback_z": fc.series_z(_pullback_series(d, WINDOW), WINDOW, BASELINE),
        "consistency_z": fc.window_z((d["mf_ratio"] > 0).astype(float), WINDOW, BASELINE),
        "vwap_z": fc.window_z(d["close_vs_vwap"], SHORT, BASELINE),
        "last_hour_z": fc.window_z(d["lh_mf"], SHORT, BASELINE),
        "trade_size_z": fc.window_z(np.log(d["avg_trade_size"]), SHORT, BASELINE),
    }


def _ret(d: pd.DataFrame, n: int) -> float:
    if len(d) <= n:
        return np.nan
    return (d["close"].iloc[-1] / d["close"].iloc[-n - 1] - 1) * 100


def _score_one(t, d, bench, bench_zs, bench_d, sector_zs, weights) -> dict:
    if len(d) < MIN_SESSIONS:
        raise ValueError(f"insufficient history ({len(d)} sessions)")
    last = d.iloc[-1]
    win = d.tail(WINDOW)

    own = _features(d)
    rel = fc.relative_zs(own, bench_zs)
    own = fc.clip_zs(own)

    # rvol is NaN on quarterly expiration days, so they never count here
    dist = win[(win["ret"] < -0.002) & (win["rvol"] > 1.2)]
    penalty = 0.15 * max(0, len(dist) - 2)
    composite, p_rel = fc.combine(rel, weights, INTERCEPT, SCALE, penalty)
    abs_comp, p_abs = fc.combine(own, weights, INTERCEPT, SCALE, penalty)
    _, p_sec = fc.combine(sector_zs, weights, INTERCEPT, SCALE) if sector_zs else (np.nan, np.nan)

    flags = []
    if pd.isna(p_rel):
        flags.append(f"benchmark {bench} unavailable: absolute score used")
        p_rel, composite = p_abs, abs_comp
    sma20 = win["close"].mean()
    ext_atr = (last["close"] - sma20) / last["atr"] if last["atr"] > 0 else np.nan
    if len(dist) >= 4:
        flags.append(f"{len(dist)} distribution days in 20")
    if not pd.isna(ext_atr) and ext_atr > 3:
        flags.append("extended >3 ATR above 20d avg")
    if last["rvol"] > 1.5 and last["clv"] < -0.5:
        flags.append("heavy selling into close today")

    abs_score = fc.to_score(p_abs)
    rel_score = fc.to_score(p_rel)
    score = fc.blend(abs_score, rel_score)
    pull = _pullback_series(d, WINDOW).iloc[-1]
    ret20 = _ret(d, WINDOW)
    bret20 = _ret(bench_d, WINDOW) if bench_d is not None else np.nan
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
        "ret_20d_pct": round(ret20, 2),
        "rel_ret_20d_pct": round(ret20 - bret20, 2) if not pd.isna(bret20) else np.nan,
        "money_flow_20d": round(float(win["mf_ratio"].mean()), 3),
        "up_vol_pct": round(float(win["up_vol_frac"].mean()) * 100, 1),
        "pullback_vol_ratio": round(float(np.exp(pull)), 2) if not pd.isna(pull) else np.nan,
        "mf_pos_days_pct": round(float((win["mf_ratio"] > 0).mean()) * 100, 1),
        "vs_vwap_10d_pct": round(float(d["close_vs_vwap"].tail(SHORT).mean()) * 100, 2),
        "dist_days": len(dist),
        "ext_atr": round(float(ext_atr), 2) if not pd.isna(ext_atr) else np.nan,
        "composite": round(composite, 3) if not pd.isna(composite) else np.nan,
        "flags": "; ".join(flags),
    }
    row.update(fc.round_zs(rel))            # relative z-scores (drive `score`)
    row.update(fc.round_zs(own, "own_"))    # stock-only z-scores
    return row


def score_trends(tickers, bars: pd.DataFrame | None = None, benchmarks: dict | None = None,
                 client=None, as_of=None, weights: dict | None = None,
                 verbose: bool = True, daily: dict | None = None) -> pd.DataFrame:
    """Score and rank trend candidates. Returns a DataFrame sorted by `score`.

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
    etf_d = {e: tables.get(e) for e in etfs}
    etf_zs = {e: (_features(d) if d is not None and len(d) >= MIN_SESSIONS else None)
              for e, d in etf_d.items()}

    rows, skipped = [], {t: r for t, r in missing.items() if t in tickers}
    for t in tickers:
        if t not in tables:
            continue
        b = bmap[t]
        sector = fc.relative_zs(etf_zs[b], etf_zs.get(MARKET), etf_vs_etf=True) if b != MARKET and etf_zs.get(b) else None
        try:
            rows.append(_score_one(t, tables[t], b, etf_zs.get(b), etf_d.get(b), sector, weights))
        except Exception as e:
            skipped[t] = str(e)
    df = fc.finalize(rows, skipped)
    if verbose:
        fc.print_report(df, "Trend candidates: buying behind the move (vs benchmark)", DISPLAY_COLS)
    return df


if __name__ == "__main__":
    import sys
    score_trends(sys.argv[1:] or ["NVDA", "LITE"])
