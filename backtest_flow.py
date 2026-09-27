"""
backtest_flow.py
Tests whether the accumulation scores from trend_confirm / dip_confirm
actually predict forward returns.

For each historical evaluation date (every `step` sessions):
  1. Pick eligible tickers with a simple proxy for your trend or dip screen
     (replaceable with your own function via `eligible=`).
  2. Score them exactly as the live scanner would, using only data up to
     that date's close.
  3. Record forward returns: enter at the NEXT session's open, exit at the
     close `h` sessions later, for each horizon h. Also record the benchmark
     ETF's and SPY's return over the same period.

Then it reports:
  - Score buckets: average forward return and win rate per score range
  - Information coefficient (IC): per-date rank correlation between a score
    and forward excess return, averaged over dates, with a t-stat
  - Top-N: the N highest-scored names per date vs the average eligible name
  - Sector filter: whether requiring sector_score >= 30 helps
  - Fitted weights: logistic regression of "beat the benchmark" on the
    feature z-scores, trained on the first 70% of dates and tested on the
    last 30%

Usage:
    from backtest_flow import run_backtest
    res = run_backtest(trend_list, mode="trend", start="2025-09-01", end="2026-09-01")
    res = run_backtest(dip_list, mode="dip", benchmarks=bench_map, save_csv="dips_bt.csv")

    res["records"]  -> one row per (date, ticker) with scores, features, returns
    res["buckets"], res["ic"], res["top_n"], res["sector_filter"], res["fit"]

IMPORTANT caveats (also printed):
  - Survivorship / selection bias: if the ticker list is today's quality
    list, the backtest only contains stocks that are good *today*.
  - Earnings are not excluded, unlike your live dip screen.
  - Horizons longer than `step` overlap between dates, which makes the
    t-stats look stronger than they are.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import flow_common as fc
from benchmarks import MARKET, benchmark_symbols, get_benchmark_map
from dip_confirm import score_dips
from trend_confirm import score_trends

SAVE_EVERY = 20            # symbols per download group (cache saved after each)
SCORE_BUCKETS = [0, 30, 45, 60, 80, 100.01]
BUCKET_LABELS = ["0-30", "30-45", "45-60", "60-80", "80-100"]
SCORE_COLS = ["score", "abs_score", "rel_score", "sector_score"]


# --------------------------------------------------------------------------
# Default candidate filters (proxies for your screens; replace as needed)
# --------------------------------------------------------------------------
def trend_eligible(d: pd.DataFrame, reg_window: int = 20, min_r2: float = 0.5) -> bool:
    """Uptrend proxy: close above its 50-day average, and a linear regression
    of log price over `reg_window` sessions with positive slope and R^2 > min_r2
    (your swing strategy's channel rule)."""
    if len(d) < 60:
        return False
    close = d["close"]
    if close.iloc[-1] <= close.tail(50).mean():
        return False
    y = np.log(close.tail(reg_window).to_numpy())
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0
    return slope > 0 and r2 > min_r2


def dip_eligible(d: pd.DataFrame, min_dip: float = 0.05, max_dip: float = 0.25) -> bool:
    """Dip proxy: close between 5% and 25% below its 20-session high."""
    if len(d) < 60:
        return False
    peak = d["close"].tail(20).max()
    dd = 1 - d["close"].iloc[-1] / peak
    return min_dip <= dd <= max_dip


# --------------------------------------------------------------------------
# Data loading with a disk cache
# --------------------------------------------------------------------------
def fetch_backtest_bars(symbols, start, cache_path: str | None = "bt_bars.pkl",
                        client=None, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """30-min bars from (start - 200 days) to today, cached on disk.

    Reuses the cache if it covers the period; symbols missing from the cache
    are downloaded and added, so a dip run after a trend run only fetches the
    new tickers. Delete the file (or refresh=True) to force a full download.
    """
    symbols = fc.normalize_tickers(symbols)
    need_from = pd.Timestamp(start) - pd.Timedelta(days=200)
    need_to = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=7)
    lookback = (pd.Timestamp(datetime.now().date()) - need_from).days

    cached = None
    if cache_path and os.path.exists(cache_path) and not refresh:
        cached = pd.read_pickle(cache_path)
        dmin, dmax = cached["date"].min(), cached["date"].max()
        if dmin > need_from + pd.Timedelta(days=10) or dmax < need_to:
            if verbose:
                print(f"Cache {cache_path} covers {dmin.date()}..{dmax.date()}, "
                      f"which doesn't cover the period; downloading again.")
            cached = None

    have = set(cached["symbol"].unique()) if cached is not None else set()
    missing = [s for s in symbols if s not in have]
    if missing:
        if verbose:
            print(f"Downloading {len(missing)} symbols, {lookback} days of 30-min bars. "
                  f"This can take several minutes; pauses for Alpaca's rate limit are "
                  f"normal. Progress is saved every {SAVE_EVERY} symbols, so an "
                  f"interrupted download resumes where it stopped.", flush=True)
        client = client or fc.get_client()
        for i in range(0, len(missing), SAVE_EVERY):
            group = missing[i:i + SAVE_EVERY]
            new = fc.fetch_intraday_bars(group, client=client, lookback_days=lookback, verbose=False)
            if len(new):
                cached = new if cached is None else pd.concat([cached, new], ignore_index=True)
                if cache_path:
                    cached.to_pickle(cache_path)
            if verbose:
                print(f"  downloaded {min(i + SAVE_EVERY, len(missing))}/{len(missing)} symbols"
                      + (f", saved to {cache_path}" if cache_path else ""), flush=True)
        if cached is None:
            raise ValueError("No bars were returned for any symbol.")
    elif verbose:
        print(f"Using cached bars from {cache_path}")
    return cached[cached["symbol"].isin(symbols)].reset_index(drop=True)


# --------------------------------------------------------------------------
# Forward returns
# --------------------------------------------------------------------------
def _fwd_return(d: pd.DataFrame, date, h: int) -> float:
    """Enter at the next session's open, exit at the close h sessions later."""
    if d is None:
        return np.nan
    pos = d.index.searchsorted(date, side="right")   # first session after date
    exit_pos = pos + h - 1
    if pos >= len(d) or exit_pos >= len(d):
        return np.nan
    entry = d["open"].iloc[pos]
    if not entry or entry <= 0:
        return np.nan
    return (d["close"].iloc[exit_pos] / entry - 1) * 100


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def _spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return np.nan
    ra, rb = a[m].rank(), b[m].rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _ic_stats(rec: pd.DataFrame, col: str, target: str) -> dict:
    per_date = rec.groupby("date").apply(lambda g: _spearman(g[col], g[target])).dropna()
    n = len(per_date)
    mean = per_date.mean() if n else np.nan
    t = mean / (per_date.std(ddof=1) / math.sqrt(n)) if n > 2 and per_date.std(ddof=1) > 0 else np.nan
    return {"ic_mean": round(mean, 4) if n else np.nan, "ic_t": round(t, 2) if not pd.isna(t) else np.nan,
            "pct_dates_positive": round((per_date > 0).mean() * 100, 1) if n else np.nan, "n_dates": n}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 5.0, iters: int = 100) -> np.ndarray:
    """L2-regularized logistic regression (Newton / IRLS). Intercept not penalized."""
    X1 = np.c_[np.ones(len(X)), X]
    w = np.zeros(X1.shape[1])
    pen = np.r_[0.0, np.full(X.shape[1], l2)]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        W = p * (1 - p)
        H = X1.T @ (X1 * W[:, None]) + np.diag(pen)
        g = X1.T @ (y - p) - pen * w
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def run_backtest(tickers, mode: str = "trend", start=None, end=None,
                 benchmarks: dict | None = None, step: int = 5,
                 horizons=(5, 10, 15), fit_horizon: int = 10, top_n: int = 5,
                 eligible=None, bars: pd.DataFrame | None = None, client=None,
                 verbose: bool = True, save_csv: str | None = None,
                 cache_path: str | None = "bt_bars.pkl") -> dict:
    """Backtest the trend or dip accumulation score.

    start/end: range of evaluation dates (default: the year ending ~1 month ago).
    step:      sessions between evaluation dates (5 = weekly).
    eligible:  function(daily_df_up_to_date) -> bool; default is the trend or
               dip proxy above. Pass your own to mirror your real screens.
    bars:      pre-fetched 30-min bars covering start-200 days .. today
               (otherwise loaded from `cache_path`, downloading what's missing).
    """
    assert mode in ("trend", "dip")
    score_fn = score_trends if mode == "trend" else score_dips
    eligible = eligible or (trend_eligible if mode == "trend" else dip_eligible)
    tickers = fc.normalize_tickers(tickers)
    horizons = tuple(sorted(set(horizons) | {fit_horizon}))
    hmax = max(horizons)

    end_ts = pd.Timestamp(end) if end else pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=int(hmax * 1.6) + 3)
    start_ts = pd.Timestamp(start) if start else end_ts - pd.Timedelta(days=365)

    if benchmarks is None:
        benchmarks = get_benchmark_map(tickers)
    bmap = {t: benchmarks.get(t, MARKET) for t in tickers}
    symbols = sorted(set(tickers) | set(benchmark_symbols(bmap)))

    if bars is None:
        bars = fetch_backtest_bars(symbols, start_ts, cache_path=cache_path,
                                   client=client, verbose=verbose)
    if verbose:
        print("Building daily tables...")
    daily = fc.build_all_daily(bars)

    ref = daily.get(MARKET)
    if ref is None:
        ref = max(daily.values(), key=len)
    sessions = ref.index
    last_ok = len(sessions) - hmax - 1
    eval_dates = [dt for i, dt in enumerate(sessions)
                  if start_ts <= dt <= end_ts and i <= last_ok][::step]
    if not eval_dates:
        raise ValueError("No evaluation dates: check start/end and that data covers them.")
    if verbose:
        print(f"Evaluating {len(eval_dates)} dates from {eval_dates[0].date()} "
              f"to {eval_dates[-1].date()}, horizons {horizons} sessions...")

    records = []
    for k, dt in enumerate(eval_dates):
        elig = []
        for t in tickers:
            d = daily.get(t)
            if d is None:
                continue
            sub = d[d.index <= dt]
            if len(sub) and sub.index[-1] == dt and eligible(sub):
                elig.append(t)
        if not elig:
            continue
        scored = score_fn(elig, benchmarks=bmap, as_of=dt, daily=daily, verbose=False)
        if len(scored) == 0:
            continue
        for _, r in scored.iterrows():
            rec = r.to_dict()
            rec["date"] = dt
            for h in horizons:
                ret = _fwd_return(daily.get(r["ticker"]), dt, h)
                bret = _fwd_return(daily.get(r["benchmark"]), dt, h)
                mret = _fwd_return(daily.get(MARKET), dt, h)
                rec[f"fwd_{h}"] = ret
                rec[f"xs_bench_{h}"] = ret - bret if not pd.isna(bret) else np.nan
                rec[f"xs_spy_{h}"] = ret - mret if not pd.isna(mret) else np.nan
            records.append(rec)
        if verbose and (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(eval_dates)} dates done")

    rec = pd.DataFrame(records)
    if len(rec) == 0:
        raise ValueError("No eligible (date, ticker) pairs; loosen the eligibility filter.")
    if save_csv:
        rec.to_csv(save_csv, index=False)

    res = {"records": rec, "mode": mode, "horizons": horizons}
    res["buckets"] = _bucket_table(rec, horizons)
    res["ic"] = _ic_table(rec, horizons)
    res["top_n"] = _top_n_table(rec, horizons, top_n)
    res["sector_filter"] = _sector_filter_table(rec, horizons)
    res["fit"] = _fit_weights(rec, fit_horizon)
    if verbose:
        print_backtest_report(res)
    return res


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def _bucket_table(rec: pd.DataFrame, horizons) -> pd.DataFrame:
    r = rec.copy()
    r["bucket"] = pd.cut(r["score"], SCORE_BUCKETS, labels=BUCKET_LABELS, right=False)
    rows = []
    for b, g in r.groupby("bucket", observed=False):
        row = {"score_bucket": b, "n": len(g)}
        for h in horizons:
            row[f"avg_ret_{h}"] = round(g[f"fwd_{h}"].mean(), 2)
            row[f"avg_xs_bench_{h}"] = round(g[f"xs_bench_{h}"].mean(), 2)
            row[f"win_{h}_%"] = round((g[f"fwd_{h}"] > 0).mean() * 100, 1) if len(g) else np.nan
            row[f"beat_bench_{h}_%"] = round((g[f"xs_bench_{h}"] > 0).mean() * 100, 1) if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _ic_table(rec: pd.DataFrame, horizons) -> pd.DataFrame:
    rows = []
    for col in SCORE_COLS:
        if col not in rec or rec[col].notna().sum() < 10:
            continue
        for h in horizons:
            st = _ic_stats(rec, col, f"xs_bench_{h}")
            rows.append({"score_col": col, "horizon": h, **st})
    return pd.DataFrame(rows)


def _top_n_table(rec: pd.DataFrame, horizons, n: int) -> pd.DataFrame:
    rows = []
    for h in horizons:
        top = rec.sort_values("score", ascending=False).groupby("date").head(n)
        top_by_date = top.groupby("date")[f"fwd_{h}"].mean()
        all_by_date = rec.groupby("date")[f"fwd_{h}"].mean()
        diff = (top_by_date - all_by_date).dropna()
        t = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 2 and diff.std(ddof=1) > 0 else np.nan
        rows.append({"horizon": h, f"top{n}_avg_ret": round(top_by_date.mean(), 2),
                     "all_eligible_avg_ret": round(all_by_date.mean(), 2),
                     "difference": round(diff.mean(), 2), "t_stat": round(t, 2) if not pd.isna(t) else np.nan,
                     "pct_dates_top_better": round((diff > 0).mean() * 100, 1), "n_dates": len(diff)})
    return pd.DataFrame(rows)


def _sector_filter_table(rec: pd.DataFrame, horizons, min_score: float = 55,
                         min_sector: float = 30) -> pd.DataFrame:
    sig = rec[rec["score"] >= min_score]
    groups = {
        f"score>={min_score:g} (all)": sig,
        f"score>={min_score:g} & sector>={min_sector:g}": sig[sig["sector_score"] >= min_sector],
        f"score>={min_score:g} & sector<{min_sector:g}": sig[sig["sector_score"] < min_sector],
    }
    rows = []
    for name, g in groups.items():
        row = {"group": name, "n": len(g)}
        for h in horizons:
            row[f"avg_ret_{h}"] = round(g[f"fwd_{h}"].mean(), 2) if len(g) else np.nan
            row[f"win_{h}_%"] = round((g[f"fwd_{h}"] > 0).mean() * 100, 1) if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_weights(rec: pd.DataFrame, h: int, train_frac: float = 0.7) -> dict:
    """Fit P(beat benchmark over h sessions) on relative and own z-scores,
    train on early dates, evaluate on later dates."""
    target = f"xs_bench_{h}"
    # Relative z-scores plus the stock's own z-scores. Features that are not
    # benchmark-relative (trade size) appear only once, as own_*.
    feats = [c for c in rec.columns if c.endswith("_z") and not c.startswith("own_")
             and c not in fc.NON_RELATIVE_FEATURES]
    feats += [c for c in rec.columns if c.startswith("own_") and c.endswith("_z")]
    r = rec.dropna(subset=[target]).copy()
    if len(r) < 200:
        return {"note": f"only {len(r)} samples; need ~200+ to fit weights reliably"}
    dates = sorted(r["date"].unique())
    split = dates[int(len(dates) * train_frac)]
    tr, te = r[r["date"] < split], r[r["date"] >= split]
    Xtr = tr[feats].fillna(0).to_numpy(float)
    Xte = te[feats].fillna(0).to_numpy(float)
    ytr = (tr[target] > 0).astype(float).to_numpy()
    yte = (te[target] > 0).astype(float).to_numpy()
    w = _fit_logistic(Xtr, ytr)
    p_te = 1 / (1 + np.exp(-(np.c_[np.ones(len(Xte)), Xte] @ w)))
    te = te.assign(fitted_prob=p_te)
    return {
        "horizon": h,
        "train_dates": f"{pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(split).date()}",
        "n_train": len(tr), "n_test": len(te),
        "base_rate_test_%": round(yte.mean() * 100, 1),
        "auc_test_fitted": round(_auc(yte, p_te), 3),
        "auc_test_current_score": round(_auc(yte, te["score"].fillna(0).to_numpy()), 3),
        "ic_test_fitted": _ic_stats(te, "fitted_prob", target)["ic_mean"],
        "ic_test_current_score": _ic_stats(te, "score", target)["ic_mean"],
        "coefficients": pd.Series(w[1:], index=feats).round(3).sort_values(),
        "intercept": round(w[0], 3),
    }


def print_backtest_report(res: dict) -> None:
    rec = res["records"]
    pd_opts = ("display.max_columns", None, "display.width", 220)
    print(f"\n========== Backtest: {res['mode']} ==========")
    print(f"{len(rec)} scored (date, ticker) pairs over {rec['date'].nunique()} dates, "
          f"{rec['ticker'].nunique()} tickers")
    with pd.option_context(*pd_opts):
        print("\n-- Score buckets (fwd = next-open to close after h sessions, %) --")
        print(res["buckets"].to_string(index=False))
        print("\n-- Information coefficient vs benchmark-excess return --")
        print("   (IC 0.02-0.05 is typical of a useful signal; |t| > 2 suggests it's not noise)")
        print(res["ic"].to_string(index=False))
        print("\n-- Top-N by score vs all eligible names --")
        print(res["top_n"].to_string(index=False))
        print("\n-- Sector filter --")
        print(res["sector_filter"].to_string(index=False))
    fit = res["fit"]
    print("\n-- Fitted weights (logistic, beat benchmark) --")
    if "note" in fit:
        print("  " + fit["note"])
    else:
        for k in ["horizon", "train_dates", "n_train", "n_test", "base_rate_test_%",
                  "auc_test_fitted", "auc_test_current_score", "ic_test_fitted", "ic_test_current_score"]:
            print(f"  {k}: {fit[k]}")
        print("  (AUC 0.5 = no skill; compare fitted vs current on the TEST period)")
        print("  coefficients:")
        print(fit["coefficients"].to_string())
    print("\nCaveats: ticker list chosen today (survivorship bias); earnings not excluded; "
          "horizons longer than the step overlap, which inflates t-stats.")
