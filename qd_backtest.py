"""
qd_backtest.py -- Quality Dip (short-term reversal) backtest
=============================================================
Hypothesis (fixed BEFORE testing, to avoid curve-fitting):
  In a long-term uptrend, a sharp short-term sell-off in a large, liquid stock
  tends to mean-revert within days. Buy the dip, sell the bounce.

Rules (variant A = base):
  Market : SPY close > SMA200
  Trend  : stock close > SMA200 (the long-term uptrend is intact)
  Dip    : RSI(2) < 10 at the close  (score = 100 - RSI2: deepest dip first)
  Entry  : next session's open            (--entry vwap: first 15m VWAP reclaim within 2 sessions)
  Exit   : first close above SMA(5)  ->  at that close
           or after 10 sessions, or a catastrophic stop 3 x ATR below entry (sizing anchor)
  Variants: B = A + CMF(20) < 0 (institutional selling into the dip)
            C = A + close above the VWAP anchored at the 50-day low (buyers still in profit)
  Fundamentals (analyst rating, beta, PEG, quality) are applied LIVE only --
  there is no free point-in-time history to test them.

Usage:
  python3 qd_backtest.py --daily-dir cache/tsm_daily                  # 8y, Tech/HC/Fin
  python3 qd_backtest.py --daily-dir cache/mds/backtest/1d            # 5y, all sectors, real SPY
  python3 qd_backtest.py --daily-dir cache/mds/backtest/1d --entry vwap   # 15m entry timing (3y)
  python3 qd_backtest.py --variant B
"""
from __future__ import annotations
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from mds_config import MDS_CONFIG as MCFG
import mds_data as D
import mds_strategy as S
import mds_backtest as B
from mds_backtest_daily import load

QD = {
    "RSI_PERIOD": 2, "RSI_MAX": 10.0, "TREND_SMA": 200, "EXIT_SMA": 5, "MAX_HOLD": 10,
    "CAT_STOP_ATR": 3.0, "CMF_MAX": 0.0, "VWAP_ENTRY_SESSIONS": 2,
}


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(df: pd.DataFrame, qd=QD) -> pd.DataFrame:
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["rsi"] = rsi(c, qd["RSI_PERIOD"])
    f["sma_t"] = c.rolling(qd["TREND_SMA"]).mean()
    f["sma_x"] = c.rolling(qd["EXIT_SMA"]).mean()
    f["atr"] = S.bxi.atr(df, 14)
    v = df["Volume"]; rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((c - df["Low"]) - (df["High"] - c)) / rng
    f["cmf"] = (mfm.fillna(0) * v).rolling(20).sum() / v.rolling(20).sum()
    f["avwap"] = S.anchored_vwap_at_window_low(df, 50)
    return f


def sim_symbol(args):
    sym, df, spy, etfsym, qd, variant, entry_mode, intra = args
    f = features(df, qd)
    mkt = S.market_regime(spy, MCFG).reindex(f.index).fillna(False).to_numpy(bool)
    O, H, L, C = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    rsi_, smat, smax, atr, cmf, av = (f[k].to_numpy(float) for k in ("rsi", "sma_t", "sma_x", "atr", "cmf", "avwap"))
    sig = mkt & (C > smat) & (rsi_ < qd["RSI_MAX"]) & np.isfinite(atr)
    if variant == "B":
        sig &= cmf < qd["CMF_MAX"]
    if variant == "C":
        sig &= C > av
    idx = df.index
    by_sess, sessions = None, None
    if entry_mode == "vwap":
        if intra is None or len(intra) < 500:
            return sym, [], {"sig": 0, "filled": 0}
        prep = S.prepare_intraday(intra, MCFG)
        by_sess = S.split_by_session(prep)
        sessions = list(by_sess.keys())
        first_sess = pd.Timestamp(sessions[MCFG["RVOL_LOOKBACK_SESSIONS"]])
    trades, st = [], {"sig": 0, "filled": 0}
    i, busy = 210, -1
    cost = MCFG["COST_BPS_PER_SIDE"] / 1e4
    while i < len(idx) - 1:
        if not sig[i] or i <= busy:
            i += 1; continue
        st["sig"] += 1
        # ---- entry
        k0, entry, etime = None, None, None
        if entry_mode == "open":
            k0, entry, etime = i + 1, O[i + 1], idx[i + 1] + pd.Timedelta(hours=9.5)
        else:
            if idx[i] < first_sess:
                i += 1; continue
            for j in range(1, qd["VWAP_ENTRY_SESSIONS"] + 1):
                if i + j >= len(idx):
                    break
                sb = by_sess.get(idx[i + j].date())
                if sb is None:
                    continue
                hit = sb[sb["base_trigger"]]
                if len(hit):
                    t = hit.index[0]
                    nxt = sb[sb.index > t]
                    if len(nxt):
                        k0, entry, etime = i + j, float(nxt["Open"].iloc[0]), nxt.index[0].tz_localize(None)
                    break
        if k0 is None:
            i += 1; continue
        st["filled"] += 1
        stop = entry - qd["CAT_STOP_ATR"] * atr[i]
        # ---- exit (daily bars from the entry day)
        exit_k, exit_px, why = None, None, None
        for k in range(k0, min(k0 + qd["MAX_HOLD"], len(idx))):
            lo = L[k] if k > k0 or entry_mode == "open" else L[k]      # entry day: conservative, full-day low
            if lo <= stop:
                exit_k, exit_px, why = k, min(O[k] if k > k0 else entry, stop), "stop"; break
            if C[k] > smax[k]:
                exit_k, exit_px, why = k, C[k], "sma5"; break
        if exit_k is None:
            exit_k = min(k0 + qd["MAX_HOLD"] - 1, len(idx) - 1)
            exit_px, why = C[exit_k], "time" if exit_k < len(idx) - 1 else "open_at_end"
        risk = entry - stop
        net = (exit_px - entry) - cost * (entry + exit_px)
        xt = idx[exit_k] + pd.Timedelta(hours=15.9)
        trades.append({"symbol": sym, "sector_etf": etfsym, "quadrant": "n/a", "score": 100 - rsi_[i],
                       "setup_date": str(idx[i].date()), "entry_time": etime, "entry": entry, "stop": stop,
                       "exit_time": xt, "exit_reason": why, "fills": [(str(xt), 1.0, float(exit_px), why)],
                       "sessions_held": exit_k - k0 + 1, "r_multiple": net / risk, "pct_return": net / entry,
                       "rsi": rsi_[i], "cmf": cmf[i], "entry_type": entry_mode})
        busy = exit_k
        i = exit_k + 1
    return sym, trades, st


def summarize(t, eq, spy, cfg, label):
    t = t.copy(); t["yr"] = pd.to_datetime(t["entry_time"]).dt.year
    def s(x):
        p = x["pct_return"] * 100
        w, l = p[p > 0].sum(), -p[p <= 0].sum()
        return f"n={len(x):5d} win={100 * (p > 0).mean():3.0f}% avg={p.mean():+.2f}% PF={w / l if l else float('nan'):.2f}"
    ret = eq["equity"].pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq["equity"].iloc[-1] / cfg["STARTING_CAPITAL"]) ** (1 / yrs) - 1
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    b = spy["Close"].loc[eq.index[0]:eq.index[-1]]; bc = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
    bdd = (b / b.cummax() - 1).min(); br = b.pct_change().dropna()
    print(f"{label} | ALL {s(t)} | <=2023 {s(t[t.yr <= 2023])} | >=2024 {s(t[t.yr >= 2024])} | "
          f"hold {t.sessions_held.mean():.1f}d | port CAGR {cagr:+.1%} DD {dd:.0%} Sh {ret.mean() / ret.std() * 252 ** .5:.2f} "
          f"exp {eq.n_positions.mean():.1f} | SPY CAGR {bc:+.1%} DD {bdd:.0%} Sh {br.mean() / br.std() * 252 ** .5:.2f}")
    print("   by year avg%: " + " ".join(f"{y}:{m * 100:+.2f}({n})" for y, m, n in
                                        zip(*[t.groupby("yr").pct_return.mean().index, t.groupby("yr").pct_return.mean(), t.groupby("yr").size()])))
    print("   exits: " + ", ".join(f"{k} {v}" for k, v in t.exit_reason.value_counts().items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-dir", default="cache/tsm_daily")
    ap.add_argument("--intra-dir", default="cache/mds/backtest/15m")
    ap.add_argument("--variant", default="A", choices=["A", "B", "C"])
    ap.add_argument("--entry", default="open", choices=["open", "vwap"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    qd = dict(QD)
    for kv in a.set:
        k, v = kv.split("="); qd[k] = type(QD[k])(float(v))
    cfg = dict(MCFG); cfg["MAX_POSITIONS"] = int(os.environ.get("QD_SLOTS", 10)); cfg["MAX_PER_SECTOR"] = max(4, cfg["MAX_POSITIONS"] // 3); cfg["MAX_POSITION_PCT"] = 1.0 / cfg["MAX_POSITIONS"]
    cfg["SIZING"] = "equal"                     # 10 slots x 10% of equity; the catastrophic stop is not a sizing anchor
    t0 = time.time()
    daily, spy, etf, sectors = load(a.daily_dir)
    if a.start:
        daily = {k: v for k, v in daily.items()}
    intra = {}
    if a.entry == "vwap":
        for s in daily:
            p = os.path.join(a.intra_dir, f"{s}.parquet")
            if os.path.exists(p):
                intra[s] = pd.read_parquet(p)
    jobs = [(s, df, spy, sectors.get(s), qd, a.variant, a.entry, intra.get(s)) for s, df in daily.items()]
    trades, st = [], {"sig": 0, "filled": 0}
    with ProcessPoolExecutor(4) as ex:
        for sym, tr, s_ in ex.map(sim_symbol, jobs, chunksize=4):
            trades += tr
            for k in st: st[k] += s_.get(k, 0)
    t = pd.DataFrame(trades)
    if a.start:
        t = t[pd.to_datetime(t.entry_time) >= a.start]
    daily_all = dict(daily); daily_all[cfg["BENCHMARK"]] = spy
    start = pd.Timestamp(a.start) if a.start else spy.index[210]
    if a.entry == "vwap":
        start = max(start, min(pd.Timestamp(v.index[0].date()) for v in intra.values()) + pd.Timedelta(days=20))
        t = t[pd.to_datetime(t.entry_time) >= start]
    eq, acc = B.simulate_portfolio(t, daily_all, cfg, start=start, end=spy.index[-1])
    label = f"QD-{a.variant} entry={a.entry} {' '.join(a.set)} [{os.path.basename(a.daily_dir.rstrip('/'))}]"
    print(f"{label}: {len(daily)} symbols, signals {st['sig']}, filled {st['filled']}, {time.time() - t0:.0f}s")
    summarize(t, eq, spy, cfg, "  signals  ")
    summarize(acc, eq, spy, cfg, "  portfolio")
    os.makedirs("results", exist_ok=True)
    tag = f"{a.variant}_{a.entry}_{os.path.basename(a.daily_dir.rstrip('/'))}{a.tag}"
    t.drop(columns=["fills"]).to_csv(f"results/qd_trades_{tag}.csv", index=False)
    eq.to_csv(f"results/qd_equity_{tag}.csv")
