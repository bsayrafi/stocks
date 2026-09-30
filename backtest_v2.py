#!/usr/bin/env python
"""
backtest_v2.py - walk-forward backtest of the enrich_html_v2 setup engine.

It REUSES the live code (enrich_html_v2.evaluate_setups and the indicator functions), so the
strategy tested is exactly the strategy that produces the reports. For every ticker and every
trading day it rebuilds what the report would have shown at that day's CLOSE (daily / 4h / 1h
bars up to that close, session VWAP, 1h MACD ...), records the signal, and simulates each BUY.

Run (from the folder that has enrich_html_v2.py and its helper modules):

    python backtest_v2.py --tickers-csv reports/Large_signal_report_Sep_29th_22_45.csv
    python backtest_v2.py --tickers AAPL MSFT NVDA --entry both --max-hold 20
    python backtest_v2.py --tickers-file tickers.txt --verify 25      # look-ahead self-check first
    python backtest_v2.py                                            # uses the TICKERS list inside this file

From your own Python code (a plain list works):

    from backtest_v2 import backtest
    if __name__ == "__main__":
        res = backtest(["AAPL", "MSFT", "NVDA"], entry="both", max_hold=20)
        print(res["trades"].head())

Outputs (folder backtest_out/): bt_trades.csv, bt_signal_days.csv, bt_summary.txt, bt_equity.png

What is and is not tested - please read:
  * Data: yfinance gives at most ~730 days of 1h bars, so the test covers roughly the last 1.5 years
    after a ~130-day warm-up. That is one market regime, not a full cycle.
  * Survivorship bias: a list of tickers picked TODAY is full of names that did well. Expect flattering
    numbers; compare the BUY rows to the ALL-days baseline printed in the report, not to zero.
  * The EPS-revision filter and news are not part of the signal, and history for them is not available,
    so they are not tested. The 52-week high uses the highest high inside the downloaded window.
  * Signals are evaluated at the daily close. Live you run mid-session, so the "turn" checks can differ.
    Entry models: next_open (conservative, default) and signal_close (assumes you buy near the close).
  * Daily-bar exits: if the stop and the target are both inside one bar, the STOP is assumed to hit first.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import enrich_html_v2 as E

# Paste your list here to run with no command-line arguments (used when no --tickers* option is given):
TICKERS = []          # e.g. ["AAPL", "MSFT", "NVDA"]

CFG = dict(E.CONFIGH)
BT = {
    "WARMUP_DAYS": 130,        # daily bars needed before the first signal (EMA50, 90-day structure, 120-day targets)
    "MAX_HOLD_DAYS": 20,       # time stop: exit at the close of the Nth trading day
    "SLIPPAGE_PCT": 0.05,      # % paid on entry and on stop / time exits
    "FWD_DAYS": (5, 10, 20),   # forward-return horizons for the signal-quality table
    "DOWNLOAD_PERIOD": "730d",
    "OUT_DIR": "backtest_out",
}


# ---------------------------------------------------------------- per-ticker preparation

def prepare(hourly: pd.DataFrame, cfg: dict) -> dict:
    """Everything that can be computed ONCE per ticker. All indicators are causal (EMA / rolling / cumulative),
    so slicing them at a given day gives the same numbers as recomputing on data cut at that day."""
    h = hourly.copy()
    h = h[~h.index.duplicated()].sort_index()
    h = h.dropna(subset=["Open", "High", "Low", "Close"])
    h["Volume"] = h["Volume"].fillna(0)
    d1 = E.resample_ohlc(h, "1D")
    h4 = E.resample_ohlc(h, "4h")
    tfs = {"1h": h, "4h": h4, "1D": d1}
    stoch = {k: E.stochastic_oscillator(v, cfg["STOCH_K_PERIOD"], cfg["STOCH_D_PERIOD"], cfg["STOCH_SMOOTH"])
             for k, v in tfs.items()}
    ma = {k: E.moving_average(v, cfg["MA_PERIOD"], cfg["MA_TYPE"]) for k, v in tfs.items()}
    vw = E.session_vwap(h, cfg.get("MARKET_TZ", "America/New_York"), cfg.get("MARKET_OPEN", "09:30"),
                        cfg.get("MARKET_CLOSE", "16:00")).ffill()
    return {
        "tfs": tfs, "stoch": stoch, "ma": ma, "vwap": vw,
        "macd1h": E.macd(h, cfg["MACD_FAST"], cfg["MACD_SLOW"], cfg["MACD_SIGNAL"]),
        "atr_d": E.average_true_range(d1, cfg["ATR_PERIOD"]),
        "hi52": d1["High"].rolling(252, min_periods=60).max(),
    }


def signal_at(prep: dict, j: int, cfg: dict) -> dict:
    """The signal the report would have produced at the close of daily bar j."""
    tfs, stoch, ma = prep["tfs"], prep["stoch"], prep["ma"]
    d1 = tfs["1D"]
    cut = d1.index[j] + pd.DateOffset(days=1)              # everything up to the end of that calendar day
    pos = {"1h": tfs["1h"].index.searchsorted(cut), "4h": tfs["4h"].index.searchsorted(cut), "1D": j + 1}
    results = {k: E.evaluate_timeframe(tfs[k].iloc[:pos[k]], stoch[k].iloc[:pos[k]], ma[k].iloc[:pos[k]], cfg)
               for k in tfs}
    h1_macd_ok = E.macd_bullish(prep["macd1h"].iloc[:pos["1h"]])
    price = results["1h"]["close"]
    v = prep["vwap"].iloc[pos["1h"] - 1]
    vwap_now = None if pd.isna(v) else round(float(v), 2)
    above_vwap = (price > vwap_now) if vwap_now is not None else None
    hi52 = prep["hi52"].iloc[j]
    return E.evaluate_setups(
        d1.iloc[:j + 1], float(prep["atr_d"].iloc[j]), results, stoch["1D"].iloc[:j + 1],
        tfs["1h"].iloc[:pos["1h"]], h1_macd_ok, above_vwap, False, cfg,
        high_52w=None if pd.isna(hi52) else float(hi52), vwap=vwap_now)


# ---------------------------------------------------------------- trade simulation

def simulate_trade(o, h, l, c, j, stop, target, buy_up_to, mode, max_hold, slip_pct):
    """One trade from the signal at bar j. Returns a dict, or None when the entry is not possible.
    next_open   : buy at the open of bar j+1 (skipped if it gaps above 'buy_up_to', to/below the stop or over the target)
    signal_close: buy at the close of bar j (the price the report showed)
    Exits (daily bars): open gaps through the stop/target fill at the open; stop and target inside the same
    bar -> stop first; otherwise stop / target price; after max_hold bars -> that bar's close."""
    n, slip = len(c), slip_pct / 100.0
    if mode == "next_open":
        if j + 1 >= n:
            return None
        entry = float(o[j + 1]) * (1 + slip)
        first = j + 1
        if buy_up_to is not None and entry > buy_up_to:
            return None
    else:
        entry = float(c[j]) * (1 + slip)
        first = j + 1
    if entry <= stop or entry >= target:
        return None
    risk = entry - stop
    last = min(first + max_hold - 1, n - 1)
    exit_px, reason, k_exit = None, None, None
    for k in range(first, last + 1):
        entry_bar = (mode == "next_open" and k == first)      # the entry bar opens AT the entry price: no gap logic
        if not entry_bar:
            if o[k] <= stop:
                exit_px, reason = float(o[k]) * (1 - slip), "stop (gap)"
            elif o[k] >= target:
                exit_px, reason = float(o[k]), "target (gap)"
        if exit_px is None:
            if l[k] <= stop:
                exit_px, reason = stop * (1 - slip), "stop"
            elif h[k] >= target:
                exit_px, reason = float(target), "target"
        if exit_px is not None:
            k_exit = k
            break
    if exit_px is None:
        k_exit = last
        exit_px, reason = float(c[last]) * (1 - slip), "time"
    return {"entry": entry, "exit": exit_px, "r": (exit_px - entry) / risk, "ret_pct": (exit_px / entry - 1) * 100,
            "exit_idx": k_exit, "hold_days": k_exit - first + 1, "reason": reason, "entry_idx": first}


# ---------------------------------------------------------------- scan one ticker

def scan_ticker(args):
    ticker, hourly, cfg, bt, modes, start, end = args
    out_days, out_trades, errors = [], [], 0
    try:
        prep = prepare(hourly, cfg)
    except Exception as e:
        return ticker, [], [], f"prepare failed: {type(e).__name__}: {e}"
    d1 = prep["tfs"]["1D"]
    o, h, l, c = (d1[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    n = len(d1)
    busy_until = {m: -1 for m in modes}                         # one open position per ticker per entry model
    for j in range(bt["WARMUP_DAYS"], n):
        date = d1.index[j]
        if (start is not None and date < start) or (end is not None and date > end):
            continue
        try:
            sig = signal_at(prep, j, cfg)
        except Exception:
            errors += 1
            continue
        rec = {"ticker": ticker, "date": date.date().isoformat(), "signal": sig["signal"], "setup": sig["setup"],
               "grade": sig["grade"], "close": c[j], "state": sig["context"]["state"], "rr": sig["rr"],
               "stop": sig["stop"], "target": sig["target"], "buy_up_to": sig["buy_up_to"],
               "gap": sig["distance"], "watch_kind": sig["watch_kind"]}
        for k in bt["FWD_DAYS"]:
            rec[f"fwd{k}"] = (c[j + k] / c[j] - 1) * 100 if j + k < n else np.nan
        out_days.append(rec)
        if sig["signal"] == "BUY":
            for m in modes:
                if j <= busy_until[m]:
                    continue
                tr = simulate_trade(o, h, l, c, j, sig["stop"], sig["target"], sig["buy_up_to"], m,
                                    bt["MAX_HOLD_DAYS"], bt["SLIPPAGE_PCT"])
                if tr is None:
                    continue
                busy_until[m] = tr["exit_idx"]
                out_trades.append({"ticker": ticker, "entry_model": m, "signal_date": rec["date"],
                                   "entry_date": d1.index[tr["entry_idx"]].date().isoformat(),
                                   "exit_date": d1.index[tr["exit_idx"]].date().isoformat(),
                                   "setup": sig["setup"], "grade": sig["grade"], "planned_rr": sig["rr"],
                                   "stop": sig["stop"], "target": sig["target"], **{k: tr[k] for k in
                                   ("entry", "exit", "r", "ret_pct", "hold_days", "reason")}})
    return ticker, out_days, out_trades, (f"{errors} day(s) failed" if errors else None)


# ---------------------------------------------------------------- statistics

def trade_stats(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    wins, losses = tr[tr["r"] > 0], tr[tr["r"] <= 0]
    gl = -losses["r"].sum()
    eq = tr.sort_values("exit_date")["r"].cumsum()
    return {
        "trades": len(tr), "win_rate_%": round(len(wins) / len(tr) * 100, 1),
        "avg_R": round(tr["r"].mean(), 3), "median_R": round(tr["r"].median(), 3),
        "avg_win_R": round(wins["r"].mean(), 2) if len(wins) else 0.0,
        "avg_loss_R": round(losses["r"].mean(), 2) if len(losses) else 0.0,
        "profit_factor": round(wins["r"].sum() / gl, 2) if gl > 0 else float("inf"),
        "total_R": round(tr["r"].sum(), 1), "max_drawdown_R": round((eq.cummax() - eq).max(), 1),
        "avg_hold_days": round(tr["hold_days"].mean(), 1),
        "avg_return_%": round(tr["ret_pct"].mean(), 2),
    }


def signal_quality(days: pd.DataFrame, fwd_cols: list) -> pd.DataFrame:
    """Forward returns (from the signal day's close) by signal, against ALL ticker-days.
    Needs no trade rules, so it shows whether BUY days beat a random day at all."""
    rows = []
    groups = [("ALL days", days)] + [(s, days[days["signal"] == s]) for s in ("BUY", "WATCH", "WAIT")]
    base = days[fwd_cols[1]].dropna()
    for name, g in groups:
        row = {"group": name, "n_days": len(g)}
        for col in fwd_cols:
            x = g[col].dropna()
            row[f"{col}_mean_%"] = round(x.mean(), 2) if len(x) else np.nan
            row[f"{col}_up_%"] = round((x > 0).mean() * 100, 1) if len(x) else np.nan
        x = g[fwd_cols[1]].dropna()
        if name != "ALL days" and len(x) > 2 and len(base) > 2:
            se = np.sqrt(x.var(ddof=1) / len(x) + base.var(ddof=1) / len(base))
            row["naive_t_vs_ALL"] = round((x.mean() - base.mean()) / se, 2) if se > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(days: pd.DataFrame, trades: pd.DataFrame, bt: dict, modes: list) -> str:
    L = []
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    fwd_cols = [f"fwd{k}" for k in bt["FWD_DAYS"]]
    L.append(f"Ticker-days evaluated: {len(days):,}   tickers: {days['ticker'].nunique()}   "
             f"period: {days['date'].min()} .. {days['date'].max()}")
    L.append("Signals: " + ", ".join(f"{k} {v:,}" for k, v in days["signal"].value_counts().items()))
    L.append("")
    L.append("SIGNAL QUALITY - forward return from the signal day's close (no trade rules; % move, up_% = share positive)")
    L.append("naive_t_vs_ALL treats overlapping windows as independent, so it flatters significance: read it loosely.")
    L.append(signal_quality(days, fwd_cols).to_string(index=False))
    buy = days[days["signal"] == "BUY"]
    if len(buy):
        L.append("")
        L.append("BUY days by setup (forward %, same horizons)")
        g = buy.groupby("setup")[fwd_cols].mean().round(2)
        g.insert(0, "n", buy.groupby("setup").size())
        L.append(g.to_string())
    for m in modes:
        tr = trades[trades["entry_model"] == m] if len(trades) else trades
        L.append("")
        L.append(f"TRADES - entry model: {m}  (max hold {bt['MAX_HOLD_DAYS']}d, slippage {bt['SLIPPAGE_PCT']}% per side)")
        if tr.empty:
            L.append("  no trades")
            continue
        L.append(pd.Series(trade_stats(tr)).to_string())
        L.append("  exits: " + ", ".join(f"{k} {v}" for k, v in tr["reason"].value_counts().items()))
        for col, title in (("setup", "by setup"), ("grade", "by grade")):
            by = tr.groupby(col).apply(lambda x: pd.Series(trade_stats(x))[["trades", "win_rate_%", "avg_R", "total_R"]])
            by["trades"] = by["trades"].astype(int)
            L.append(f"\n  {title}:\n" + by.to_string())
        tr = tr.assign(month=tr["exit_date"].str[:7])
        by = tr.groupby("month")["r"].agg(trades="count", total_R="sum", avg_R="mean").round(2)
        L.append("\n  by exit month:\n" + by.to_string())
    return "\n".join(L)


# ---------------------------------------------------------------- self-check for look-ahead

def verify(hourly_by_ticker: dict, cfg: dict, bt: dict, samples: int) -> bool:
    """Recompute the signal from scratch on data CUT at the day's close (no shared precomputation) and
    compare with the fast path. A mismatch means look-ahead or a bug."""
    rng = np.random.default_rng(1)
    bad = tot = 0
    for t, hourly in hourly_by_ticker.items():
        prep = prepare(hourly, cfg)
        d1 = prep["tfs"]["1D"]
        if len(d1) <= bt["WARMUP_DAYS"] + 2:
            continue
        for j in rng.choice(np.arange(bt["WARMUP_DAYS"], len(d1)), size=min(samples, len(d1) - bt["WARMUP_DAYS"]), replace=False):
            fast = signal_at(prep, int(j), cfg)
            cutoff = d1.index[j] + pd.DateOffset(days=1)
            cut = prep["tfs"]["1h"][prep["tfs"]["1h"].index < cutoff]
            slow_prep = prepare(cut, cfg)
            slow = signal_at(slow_prep, len(slow_prep["tfs"]["1D"]) - 1, cfg)
            keys = ("signal", "setup", "rr", "stop", "target", "grade", "distance")
            tot += 1
            if any(fast.get(k) != slow.get(k) for k in keys):
                bad += 1
                print(f"  MISMATCH {t} {d1.index[j].date()}: " +
                      "; ".join(f"{k}: {fast.get(k)} vs {slow.get(k)}" for k in keys if fast.get(k) != slow.get(k)))
    print(f"verify: {tot - bad}/{tot} sampled days identical between the fast path and a from-scratch recompute")
    return bad == 0


# ---------------------------------------------------------------- main

def load_tickers(a) -> list:
    t = list(a.tickers or [])
    if a.tickers_file:
        t += [x.strip().upper() for x in open(a.tickers_file, encoding="utf-8").read().replace(",", "\n").split() if x.strip()]
    if a.tickers_csv:
        df = pd.read_csv(a.tickers_csv)
        t += [str(x).strip().upper() for x in df["ticker"].dropna().unique()]
    return t


def backtest(tickers, entry="both", max_hold=None, slippage=None, start=None, end=None, workers=None,
             verify_n=0, cache=None, refresh=False):
    """Run the whole backtest from Python. `tickers` is a plain list, e.g. ["AAPL", "MSFT"].
    Returns {"days": DataFrame, "trades": DataFrame, "summary": str}, or None for a verify run.
    On Windows call it from inside `if __name__ == "__main__":` (worker processes re-import the script)."""
    tickers = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not tickers:
        raise ValueError("no tickers given")
    bt = dict(BT)
    if max_hold is not None:
        bt["MAX_HOLD_DAYS"] = int(max_hold)
    if slippage is not None:
        bt["SLIPPAGE_PCT"] = float(slippage)
    modes = ["next_open", "signal_close"] if entry == "both" else [entry]
    cache = cache or os.path.join(bt["OUT_DIR"], "hourly_cache.pkl")
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(bt["OUT_DIR"], exist_ok=True)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)

    hourly = {}
    if os.path.exists(cache) and not refresh:
        hourly = pd.read_pickle(cache)
        print(f"loaded cached 1h data for {len(hourly)} tickers from {cache}")
    missing = [t for t in tickers if t not in hourly]
    if missing:
        print(f"downloading {len(missing)} tickers ({bt['DOWNLOAD_PERIOD']} of 1h bars) ...")
        hourly.update(E.prefetch_hourly_data(missing, bt["DOWNLOAD_PERIOD"], "1h"))
        pd.to_pickle(hourly, cache)
    hourly = {t: hourly[t] for t in tickers if t in hourly}
    print(f"{len(hourly)}/{len(tickers)} tickers have data")
    if verify_n:
        ok = verify(dict(list(hourly.items())[:5]), CFG, bt, verify_n)
        if not ok:
            sys.exit(1)
        return None

    start_ts = pd.Timestamp(start, tz="America/New_York") if start else None
    end_ts = pd.Timestamp(end, tz="America/New_York") if end else None
    jobs = [(t, df, CFG, bt, modes, start_ts, end_ts) for t, df in hourly.items()]
    t0 = time.time()
    days, trades = [], []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, (t, d, tr, err) in enumerate(ex.map(scan_ticker, jobs), 1):
                days += d
                trades += tr
                print(f"[{k}/{len(jobs)}] {t}: {len(d)} days, {len(tr)} trades" + (f"  ({err})" if err else ""), flush=True)
    else:
        for k, job in enumerate(jobs, 1):
            t, d, tr, err = scan_ticker(job)
            days += d
            trades += tr
            print(f"[{k}/{len(jobs)}] {t}: {len(d)} days, {len(tr)} trades" + (f"  ({err})" if err else ""), flush=True)
    print(f"scan finished in {time.time() - t0:.0f}s")

    days = pd.DataFrame(days)
    trades = pd.DataFrame(trades)
    if days.empty:
        print("no signal days were produced (not enough history?)")
        return {"days": days, "trades": trades, "summary": ""}
    summary = build_summary(days, trades, bt, modes)
    print("\n" + summary)
    out = bt["OUT_DIR"]
    days[days["signal"].isin(["BUY", "WATCH"])].to_csv(os.path.join(out, "bt_signal_days.csv"), index=False)
    if not trades.empty:
        trades.to_csv(os.path.join(out, "bt_trades.csv"), index=False)
    with open(os.path.join(out, "bt_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not trades.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for m in modes:
                tm = trades[trades["entry_model"] == m].sort_values("exit_date")
                if len(tm):
                    ax.plot(pd.to_datetime(tm["exit_date"]), tm["r"].cumsum(), label=f"{m} ({len(tm)} trades)")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_ylabel("cumulative R")
            ax.set_title("BUY signals - cumulative R by exit date")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out, "bt_equity.png"), dpi=120)
    except Exception as e:
        print(f"(equity chart skipped: {type(e).__name__})")
    print(f"\nfiles written to {out}/")
    return {"days": days, "trades": trades, "summary": summary}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest the enrich_html_v2 setup engine")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--tickers-file")
    ap.add_argument("--tickers-csv", help="a signal-report CSV; uses its 'ticker' column")
    ap.add_argument("--entry", default="both", choices=["both", "next_open", "signal_close"])
    ap.add_argument("--max-hold", type=int, default=BT["MAX_HOLD_DAYS"])
    ap.add_argument("--slippage", type=float, default=BT["SLIPPAGE_PCT"], help="percent per side")
    ap.add_argument("--start", help="first signal date, YYYY-MM-DD")
    ap.add_argument("--end", help="last signal date, YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--verify", type=int, default=0, metavar="N", help="look-ahead self-check on N random days per ticker, then exit")
    ap.add_argument("--cache", default=None, help="downloaded 1h bars are cached here")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    a = ap.parse_args(argv)
    tickers = load_tickers(a) or list(TICKERS)          # no CLI source -> the TICKERS list at the top of this file
    if not tickers:
        ap.error("no tickers: use --tickers / --tickers-file / --tickers-csv, or fill TICKERS at the top of the script")
    backtest(tickers, entry=a.entry, max_hold=a.max_hold, slippage=a.slippage, start=a.start, end=a.end,
             workers=a.workers, verify_n=a.verify, cache=a.cache, refresh=a.refresh)


if __name__ == "__main__":
    main()
