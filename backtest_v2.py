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
    res = backtest(["AAPL", "MSFT", "NVDA"], entry="both", max_hold=20)       # one process: safe anywhere
    print(res["trades"].head())

    # faster on many tickers - this form needs the guard:
    if __name__ == "__main__":
        res = backtest(my_list, workers=4)

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
import multiprocessing as mp
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

import enrich_html_v2 as E

# Paste your list here to run with no command-line arguments (used when no --tickers* option is given):
TICKERS = []          # e.g. ["AAPL", "MSFT", "NVDA"]

# Ways of relaxing the individual confirmation ITEMS (engine config overrides). Each is combined with the minimum
# confirmation counts in `conf_levels` (typically 2 and 3). "live" (the report's rules) is always included.
ITEM_VARIANTS = {
    "vol_tail2":    {"DIP_TAIL_DAYS": 2},                                    # volume fading over the last 2 dip days (live: 2 on short dips, else 3)
    "reset2_stoch": {"RESET_BARS": 2},                                       # stochastic %K <= 40 at least once in the last 2 bars (live: 5)
    "reset2_rsi":   {"RESET_INDICATOR": "rsi", "RESET_BARS": 2},             # RSI14 <= 40 at least once in the last 2 bars
    "reset5_rsi":   {"RESET_INDICATOR": "rsi", "RESET_BARS": 5},             # RSI14 <= 40 in the last 5 bars (RSI in place of stochastic)
    "value_1.0atr": {"VALUE_ZONE_ATR": 1.0},                                 # pullback low within 1.0 ATR of a value level (live: 0.5)
    "all_relaxed":  {"DIP_TAIL_DAYS": 2, "RESET_BARS": 2, "VALUE_ZONE_ATR": 1.0},
}

CFG = dict(E.CONFIGH)
BT = {
    "WARMUP_DAYS": 130,        # daily bars needed before the first signal (EMA50, 90-day structure, 120-day targets)
    "MAX_HOLD_DAYS": 20,       # time stop: exit at the close of the Nth trading day
    "SLIPPAGE_PCT": 0.05,      # % paid on entry and on stop / time exits
    "FWD_DAYS": (5, 10, 20),   # forward-return horizons for the signal-quality table
    "DOWNLOAD_PERIOD": "730d",
    "EXITS": ("fixed", "breakeven", "atr_trail", "ema_trail"),
    "TRAIL_ACT_R": 1.0,        # trailing / break-even starts once price has gained this many R
    "TRAIL_ATR_MULT": 2.5,     # atr_trail: stop = highest high since entry - this many daily ATRs
    "TRAIL_MAX_HOLD": None,    # time stop for the trailing exits without a target (None = same as MAX_HOLD_DAYS)
    "CONTROL_K": 10,           # matched random days per BUY trade (0 = no control)
    "CONTROL_WINDOW": 20,      # ... drawn from the same ticker within +/- this many trading days of the signal
    "VARIANTS": None,          # {name: engine-config overrides}; used together with CONF_LEVELS
    "CONF_LEVELS": None,       # e.g. (0, 1, 2, 3, 4): also trade BUY-type days needing at least k confirmations (sweep)
    "TRADE_ON": "BUY",         # "BUY" (turn confirmed) or "PRE_TURN" (setup complete, R:R fine, only the turn gate still failing)
    "GRADES": None,            # e.g. ("A",): only BUY signals with these grades become trades
    "SETUPS": None,            # e.g. ("PULLBACK",): only these setup types become trades
    "BASE_EXIT": None,         # exit model that decides which signals become trades (None = "fixed" if present, else the first)
    "DETAIL_EXIT": "fixed",    # exit model whose by-setup / by-grade / by-month tables are printed
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


def signals_at(prep: dict, j: int, cfg: dict, variants: dict = None) -> dict:
    """The signal(s) the report would have produced at the close of daily bar j: {"live": ..., variant: ...}.
    The expensive part (indicators, timeframe evaluation) is done once; each variant only re-runs the setup engine
    with some config keys overridden (see ITEM_VARIANTS)."""
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

    shared = {}                                            # reused by every variant of this day

    def run(c_):
        return E.evaluate_setups(
            d1.iloc[:j + 1], float(prep["atr_d"].iloc[j]), results, stoch["1D"].iloc[:j + 1],
            tfs["1h"].iloc[:pos["1h"]], h1_macd_ok, above_vwap, False, c_,
            high_52w=None if pd.isna(hi52) else float(hi52), vwap=vwap_now, shared=shared)

    out = {"live": run(cfg)}
    for name, overrides in (variants or {}).items():
        if name != "live":
            out[name] = run({**cfg, **overrides})
    return out


def signal_at(prep: dict, j: int, cfg: dict) -> dict:
    """The live-rules signal at the close of daily bar j."""
    return signals_at(prep, j, cfg)["live"]


# ---------------------------------------------------------------- trade simulation

SETUP_TYPES = ("FRESH_BREAKOUT", "SUPPORT_BOUNCE", "MA_BOUNCE", "PULLBACK", "MA_RECLAIM")

EXIT_MODELS = {
    "fixed": "initial stop + target (baseline)",
    "breakeven": "initial stop + target; stop moves to the entry price after +TRAIL_ACT_R",
    "atr_trail": "no target; after +TRAIL_ACT_R the stop trails the highest high by TRAIL_ATR_MULT daily ATRs",
    "ema_trail": "no target; after +TRAIL_ACT_R the stop trails the daily EMA20",
}


def simulate_trade(o, h, l, c, j, stop, target, buy_up_to, mode, max_hold, slip_pct,
                   exit="fixed", atr=None, ema=None, act_r=1.0, atr_mult=2.5):
    """One trade from the signal at bar j. Returns a dict, or None when the entry is not possible.
    Entry  next_open   : buy at the open of bar j+1 (skipped if it gaps above 'buy_up_to', to/below the stop or over the target)
           signal_close: buy at the close of bar j (the price the report showed)
    Exit   see EXIT_MODELS. R is always measured against the INITIAL stop. A trailing stop is updated after a bar
           closes and only protects the NEXT bar (no look-ahead). Open gaps through a stop / target fill at the open;
           stop and target inside one bar -> stop first; after max_hold bars -> that bar's close.
    The entries are identical for every exit model, so the models are directly comparable."""
    n, slip = len(c), slip_pct / 100.0
    if j + 1 >= n:                       # a signal on the very last bar has no following bar to trade or exit on
        return None
    if mode == "next_open":
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
    use_target = exit in ("fixed", "breakeven")
    last = min(first + max_hold - 1, n - 1)
    cur_stop, hh, ll = stop, entry, entry
    exit_px, reason, k_exit = None, None, None
    for k in range(first, last + 1):
        entry_bar = (mode == "next_open" and k == first)      # the entry bar opens AT the entry price: no gap logic
        moved = cur_stop != stop
        if not entry_bar:
            if o[k] <= cur_stop:
                exit_px, reason = float(o[k]) * (1 - slip), ("trail (gap)" if moved else "stop (gap)")
            elif use_target and o[k] >= target:
                exit_px, reason = float(o[k]), "target (gap)"
        if exit_px is None:
            if l[k] <= cur_stop:
                exit_px, reason = cur_stop * (1 - slip), ("trail" if moved else "stop")
            elif use_target and h[k] >= target:
                exit_px, reason = float(target), "target"
        if exit_px is not None:
            k_exit = k
            ll = min(ll, float(l[k]))
            break
        hh, ll = max(hh, float(h[k])), min(ll, float(l[k]))
        if exit != "fixed" and hh >= entry + act_r * risk:      # armed: raise the stop for the NEXT bar
            if exit == "breakeven":
                cur_stop = max(cur_stop, entry)
            elif exit == "atr_trail" and atr is not None and not np.isnan(atr[k]):
                cur_stop = max(cur_stop, hh - atr_mult * float(atr[k]))
            elif exit == "ema_trail" and ema is not None and not np.isnan(ema[k]):
                cur_stop = max(cur_stop, float(ema[k]))
    if exit_px is None:
        k_exit = last
        exit_px, reason = float(c[last]) * (1 - slip), "time"
    return {"entry": entry, "exit": exit_px, "r": (exit_px - entry) / risk, "ret_pct": (exit_px / entry - 1) * 100,
            "exit_idx": k_exit, "hold_days": k_exit - first + 1, "reason": reason, "entry_idx": first,
            "mfe_r": (hh - entry) / risk, "mae_r": (entry - ll) / risk}


# ---------------------------------------------------------------- matched random-day control

def add_controls(trades, sig_by_j, o, h, l, c, atr_arr, ema20, bt, ticker):
    """For every BUY trade, replay the SAME trade on K random other days of the SAME ticker within +/- WINDOW trading
    days (days that were not BUY signals): same entry model and exit model, and the same stop distance (in ATRs) and
    target multiple (in R). Each trade gets ctrl_r = the mean R of its matched days. BUY minus ctrl_r then measures the
    value of the SIGNAL's timing, with the stock's drift and the market regime around that time held constant."""
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    K, W = int(bt["CONTROL_K"]), int(bt["CONTROL_WINDOW"])
    days_ok = sorted(sig_by_j)
    n = len(c)
    for tr in trades:
        j = tr["_j"]
        cand = [d for d in days_ok if abs(d - j) <= W and d != j and sig_by_j[d] not in ("BUY", "TRIG") and d + 1 < n]
        if not cand:
            tr["ctrl_r"], tr["ctrl_n"] = np.nan, 0
            continue
        picks = rng.choice(cand, size=min(K, len(cand)), replace=False)
        hold = bt["MAX_HOLD_DAYS"]
        if tr["exit_model"] in ("atr_trail", "ema_trail") and bt.get("TRAIL_MAX_HOLD"):
            hold = bt["TRAIL_MAX_HOLD"]
        rs = []
        for d in picks:
            px = float(c[d])
            risk = tr["_risk_atr"] * float(atr_arr[d])
            if not np.isfinite(risk) or risk <= 0:
                continue
            res = simulate_trade(o, h, l, c, int(d), px - risk, px + tr["_rr"] * risk, None, tr["entry_model"], hold,
                                 bt["SLIPPAGE_PCT"], exit=tr["exit_model"], atr=atr_arr, ema=ema20,
                                 act_r=bt["TRAIL_ACT_R"], atr_mult=bt["TRAIL_ATR_MULT"])
            if res is not None:
                rs.append(res["r"])
        tr["ctrl_r"], tr["ctrl_n"] = (float(np.mean(rs)) if rs else np.nan), len(rs)


def control_by(trades: pd.DataFrame, entry: str, exit_: str, col: str) -> str:
    """Paired BUY-minus-control R per setup / grade for one entry + exit model."""
    t = trades[(trades["entry_model"] == entry) & (trades["exit_model"] == exit_)].dropna(subset=["ctrl_r"])
    rows = []
    for key, g in t.groupby(col):
        d = g["r"] - g["ctrl_r"]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 2 else np.nan
        rows.append({col: key, "trades": len(g), "BUY_avg_R": round(g["r"].mean(), 3),
                     "control_avg_R": round(g["ctrl_r"].mean(), 3), "BUY-control_R": round(d.mean(), 3),
                     "t": round(d.mean() / se, 2) if se and se > 0 else np.nan})
    return pd.DataFrame(rows).to_string(index=False) if rows else ""


def control_df(trades: pd.DataFrame, modes: list, exits: list) -> pd.DataFrame:
    if trades.empty or "ctrl_r" not in trades or trades["ctrl_r"].notna().sum() == 0:
        return pd.DataFrame()
    rows = []
    for m in modes:
        for x in exits:
            t = trades[(trades["entry_model"] == m) & (trades["exit_model"] == x)].dropna(subset=["ctrl_r"])
            if len(t) < 5:
                continue
            d = t["r"] - t["ctrl_r"]
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"entry": m, "exit": x, "trades": len(t), "BUY_avg_R": round(t["r"].mean(), 3),
                         "control_avg_R": round(t["ctrl_r"].mean(), 3), "BUY-control_R": round(d.mean(), 3),
                         "std_err": round(se, 3), "t": round(d.mean() / se, 2) if se > 0 else np.nan,
                         "BUY_beats_ctrl_%": round((d > 0).mean() * 100, 1)})
    return pd.DataFrame(rows)


def control_table(trades: pd.DataFrame, modes: list, exits: list) -> str:
    df = control_df(trades, modes, exits)
    return df.to_string(index=False) if len(df) else ""


def halves_table(days: pd.DataFrame, trades: pd.DataFrame, split: str, modes: list, exits: list) -> str:
    """The same trades and controls, cut at the split date: does the result hold in both halves?"""
    parts = [("FULL", trades), (f"before {split}", trades[trades["signal_date"] < split]),
             (f"from {split}", trades[trades["signal_date"] >= split])]
    frames = []
    for name, t in parts:
        df = control_df(t, modes, exits)
        if len(df):
            df.insert(0, "period", name)
            frames.append(df)
    if not frames:
        return ""
    out = pd.concat(frames, ignore_index=True)
    out = out[["period", "entry", "exit", "trades", "BUY_avg_R", "control_avg_R", "BUY-control_R", "t"]]
    return out.sort_values(["entry", "exit", "period"], kind="stable").to_string(index=False)


# ---------------------------------------------------------------- scan one ticker

def scan_ticker(args):
    """Wrapper: one misbehaving ticker is reported and skipped instead of killing the whole run."""
    try:
        return _scan_ticker(args)
    except Exception as e:
        import traceback
        return args[0], [], [], f"SKIPPED - {type(e).__name__}: {e} ({traceback.format_exc().splitlines()[-3].strip()})"


def _scan_ticker(args):
    ticker, hourly, cfg, bt, modes, start, end = args
    out_days, out_trades, errors = [], [], 0
    sig_by_j = {}
    levels = list(bt.get("CONF_LEVELS") or [])                  # confirmation sweep: minimum confirmation counts to test
    variants = dict(bt.get("VARIANTS") or {}) if levels else {}
    vnames = ["live"] + [v for v in variants if v != "live"]
    sig_by_j_k = {(v, k): {} for v in vnames for k in levels}
    try:
        prep = prepare(hourly, cfg)
    except Exception as e:
        return ticker, [], [], f"prepare failed: {type(e).__name__}: {e}"
    d1 = prep["tfs"]["1D"]
    o, h, l, c = (d1[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    n = len(d1)
    atr_arr = prep["atr_d"].to_numpy(dtype=float)
    ema20 = d1["Close"].ewm(span=20, adjust=False).mean().to_numpy(dtype=float)
    exits = list(bt["EXITS"])
    base_exit = bt.get("BASE_EXIT") or ("fixed" if "fixed" in exits else exits[0])
    grades, setups = bt.get("GRADES"), bt.get("SETUPS")
    busy_until = {m: -1 for m in modes}                         # one open position per ticker per entry model (baseline exit)
    busy_k = {(m, v, k): -1 for m in modes for v in vnames for k in levels}   # ... and per variant / level in the sweep

    def open_trades(j, plan, m, busy, key, extra):
        """Simulate one entry with every exit model. The baseline exit decides whether the entry is taken (one open
        position per ticker); every other exit model then trades the SAME entry."""
        sims = {}
        for x in [base_exit] + [e for e in exits if e != base_exit]:
            hold = bt["MAX_HOLD_DAYS"]
            if x in ("atr_trail", "ema_trail") and bt.get("TRAIL_MAX_HOLD"):
                hold = bt["TRAIL_MAX_HOLD"]
            tr = simulate_trade(o, h, l, c, j, plan["stop"], plan["target"], plan["buy_up_to"], m,
                                hold, bt["SLIPPAGE_PCT"], exit=x, atr=atr_arr, ema=ema20,
                                act_r=bt["TRAIL_ACT_R"], atr_mult=bt["TRAIL_ATR_MULT"])
            if tr is None:
                break                                           # entry not possible (same for every exit model)
            sims[x] = tr
        if base_exit not in sims:
            return
        busy[key] = sims[base_exit]["exit_idx"]
        for x, tr in sims.items():
            out_trades.append({"ticker": ticker, "entry_model": m, "exit_model": x, "signal_date": d1.index[j].date().isoformat(),
                               "entry_date": d1.index[tr["entry_idx"]].date().isoformat(),
                               "exit_date": d1.index[tr["exit_idx"]].date().isoformat(),
                               "setup": plan["setup"], "grade": plan.get("grade"), "planned_rr": plan["rr"],
                               "stop": plan["stop"], "target": plan["target"], **{k: tr[k] for k in
                               ("entry", "exit", "r", "ret_pct", "hold_days", "reason", "mfe_r", "mae_r")},
                               "_j": j, "_risk_atr": (c[j] - plan["stop"]) / atr_arr[j],
                               "_rr": (plan["target"] - c[j]) / (c[j] - plan["stop"]), **extra})

    for j in range(bt["WARMUP_DAYS"], n):
        date = d1.index[j]
        if (start is not None and date < start) or (end is not None and date > end):
            continue
        try:
            sigs = signals_at(prep, j, cfg, variants)
            sig = sigs["live"]
        except Exception:
            errors += 1
            continue
        rec = {"ticker": ticker, "date": date.date().isoformat(), "signal": sig["signal"], "setup": sig["setup"],
               "grade": sig["grade"], "close": c[j], "state": sig["context"]["state"], "rr": sig["rr"],
               "stop": sig["stop"], "target": sig["target"], "buy_up_to": sig["buy_up_to"],
               "gap": sig["distance"], "watch_kind": sig["watch_kind"]}
        is_buy, is_pre = sig["signal"] == "BUY", bool(sig.get("pre_turn"))
        rec["pre_turn"] = is_pre
        trig_type = (is_buy if bt.get("TRADE_ON", "BUY") == "BUY" else is_pre)      # is this the kind of day we trade?
        # the grade filter only makes sense for BUY (WATCH-type days are always graded C)
        rec["selected"] = bool(trig_type and (grades is None or not is_buy or sig["grade"] in grades)
                               and (setups is None or sig["setup"] in setups))
        rec["trigger"] = "BUY" if is_buy else ("PRE_TURN" if is_pre else None)
        cand0 = sig.get("conf_candidate")
        rec["cand_confirms"] = cand0["n_confirms"] if cand0 else np.nan
        for k in bt["FWD_DAYS"]:
            rec[f"fwd{k}"] = (c[j + k] / c[j] - 1) * 100 if j + k < n else np.nan
        out_days.append(rec)
        sig_by_j[j] = "TRIG" if trig_type else "OTHER"
        if rec["selected"]:
            plan = {"stop": sig["stop"], "target": sig["target"], "buy_up_to": sig["buy_up_to"], "setup": sig["setup"],
                    "grade": sig["grade"], "rr": sig["rr"]}
            for m in modes:
                if j > busy_until[m]:
                    open_trades(j, plan, m, busy_until, m, {})
        for v in vnames:                                        # confirmation sweep: same BUY logic, minimum count k
            cand = sigs[v].get("conf_candidate")
            for k in levels:
                ok_k = bool(cand is not None and cand["n_confirms"] >= k and (setups is None or cand["type"] in setups))
                sig_by_j_k[(v, k)][j] = "TRIG" if ok_k else "OTHER"
                if ok_k:
                    plan = {"stop": cand["stop"], "target": cand["target"], "buy_up_to": cand["buy_up_to"],
                            "setup": cand["type"], "grade": None, "rr": cand["rr"]}
                    for m in modes:
                        if j > busy_k[(m, v, k)]:
                            open_trades(j, plan, m, busy_k, (m, v, k),
                                        {"sweep": True, "variant": v, "conf_min": k, "n_confirms": cand["n_confirms"]})
    K = int(bt.get("CONTROL_K") or 0)
    if K and out_trades:
        normal = [t for t in out_trades if not t.get("sweep")]
        if normal:
            add_controls(normal, sig_by_j, o, h, l, c, atr_arr, ema20, bt, ticker)
        for v in vnames:
            for k in levels:
                grp = [t for t in out_trades if t.get("sweep") and t["variant"] == v and t["conf_min"] == k]
                if grp:
                    add_controls(grp, sig_by_j_k[(v, k)], o, h, l, c, atr_arr, ema20, bt, ticker)
    for tr in out_trades:
        for k in ("_j", "_risk_atr", "_rr"):
            tr.pop(k, None)
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
        "losers_that_hit_+1R_%": (round((losses["mfe_r"] >= 1).mean() * 100, 1)
                                  if "mfe_r" in tr and len(losses) else None),
    }


def signal_quality(days: pd.DataFrame, fwd_cols: list) -> pd.DataFrame:
    """Forward returns (from the signal day's close) by signal, against ALL ticker-days.
    Needs no trade rules, so it shows whether BUY days beat a random day at all."""
    rows = []
    groups = [("ALL days", days)] + [(s, days[days["signal"] == s]) for s in ("BUY", "WATCH", "WAIT")]
    if "pre_turn" in days and int(days["pre_turn"].sum()):
        groups.insert(2, ("PRE_TURN (part of WATCH)", days[days["pre_turn"]]))
    if "selected" in days and days["selected"].any():
        sel_is_buy = (days.loc[days["selected"], "signal"] == "BUY").all()
        same_as_pre = "pre_turn" in days and bool((days["selected"] == days["pre_turn"]).all())
        if not same_as_pre and (not sel_is_buy or int(days["selected"].sum()) != int((days["signal"] == "BUY").sum())):
            groups.insert(2, ("TRADED days (selected)", days[days["selected"]]))
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


def sweep_tables(sweep: pd.DataFrame, bt: dict, modes: list, split: str = None) -> str:
    """Does the confirmation setup matter? Same BUY logic; the minimum confirmation count and the confirmation ITEMS
    (variants) are varied. Every row uses the same stops, the same exits and its own matched-random-day control."""
    if sweep.empty:
        return ""
    exits = [x for x in bt["EXITS"] if x in set(sweep["exit_model"])]
    dx = bt.get("DETAIL_EXIT") if bt.get("DETAIL_EXIT") in exits else exits[0]
    variants = ["live"] + [v for v in sweep["variant"].unique() if v != "live"]
    L = ["", "CONFIRMATION SWEEP - BUY-type days (structure, turn, R:R and extension all pass) with AT LEAST k confirmations.",
         "  live = the report's rules. k = 3 is the live minimum (of 4). Variants relax individual confirmation items:",
         "  " + "; ".join(f"{n}: {ov}" for n, ov in (bt.get("VARIANTS") or {}).items()),
         "  Reading: 'diff' is BUY minus its matched-random-day control (R). Trust a variant only if diff has the same sign in both halves.",
         "  Many rows are shown; some will look good by luck. Compare the pattern across rows, not the best single row."]

    def row(df, v, k, m, x):
        t = df[(df["variant"] == v) & (df["conf_min"] == k) & (df["entry_model"] == m) & (df["exit_model"] == x)].dropna(subset=["ctrl_r"])
        if len(t) < 5:
            return None
        d = t["r"] - t["ctrl_r"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        return t, d, se

    for m in modes:
        rows = []
        for v in variants:
            for k in sorted(sweep["conf_min"].unique()):
                got = row(sweep, v, k, m, dx)
                if got is None:
                    continue
                t, d, se = got
                r = {"variant": v, "min_conf": int(k), "trades": len(t), "avg_R": round(t["r"].mean(), 3),
                     "control_R": round(t["ctrl_r"].mean(), 3), "diff": round(d.mean(), 3),
                     "t": round(d.mean() / se, 2) if se > 0 else np.nan}
                if split:
                    for nm, msk in (("diff_H1", t["signal_date"] < split), ("diff_H2", t["signal_date"] >= split)):
                        r[nm] = round(d[msk].mean(), 3) if msk.sum() >= 5 else np.nan
                rows.append(r)
        if rows:
            L.append(f"\n  ENTRY {m}, EXIT {dx}" + (f"   (H1 = before {split}, H2 = from {split})" if split else ""))
            L.append(pd.DataFrame(rows).to_string(index=False))

    other = [x for x in exits if x != dx]
    if other:
        rows = []
        for m in modes:
            for v in variants:
                for k in sorted(sweep["conf_min"].unique()):
                    for x in other:
                        got = row(sweep, v, k, m, x)
                        if got is None:
                            continue
                        t, d, se = got
                        rows.append({"entry": m, "variant": v, "min_conf": int(k), "exit": x, "trades": len(t),
                                     "avg_R": round(t["r"].mean(), 3), "diff": round(d.mean(), 3),
                                     "t": round(d.mean() / se, 2) if se > 0 else np.nan})
        if rows:
            L.append("\n  OTHER EXITS (same rows):")
            L.append(pd.DataFrame(rows).to_string(index=False))

    live = sweep[sweep["variant"] == "live"]
    if len(live):
        lowest = live[live["conf_min"] == live["conf_min"].min()]
        for m in modes:
            t = lowest[(lowest["entry_model"] == m) & (lowest["exit_model"] == dx)].dropna(subset=["ctrl_r"])
            rows = []
            for nc, g in t.groupby("n_confirms"):
                d = g["r"] - g["ctrl_r"]
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 2 else np.nan
                rows.append({"n_confirms": int(nc), "trades": len(g), "avg_R": round(g["r"].mean(), 3),
                             "control_R": round(g["ctrl_r"].mean(), 3), "diff": round(d.mean(), 3),
                             "t": round(d.mean() / se, 2) if se and se > 0 else np.nan})
            if rows:
                L.append(f"\n  EXACT confirmation count, live rules ({m}, exit {dx}; trades taken at min_conf = {int(live['conf_min'].min())}):")
                L.append(pd.DataFrame(rows).to_string(index=False))
    return "\n".join(L)


def exit_rows(trades: pd.DataFrame, modes: list, exits: list) -> pd.DataFrame:
    rows = []
    for m in modes:
        for x in exits:
            tr = trades[(trades["entry_model"] == m) & (trades["exit_model"] == x)]
            if tr.empty:
                continue
            st = trade_stats(tr)
            rows.append({"entry": m, "exit": x, "trades": st["trades"], "win_%": st["win_rate_%"], "avg_R": st["avg_R"],
                         "median_R": st["median_R"], "avg_win_R": st["avg_win_R"], "avg_loss_R": st["avg_loss_R"],
                         "PF": st["profit_factor"], "total_R": st["total_R"], "maxDD_R": st["max_drawdown_R"],
                         "hold_d": st["avg_hold_days"]})
    return pd.DataFrame(rows)


def build_summary(days: pd.DataFrame, trades: pd.DataFrame, bt: dict, modes: list) -> str:
    L = []
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    fwd_cols = [f"fwd{k}" for k in bt["FWD_DAYS"]]
    L.append(f"Ticker-days evaluated: {len(days):,}   tickers: {days['ticker'].nunique()}   "
             f"period: {days['date'].min()} .. {days['date'].max()}")
    L.append("Signals: " + ", ".join(f"{k} {v:,}" for k, v in days["signal"].value_counts().items()))
    if bt.get("TRADE_ON", "BUY") != "BUY":
        L.append(f"TRADING ON: {bt['TRADE_ON']}  (setup complete, R:R and confirmations fine - only the daily/1h turn still missing); "
                 f"{int(days['selected'].sum()):,} selected days; PRE_TURN days: {int(days['pre_turn'].sum()):,}")
    if bt.get("GRADES") or bt.get("SETUPS"):
        L.append(f"TRADE FILTERS: grades {list(bt['GRADES']) if bt.get('GRADES') else 'any'}, "
                 f"setups {list(bt['SETUPS']) if bt.get('SETUPS') else 'any'}  ->  {int(days['selected'].sum()):,} selected BUY days")
    L.append("")
    L.append("SIGNAL QUALITY - forward return from the signal day's close (no trade rules; % move, up_% = share positive)")
    L.append("naive_t_vs_ALL treats overlapping windows as independent, so it flatters significance: read it loosely.")
    L.append(signal_quality(days, fwd_cols).to_string(index=False))
    st = days.groupby("state")[fwd_cols].mean().round(2)
    st.insert(0, "n_days", days.groupby("state").size())
    L.append("")
    L.append("ALL days by daily TREND STATE (does the trend label predict anything?)  forward %")
    L.append(st.to_string())
    gd = days.dropna(subset=["gap"]).copy()
    if len(gd):
        gd["gap_bucket"] = pd.cut(gd["gap"], [-0.01, 0.01, 1.5, 3.5, 100], labels=["0 (BUY)", "0.1-1.5", "1.6-3.5", ">3.5"])
        gg = gd.groupby("gap_bucket", observed=True)[fwd_cols].mean().round(2)
        gg.insert(0, "n_days", gd.groupby("gap_bucket", observed=True).size())
        L.append("")
        L.append("Days WITH a setup, by GAP (distance from a BUY): does a smaller gap mean better forward returns?  forward %")
        L.append(gg.to_string())
    buy = days[days["signal"] == "BUY"]
    if len(buy):
        L.append("")
        L.append("BUY days by setup (forward %, same horizons)")
        g = buy.groupby("setup")[fwd_cols].mean().round(2)
        g.insert(0, "n", buy.groupby("setup").size())
        L.append(g.to_string())
    if trades.empty:
        L.append("\nno trades")
        return "\n".join(L)

    exits = [x for x in bt["EXITS"] if x in set(trades["exit_model"])]
    L.append("")
    L.append(f"EXIT MODEL COMPARISON  (identical entries, different exits; slippage {bt['SLIPPAGE_PCT']}% per side; R = initial risk)")
    L.append("  entries are chosen with the baseline exit (one open position per ticker); a trailing exit that holds longer may overlap the next entry,")
    L.append("  so total_R / maxDD_R for the trailing models assume you can hold overlapping positions.")
    L.append(f"  trailing arms after +{bt['TRAIL_ACT_R']:g}R; atr_trail = highest high - {bt['TRAIL_ATR_MULT']:g} x daily ATR; "
             f"max hold {bt['MAX_HOLD_DAYS']}d" + (f" (trailing exits: {bt['TRAIL_MAX_HOLD']}d)" if bt.get("TRAIL_MAX_HOLD") else ""))
    rows = []
    for m in modes:
        for x in exits:
            tr = trades[(trades["entry_model"] == m) & (trades["exit_model"] == x)]
            if tr.empty:
                continue
            st = trade_stats(tr)
            rows.append({"entry": m, "exit": x, "trades": st["trades"], "win_%": st["win_rate_%"], "avg_R": st["avg_R"],
                         "median_R": st["median_R"], "avg_win_R": st["avg_win_R"], "avg_loss_R": st["avg_loss_R"],
                         "PF": st["profit_factor"], "total_R": st["total_R"], "maxDD_R": st["max_drawdown_R"],
                         "hold_d": st["avg_hold_days"]})
    L.append(pd.DataFrame(rows).to_string(index=False))
    ctl = control_table(trades, modes, exits)
    if ctl:
        L.append("")
        L.append(f"CONTROL - is the SIGNAL better than a random day? Each BUY trade is replayed on {bt['CONTROL_K']} random non-BUY days of the")
        L.append(f"  same ticker within +/-{bt['CONTROL_WINDOW']} trading days (same stop distance in ATRs, same target multiple, same exit model).")
        L.append("  BUY-control_R is the average paired difference; |t| below ~2 means the signal's timing adds nothing measurable.")
        L.append(ctl)
    detail_x = bt.get("DETAIL_EXIT") if bt.get("DETAIL_EXIT") in exits else exits[0]
    if ctl:
        for m in modes:
            for col in ("setup", "grade"):
                tb = control_by(trades, m, detail_x, col)
                if tb:
                    L.append(f"\n  BUY minus control by {col}  ({m}, exit {detail_x}):\n" + tb)
    fx = trades[trades["exit_model"] == "fixed"] if "fixed" in exits else None
    if fx is not None and len(fx) and "mfe_r" in fx:
        los = fx[fx["r"] <= 0]
        L.append("")
        L.append("  how much do losing trades give back? (fixed exit, 'mfe' = best unrealised gain before the exit bar)")
        L.append("  share of losers that first reached: " + ", ".join(
            f"+{t:g}R {(los['mfe_r'] >= t).mean() * 100:.0f}%" for t in (0.5, 1.0, 1.5, 2.0)) + f"   (losers: {len(los)})")

    detail = bt.get("DETAIL_EXIT", "fixed")
    if detail not in exits:
        detail = exits[0]
    for m in modes:
        tr = trades[(trades["entry_model"] == m) & (trades["exit_model"] == detail)]
        L.append("")
        hold_d = bt["TRAIL_MAX_HOLD"] if (detail in ("atr_trail", "ema_trail") and bt.get("TRAIL_MAX_HOLD")) else bt["MAX_HOLD_DAYS"]
        L.append(f"DETAIL - entry: {m}, exit: {detail}  (max hold {hold_d}d, slippage {bt['SLIPPAGE_PCT']}% per side)")
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


def _load_cache(path: str) -> dict:
    """Read the download cache; a missing, half-written or corrupt file just means 'download again'."""
    if not os.path.exists(path):
        return {}
    try:
        obj = pd.read_pickle(path)
        if isinstance(obj, dict):
            return obj
        raise ValueError("not a dict")
    except Exception as e:
        print(f"cache {path} is unreadable ({type(e).__name__}) - ignoring it and downloading again")
        try:
            os.replace(path, path + ".corrupt")
        except OSError:
            pass
        return {}


def _save_cache(obj: dict, path: str) -> None:
    """Atomic write (temp file, then rename) so an interrupted run can never leave a half-written cache."""
    tmp = f"{path}.{os.getpid()}.tmp"
    pd.to_pickle(obj, tmp)
    os.replace(tmp, path)


def backtest(tickers, entry="both", max_hold=None, slippage=None, start=None, end=None, workers=None,
             verify_n=0, cache=None, refresh=False, exits="all", trail_activate_r=None, trail_atr_mult=None,
             trail_max_hold=None, detail_exit=None, grades=None, setups=None, base_exit=None,
             control_k=None, control_window=None, trade_on=None, split_date=None, tag=None, conf_levels=None, variants=None):
    """Run the whole backtest from Python. `tickers` is a plain list, e.g. ["AAPL", "MSFT"].
    Returns {"days": DataFrame, "trades": DataFrame, "summary": str}, or None for a verify run.
    workers=None (default) runs in ONE process, which is safe from any script. workers=N>1 uses N worker
    processes; then the calling script MUST wrap the call in `if __name__ == "__main__":` (macOS/Windows
    start workers by re-importing it). If the workers crash, this falls back to a single process."""
    if mp.parent_process() is not None:      # we are a worker that re-imported an unguarded script: do nothing, fail fast
        raise RuntimeError("backtest() was called inside a worker process - put the call under "
                           "`if __name__ == \"__main__\":` in your script")
    tickers = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not tickers:
        raise ValueError("no tickers given")
    bt = dict(BT)
    if trade_on and str(trade_on).upper() == "PRE_TURN":
        bt["OUT_DIR"] = BT["OUT_DIR"] + "_pre_turn"
    if tag:
        bt["OUT_DIR"] = bt["OUT_DIR"] + "_" + str(tag)
    if max_hold is not None:
        bt["MAX_HOLD_DAYS"] = int(max_hold)
    if slippage is not None:
        bt["SLIPPAGE_PCT"] = float(slippage)
    if exits in ("all", None):
        bt["EXITS"] = tuple(BT["EXITS"])
    else:
        ex_list = [exits] if isinstance(exits, str) else list(exits)
        bad = [x for x in ex_list if x not in EXIT_MODELS]
        if bad:
            raise ValueError(f"unknown exit model(s) {bad}; choose from {list(EXIT_MODELS)}")
        bt["EXITS"] = tuple(ex_list)
    if trail_activate_r is not None:
        bt["TRAIL_ACT_R"] = float(trail_activate_r)
    if trail_atr_mult is not None:
        bt["TRAIL_ATR_MULT"] = float(trail_atr_mult)
    if trail_max_hold is not None:
        bt["TRAIL_MAX_HOLD"] = int(trail_max_hold)
    if detail_exit:
        bt["DETAIL_EXIT"] = detail_exit
    if grades:
        bt["GRADES"] = tuple(str(g).upper() for g in ([grades] if isinstance(grades, str) else grades))
    if setups:
        want = tuple(str(x).upper().replace(" ", "_") for x in ([setups] if isinstance(setups, str) else setups))
        bad = [x for x in want if x not in SETUP_TYPES]
        if bad:
            raise ValueError(f"unknown setup(s) {bad}; choose from {list(SETUP_TYPES)}")
        bt["SETUPS"] = want
    if trade_on:
        if str(trade_on).upper() not in ("BUY", "PRE_TURN"):
            raise ValueError("trade_on must be 'BUY' or 'PRE_TURN'")
        bt["TRADE_ON"] = str(trade_on).upper()
    if conf_levels:
        bt["CONF_LEVELS"] = tuple(sorted({int(k) for k in conf_levels}))
    if variants:
        bt["VARIANTS"] = dict(ITEM_VARIANTS) if variants in ("items", "all") else dict(variants)
    if control_k is not None:
        bt["CONTROL_K"] = int(control_k)
    if control_window is not None:
        bt["CONTROL_WINDOW"] = int(control_window)
    if base_exit:
        if base_exit not in bt["EXITS"]:
            raise ValueError(f"base_exit {base_exit!r} must be one of the exits being run: {list(bt['EXITS'])}")
        bt["BASE_EXIT"] = base_exit
        if not detail_exit:
            bt["DETAIL_EXIT"] = base_exit          # show the detail tables for the exit that drives the trades
    if bt.get("DETAIL_EXIT") not in bt["EXITS"]:
        bt["DETAIL_EXIT"] = bt.get("BASE_EXIT") or bt["EXITS"][0]
    modes = ["next_open", "signal_close"] if entry == "both" else [entry]
    cache = cache or os.path.join(BT["OUT_DIR"], "hourly_cache.pkl")      # one shared download cache for every run
    workers = 1 if workers is None else int(workers)
    os.makedirs(bt["OUT_DIR"], exist_ok=True)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)

    hourly = {}
    if not refresh:
        hourly = _load_cache(cache)
        if hourly:
            print(f"loaded cached 1h data for {len(hourly)} tickers from {cache}")
    missing = [t for t in tickers if t not in hourly]
    if missing:
        print(f"downloading {len(missing)} tickers ({bt['DOWNLOAD_PERIOD']} of 1h bars) ...")
        hourly.update(E.prefetch_hourly_data(missing, bt["DOWNLOAD_PERIOD"], "1h"))
        _save_cache(hourly, cache)
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

    def run_sequential():
        for k, job in enumerate(jobs, 1):
            t, d, tr, err = scan_ticker(job)
            days.extend(d)
            trades.extend(tr)
            print(f"[{k}/{len(jobs)}] {t}: {len(d)} days, {len(tr)} trades" + (f"  ({err})" if err else ""), flush=True)

    if workers > 1 and len(jobs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for k, (t, d, tr, err) in enumerate(ex.map(scan_ticker, jobs), 1):
                    days.extend(d)
                    trades.extend(tr)
                    print(f"[{k}/{len(jobs)}] {t}: {len(d)} days, {len(tr)} trades" + (f"  ({err})" if err else ""), flush=True)
        except BrokenProcessPool:
            print("\nworker processes crashed (usually the calling script has no `if __name__ == \"__main__\":` "
                  "guard). Falling back to a single process - slower but safe.\n", flush=True)
            days.clear()
            trades.clear()
            run_sequential()
    else:
        run_sequential()
    print(f"scan finished in {time.time() - t0:.0f}s")

    days = pd.DataFrame(days)
    trades = pd.DataFrame(trades)
    sweep = pd.DataFrame()
    if len(trades) and "sweep" in trades:
        is_sw = trades["sweep"].fillna(False).astype(bool)
        sweep = trades[is_sw].drop(columns=["sweep"]).reset_index(drop=True)
        trades = trades[~is_sw].drop(columns=["sweep", "conf_min", "n_confirms"], errors="ignore").reset_index(drop=True)
    if days.empty:
        print("no signal days were produced (not enough history?)")
        return {"days": days, "trades": trades, "summary": ""}
    summary = build_summary(days, trades, bt, modes)
    if len(sweep):
        summary += "\n" + sweep_tables(sweep, bt, modes, str(pd.Timestamp(split_date).date()) if split_date else None)
    print("\n" + summary)
    out = bt["OUT_DIR"]
    days[days["signal"].isin(["BUY", "WATCH"])].to_csv(os.path.join(out, "bt_signal_days.csv"), index=False)
    days.to_csv(os.path.join(out, "bt_all_days.csv.gz"), index=False, compression="gzip")
    if not trades.empty:
        trades.to_csv(os.path.join(out, "bt_trades.csv"), index=False)
    if len(sweep):
        sweep.to_csv(os.path.join(out, "bt_conf_sweep_trades.csv"), index=False)
    if split_date:
        sd = str(pd.Timestamp(split_date).date())
        ht = halves_table(days, trades, sd, modes, list(bt["EXITS"]))
        if ht:
            block = (f"\n\nHALF-BY-HALF - same trades, same controls, cut at signal date {sd}. The result is only believable if it "
                     f"points the same way in both halves.\n" + ht)
            for nm, tt in ((f"BEFORE {sd}", trades[trades["signal_date"] < sd]),
                           (f"FROM {sd}", trades[trades["signal_date"] >= sd])):
                er = exit_rows(tt, modes, [x for x in bt["EXITS"] if x in set(tt["exit_model"])]) if len(tt) else pd.DataFrame()
                if len(er):
                    block += f"\n\nEXIT COMPARISON - {nm}\n" + er.to_string(index=False)
            summary += block
            print(block)
    with open(os.path.join(out, "bt_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not trades.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for m in modes:
                for x in bt["EXITS"]:
                    tm = trades[(trades["entry_model"] == m) & (trades["exit_model"] == x)].sort_values("exit_date")
                    if len(tm):
                        ax.plot(pd.to_datetime(tm["exit_date"]), tm["r"].cumsum(), label=f"{m} / {x} ({len(tm)})",
                                ls="-" if m == modes[0] else "--")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_ylabel("cumulative R")
            ax.set_title("BUY signals - cumulative R by exit date")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out, "bt_equity.png"), dpi=120)
    except Exception as e:
        print(f"(equity chart skipped: {type(e).__name__})")
    print(f"\nfiles written to {out}/")
    return {"days": days, "trades": trades, "summary": summary, "sweep": sweep}


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
    ap.add_argument("--exits", nargs="*", default=["all"], help="exit models to compare: fixed breakeven atr_trail ema_trail (default all)")
    ap.add_argument("--trail-activate", type=float, default=None, help="R gained before the trailing / break-even stop arms (default 1.0)")
    ap.add_argument("--trail-atr", type=float, default=None, help="ATR multiple for atr_trail (default 2.5)")
    ap.add_argument("--trail-max-hold", type=int, default=None, help="time stop in days for atr_trail / ema_trail (default = --max-hold)")
    ap.add_argument("--conf-levels", nargs="*", type=int, default=None, help="confirmation sweep, e.g. --conf-levels 2 3")
    ap.add_argument("--variants", default=None, choices=["items"], help="also test the relaxed confirmation items (ITEM_VARIANTS)")
    ap.add_argument("--trade-on", default=None, choices=["BUY", "PRE_TURN"], help="which days become trades (default BUY)")
    ap.add_argument("--split-date", default=None, help="YYYY-MM-DD: also report the two halves separately")
    ap.add_argument("--tag", default=None, help="suffix for the output folder, e.g. nobreakout")
    ap.add_argument("--control-k", type=int, default=None, help="matched random days per BUY trade for the control (default 10, 0 = off)")
    ap.add_argument("--control-window", type=int, default=None, help="control days come from +/- this many trading days (default 20)")
    ap.add_argument("--grades", nargs="*", default=None, help="only trade these grades, e.g. A")
    ap.add_argument("--setups", nargs="*", default=None, help="only trade these setups: PULLBACK FRESH_BREAKOUT SUPPORT_BOUNCE MA_BOUNCE MA_RECLAIM")
    ap.add_argument("--base-exit", default=None, help="exit model that decides which signals become trades (default fixed)")
    ap.add_argument("--detail-exit", default=None, help="exit model for the by-setup / grade / month tables (default fixed)")
    a = ap.parse_args(argv)
    tickers = load_tickers(a) or list(TICKERS)          # no CLI source -> the TICKERS list at the top of this file
    if not tickers:
        ap.error("no tickers: use --tickers / --tickers-file / --tickers-csv, or fill TICKERS at the top of the script")
    workers = a.workers if a.workers is not None else max(1, (os.cpu_count() or 2) - 1)    # the CLI has the __main__ guard
    backtest(tickers, entry=a.entry, max_hold=a.max_hold, slippage=a.slippage, start=a.start, end=a.end,
             workers=workers, verify_n=a.verify, cache=a.cache, refresh=a.refresh,
             exits=(a.exits[0] if a.exits == ["all"] else a.exits), trail_activate_r=a.trail_activate,
             trail_atr_mult=a.trail_atr, trail_max_hold=a.trail_max_hold, detail_exit=a.detail_exit,
             grades=a.grades, setups=a.setups, base_exit=a.base_exit,
             control_k=a.control_k, control_window=a.control_window, trade_on=a.trade_on,
             split_date=a.split_date, tag=a.tag, conf_levels=a.conf_levels, variants=a.variants)


if __name__ == "__main__":
    main()
