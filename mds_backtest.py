"""
mds_backtest.py
===============
Backtest of the Multi-Day Swing strategy using the SAME functions as the live
scanner (mds_strategy.py), on Alpaca SIP daily + 15m bars.

Walk-forward, no look-ahead:
  - Daily setup on day D uses daily bars <= D and 15m bars <= D's close.
  - On each later session the ACTIVE setup is the most recent one from the last
    SETUP_VALID_SESSIONS sessions -- exactly what the live watchlist does (a new
    daily scan replaces yesterday's levels).
  - Entry = open of the 15m bar AFTER the trigger bar closes.
  - Stop / TP checked bar by bar on 15m; if one bar touches both, the stop wins
    (conservative). Gaps through a level fill at the open.
  - End-of-session exits: close below the channel's lower band (the trend
    thesis is broken) or MAX_HOLD_SESSIONS reached.

What the backtest CANNOT test honestly (and how it's handled):
  - Analyst ratings / beta / PEG have no free point-in-time history. By default
    the fundamentals gate is OFF in the backtest. --fundamentals-snapshot applies
    TODAY's gate to the whole history: that's look-ahead (stocks rated Strong Buy
    today often got there by going up) -- use it only to compare, never to judge.
  - Finviz universes are today's survivors. --universe sp500-pit uses the
    survivorship-free S&P 500 membership file already in the repo.

Usage:
    python3 mds_backtest.py cache    --universe finviz --size 2 --daily-years 5 --intraday-years 3
    python3 mds_backtest.py cache    --universe sp500-pit --intraday-years 3
    python3 mds_backtest.py cache    --tickers NVDA MSFT AAPL
    python3 mds_backtest.py simulate [--no-sector] [--no-market] [--fundamentals-snapshot] [--start 2024-01-01]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from mds_config import MDS_CONFIG as CFG
import mds_data as D
import mds_strategy as S

DEFAULT_CACHE = os.path.join(CFG["CACHE_DIR"], "backtest")


# =========================================================================== phase A: per-symbol trades

def simulate_trade(setup: S.Setup, trig: dict, by_session: dict, sessions: list, feats: pd.DataFrame,
                   cfg=CFG) -> dict | None:
    """Manage one position from the trigger to the exit, on 15m bars."""
    t_time = trig["time"]
    s_idx = sessions.index(t_time.date())
    bars = by_session[sessions[s_idx]]
    after = bars[bars.index > t_time]
    if after.empty:                                       # trigger on the session's last bar -> next open
        if s_idx + 1 >= len(sessions):
            return None
        s_idx += 1
        after = by_session[sessions[s_idx]]
    entry_time = after.index[0]
    entry = float(after["Open"].iloc[0])
    stop0 = setup.stop
    if entry <= stop0 or entry >= setup.tp1:
        return None                                       # gapped through a level before we could fill
    risk = entry - stop0
    stop, remaining, tp1_hit = stop0, 1.0, False
    fills = []                                            # (time, fraction, price, reason)
    use_targets = cfg.get("USE_TARGETS", True)            # False: no TP1/TP2, exit by trail / channel / time
    trail = cfg.get("TRAIL_ATR", 0)                       # >0: chandelier stop = highest close - trail * daily ATR
    hi_close = entry

    held = 0
    for k in range(s_idx, len(sessions)):
        sess = sessions[k]
        sb = by_session[sess]
        if k == s_idx:
            sb = sb[sb.index >= entry_time]
        o, h, l, c = (sb[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
        for i in range(len(sb)):
            if l[i] <= stop:
                px = min(o[i], stop)
                fills.append((sb.index[i], remaining, px, "breakeven" if tp1_hit and stop >= entry else ("trail" if stop > stop0 else "stop")))
                remaining = 0.0
                break
            if use_targets and not tp1_hit and h[i] >= setup.tp1:
                frac = cfg["TP1_FRACTION"]
                fills.append((sb.index[i], frac, max(o[i], setup.tp1), "tp1"))
                remaining -= frac
                tp1_hit = True
                if cfg["MOVE_STOP_TO_BE_AT_TP1"]:
                    stop = max(stop, entry)
            if use_targets and tp1_hit and remaining > 0 and h[i] >= setup.tp2:
                fills.append((sb.index[i], remaining, max(o[i], setup.tp2), "tp2"))
                remaining = 0.0
                break
        if remaining <= 1e-9:
            break
        held += 1
        day = pd.Timestamp(sess)
        close = float(c[-1])
        if trail > 0:                                     # ratchet at the session close
            hi_close = max(hi_close, close)
            stop = max(stop, hi_close - trail * setup.atr)
        if cfg["EXIT_ON_CLOSE_BELOW_LRC_LOWER"] and day in feats.index:
            lower = feats.at[day, "lrc_lower"]
            if np.isfinite(lower) and close < lower:
                fills.append((sb.index[-1], remaining, close, "lrc_break"))
                remaining = 0.0
                break
        if held >= cfg["MAX_HOLD_SESSIONS"]:
            fills.append((sb.index[-1], remaining, close, "time"))
            remaining = 0.0
            break
    if remaining > 1e-9:                                  # ran out of data: mark as open at last close
        last = by_session[sessions[-1]]
        fills.append((last.index[-1], remaining, float(last["Close"].iloc[-1]), "open_at_end"))

    cost = cfg["COST_BPS_PER_SIDE"] / 10000
    gross = sum(f * (p - entry) for _, f, p, _ in fills)
    costs = cost * (entry + sum(f * p for _, f, p, _ in fills))
    net = gross - costs
    return {
        "symbol": setup.symbol, "sector_etf": setup.sector_etf, "quadrant": setup.quadrant,
        "setup_date": setup.date, "trigger_time": str(t_time), "entry_time": entry_time, "entry": round(entry, 4),
        "stop": stop0, "stop_src": setup.stop_src, "tp1": setup.tp1, "tp1_src": setup.tp1_src, "tp2": setup.tp2,
        "rr_at_trigger": trig["rr"], "rvol": trig.get("rvol"), "entry_type": trig.get("entry_type"), "score": setup.score, "lrc_r2": setup.lrc_r2,
        "flow_n": setup.flow_n, "exit_time": fills[-1][0], "exit_reason": fills[-1][3],
        "fills": [(str(t), float(f), round(float(p), 4), r) for t, f, p, r in fills],
        "sessions_held": held, "r_multiple": net / risk, "pct_return": net / entry,
    }


def symbol_trades(sym: str, daily: pd.DataFrame, intra: pd.DataFrame, spy: pd.DataFrame, rot: dict | None,
                  sector: str | None, cfg=CFG, use_market=True, start=None) -> tuple[list[dict], dict]:
    stats = {"days": 0, "tech_ok": 0, "setups": 0, "triggered": 0, "invalidated": 0}
    if daily is None or intra is None or len(daily) < 260 or len(intra) < 500:
        return [], stats
    feats = S.daily_features(daily, spy["Close"], cfg)
    mkt = S.market_regime(spy, cfg).reindex(feats.index).fillna(False) if use_market else None
    prep = S.prepare_intraday(intra, cfg)
    by_session = S.split_by_session(prep)
    sessions = list(by_session.keys())
    if len(sessions) <= cfg["VP_SESSIONS"] + 1:
        return [], stats
    first_ok = pd.Timestamp(sessions[cfg["VP_SESSIONS"]])
    if start is not None:
        first_ok = max(first_ok, pd.Timestamp(start))
    sess_set = set(sessions)

    ok = feats["tech_ok"].copy()
    if mkt is not None:
        ok &= mkt
    cand_days = set(feats.index[ok & (feats.index >= first_ok)])
    stats["days"] = int((feats.index >= first_ok).sum())
    stats["tech_ok"] = len(cand_days)

    trades = []
    active: S.Setup | None = None
    active_age = 0
    busy_until = None
    for k, sess in enumerate(sessions):
        day = pd.Timestamp(sess)
        if day < first_ok:
            continue
        # 1) intraday: does the active setup trigger / break today?
        if active is not None:
            active_age += 1
            res = S.scan_session(by_session[sess], active, cfg)
            if res is not None:
                if res["status"] == "triggered":
                    stats["triggered"] += 1
                    tr = simulate_trade(active, res, by_session, sessions, feats, cfg)
                    if tr is not None:
                        trades.append(tr)
                        busy_until = pd.Timestamp(tr["exit_time"])
                else:
                    stats["invalidated"] += 1
                active = None
            elif active_age >= cfg["SETUP_VALID_SESSIONS"]:
                active = None
        # 2) after the close: new daily setup (replaces yesterday's levels)
        if day in cand_days and day in feats.index and sess in sess_set:
            if busy_until is not None and busy_until.date() >= sess:
                continue                                  # already in a position in this symbol
            s, _ = S.evaluate_setup(sym, daily, feats, day, intra, sector, rot, cfg, by_session=by_session)
            if s is not None:
                stats["setups"] += 1
                active, active_age = s, 0
    return trades, stats


# =========================================================================== phase B: portfolio

def simulate_portfolio(trades: pd.DataFrame, daily: dict[str, pd.DataFrame], cfg=CFG,
                       start=None, end=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared capital, risk-based sizing (RISK_PCT_PER_TRADE of equity between entry
    and stop), position / per-symbol / per-sector caps, daily mark-to-market.
    Exits are processed before entries that happen at the same timestamp."""
    if trades.empty:
        return pd.DataFrame(), trades
    trades = trades.copy()
    trades["_entry_ts"] = trades["entry_time"].map(_naive)
    trades = trades.sort_values(["_entry_ts", "score"], ascending=[True, False]).reset_index(drop=True)
    cost = cfg["COST_BPS_PER_SIDE"] / 10000
    cash = float(cfg["STARTING_CAPITAL"])
    open_pos: dict[int, dict] = {}
    accepted: list[dict] = []
    last_px: dict[str, float] = {}

    entries_by_day: dict[pd.Timestamp, list[int]] = {}
    for i, ts in trades["_entry_ts"].items():
        entries_by_day.setdefault(ts.normalize(), []).append(i)
    first = pd.Timestamp(start) if start is not None else trades["_entry_ts"].min().normalize()
    last = pd.Timestamp(end) if end is not None else max(_naive(t).normalize() for t in trades["exit_time"])
    syms = [s for s in trades["symbol"].unique() if s in daily]
    ref = [daily[cfg["BENCHMARK"]]] if cfg["BENCHMARK"] in daily else [daily[s] for s in syms]
    all_days = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in ref])))
    all_days = all_days[(all_days >= first) & (all_days <= last)]
    exit_q: list[tuple] = []                                 # (ts, idx, frac, px) for accepted trades
    _curve: list[dict] = []

    def equity_now():
        return cash + sum(p["shares_left"] * last_px.get(p["symbol"], p["entry"]) for p in open_pos.values())

    for day in all_days:
        nxt_day = day + pd.Timedelta(days=1)
        ev = [(ts, 0, idx, (fr, px)) for ts, idx, fr, px in exit_q if ts < nxt_day]
        exit_q = [e for e in exit_q if e[0] >= nxt_day]
        ev += [(trades.at[i, "_entry_ts"], 1, i, None) for i in entries_by_day.get(day, [])]
        ev.sort(key=lambda e: (e[0], e[1], -trades.at[e[2], "score"] if e[1] else 0))
        for ts, kind, idx, payload in ev:
            if kind == 0:
                p = open_pos.get(idx)
                if p is None:
                    continue
                frac, px = payload
                sh = p["shares_left"] if frac >= p["frac_left"] - 1e-9 else min(p["shares_left"], int(round(p["shares"] * frac)))
                p["frac_left"] -= frac
                cash += sh * px * (1 - cost)
                p["shares_left"] -= sh
                p["pnl"] += sh * (px - p["entry"]) - cost * sh * px
                if p["shares_left"] <= 0:
                    accepted[p["acc_i"]]["pnl"] = p["pnl"]
                    del open_pos[idx]
                continue
            t = trades.loc[idx]
            if len(open_pos) >= cfg["MAX_POSITIONS"]:
                continue
            if any(p["symbol"] == t["symbol"] for p in open_pos.values()):
                continue
            if t["sector_etf"] and sum(p["sector_etf"] == t["sector_etf"] for p in open_pos.values()) >= cfg["MAX_PER_SECTOR"]:
                continue
            eq = equity_now()
            if cfg.get("SIZING", "risk") == "equal":        # fixed notional: MAX_POSITION_PCT of equity per position
                shares = int(eq * cfg["MAX_POSITION_PCT"] / t["entry"])
            else:
                shares = int(eq * cfg["RISK_PCT_PER_TRADE"] / (t["entry"] - t["stop"]))
            shares = min(shares, int(eq * cfg["MAX_POSITION_PCT"] / t["entry"]), int(cash / (t["entry"] * (1 + cost))))
            if shares <= 0:
                continue
            cash -= shares * t["entry"] * (1 + cost)
            acc = t.drop(labels=["_entry_ts"]).to_dict()
            acc.update({"shares": shares, "pnl": -cost * shares * t["entry"]})
            accepted.append(acc)
            open_pos[idx] = {"symbol": t["symbol"], "sector_etf": t["sector_etf"], "entry": t["entry"], "shares": shares,
                             "shares_left": shares, "frac_left": 1.0, "pnl": acc["pnl"], "acc_i": len(accepted) - 1}
            last_px[t["symbol"]] = t["entry"]
            for ft, fr, fp, _ in t["fills"]:
                exit_q.append((_naive(ft), idx, fr, fp))
        for p in open_pos.values():                          # mark to market at the close
            df = daily.get(p["symbol"])
            if df is not None and day in df.index:
                last_px[p["symbol"]] = float(df.at[day, "Close"])
        _curve.append({"date": day, "equity": equity_now(), "cash": cash, "n_positions": len(open_pos)})
    eq = pd.DataFrame(_curve).set_index("date")
    return eq, pd.DataFrame(accepted)


def _naive(ts) -> pd.Timestamp:
    """Timezone-free New York wall-clock time (daily bars are tz-naive dates)."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert("America/New_York").tz_localize(None) if ts.tzinfo else ts


def stats_report(eq: pd.DataFrame, acc: pd.DataFrame, all_trades: pd.DataFrame, bench: pd.Series | None, cfg=CFG) -> str:
    L = []
    if eq.empty:
        return "no trades"
    start, end = eq["equity"].iloc[0], eq["equity"].iloc[-1]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (end / cfg["STARTING_CAPITAL"]) ** (1 / yrs) - 1
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    ret = eq["equity"].pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    L.append(f"Period            {eq.index[0].date()} -> {eq.index[-1].date()}  ({yrs:.1f}y)")
    L.append(f"Equity            {cfg['STARTING_CAPITAL']:,.0f} -> {end:,.0f}   CAGR {cagr:.1%}   MaxDD {dd:.1%}   Sharpe {sharpe:.2f}")
    L.append(f"Exposure          avg {eq['n_positions'].mean():.1f} positions, in market {(eq['n_positions'] > 0).mean():.0%} of days")
    if bench is not None and len(bench):
        b = bench.loc[eq.index[0]:eq.index[-1]]
        if len(b) > 1:
            bc = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
            bdd = (b / b.cummax() - 1).min()
            L.append(f"Benchmark SPY     CAGR {bc:.1%}   MaxDD {bdd:.1%}")
    for name, t in (("All signals (unconstrained)", all_trades), ("Portfolio (taken)", acc)):
        if t.empty:
            continue
        r = t["r_multiple"]
        wins, losses = r[r > 0], r[r <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
        L.append(f"\n{name}: {len(t)} trades")
        L.append(f"  win rate {len(wins) / len(t):.0%}   avg R {r.mean():+.2f}   median R {r.median():+.2f}   "
                 f"profit factor {pf:.2f}   avg hold {t['sessions_held'].mean():.1f} sessions")
        L.append("  exit reasons: " + ", ".join(f"{k} {v}" for k, v in t["exit_reason"].value_counts().items()))
        by_year = t.assign(y=pd.to_datetime(t["entry_time"].astype(str).str[:10]).dt.year).groupby("y")["r_multiple"]
        L.append("  by year (n, avg R): " + ", ".join(f"{y}: {n} / {m:+.2f}" for y, n, m in
                                                     zip(by_year.mean().index, by_year.size(), by_year.mean())))
        if "entry_type" in t:
            q = t.groupby("entry_type")["r_multiple"].agg(["size", "mean"])
            L.append("  by entry type: " + ", ".join(f"{i}: {int(r_['size'])} / {r_['mean']:+.2f}" for i, r_ in q.iterrows()))
        if "quadrant" in t:
            q = t.groupby("quadrant")["r_multiple"].agg(["size", "mean"])
            L.append("  by quadrant: " + ", ".join(f"{i}: {int(r_['size'])} / {r_['mean']:+.2f}" for i, r_ in q.iterrows()))
    return "\n".join(L)


# =========================================================================== CLI

def _worker(args):
    sym, daily, intra, spy, rot, sector, cfg, use_market, start = args
    try:
        return sym, *symbol_trades(sym, daily, intra, spy, rot, sector, cfg, use_market, start)
    except Exception as e:  # noqa: BLE001
        return sym, [], {"error": str(e)}


def cmd_cache(a):
    os.makedirs(a.cache_dir, exist_ok=True)
    sectors = {}
    if a.tickers:
        syms = a.tickers
    elif a.universe == "sp500-pit":
        syms, sectors = D.sp500_pit_universe(a.intraday_years)
    else:
        u = D.finviz_universe(a.size)
        syms = u["Ticker"].tolist()
        sectors = dict(zip(u["Ticker"], u["Sector"]))
    if a.tickers:
        try:
            import yfinance as yf  # best-effort sector lookup for an explicit list
            for s in syms:
                sectors[s] = (yf.Ticker(s).info or {}).get("sector")
        except Exception:
            pass
    path = os.path.join(a.cache_dir, "sectors.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({k: v for k, v in sectors.items() if v})
    json.dump(old, open(path, "w"), indent=1)
    all_syms = list(dict.fromkeys([CFG["BENCHMARK"]] + CFG["SECTOR_ETFS"] + syms))
    print(f"caching {len(all_syms)} symbols -> {a.cache_dir}")
    D.build_cache(all_syms, a.daily_years, a.intraday_years, a.cache_dir, refresh=a.refresh)


def cmd_simulate(a):
    cfg = dict(CFG)
    if a.no_sector:
        cfg["REQUIRE_SECTOR_ROTATION"] = False
    if a.quad:
        cfg["ALLOWED_QUADRANTS"] = tuple(a.quad.split(","))
    for kv in a.set or []:                                # e.g. --set TRAIL_ATR=3 USE_TARGETS=0 MAX_HOLD_SESSIONS=40
        k, v = kv.split("=")
        old = CFG.get(k)
        if isinstance(old, bool) or v in ("True", "False"):
            cfg[k] = v in ("1", "True")
        elif isinstance(old, tuple) or "," in v:
            cfg[k] = tuple(x for x in v.split(",") if x)
        else:
            try:
                cfg[k] = type(old)(float(v)) if isinstance(old, (int, float)) else float(v)
            except ValueError:
                cfg[k] = v
    # symbol_trades imports the module-level CFG defaults through S; pass cfg explicitly (already done)
    t0 = time.time()
    daily, intra = D.load_cache(None, a.cache_dir)
    sectors = json.load(open(os.path.join(a.cache_dir, "sectors.json"))) if os.path.exists(os.path.join(a.cache_dir, "sectors.json")) else {}
    spy = daily.get(cfg["BENCHMARK"])
    if spy is None:
        raise SystemExit(f"{cfg['BENCHMARK']} missing from cache")
    etf_close = {e: daily[e]["Close"] for e in cfg["SECTOR_ETFS"] if e in daily}
    rot = S.sector_rotation(etf_close, spy["Close"], cfg) if etf_close else None
    skip = set([cfg["BENCHMARK"]] + cfg["SECTOR_ETFS"])
    syms = [s for s in daily if s not in skip and s in intra]
    if a.tickers:
        syms = [s for s in syms if s in a.tickers]

    if a.fundamentals_snapshot:
        print("!! --fundamentals-snapshot: applying TODAY's fundamentals to all history (look-ahead bias)")
        keep = []
        for s in syms:
            g = S.fundamentals_gate(S.fetch_fundamental_snapshot(s), cfg)
            if g["pass"] or all(f.startswith("earnings_in") for f in g["fails"]):
                keep.append(s)
        print(f"   {len(keep)}/{len(syms)} symbols pass today's fundamentals gate")
        syms = keep

    jobs = [(s, daily[s], intra[s], spy, rot, sectors.get(s), cfg, not a.no_market, a.start) for s in syms]
    all_trades, funnel = [], {"days": 0, "tech_ok": 0, "setups": 0, "triggered": 0, "invalidated": 0}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for sym, trades, st in ex.map(_worker, jobs, chunksize=4):
            if "error" in st:
                print(f"  [{sym}] error: {st['error']}")
                continue
            all_trades += trades
            for k in funnel:
                funnel[k] += st.get(k, 0)
    print(f"simulated {len(syms)} symbols in {time.time() - t0:.0f}s")
    print("funnel (symbol-days): " + " -> ".join(f"{k} {v}" for k, v in funnel.items()))
    tdf = pd.DataFrame(all_trades)
    if tdf.empty:
        print("no trades")
        return
    # equity curve spans the whole tradable window, not just first-to-last trade
    win_start = max(pd.Timestamp(a.start) if a.start else pd.Timestamp.min,
                    min(pd.Timestamp(intra[s].index[0].date()) for s in syms) + pd.Timedelta(days=int(cfg["VP_SESSIONS"] * 1.45)))
    win_end = max(pd.Timestamp(intra[s].index[-1].date()) for s in syms)
    eq, acc = simulate_portfolio(tdf, daily, cfg, win_start, win_end)
    txt = stats_report(eq, acc, tdf, spy["Close"], cfg)
    print(txt)
    os.makedirs("results", exist_ok=True)
    tag = a.tag or time.strftime("%Y%m%d_%H%M")
    tdf.to_csv(f"results/mds_signals_{tag}.csv", index=False)
    acc.to_csv(f"results/mds_trades_{tag}.csv", index=False)
    eq.to_csv(f"results/mds_equity_{tag}.csv")
    open(f"results/mds_stats_{tag}.txt", "w").write(txt)
    print(f"\nsaved results/mds_*_{tag}.*")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--universe", choices=["finviz", "sp500-pit"], default="finviz")
    c.add_argument("--size", type=int, default=2, help="finviz market-cap bucket (0 small, 1 mid, 2 +mid, 3 +micro)")
    c.add_argument("--tickers", nargs="*")
    c.add_argument("--daily-years", type=float, default=5)
    c.add_argument("--intraday-years", type=float, default=3)
    c.add_argument("--cache-dir", default=DEFAULT_CACHE)
    c.add_argument("--refresh", action="store_true")
    s = sub.add_parser("simulate")
    s.add_argument("--cache-dir", default=DEFAULT_CACHE)
    s.add_argument("--tickers", nargs="*")
    s.add_argument("--start", default=None)
    s.add_argument("--no-sector", action="store_true")
    s.add_argument("--no-market", action="store_true")
    s.add_argument("--fundamentals-snapshot", action="store_true")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--tag", default=None)
    s.add_argument("--quad", default=None, help="comma list of allowed RRG quadrants, e.g. Leading,Improving")
    s.add_argument("--set", nargs="*", help="override MDS_CONFIG keys, e.g. TRAIL_ATR=3 USE_TARGETS=0")
    a = p.parse_args()
    cmd_cache(a) if a.cmd == "cache" else cmd_simulate(a)


if __name__ == "__main__":
    main()
