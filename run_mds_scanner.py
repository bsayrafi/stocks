"""
run_mds_scanner.py
==================
Live runner for the Multi-Day Swing strategy. Two modes:

  daily  (run after the close, ~16:30 ET / 23:30 Israel)
         finviz universe -> market regime -> sector rotation -> trend + flow ->
         fundamentals -> volume profile + S/R levels -> watchlist + HTML report + ntfy
  entry  (run every 15 minutes during the session; exits quietly outside it)
         watchlist -> 15m bars -> VWAP/EMA9/RVOL trigger -> ntfy alert with
         entry / stop / TP1 / TP2 / R:R

Usage:
    python3 run_mds_scanner.py daily --size 2
    python3 run_mds_scanner.py daily --tickers NVDA MSFT ANET
    python3 run_mds_scanner.py entry
    ./run_local_buyentry.sh run_mds_scanner.py daily --size 2      # with keys from .env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import pandas as pd

sys.path.insert(0, os.getcwd())

from mds_config import MDS_CONFIG as CFG
import mds_data as D
import mds_strategy as S
import mds_report as R

WATCHLIST = os.path.join(CFG["LIVE_DIR"], "mds_watchlist.json")
ALERT_STATE = os.path.join(CFG["LIVE_DIR"], "mds_alert_state.json")
SIZE_NAME = {0: "Small", 1: "Mid", 2: "Large", 3: "Micro+"}


# --------------------------------------------------------------------------- helpers

def ntfy_text(title: str, message: str, priority: str = "default", tags: str = "") -> None:
    import requests
    topic = os.environ.get("NTFY_TOPIC")
    if not CFG["NTFY_ENABLED"] or not topic:
        print(f"[ntfy off] {title}\n{message}")
        return
    try:
        requests.post(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"), timeout=15,
                      headers={"Title": title.encode("ascii", "ignore").decode(), "Priority": priority, "Tags": tags})
    except Exception as e:  # noqa: BLE001
        print(f"ntfy failed: {e}")


def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def setup_from_dict(d: dict) -> S.Setup:
    names = {f.name for f in fields(S.Setup)}
    return S.Setup(**{k: v for k, v in d.items() if k in names})


def sessions_since(date_str: str, calendar: pd.DatetimeIndex) -> int:
    return int((calendar > pd.Timestamp(date_str)).sum())


# --------------------------------------------------------------------------- daily

def run_daily(a) -> None:
    t0 = time.time()
    cfg = CFG
    # 1. universe
    if a.tickers:
        tickers, sectors, uni_name = a.tickers, {}, "custom"
    else:
        u = D.finviz_universe(a.size)
        tickers, sectors = u["Ticker"].tolist(), dict(zip(u["Ticker"], u["Sector"]))
        uni_name = f"finviz {SIZE_NAME.get(a.size, a.size)}"
    print(f"universe: {len(tickers)} tickers ({uni_name})")
    funnel = {"universe": len(tickers)}

    # 2. daily bars (+ SPY + sector ETFs)
    daily = D.load_daily(list(dict.fromkeys(tickers + [cfg["BENCHMARK"]] + cfg["SECTOR_ETFS"])), a.source)
    spy = daily.get(cfg["BENCHMARK"])
    if spy is None:
        raise SystemExit("could not load SPY")
    asof = spy.index[-1]
    market_ok = bool(S.market_regime(spy, cfg).iloc[-1])
    rot = S.sector_rotation({e: daily[e]["Close"] for e in cfg["SECTOR_ETFS"] if e in daily}, spy["Close"], cfg)
    sec_table = S.sector_table(rot)
    print(sec_table.to_string(index=False))

    # sector for --tickers lists comes from yfinance .info (fetched anyway for fundamentals)
    snaps: dict[str, dict] = {}
    if not sectors:
        with ThreadPoolExecutor(cfg["MAX_WORKERS"]) as ex:
            for t, sn in zip(tickers, ex.map(S.fetch_fundamental_snapshot, tickers)):
                snaps[t] = sn
                sectors[t] = (sn.get("info") or {}).get("sector")

    # 3. technical + sector gates (cheap, all names)
    feats, stage1, near = {}, [], []
    for t in tickers:
        df = daily.get(t)
        if df is None or len(df) < 260 or df.index[-1] != asof:
            continue
        f = S.daily_features(df, spy["Close"], cfg)
        feats[t] = f
        row = f.iloc[-1]
        sok, quad = S.sector_ok(rot, S.sector_etf(sectors.get(t)), asof, cfg)
        why = S.tech_fail_reasons(row, cfg) if not bool(row["tech_ok"]) else []
        if not sok:
            why.append(f"sector {S.sector_etf(sectors.get(t))}:{quad}")
        if not why:
            stage1.append(t)
        elif len(why) == 1:
            near.append({"symbol": t, "sector": sectors.get(t), "why": why[0] + " (fundamentals not checked)"})
    funnel["with data"] = len(feats)
    funnel["trend+flow+sector"] = len(stage1)

    # 4. fundamentals (only for survivors)
    missing = [t for t in stage1 if t not in snaps]
    with ThreadPoolExecutor(cfg["MAX_WORKERS"]) as ex:
        for t, sn in zip(missing, ex.map(S.fetch_fundamental_snapshot, missing)):
            snaps[t] = sn
    stage2, fund = [], {}
    for t in stage1:
        g = S.fundamentals_gate(snaps[t], cfg)
        fund[t] = g
        if g["pass"]:
            stage2.append(t)
        elif len(g["fails"]) == 1:
            near.append({"symbol": t, "sector": sectors.get(t), "why": g["fails"][0]})
    funnel["fundamentals"] = len(stage2)

    # 5. 15m bars -> volume profile -> levels
    intra = D.load_intraday(stage2, a.source) if stage2 else {}
    setups = []
    for t in stage2:
        s, why = S.evaluate_setup(t, daily[t], feats[t], asof, intra.get(t), sectors.get(t), rot, cfg)
        if s is None:
            near.append({"symbol": t, "sector": sectors.get(t), "why": why[0] if why else "?"})
            continue
        d = s.to_dict()
        d["fundamentals"] = fund[t]
        setups.append(d)
    setups.sort(key=lambda d: -d["score"])
    funnel["setups (levels, R:R)"] = len(setups)
    if not market_ok and cfg["REQUIRE_MARKET_UPTREND"]:
        print("!! SPY below SMA200 -- setups listed for information, NOT added to the watchlist")

    # 6. watchlist: today's setups + still-valid older ones not replaced today
    old = _load_json(WATCHLIST, {"setups": []})
    state = _load_json(ALERT_STATE, {})
    new_syms = {d["symbol"] for d in setups}
    carried = [d for d in old.get("setups", [])
               if d["symbol"] not in new_syms
               and sessions_since(d["date"], spy.index) < cfg["SETUP_VALID_SESSIONS"]
               and state.get(f'{d["symbol"]}|{d["date"]}', {}).get("status") not in ("triggered", "invalidated")]
    live = (setups if (market_ok or not cfg["REQUIRE_MARKET_UPTREND"]) else []) + carried
    _save_json(WATCHLIST, {"generated": pd.Timestamp.now(tz=D.ET).isoformat(), "asof": str(asof.date()), "setups": live})

    # 7. report
    stamp = time.strftime("%b_%d_%H_%M")
    title = f"MDS {SIZE_NAME.get(a.size, 'Custom') if not a.tickers else 'Custom'} swing setups {asof.date()}"
    near_df = pd.DataFrame(near).drop_duplicates("symbol") if near else pd.DataFrame(columns=["symbol", "sector", "why"])
    html_doc = R.build_html(title, {"market_ok": market_ok, "generated": time.strftime("%Y-%m-%d %H:%M"),
                                    "universe": uni_name, "asof": str(asof.date())},
                            setups, sec_table, near_df, daily, funnel)
    os.makedirs(cfg["REPORT_DIR"], exist_ok=True)
    path = os.path.join(cfg["REPORT_DIR"], f"MDS_{SIZE_NAME.get(a.size, 'Custom') if not a.tickers else 'Custom'}_{stamp}.html")
    with open(path, "w") as f:
        f.write(html_doc)
    pd.DataFrame([{k: v for k, v in d.items() if k not in ("fundamentals", "supports", "resistances")} for d in setups]
                 ).to_csv(path.replace(".html", ".csv"), index=False)
    print(f"\nfunnel: {funnel}\nreport: {path}\nwatchlist: {len(live)} setups ({len(carried)} carried) -> {WATCHLIST}")
    print(f"done in {time.time() - t0:.0f}s")

    if cfg["NTFY_ENABLED"] and not a.no_ntfy:
        try:
            from enrich_html import send_ntfy_file
            msg = f"{len(setups)} setups: " + (", ".join(f'{d["symbol"]} <={d["max_entry"]:.2f}' for d in setups[:12]) or "none")
            send_ntfy_file(path, title=title, message=msg, tags="chart_with_upwards_trend",
                           priority="high" if setups else "default")
        except Exception as e:  # noqa: BLE001
            print(f"ntfy failed: {e}")


# --------------------------------------------------------------------------- entry

def in_session(now: pd.Timestamp) -> bool:
    m = now.hour * 60 + now.minute
    return now.weekday() < 5 and S._hhmm(CFG["SESSION_OPEN"]) + 15 <= m <= S._hhmm(CFG["SESSION_CLOSE"]) + 5


def run_entry(a) -> None:
    now = D.now_et()
    if not a.force and not in_session(now):
        print(f"{now:%Y-%m-%d %H:%M} ET: market closed -- nothing to do")
        return
    wl = _load_json(WATCHLIST, {"setups": []})
    state = _load_json(ALERT_STATE, {})
    setups = [setup_from_dict(d) for d in wl.get("setups", [])
              if state.get(f'{d["symbol"]}|{d["date"]}', {}).get("status") not in ("triggered", "invalidated")]
    if not setups:
        print("watchlist empty")
        return
    intra = D.load_intraday([s.symbol for s in setups], a.source, days=CFG["RVOL_LOOKBACK_SESSIONS"] + 8)
    rows = []
    for s in setups:
        df = intra.get(s.symbol)
        if df is None:
            rows.append((s.symbol, "no data", ""))
            continue
        prep = S.prepare_intraday(df)
        res = S.find_entry(prep, s)
        key = f"{s.symbol}|{s.date}"
        last = prep.iloc[-1]
        info = f'last {last["Close"]:.2f} vwap {last["vwap"]:.2f} rvol {last["rvol"]:.2f}' if pd.notna(last["rvol"]) else ""
        rows.append((s.symbol, res["status"], info))
        if res["status"] in ("triggered", "invalidated") and state.get(key, {}).get("status") != res["status"]:
            state[key] = {"status": res["status"], "time": str(res.get("time")), **{k: v for k, v in res.items() if k not in ("status", "time")}}
            if res["status"] == "triggered":
                px = res["trigger_close"]
                risk = px - s.stop
                how = res.get("entry_type", "")
                if res.get("level"):
                    how += f' @ {res["level"]:.2f}'
                msg = (f'{s.symbol} 15m {how} {pd.Timestamp(res["time"]):%H:%M} ET @ {px:.2f}\n'
                       f'Stop {s.stop:.2f} ({s.stop_src}) -{risk / px * 100:.1f}%\n'
                       f'TP1 {s.tp1:.2f} ({s.tp1_src}) R:R {res["rr"]:.2f}\nTP2 {s.tp2:.2f} ({s.tp2_src})\n'
                       f'VWAP {res["vwap"]:.2f} RVOL {res["rvol"]:.2f} | sector {s.sector_etf} {s.quadrant}\n'
                       f'Size: shares = risk$ / {risk:.2f}')
                ntfy_text(f"MDS BUY {s.symbol}", msg, priority="high", tags="rotating_light")
            else:
                ntfy_text(f"MDS void {s.symbol}", f"{s.symbol} traded through stop {s.stop:.2f} before a trigger -- setup removed",
                          priority="low", tags="x")
    _save_json(ALERT_STATE, state)
    print(f"{now:%Y-%m-%d %H:%M} ET")
    for r in rows:
        print(f"  {r[0]:6s} {r[1]:12s} {r[2]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["daily", "entry"])
    p.add_argument("--size", type=int, default=2, help="finviz market-cap bucket: 0 small, 1 mid, 2 +mid, 3 +micro")
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--source", choices=["yfinance", "alpaca"], default=None)
    p.add_argument("--no-ntfy", action="store_true")
    p.add_argument("--force", action="store_true", help="entry mode: run even outside market hours")
    a = p.parse_args()
    if a.no_ntfy:
        CFG["NTFY_ENABLED"] = False
    run_daily(a) if a.mode == "daily" else run_entry(a)


if __name__ == "__main__":
    main()
