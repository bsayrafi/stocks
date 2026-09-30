#!/usr/bin/env python
"""
portfolio_sim.py - turn the backtest's R-multiple trades into a dollar account.

    python portfolio_sim.py backtest_out/bt_trades.csv
    python portfolio_sim.py backtest_out/bt_trades.csv --equity 10000 --setups PULLBACK --exit ema_trail --risk 0.25 0.5 1.0

What it does
  * takes the trades of ONE entry model / exit model / setup list from bt_trades.csv (default: signal_close, ema_trail, PULLBACK);
  * sizes every trade so that hitting the initial stop loses `risk` % of the account at that moment
    (position = risk$ / stop distance), capped at --max-pos % of the account per position, at --max-positions open at once,
    and at the cash available (no margin): a trade that does not fit is skipped or shrunk;
  * reports final equity, CAGR, closed-trade max drawdown, how much of the account was deployed;
  * --bootstrap: resamples the trades to show how wide the range of outcomes is, both with the observed edge and with the edge
    reduced to what matched random days earned (= "the signal added nothing").

What it cannot do: it marks open positions at cost (so the drawdown is understated), ignores commissions, taxes and
dividends, and assumes you follow every signal exactly. It is a replay of the past, not a forecast.
"""
import argparse
import numpy as np
import pandas as pd


def load(path, entry, exit_, setups):
    t = pd.read_csv(path)
    t = t[(t["entry_model"] == entry) & (t["exit_model"] == exit_)]
    if setups:
        t = t[t["setup"].isin(setups)]
    t = t.dropna(subset=["entry", "stop", "ret_pct"]).copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    return t.sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def simulate(t, equity0=10000.0, risk_pct=0.5, max_pos_pct=25.0, max_positions=8):
    cash, open_pos, curve, taken, skipped = equity0, [], [], 0, 0
    exposure_log = []                                    # (date, deployed fraction of equity) at every event
    last_date = None

    def close_until(date):
        nonlocal cash
        for p in sorted([p for p in open_pos if p["exit"] <= date], key=lambda p: p["exit"]):
            cash += p["notional"] + p["pnl"]
            open_pos.remove(p)
            curve.append((p["exit"], cash + sum(q["notional"] for q in open_pos)))

    for _, r in t.iterrows():
        close_until(r["signal_date"])
        eq = cash + sum(p["notional"] for p in open_pos)
        exposure_log.append((r["signal_date"], sum(p["notional"] for p in open_pos) / eq))
        risk_frac = (r["entry"] - r["stop"]) / r["entry"]
        if risk_frac <= 0 or len(open_pos) >= max_positions:
            skipped += 1
            continue
        notional = min(eq * risk_pct / 100 / risk_frac, eq * max_pos_pct / 100, cash)
        if notional < 0.02 * eq:
            skipped += 1
            continue
        cash -= notional
        open_pos.append({"exit": r["exit_date"], "notional": notional, "pnl": notional * r["ret_pct"] / 100})
        taken += 1
    close_until(pd.Timestamp("2100-01-01"))
    final = cash
    cv = pd.Series({d: v for d, v in curve}).sort_index() if curve else pd.Series(dtype=float)
    peak = cv.cummax() if len(cv) else cv
    dd = ((cv / peak) - 1).min() * 100 if len(cv) else 0.0
    years = max((t["exit_date"].max() - t["signal_date"].min()).days / 365.25, 1e-9)
    ex = pd.DataFrame(exposure_log, columns=["d", "x"])
    return {"final": final, "return_%": (final / equity0 - 1) * 100, "CAGR_%": ((final / equity0) ** (1 / years) - 1) * 100,
            "max_DD_%": dd, "taken": taken, "skipped": skipped, "years": years,
            "avg_deployed_%": ex["x"].mean() * 100 if len(ex) else 0, "peak_deployed_%": ex["x"].max() * 100 if len(ex) else 0}


def bootstrap(t, equity0, risk_pct, n=5000, seed=1):
    """Fixed-fraction compounding over resampled trades (no capacity limit): the spread of outcomes for the SAME number of trades."""
    rng = np.random.default_rng(seed)
    r = t["r"].to_numpy(dtype=float)
    out = {}
    scen = {"observed edge": r}
    if "ctrl_r" in t and t["ctrl_r"].notna().any():
        scen["edge reduced to matched-random-day level"] = r - (r.mean() - t["ctrl_r"].mean())
    scen["no edge at all (mean R = 0)"] = r - r.mean()
    for name, x in scen.items():
        idx = rng.integers(0, len(x), size=(n, len(x)))
        g = np.log1p(np.clip(risk_pct / 100 * x[idx], -0.99, None)).sum(axis=1)
        fin = equity0 * np.exp(g)
        out[name] = (x.mean(), np.percentile(fin, [5, 25, 50, 75, 95]), (fin < equity0).mean() * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trades_csv")
    ap.add_argument("--equity", type=float, default=10000)
    ap.add_argument("--entry", default="signal_close")
    ap.add_argument("--exit", default="ema_trail")
    ap.add_argument("--setups", nargs="*", default=["PULLBACK"])
    ap.add_argument("--risk", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--max-pos", type=float, default=25.0, help="max %% of the account in one position")
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--bootstrap", action="store_true")
    a = ap.parse_args()
    t = load(a.trades_csv, a.entry, a.exit, a.setups)
    if t.empty:
        raise SystemExit("no trades match those filters")
    print(f"{len(t)} trades ({a.entry} entry, {a.exit} exit, setups {a.setups}); signals {t['signal_date'].min().date()} .. {t['exit_date'].max().date()}")
    print(f"average R {t['r'].mean():+.3f}, average stop distance {((t['entry'] - t['stop']) / t['entry']).mean() * 100:.1f}% of price, "
          f"average hold {t['hold_days'].mean():.1f} days\n")
    rows = []
    for rp in a.risk:
        s = simulate(t, a.equity, rp, a.max_pos, a.max_positions)
        rows.append({"risk_per_trade_%": rp, "final_$": round(s["final"]), "return_%": round(s["return_%"], 1), "CAGR_%": round(s["CAGR_%"], 1),
                     "max_DD_%": round(s["max_DD_%"], 1), "trades_taken": s["taken"], "skipped": s["skipped"],
                     "avg_deployed_%": round(s["avg_deployed_%"]), "peak_deployed_%": round(s["peak_deployed_%"])})
    print(f"REPLAY of ${a.equity:,.0f} (max {a.max_pos:g}% per position, max {a.max_positions} positions, no margin):")
    print(pd.DataFrame(rows).to_string(index=False))
    if a.bootstrap:
        rp = a.risk[min(1, len(a.risk) - 1)]
        print(f"\nBOOTSTRAP at {rp:g}% risk per trade, {len(t)} trades, 5,000 resamples (final account, no capacity limit):")
        for name, (m, pct, ploss) in bootstrap(t, a.equity, rp).items():
            print(f"  {name:45s} mean R {m:+.3f} | 5%: ${pct[0]:>8,.0f}  25%: ${pct[1]:>8,.0f}  median: ${pct[2]:>8,.0f}  75%: ${pct[3]:>8,.0f}  95%: ${pct[4]:>8,.0f} | chance of ending below start: {ploss:.0f}%")


if __name__ == "__main__":
    main()
