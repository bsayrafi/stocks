"""
trend_tsm_live_signal.py
============================
Daily signal generator for the TSM trend-following strategy (trend_tsm_backtest.py).
ADVISORY ONLY -- this places no broker orders, ever. It tells you what the
validated strategy says to do; you (or your own separate execution code) decide
whether to actually act on it.

WHY ONCE A DAY, NOT HOURLY: daily_trend_confirmed is built from DAILY bars (the
20/60/120/250-day momentum horizons, OBV slope), explicitly shifted by one row so
a value is only ever based on the PRIOR day's completed close. That value cannot
change until the next daily bar closes -- running this hourly would recompute the
identical signal five times before there's anything new to compute. Run it once,
any time between today's close and tomorrow's open; the model always trades at
"next session's open," so there's no rush.

WHY NO STOP-LOSS / TAKE-PROFIT NOTIFICATION: we tested adding one (fixed and
trailing ATR stops, several multiples) against this exact strategy and every
configuration made total returns WORSE -- it cut off the big winners that dip
hard before recovering more than it saved on losers (see trend_tsm_backtest.py's
SIM_CONFIG comment for the full numbers). The only validated exit is the trend
signal flipping off. This script instead sends an INFORMATIONAL drawdown flag
on open positions -- awareness only, never a suggested action -- per your call
to keep risk-awareness passive rather than reintroducing a stop.

WHY TODAY'S UNIVERSE, NOT THE SURVIVORSHIP-FREE ONE: survivorship_free_universe.py
exists to fix a BACKTESTING bug (today's index membership can't tell you which
stocks used to be members years ago). Going forward, there's nothing to correct --
today's real S&P 500 sector membership is exactly the live universe you want. This
script uses fvg_data_pipeline_hourly.load_tickers_by_sector() directly, same as
before the backtest fix, plus every symbol the shadow portfolio currently holds
(even one that's since dropped out of the sector list) so open positions can
still be evaluated for an exit.

WHAT IT MAINTAINS: a local "shadow portfolio" (live/tsm_live_state.json) mirroring
exactly what trend_tsm_backtest.simulate_portfolio() would do if every signal were
taken -- same ATR position sizing, same MAX_POSITIONS cap, against a notional
--capital you choose. This is bookkeeping for the notification only; it is NOT
connected to your broker account and places no orders. If you're paper- or
live-trading this alongside it, treat the digest as instructions and manage your
actual broker positions yourself.

EACH RUN SENDS ONE DIGEST PUSH (via ntfy.sh, not one push per symbol) with up to
four sections:
  - CONFIRMED ENTRIES   -- signal fired and passed the ADX/RSI entry filter,
                            and a shadow-portfolio slot was free to take it
  - CANDIDATES          -- signal fired and passed the ADX/RSI entry filter,
                            but the shadow portfolio was full (MAX_POSITIONS) --
                            these would have been entries with a free slot
  - CONFIRMED EXITS     -- a held position's trend just flipped off
  - WATCHLIST           -- momentum turned positive but not yet confirmed, or
                            confirmed but filtered out (weak/late setup) --
                            "this looks like it might be starting," not a signal
  - RISK AWARENESS       -- held positions down more than --drawdown-alert-pct
                            from entry. Informational only, no action implied.

Setup:
  - Same .env as the rest of this repo (APCA_API_KEY_ID / APCA_API_SECRET_KEY).
  - Set NTFY_TOPIC (env var or --ntfy-topic) to a PRIVATE, hard-to-guess string --
    ntfy.sh topics are public to anyone who knows the name. Install the ntfy app
    (iOS/Android), subscribe to that same topic name, done -- no account needed.

Usage:
    python3 trend_tsm_live_signal.py --sectors Technology "Health Care" Financials --capital 100000
    # schedule this once daily after market close -- see the accompanying launchd setup
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from trend_data_pipeline import (
    CONFIG as BASE_CONFIG, fetch_bars, add_indicators, build_daily_trend_context,
    run_with_timeout,
)
from trend_tsm_backtest import SIM_CONFIG, passes_entry_filters
from trend_maturity import compute_maturity, describe_stage
from fvg_data_pipeline_hourly import load_tickers_by_sector
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))   # .env in the folder you run from (or a parent)
    load_dotenv()                            # .env next to this file (or a parent)
except ImportError:
    pass


STATE_PATH = "live/tsm_live_state.json"
LIVE_LOOKBACK_YEARS = 2   # enough for the 250-day momentum horizon + 200-day EMA with room to spare
DRAWDOWN_ALERT_PCT_DEFAULT = 0.15  # informational only -- see docstring


def load_state(path=STATE_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"positions": {}, "cash": None, "last_run_date": None}


def save_state(state, path=STATE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def fetch_symbol_context(client, symbol, cfg):
    daily_raw, err = run_with_timeout(fetch_bars, client, symbol, TimeFrame.Day, LIVE_LOOKBACK_YEARS, cfg)
    if err or daily_raw is None or len(daily_raw) < 260:
        return None
    ind = add_indicators(daily_raw, cfg)
    ctx = build_daily_trend_context(daily_raw, cfg)
    tz = cfg["MARKET_TZ"]
    idx = ind.index
    dates = idx.tz_convert(tz).date if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz).date
    ind = ind.set_index(pd.Index(dates, name="date"))
    merged = ind.join(ctx, how="left")
    return compute_maturity(merged)  # adds days_above_ema200, pct_in_252d_range, horizon_skew -- informational, see trend_maturity.py


def send_ntfy(topic, title, message, priority="default"):
    if not topic:
        print("NTFY_TOPIC not set -- printing digest instead of pushing:\n")
        print(f"{title}\n{message}")
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            timeout=10,
        )
    except Exception as e:
        print(f"ntfy push failed ({e}) -- digest was:\n{title}\n{message}")


def entry_quality_notes(adx, rsi14, sim_cfg):
    """Short, descriptive feedback tags about how close an entry sits to the
    configured entry-filter edges. Purely informational (like trend_maturity's
    stage label) -- these thresholds are derived directly from the filter
    itself (75% of the ADX ceiling, 5 RSI points above the exclusion floor),
    not independently validated cutoffs, so they explain the signal rather
    than gate it."""
    notes = []
    max_adx = sim_cfg.get("ENTRY_MAX_ADX")
    if max_adx and pd.notna(adx) and adx >= 0.75 * max_adx:
        notes.append("ADX near filter ceiling")
    rsi_range = sim_cfg.get("ENTRY_RSI_EXCLUDE_RANGE")
    if rsi_range and pd.notna(rsi14) and rsi14 < rsi_range[1] + 5:
        notes.append("RSI near filter floor")
    return notes


def export_full_digest_csv(today_str, entries, exits, risk_flags, watchlist, candidates, export_dir="live"):
    """Write every entry/candidate/exit/risk-flag/watchlist row -- untruncated --
    to a dated CSV, in the same directory as the log/state files. The console/
    ntfy digest below trims the watchlist for readability in a push
    notification; this file never does, so it's the source of truth for
    anything downstream (e.g. an Excel export)."""
    os.makedirs(export_dir, exist_ok=True)
    rows = []
    for e in entries:
        rows.append({
            "section": "confirmed_entry", "symbol": e["symbol"], "price": e.get("price"),
            "shares": e.get("shares"), "composite_momentum": e.get("composite_momentum"),
            "adx": e.get("adx"), "rsi14": e.get("rsi14"), "stage": e.get("stage"),
            "quality_notes": "; ".join(e.get("quality_notes", [])), "reason": "", "entry_date": "",
            "unrealized_pct": "", "as_of": e.get("as_of"), "score": e.get("score"), "rank": e.get("rank"),
        })
    for c in candidates:
        rows.append({
            "section": "candidate", "symbol": c["symbol"], "price": c.get("price"),
            "shares": c.get("shares"), "composite_momentum": c.get("composite_momentum"),
            "adx": c.get("adx"), "rsi14": c.get("rsi14"), "stage": c.get("stage"),
            "quality_notes": "; ".join(c.get("quality_notes", [])), "reason": c.get("reason"),
            "entry_date": "", "unrealized_pct": "", "as_of": c.get("as_of"), "score": c.get("score"), "rank": c.get("rank"),
        })
    for x in exits:
        rows.append({
            "section": "confirmed_exit", "symbol": x["symbol"], "price": x.get("last_close"),
            "shares": "", "composite_momentum": "", "adx": "", "rsi14": "", "stage": "",
            "quality_notes": "", "reason": "", "entry_date": x.get("entry_date"),
            "unrealized_pct": x.get("unrealized_pct"), "as_of": x.get("as_of"), "score": "", "rank": "",
        })
    for r in risk_flags:
        rows.append({
            "section": "risk_flag", "symbol": r["symbol"], "price": "", "shares": "",
            "composite_momentum": "", "adx": "", "rsi14": "", "stage": "", "quality_notes": "",
            "reason": "trend still confirmed, exit rule unchanged", "entry_date": r.get("entry_date"),
            "unrealized_pct": r.get("unrealized_pct"), "as_of": r.get("as_of"), "score": "", "rank": "",
        })
    for w in watchlist:
        rows.append({
            "section": "watchlist", "symbol": w["symbol"], "price": "", "shares": "",
            "composite_momentum": w.get("composite_momentum"), "adx": w.get("adx"),
            "rsi14": w.get("rsi14"), "stage": "", "quality_notes": "", "reason": w.get("reason"),
            "entry_date": "", "unrealized_pct": "", "as_of": w.get("as_of"), "score": "", "rank": "",
        })
    path = os.path.join(export_dir, f"tsm_live_{today_str}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def score_slot_seekers(seekers):
    """Attach a 0-1 composite score to every confirmed & filter-passing signal
    from today's run (mutates seekers in place), so scarce slots and the day's
    remaining cash go to the best-looking setups first instead of whichever
    ticker happens to sort first alphabetically.

    CAVEAT -- read before trusting this too much: the per-trade analysis in
    trend_tsm_backtest.py already tested whether composite_momentum (or the
    other available indicators) can rank *which* confirmed trade will turn out
    more profitable, and found none of them do. This score does not overturn
    that finding -- it's a principled tiebreaker for capital allocation
    ("something beats alphabetical order"), not a validated way to pick
    winners. Don't expect it to raise returns on its own; it just replaces an
    arbitrary rule with a documented one.

    Method: percentile-rank each factor across today's pool (0..1) and average
    the three, equally weighted:
      - momentum : higher composite_momentum ranks higher (the core signal).
      - ADX      : LOWER ranks higher -- mirrors why ENTRY_MAX_ADX exists at
                   all (an overextended, high-ADX entry hurt returns in
                   backtest), so a fresher trend outranks a more forceful one
                   even when both clear the same ceiling.
      - RSI      : higher ranks higher -- mirrors why excluding RSI<50
                   improved returns, i.e. more relative strength (within the
                   passing range) outranks less.
    """
    if not seekers:
        return
    df = pd.DataFrame(seekers)
    mom = pd.to_numeric(df["composite_momentum"], errors="coerce")
    adx = pd.to_numeric(df["adx"], errors="coerce")
    rsi = pd.to_numeric(df["rsi14"], errors="coerce")
    ranks = pd.concat([mom.rank(pct=True), (-adx).rank(pct=True), rsi.rank(pct=True)], axis=1)
    scores = ranks.mean(axis=1, skipna=True).fillna(0.5)
    for seeker, s in zip(seekers, scores):
        seeker["score"] = round(float(s), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sectors", nargs="*", default=BASE_CONFIG["SECTORS"])
    parser.add_argument("--capital", type=float, default=SIM_CONFIG["STARTING_CAPITAL"],
                         help="Notional capital the shadow portfolio sizes positions against.")
    parser.add_argument("--max-positions", type=int, default=SIM_CONFIG["MAX_POSITIONS"])
    parser.add_argument("--risk-pct", type=float, default=SIM_CONFIG["RISK_PCT_PER_TRADE"])
    parser.add_argument("--entry-max-adx", type=float, default=SIM_CONFIG["ENTRY_MAX_ADX"])
    parser.add_argument("--entry-rsi-exclude", type=float, nargs=2, default=SIM_CONFIG["ENTRY_RSI_EXCLUDE_RANGE"],
                         metavar=("LOW", "HIGH"))
    parser.add_argument("--drawdown-alert-pct", type=float, default=DRAWDOWN_ALERT_PCT_DEFAULT,
                         help="Informational-only flag threshold; no action is implied or taken.")
    parser.add_argument("--ntfy-topic", default=os.environ.get("NTFY_TOPIC"))
    parser.add_argument("--state-path", default=STATE_PATH)
    args = parser.parse_args()

    sim_cfg = dict(SIM_CONFIG)
    sim_cfg["MAX_POSITIONS"] = args.max_positions
    sim_cfg["RISK_PCT_PER_TRADE"] = args.risk_pct
    sim_cfg["ENTRY_MAX_ADX"] = args.entry_max_adx
    sim_cfg["ENTRY_RSI_EXCLUDE_RANGE"] = tuple(args.entry_rsi_exclude) if args.entry_rsi_exclude else None

    # .env uses Alpaca's standard names (APCA_*); older names kept as fallbacks
    api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise SystemExit("Alpaca keys not found: set APCA_API_KEY_ID / APCA_API_SECRET_KEY in your .env file.")
    client = StockHistoricalDataClient(api_key, api_secret)

    state = load_state(args.state_path)
    if state["cash"] is None:
        state["cash"] = args.capital
    held_symbols = set(state["positions"].keys())

    universe = set()
    for sector in args.sectors:
        universe.update(load_tickers_by_sector(sector, verbose=False))
    universe |= held_symbols  # always re-evaluate anything currently held, even if it left the sector list
    print(f"Live universe: {len(universe)} symbols ({len(held_symbols)} currently held)")

    entries, exits, watchlist, risk_flags, candidates = [], [], [], [], []
    slot_seekers = []  # confirmed & filter-passing -- not yet assigned a slot/cash
    today_str = datetime.now(timezone.utc).date().isoformat()

    for symbol in sorted(universe):
        ctx = fetch_symbol_context(client, symbol, BASE_CONFIG)
        if ctx is None or ctx.empty:
            continue
        last = ctx.iloc[-1]
        last_date = str(ctx.index[-1])
        confirmed = bool(last.get("daily_trend_confirmed", False))

        if symbol in state["positions"]:
            pos = state["positions"][symbol]
            unrealized_pct = (last["close"] - pos["entry_price"]) / pos["entry_price"]
            if not confirmed:
                exits.append({
                    "symbol": symbol, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                    "last_close": last["close"], "unrealized_pct": unrealized_pct, "as_of": last_date,
                })
                proceeds = pos["shares"] * last["close"]
                state["cash"] += proceeds
                del state["positions"][symbol]
            else:
                if unrealized_pct <= -abs(args.drawdown_alert_pct):
                    risk_flags.append({
                        "symbol": symbol, "entry_date": pos["entry_date"], "unrealized_pct": unrealized_pct,
                        "as_of": last_date,
                    })
            continue

        passes_filter = passes_entry_filters(last, sim_cfg)
        atr = last.get("atr")
        if confirmed and passes_filter and pd.notna(atr) and atr > 0:
            # Defer the slot/cash decision until every symbol has been seen and
            # scored (see score_slot_seekers below) -- alphabetical iteration
            # order should never decide who gets a scarce slot or the day's
            # remaining cash.
            slot_seekers.append({
                "symbol": symbol, "price": last["close"], "atr": float(atr),
                "composite_momentum": last.get("composite_momentum"), "adx": last.get("adx"),
                "rsi14": last.get("rsi14"), "as_of": last_date,
                "stage": describe_stage(last.get("days_above_ema200"), last.get("pct_in_252d_range")),
            })
        elif confirmed and not passes_filter:
            watchlist.append({
                "symbol": symbol, "reason": "confirmed but filtered (weak/late setup)",
                "adx": last.get("adx"), "rsi14": last.get("rsi14"), "as_of": last_date,
            })
        elif not confirmed and last.get("composite_momentum", -1) > 0 and last.get("obv_slope_norm", -1) <= 0:
            watchlist.append({
                "symbol": symbol, "reason": "momentum positive, volume not confirming yet",
                "composite_momentum": last.get("composite_momentum"), "as_of": last_date,
            })

    # -- score every confirmed & filter-passing signal, then allocate slots/cash
    # in score order (best first) instead of alphabetical order. --------------
    score_slot_seekers(slot_seekers)
    slot_seekers.sort(key=lambda s: s["score"], reverse=True)
    for rank, seeker in enumerate(slot_seekers, start=1):
        portfolio_full = len(state["positions"]) >= sim_cfg["MAX_POSITIONS"]
        dollar_risk = state["cash"] * sim_cfg["RISK_PCT_PER_TRADE"]
        hyp_shares = int(dollar_risk / (sim_cfg["SIZING_ATR_MULT"] * seeker["atr"]))
        cost_basis = hyp_shares * seeker["price"]
        quality_notes = entry_quality_notes(seeker["adx"], seeker["rsi14"], sim_cfg)

        if not portfolio_full and hyp_shares > 0 and cost_basis <= state["cash"]:
            state["positions"][seeker["symbol"]] = {
                "shares": hyp_shares, "entry_price": float(seeker["price"]),
                "entry_date": today_str, "atr_at_entry": seeker["atr"],
                "entry_stage": seeker["stage"],
            }
            state["cash"] -= cost_basis
            entries.append({
                "symbol": seeker["symbol"], "price": seeker["price"], "shares": hyp_shares,
                "composite_momentum": seeker["composite_momentum"], "adx": seeker["adx"],
                "rsi14": seeker["rsi14"], "as_of": seeker["as_of"], "stage": seeker["stage"],
                "quality_notes": quality_notes, "score": seeker["score"], "rank": rank,
            })
        else:
            if portfolio_full:
                reason = "confirmed & passes filter, but no free slot (portfolio full)"
            else:
                reason = "confirmed & passes filter, slot available but insufficient remaining cash"
                quality_notes.append("would size to 0 sh at current remaining cash" if hyp_shares == 0
                                      else "cost exceeds remaining cash")
            candidates.append({
                "symbol": seeker["symbol"], "price": seeker["price"], "shares": hyp_shares,
                "composite_momentum": seeker["composite_momentum"], "adx": seeker["adx"],
                "rsi14": seeker["rsi14"], "as_of": seeker["as_of"], "stage": seeker["stage"],
                "quality_notes": quality_notes, "reason": reason, "score": seeker["score"], "rank": rank,
            })

    state["last_run_date"] = today_str
    save_state(state, args.state_path)
    export_path = export_full_digest_csv(today_str, entries, exits, risk_flags, watchlist, candidates)

    lines = [f"TSM daily signal -- {today_str}"]
    if entries:
        lines.append(f"\nCONFIRMED ENTRIES ({len(entries)}) -- ranked by score, buy at next open:")
        for e in entries:
            note_suffix = f"  [{'; '.join(e['quality_notes'])}]" if e.get("quality_notes") else ""
            lines.append(f"  #{e['rank']} {e['symbol']}: {e['shares']} sh @ ~{e['price']:.2f}  "
                         f"(score={e['score']:.2f}, mom={e['composite_momentum']:.2f}, ADX={e['adx']:.0f}, RSI={e['rsi14']:.0f})  -- {e['stage']}{note_suffix}")
        total_cost = sum(e['shares'] * e['price'] for e in entries)
        pct_of_capital = total_cost / args.capital if args.capital else 0.0
        lines.append(f"  -> total cost if all filled: ${total_cost:,.2f} "
                     f"({pct_of_capital:.1%} of ${args.capital:,.2f} capital)")
        below_200d = [e['symbol'] for e in entries if 'below its 200d avg' in e['stage']]
        if below_200d:
            lines.append(f"  -> below 200d avg despite confirmed signal (lower conviction): {', '.join(below_200d)}")
    if candidates:
        lines.append(f"\nCANDIDATES ({len(candidates)}) -- confirmed & filter-passing, ranked by score, "
                     f"but didn't make the cut today (portfolio full at {sim_cfg['MAX_POSITIONS']} positions, "
                     f"or insufficient remaining cash):")
        for c in candidates:
            note_suffix = f"  [{'; '.join(c['quality_notes'])}]" if c.get("quality_notes") else ""
            lines.append(f"  #{c['rank']} {c['symbol']}: {c['shares']} sh @ ~{c['price']:.2f} if a slot/cash frees up  "
                         f"(score={c['score']:.2f}, mom={c['composite_momentum']:.2f}, ADX={c['adx']:.0f}, RSI={c['rsi14']:.0f})  "
                         f"-- {c['stage']}{note_suffix}")
    if exits:
        lines.append(f"\nCONFIRMED EXITS ({len(exits)}) -- sell at next open:")
        for x in exits:
            lines.append(f"  {x['symbol']}: entered {x['entry_date']} @ {x['entry_price']:.2f}, "
                         f"now {x['last_close']:.2f} ({x['unrealized_pct']:+.1%})")
    if risk_flags:
        lines.append(f"\nRISK AWARENESS -- informational only, no action ({len(risk_flags)}):")
        for r in risk_flags:
            lines.append(f"  {r['symbol']}: {r['unrealized_pct']:+.1%} since {r['entry_date']} "
                         f"-- trend still confirmed, exit rule unchanged")
    if watchlist:
        lines.append(f"\nWATCHLIST -- not a signal, just building ({len(watchlist)}):")
        for w in watchlist[:15]:
            lines.append(f"  {w['symbol']}: {w['reason']}")
        if len(watchlist) > 15:
            lines.append(f"  ...+{len(watchlist)-15} more (full list in {export_path})")
    if not (entries or candidates or exits or risk_flags or watchlist):
        lines.append("\nNothing to report -- no new signals, no held positions crossed a flag.")
    lines.append(f"\nFull untruncated data (all sections): {export_path}")

    message = "\n".join(lines)
    title = f"TSM: {len(entries)} entry / {len(candidates)} candidate / {len(exits)} exit / {len(risk_flags)} risk flag"
    priority = "high" if (entries or exits) else "default"
    send_ntfy(args.ntfy_topic, title, message, priority)
    print(message)


if __name__ == "__main__":
    main()
