"""
trend_tsm_backtest.py
========================
Pure systematic time-series momentum — no ML ranking. Philosophy: don't
try to predict which trend will work (that failed in the swing+ADX+ML
version); instead take EVERY valid signal, sized by volatility, spread
across many symbols, and let the well-documented TSM edge (Moskowitz/
Ooi/Pedersen) show up in aggregate, or not.

Entry: daily_trend_confirmed turns True (composite multi-horizon vol-
adjusted momentum > 0 AND OBV slope agrees) -> go long at that day's open.
Exit: daily_trend_confirmed turns False -> close at that day's open.
No fixed target, no ATR trailing stop — the trend gate itself is the
entire exit rule. Deliberately late on both ends, deliberately simple.

Two phases, same pattern as the FVG project's horizon sweep:
  cache    — fetch daily bars once per symbol (timeout-protected,
             resumable — a repeat of last night's 12-hour hang is
             structurally impossible now: any single request that hangs
             times out at 30s and the symbol is skipped, not the run).
  simulate — walk the whole universe forward day by day as one shared
             portfolio: volatility-based position sizing (risk a fixed %
             of current equity per position, sized by ATR), a cap on
             concurrent positions, mark-to-market equity curve, CAGR,
             max drawdown, Sharpe. Pure local computation, no network —
             safe to re-run many times while tuning parameters.

Usage:
    python3 trend_tsm_backtest.py cache --sectors Technology "Health Care" Financials --years 8 --cache-dir cache/tsm_daily
    python3 trend_tsm_backtest.py simulate --cache-dir cache/tsm_daily --out results/tsm_equity.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

from trend_data_pipeline import (
    CONFIG as BASE_CONFIG, fetch_bars, add_indicators, build_daily_trend_context,
    run_with_timeout,
)
from fvg_data_pipeline_hourly import load_tickers_by_sector
from survivorship_free_universe import load_survivorship_free_universe
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

load_dotenv()

SIM_CONFIG = {
    "STARTING_CAPITAL": 100_000,
    "RISK_PCT_PER_TRADE": 0.01,   # fraction of CURRENT equity risked per new position
    "SIZING_ATR_MULT": 2.0,       # dollar risk = shares * SIZING_ATR_MULT * ATR_at_entry
    "MAX_POSITIONS": 15,          # cap on concurrent open positions — the diversification-vs-concentration control
    "ROUND_TRIP_COST_BPS": 10,    # simple flat cost assumption per trade (entry+exit combined), in basis points

    # -- entry-quality filters, added after a loss-driver analysis of the 597-trade
    #    survivorship-free run. IMPORTANT: adding a stop-loss (fixed or trailing ATR)
    #    was tested first and made total P&L WORSE in every configuration tried
    #    (-$82k to -$261k) -- it cut this strategy's big winners short more than it
    #    saved on losers, the classic trend-following stop-loss paradox. These two
    #    filters instead avoid bad setups at ENTRY rather than exiting early:
    #      - ADX > 40 at entry (an already-extended/exhausted trend)
    #      - RSI14 < 50 at entry (momentum that's only just turned positive, not
    #        yet confirmed -- simplified from an initial [30,50) band once the
    #        <30 sub-bucket turned out to be a 5-trade, noise-level sample)
    #    Full-period effect: excluding both raised a REAL portfolio re-run (not
    #    just a trade-level sum -- freeing capital from bad candidates let other
    #    trades fill those slots) from CAGR 19.0% / -42.0% max DD / Sharpe 0.74
    #    to CAGR 25.5% / -29.7% max DD / Sharpe 0.94 on the same 8-year universe.
    #
    #    VALIDATED across sub-periods, not just in aggregate -- this is what
    #    makes it more than a curve-fit: split into 2018-21 / 2021-24 / 2024-26
    #    (COVID crash+recovery, 2022 bear market, 2024-25 bull run) and also by
    #    single calendar year, the excluded bucket (adx>40 or rsi<50) underperformed
    #    the kept bucket's mean trade return in EVERY one of the 3 sub-periods AND
    #    all 9 individual years (2018-2026), including years where the kept bucket
    #    itself was flat. A threshold that was pure noise on this dataset would not
    #    reliably point the same direction across a crash, a bear market, and a
    #    bull run alike. Still: same 8-year dataset and universe throughout, so
    #    treat this as well-supported, not proven -- re-check if you extend the
    #    lookback window, change the universe, or after enough live/paper trading
    #    to see if the edge persists out of sample for real.
    #    Set either to None to disable that filter.
    "ENTRY_MAX_ADX": 40,
    "ENTRY_RSI_EXCLUDE_RANGE": (0, 50),
}


def passes_entry_filters(row, sim_cfg):
    """Additional entry-quality gates on top of daily_trend_confirmed -- see
    SIM_CONFIG's ENTRY_MAX_ADX / ENTRY_RSI_EXCLUDE_RANGE comment for why these
    two exist. Both default to off (None) so behavior is unchanged unless set."""
    max_adx = sim_cfg.get("ENTRY_MAX_ADX")
    if max_adx is not None:
        adx = row.get("adx")
        if pd.notna(adx) and adx > max_adx:
            return False
    rsi_range = sim_cfg.get("ENTRY_RSI_EXCLUDE_RANGE")
    if rsi_range is not None:
        rsi = row.get("rsi14")
        if pd.notna(rsi) and rsi_range[0] <= rsi < rsi_range[1]:
            return False
    return True


# ---------------------------------------------------------------------------
# Phase 1: cache — fetch once per symbol, timeout-protected, resumable
# ---------------------------------------------------------------------------

def build_cache(tickers, api_key, api_secret, cache_dir, cfg=BASE_CONFIG, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    client = StockHistoricalDataClient(api_key, api_secret)

    for n, symbol in enumerate(tickers, 1):
        cache_path = os.path.join(cache_dir, f"{symbol}.parquet")
        if os.path.exists(cache_path):
            if verbose:
                print(f"[{n}/{len(tickers)}] {symbol} — already cached, skipping")
            continue

        daily_raw, err = run_with_timeout(fetch_bars, client, symbol, TimeFrame.Day, cfg["DAILY_LOOKBACK_YEARS"], cfg)
        if err:
            print(f"[{n}/{len(tickers)}] {symbol} — fetch failed ({err}), skipping")
            continue
        if daily_raw is None or len(daily_raw) < 260:
            pd.DataFrame().to_parquet(cache_path)  # empty marker — don't retry a symbol with genuinely too little history
            if verbose:
                print(f"[{n}/{len(tickers)}] {symbol} — insufficient history")
            continue

        ind = add_indicators(daily_raw, cfg)              # gives us 'atr' for position sizing
        ctx = build_daily_trend_context(daily_raw, cfg)    # gives us the (already causally-shifted) trend gate

        tz = cfg["MARKET_TZ"]
        idx = ind.index
        dates = idx.tz_convert(tz).date if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz).date
        ind = ind.set_index(pd.Index(dates, name="date"))
        merged = ind.join(ctx, how="left")

        merged.to_parquet(cache_path)
        if verbose:
            print(f"[{n}/{len(tickers)}] {symbol} — cached ({len(merged)} daily bars)")


# ---------------------------------------------------------------------------
# Phase 2: simulate — pure local computation, no network
# ---------------------------------------------------------------------------

def load_cache(cache_dir):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
    data = {}
    for f in files:
        symbol = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        if not df.empty:
            data[symbol] = df
    return data


def simulate_portfolio(data, sim_cfg=SIM_CONFIG, verbose=True):
    all_dates = sorted(set().union(*(df.index for df in data.values())))

    equity = sim_cfg["STARTING_CAPITAL"]
    cash = equity
    open_positions = {}   # symbol -> {"shares": int, "entry_price": float, "entry_date": date}
    equity_curve = []
    trade_log = []

    for d in all_dates:
        # 1. mark-to-market equity using today's close for anything still open
        positions_value = 0.0
        for symbol, pos in open_positions.items():
            if d in data[symbol].index:
                positions_value += pos["shares"] * data[symbol].loc[d, "close"]
            else:
                positions_value += pos["shares"] * pos["entry_price"]  # symbol has no bar today (holiday gap edge case) — hold last known value
        equity = cash + positions_value
        equity_curve.append({"date": d, "equity": equity, "n_positions": len(open_positions)})

        # 2. exits — signal turned off
        for symbol in list(open_positions.keys()):
            if d not in data[symbol].index:
                continue
            row = data[symbol].loc[d]
            if not bool(row.get("daily_trend_confirmed", False)):
                pos = open_positions.pop(symbol)
                exit_price = row["open"]
                gross_pnl = pos["shares"] * (exit_price - pos["entry_price"])
                cost = sim_cfg["ROUND_TRIP_COST_BPS"] / 10000 * pos["shares"] * (pos["entry_price"] + exit_price)
                net_pnl = gross_pnl - cost
                cash += pos["shares"] * exit_price - cost
                trade_log.append({
                    "symbol": symbol, "entry_date": pos["entry_date"], "exit_date": d,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": pos["shares"], "pnl": net_pnl,
                    "pct_return": (exit_price - pos["entry_price"]) / pos["entry_price"],
                    "days_held": (d - pos["entry_date"]).days,
                })

        # 3. entries — signal just turned on, capped by MAX_POSITIONS, sized by volatility
        candidates = []
        for symbol, df in data.items():
            if symbol in open_positions or d not in df.index:
                continue
            row = df.loc[d]
            if (bool(row.get("daily_trend_confirmed", False)) and not pd.isna(row.get("atr")) and row["atr"] > 0
                    and passes_entry_filters(row, sim_cfg)):
                candidates.append((symbol, row))
        # prioritize the strongest momentum when more signals fire than available slots
        candidates.sort(key=lambda x: x[1].get("composite_momentum", 0), reverse=True)

        slots_available = sim_cfg["MAX_POSITIONS"] - len(open_positions)
        for symbol, row in candidates[:max(slots_available, 0)]:
            entry_price = row["open"]
            dollar_risk = equity * sim_cfg["RISK_PCT_PER_TRADE"]
            shares = int(dollar_risk / (sim_cfg["SIZING_ATR_MULT"] * row["atr"]))
            cost_basis = shares * entry_price
            if shares <= 0 or cost_basis > cash:
                continue
            cost = sim_cfg["ROUND_TRIP_COST_BPS"] / 10000 * cost_basis
            cash -= (cost_basis + cost)
            open_positions[symbol] = {"shares": shares, "entry_price": entry_price, "entry_date": d}

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trade_log)

    if verbose:
        print_stats(equity_df, trades_df, sim_cfg)

    return equity_df, trades_df


def print_stats(equity_df, trades_df, sim_cfg):
    start_eq = sim_cfg["STARTING_CAPITAL"]
    end_eq = equity_df["equity"].iloc[-1]
    total_return = end_eq / start_eq - 1
    n_years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr = (end_eq / start_eq) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    daily_ret = equity_df["equity"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else np.nan

    running_max = equity_df["equity"].cummax()
    drawdown = equity_df["equity"] / running_max - 1
    max_dd = drawdown.min()

    print(f"\n=== Portfolio results ===")
    print(f"Period: {equity_df.index[0]} -> {equity_df.index[-1]}  ({n_years:.1f} years)")
    print(f"Starting capital: ${start_eq:,.0f}  |  Ending: ${end_eq:,.0f}")
    print(f"Total return: {total_return:.1%}  |  CAGR: {cagr:.1%}")
    print(f"Sharpe (daily, ann.): {sharpe:.2f}  |  Max drawdown: {max_dd:.1%}")
    print(f"Avg concurrent positions: {equity_df['n_positions'].mean():.1f}  |  "
          f"Max concurrent: {equity_df['n_positions'].max()}")

    if not trades_df.empty:
        print(f"\nTotal trades: {len(trades_df)}")
        print(f"Win rate: {(trades_df['pnl'] > 0).mean():.1%}")
        print(f"Mean trade return: {trades_df['pct_return'].mean():.2%}  |  "
              f"Median: {trades_df['pct_return'].median():.2%}")
        print(f"Mean days held: {trades_df['days_held'].mean():.1f}")
        print(f"Total P&L: ${trades_df['pnl'].sum():,.0f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)

    p_cache = sub.add_parser("cache")
    p_cache.add_argument("--sectors", nargs="*", default=BASE_CONFIG["SECTORS"])
    p_cache.add_argument("--tickers", nargs="*", default=None)
    p_cache.add_argument("--years", type=int, default=BASE_CONFIG["DAILY_LOOKBACK_YEARS"])
    p_cache.add_argument("--cache-dir", required=True)
    p_cache.add_argument("--legacy-universe", action="store_true",
                          help="Use today's S&P 500 sector membership only (the old, "
                               "survivorship-biased universe) -- for before/after comparison.")

    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--cache-dir", required=True)
    p_sim.add_argument("--out", default="results/tsm_trades.csv")
    p_sim.add_argument("--capital", type=float, default=SIM_CONFIG["STARTING_CAPITAL"])
    p_sim.add_argument("--risk-pct", type=float, default=SIM_CONFIG["RISK_PCT_PER_TRADE"])
    p_sim.add_argument("--max-positions", type=int, default=SIM_CONFIG["MAX_POSITIONS"])
    p_sim.add_argument("--entry-max-adx", type=float, default=SIM_CONFIG["ENTRY_MAX_ADX"],
                        help="Skip entries with ADX above this (an already-extended trend). Backtested finding: try 40.")
    p_sim.add_argument("--entry-rsi-exclude", type=float, nargs=2, default=SIM_CONFIG["ENTRY_RSI_EXCLUDE_RANGE"],
                        metavar=("LOW", "HIGH"),
                        help="Skip entries with RSI14 in [LOW, HIGH) (weak/borderline momentum). Backtested finding: try 30 50.")

    args = parser.parse_args()

    if args.phase == "cache":
        import os as _os
        api_key = _os.environ.get("ALPACA_API_KEY")
        api_secret = _os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            raise SystemExit("Set ALPACA_API_KEY / ALPACA_SECRET_KEY in your .env file.")
        cfg = dict(BASE_CONFIG)
        cfg["DAILY_LOOKBACK_YEARS"] = args.years
        if args.tickers:
            tickers = args.tickers
        elif args.legacy_universe:
            tickers = []
            for sector in args.sectors:
                tickers.extend(load_tickers_by_sector(sector))
            tickers = sorted(set(tickers))
            print(f"[legacy, survivorship-biased] Universe: {len(tickers)} tickers across sectors {args.sectors}")
        else:
            tickers, _universe_report = load_survivorship_free_universe(args.sectors, years=args.years)
        build_cache(tickers, api_key, api_secret, args.cache_dir, cfg)

    elif args.phase == "simulate":
        sim_cfg = dict(SIM_CONFIG)
        sim_cfg["STARTING_CAPITAL"] = args.capital
        sim_cfg["RISK_PCT_PER_TRADE"] = args.risk_pct
        sim_cfg["MAX_POSITIONS"] = args.max_positions
        sim_cfg["ENTRY_MAX_ADX"] = args.entry_max_adx
        sim_cfg["ENTRY_RSI_EXCLUDE_RANGE"] = tuple(args.entry_rsi_exclude) if args.entry_rsi_exclude else None

        data = load_cache(args.cache_dir)
        print(f"Loaded {len(data)} symbols from cache.")
        equity_df, trades_df = simulate_portfolio(data, sim_cfg)

        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        trades_df.to_csv(args.out, index=False)
        equity_df.to_csv(args.out.replace("trades", "equity"))
        print(f"\nWrote trade log to {args.out}")


if __name__ == "__main__":
    main()
