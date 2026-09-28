"""
macro_regime.py

How are oil and bond yields moving, and how are stocks / sectors reacting to them?

Usage:
    from macro_regime import find_macro_impact

    res = find_macro_impact()                        # oil + 10Y yield vs 11 sectors
    res["factors"]        # what oil / yields are doing now
    res["sensitivity"]    # per-sector correlation, beta and a separate tilt for each factor

    find_macro_impact(universe="industries")         # same, for ~30 industry ETFs
    find_macro_impact(factors={                      # add more factors
        "oil":    ("CL=F", "price"),
        "brent":  ("BZ=F", "price"),
        "y10":    ("^TNX", "yield"),
        "y3m":    ("^IRX", "yield"),
        "dollar": ("DX-Y.NYB", "price"),
    })

Requires: pip install yfinance pandas numpy
(sector_rotation.py must sit in the same folder.)
"""

import numpy as np
import pandas as pd

from sector_rotation import _resolve_universe

# name -> (yahoo ticker, kind).  kind = "price" (pct changes) or "yield" (bps changes)
DEFAULT_FACTORS = {
    "oil": ("CL=F", "price"),     # WTI crude futures
    "y10": ("^TNX", "yield"),     # US 10-year Treasury yield
}


def _download_close(tickers, trading_days):
    import yfinance as yf

    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(trading_days * 1.6) + 30)
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="column")
    return raw["Close"].dropna(how="all")


def find_macro_impact(
    factors: dict = None,
    universe="sectors",
    benchmark: str = "SPY",
    window: int = 60,
    long_window: int = 250,
    short_days: int = 5,
    prices: pd.DataFrame = None,
    factor_data: pd.DataFrame = None,
    sort_by: str = None,
    relative: bool = True,
    top_n: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Parameters
    ----------
    factors      : {name: (ticker, "price"|"yield")}. Default: oil (CL=F) and 10Y yield (^TNX).
    universe     : "sectors" | "tech" | "industries" | dict | list of tickers
    benchmark    : market proxy, included in the output (default SPY)
    window       : recent window (trading days) for correlation/beta
    long_window  : longer window to compare against, to spot regime changes
    short_days   : window used to decide if a factor is currently rising/falling
    prices, factor_data : optional DataFrames of levels (columns = tickers) to skip downloads

    Returns
    -------
    {"factors": DataFrame, "sensitivity": DataFrame}

    sensitivity columns, per factor f:
      f_corr       correlation of daily returns with the factor's daily change (recent window)
      f_corr_long  same over the long window
      f_beta       % move in the group per 1% move in a price factor / per 10bp move in a yield
      f_tilt       current move strength of the factor (capped at +-2) x f_corr.
                   Positive = the factor's current move has been a tailwind for the group,
                   negative = headwind. Each factor gets its own tilt; they are never
                   summed or blended.
    plus:
      regime_shift factors whose recent correlation has flipped sign vs the long window

    sort_by : factor name whose tilt orders the sensitivity table (default: first factor).
    relative : if True (default), each group's returns are measured RELATIVE to the benchmark
               before correlating with the factors. This removes the market-wide effect
               (e.g. rising yields hitting every sector) so corr/beta/tilt show what is
               specific to each group: positive = the group tends to outperform the
               benchmark when the factor rises. The benchmark row itself always uses
               its absolute returns, so it shows how the market reacts.
    """
    factors = factors or DEFAULT_FACTORS
    names = _resolve_universe(universe)
    names.pop(benchmark, None)
    etfs = list(names) + [benchmark]
    f_tickers = [t for t, _ in factors.values()]

    if prices is None:
        prices = _download_close(etfs, long_window + 10)
    if factor_data is None:
        factor_data = _download_close(f_tickers, long_window + 10)

    # Futures and ETFs trade on slightly different calendars: align on common days
    data = pd.concat([prices[etfs], factor_data[f_tickers]], axis=1).ffill(limit=3).dropna()
    if len(data) < window + short_days + 5:
        raise ValueError(f"Not enough overlapping data ({len(data)} rows).")
    long_window = min(long_window, len(data) - 2)

    # --- factor levels and daily changes (yields in bps, prices in %) ---
    lvl, chg = pd.DataFrame(index=data.index), pd.DataFrame(index=data.index)
    for f, (tkr, kind) in factors.items():
        s = data[tkr].astype(float)
        if kind == "yield":
            if s.median() > 20:          # some feeds quote yield x10
                s = s / 10
            lvl[f], chg[f] = s, s.diff() * 100
        else:
            lvl[f], chg[f] = s, s.pct_change() * 100
    rets = data[etfs].pct_change() * 100
    rets, chg, lvl = rets.iloc[1:], chg.iloc[1:], lvl.iloc[1:]
    if relative:
        bench_ret = rets[benchmark].copy()
        rets = rets.sub(bench_ret, axis=0)
        rets[benchmark] = bench_ret          # benchmark row stays absolute

    # --- what are the factors doing now? ---
    rows, z_now = [], {}
    for f, (tkr, kind) in factors.items():
        c = chg[f]
        z = c.iloc[-short_days:].sum() / (c.iloc[-long_window:].std() * np.sqrt(short_days))
        z_now[f] = float(np.clip(z, -2, 2))
        if kind == "yield":
            d_short, d_win, unit = c.iloc[-short_days:].sum(), c.iloc[-window:].sum(), "bps"
        else:
            d_short = (lvl[f].iloc[-1] / lvl[f].iloc[-short_days - 1] - 1) * 100
            d_win = (lvl[f].iloc[-1] / lvl[f].iloc[-window - 1] - 1) * 100
            unit = "%"
        rows.append({
            "factor": f, "ticker": tkr, "level": lvl[f].iloc[-1],
            f"chg_{short_days}d": d_short, f"chg_{window}d": d_win, "unit": unit,
            "z_score": z, "state": "Rising" if z > 1 else "Falling" if z < -1 else "Flat",
        })
    factors_df = pd.DataFrame(rows).set_index("factor")

    # --- how do groups react to each factor? ---
    rec, lng = slice(-window, None), slice(-long_window, None)
    out = {}
    for g in etfs:
        row = {"name": names.get(g, "Benchmark")}
        shifts = []
        for f, (tkr, kind) in factors.items():
            y_r, x_r = rets[g].iloc[rec], chg[f].iloc[rec]
            y_l, x_l = rets[g].iloc[lng], chg[f].iloc[lng]
            c_r, c_l = y_r.corr(x_r), y_l.corr(x_l)
            beta = y_r.cov(x_r) / x_r.var() * (10 if kind == "yield" else 1)
            row[f"{f}_corr"], row[f"{f}_corr_long"], row[f"{f}_beta"] = c_r, c_l, beta
            row[f"{f}_tilt"] = z_now[f] * c_r
            if np.sign(c_r) != np.sign(c_l) and min(abs(c_r), abs(c_l)) > 0.1:
                shifts.append(f)
        row["regime_shift"] = ",".join(shifts)
        out[g] = row
    sens = pd.DataFrame(out).T
    for c in sens.columns:
        if c not in ("name", "regime_shift"):
            sens[c] = sens[c].astype(float)
    sort_by = sort_by or next(iter(factors))
    sens = pd.concat([sens.loc[[benchmark]],
                      sens.drop(benchmark).sort_values(f"{sort_by}_tilt", ascending=False)])

    if verbose:
        print("Macro factors now:")
        print(factors_df.round(2).to_string(), "\n")
        b = sens.loc[benchmark]
        print(f"{benchmark} reaction (recent {window}d vs long {long_window}d correlation):")
        for f in factors:
            print(f"  {f:8} recent={b[f'{f}_corr']:+.2f}  long={b[f'{f}_corr_long']:+.2f}")
        rest = sens.drop(benchmark)
        for f in factors:
            col = f"{f}_tilt"
            print(f"\n[{f}] currently {factors_df.loc[f, 'state']}")
            print(f"  Biggest tailwinds ({col} > 0):")
            for t, r in rest.sort_values(col, ascending=False).head(top_n).iterrows():
                print(f"    {t:12} {r['name']:24} {col}={r[col]:+.2f}")
            print(f"  Biggest headwinds ({col} < 0):")
            for t, r in rest.sort_values(col).head(top_n).iterrows():
                print(f"    {t:12} {r['name']:24} {col}={r[col]:+.2f}")
        flips = rest[rest["regime_shift"] != ""]
        if len(flips):
            print("\nRelationship has flipped vs long-run:",
                  "; ".join(f"{t} ({r['regime_shift']})" for t, r in flips.iterrows()))

    return {"factors": factors_df, "sensitivity": sens}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    res = find_macro_impact()
    print()
    print(res["sensitivity"])
