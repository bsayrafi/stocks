"""
market_scan.py

One call that combines:
  - sector_rotation.find_sector_rotation  (where is money rotating?)
  - macro_regime.find_macro_impact        (is each factor helping or hurting each group?)

Oil, yields (and any other factor) are kept SEPARATE. Each gets its own tilt column and
its own signal column; nothing is blended across factors, and the ranking score is the
pure rotation score.

Usage:
    from market_scan import scan_market

    res = scan_market()                                   # 11 sectors
    res["table"]                                          # ranked by rotation score
    res["factors"]                                        # what oil / yields are doing

    scan_market(universe="tech", benchmark="XLK",
                baskets={"AI compute": ["NVDA", "AVGO", "AMD", "TSM"],
                         "AI apps":    ["MSFT", "PLTR", "CRM", "NOW"]})

Requires: pip install yfinance pandas numpy
(sector_rotation.py and macro_regime.py must be in the same folder.)
"""

import pandas as pd

from sector_rotation import (
    find_sector_rotation, _resolve_universe, _download, _build_basket,
)
from macro_regime import find_macro_impact, DEFAULT_FACTORS, _download_close


def _signal(quadrant, rotation_score, tilt):
    rotating_in = quadrant in ("Leading", "Improving") and rotation_score >= 0.5
    tailwind = pd.notna(tilt) and tilt > 0
    if rotating_in and tailwind:
        return "In + tailwind"
    if rotating_in:
        return "In, headwind"
    if tailwind:
        return "Tailwind only"
    return "Weak"


def scan_market(
    universe="sectors",
    benchmark: str = "SPY",
    baskets: dict = None,
    factors: dict = None,
    lookback_days: int = 60,
    short_days: int = 20,
    macro_window: int = 60,
    macro_long_window: int = 250,
    macro_short_days: int = 5,
    top_n: int = 3,
    prices: pd.DataFrame = None,
    volumes: pd.DataFrame = None,
    factor_data: pd.DataFrame = None,
    verbose: bool = True,
) -> dict:
    """
    Rank groups by rotation strength and show each macro factor separately.

    Table columns:
      rotation columns   name, rel_ret_long, rel_ret_short, rs_ratio, rs_momentum,
                         flow_proxy, quadrant, rotation_score  (sorted by rotation_score)
      per factor f       f_corr, f_tilt, f_signal
      regime_shift       factors whose recent correlation flipped sign vs the long window

    f_signal (per factor, independent of the other factors):
      In + tailwind   rotating in and factor f's current move is helping the group
      In, headwind    rotating in but factor f's current move works against it
      Tailwind only   not rotating in, but factor f is helping
      Weak            neither
      Factor flat     the factor isn't moving, so no tailwind/headwind call is made

    Returns {"table": DataFrame, "factors": DataFrame}
    """
    names = _resolve_universe(universe)
    names.pop(benchmark, None)
    baskets = baskets or {}
    factors = factors or DEFAULT_FACTORS

    members = sorted({m for ms in baskets.values() for m in ms})
    tickers = sorted(set(names) | {benchmark} | set(members))

    # One download shared by both analyses (long enough for the macro window)
    if prices is None or volumes is None:
        prices, volumes = _download(tickers, max(macro_long_window, lookback_days), short_days)
    if factor_data is None:
        factor_data = _download_close([t for t, _ in factors.values()], macro_long_window + 10)

    # 1) rotation
    rot = find_sector_rotation(
        universe=names, benchmark=benchmark, baskets=baskets,
        lookback_days=lookback_days, short_days=short_days,
        prices=prices, volumes=volumes, verbose=False,
    ).rename(columns={"score": "rotation_score"})

    # 2) macro. Baskets are added as synthetic price columns so they get factor tilts too.
    macro_prices = prices.copy()
    p_ff = prices.ffill()
    v_fill = volumes.reindex(prices.index).fillna(0)
    macro_names = {
        t: n for t, n in names.items()
        if t in prices.columns and prices[t].count() >= macro_window + macro_short_days + 5
    }
    for bname, ms in baskets.items():
        ms = [m for m in ms if m in p_ff.columns and m in v_fill.columns and p_ff[m].count() > 0]
        if ms:
            macro_prices[bname], _ = _build_basket(p_ff, v_fill, ms)
            macro_names[bname] = f"Basket ({len(ms)} stocks)"

    macro = find_macro_impact(
        factors=factors, universe=macro_names, benchmark=benchmark,
        window=macro_window, long_window=macro_long_window, short_days=macro_short_days,
        prices=macro_prices, factor_data=factor_data, verbose=False,
    )
    sens = macro["sensitivity"].drop(benchmark)

    # 3) merge: separate columns per factor, no blending
    keep = ["regime_shift"]
    for f in factors:
        keep += [f"{f}_corr", f"{f}_tilt"]
    df = rot.join(sens[keep], how="left")
    fstate = macro["factors"]["state"]
    for f in factors:
        df[f"{f}_signal"] = [
            "Factor flat" if fstate[f] == "Flat" else _signal(q, s, t)
            for q, s, t in zip(df["quadrant"], df["rotation_score"], df[f"{f}_tilt"])
        ]
    df = df.sort_values("rotation_score", ascending=False)

    if verbose:
        fdf = macro["factors"]
        print(f"Market scan vs {benchmark}\n")
        print("Macro factors:")
        for name, r in fdf.iterrows():
            print(f"  {name:8} {r['level']:8.2f}  {r['state']:8} "
                  f"({r[f'chg_{macro_short_days}d']:+.1f}{r['unit']} over {macro_short_days}d)")
        flat = [n for n, r in fdf.iterrows() if r["state"] == "Flat"]
        if flat:
            print(f"  (flat, so their tilt carries little information now: {', '.join(flat)})")

        def show(title, rows):
            print(title)
            for t, r in rows.iterrows():
                tilts = "  ".join(f"{f}:{r[f'{f}_tilt']:+.2f}" for f in factors)
                flip = "  [macro relationship flipped]" if r["regime_shift"] else ""
                print(f"  {t:12} {r['name']:24} {r['quadrant']:10} "
                      f"rot={r['rotation_score']:.2f}  tilt {tilts}{flip}")

        print()
        show(f"Top {top_n} by rotation:", df.head(top_n))
        print()
        show(f"Bottom {top_n} by rotation:", df.tail(top_n).iloc[::-1])
        for f in factors:
            if fstate[f] == "Flat":
                print(f"\n[{f}] flat: no tailwind/headwind signal")
                continue
            inn = df[df[f"{f}_signal"] == "In + tailwind"].index
            fragile = df[df[f"{f}_signal"] == "In, headwind"].index
            print(f"\n[{f}] rotating in + tailwind: {', '.join(inn) or 'none'}")
            print(f"[{f}] rotating in, headwind:  {', '.join(fragile) or 'none'}")

    return {"table": df, "factors": macro["factors"]}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    res = scan_market()
    print()
    print(res["table"])
