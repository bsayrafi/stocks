"""
sector_rotation.py

Find which sectors / industries / themes money is rotating into, measured
relative to a benchmark.

Usage:
    from sector_rotation import find_sector_rotation

    find_sector_rotation()                                   # 11 GICS sectors vs SPY
    find_sector_rotation(universe="industries")              # ~30 industry ETFs vs SPY
    find_sector_rotation(universe="tech", benchmark="XLK")   # inside tech: semis vs software vs AI...
    find_sector_rotation(universe="sectors+tech")            # 11 sectors + tech breakdown, vs SPY

    # Your own themes as equal-weight baskets (best way to isolate "AI"):
    find_sector_rotation(
        universe="tech", benchmark="XLK",
        baskets={"AI compute": ["NVDA", "AVGO", "AMD", "TSM"],
                 "AI apps":    ["MSFT", "PLTR", "CRM", "NOW"]},
    )

Requires: pip install yfinance pandas numpy
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- universes
SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}

# Inside tech. Use benchmark="XLK" to see rotation *within* tech.
TECH_SUBSECTORS = {
    "SMH": "Semiconductors (VanEck)",
    "SOXX": "Semiconductors (iShares)",
    "IGV": "Software",
    "AIQ": "AI & Big Data",
    "BOTZ": "Robotics & AI",
    "CIBR": "Cybersecurity",
    "SKYY": "Cloud Computing",
    "FDN": "Internet",
    "XSD": "Semis (equal-weight)",
}

INDUSTRY_ETFS = {
    **TECH_SUBSECTORS,
    "XBI": "Biotech", "IHI": "Medical Devices", "IHF": "Health Providers",
    "KRE": "Regional Banks", "KBE": "Banks", "IAI": "Brokers/Exchanges",
    "XOP": "Oil & Gas E&P", "OIH": "Oil Services", "URA": "Uranium",
    "XME": "Metals & Mining", "GDX": "Gold Miners", "COPX": "Copper Miners",
    "ITA": "Aerospace & Defense", "IYT": "Transportation", "JETS": "Airlines",
    "ITB": "Homebuilders", "XRT": "Retail", "PBJ": "Food & Beverage",
    "TAN": "Solar", "ICLN": "Clean Energy", "LIT": "Lithium & Batteries",
}

UNIVERSES = {
    "sectors": SECTOR_ETFS,
    "tech": TECH_SUBSECTORS,
    "industries": INDUSTRY_ETFS,
    # 11 sectors AND the tech breakdown in one table (XLK sits next to its own sub-industries;
    # use benchmark="SPY" so every line means the same thing)
    "sectors+tech": {**SECTOR_ETFS, **TECH_SUBSECTORS},
}


# Themes: ETFs that hold largely the same stocks share a theme. With dedupe_themes=True only
# the ETF with the largest average dollar volume in each theme is kept, so near-duplicates
# (SMH / SOXX / XSD) count as one signal. Tickers not listed here are their own theme.
THEMES = {
    "SMH": "Semiconductors", "SOXX": "Semiconductors", "XSD": "Semiconductors",
    "AIQ": "AI", "BOTZ": "AI",
    "IGV": "Software", "CIBR": "Cybersecurity", "SKYY": "Cloud", "FDN": "Internet",
    "KRE": "Banks", "KBE": "Banks",
    "ICLN": "Clean Energy", "TAN": "Clean Energy",
}


# ------------------------------------------------------------------ helpers
def _resolve_universe(universe):
    if isinstance(universe, str):
        return dict(UNIVERSES[universe])
    if isinstance(universe, dict):
        return dict(universe)
    return {t: t for t in universe}  # plain list of tickers


def _download(tickers, lookback_days, short_days):
    import yfinance as yf

    bars_needed = lookback_days + short_days + 10
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(bars_needed * 1.6) + 30)
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="column")
    return raw["Close"].dropna(how="all"), raw["Volume"].dropna(how="all")


def _build_basket(prices, volumes, members):
    """Equal-weight basket: synthetic price index + summed dollar volume."""
    rets = prices[members].pct_change().mean(axis=1).fillna(0)
    level = 100 * (1 + rets).cumprod()
    dollar_vol = (prices[members] * volumes[members]).sum(axis=1)
    return level, dollar_vol / level  # price * "volume" == basket dollar volume


def _quadrant(rs_ratio, rs_mom):
    if rs_ratio >= 100 and rs_mom >= 100:
        return "Leading"
    if rs_ratio < 100 and rs_mom >= 100:
        return "Improving"   # classic early rotation-in signal
    if rs_ratio >= 100 and rs_mom < 100:
        return "Weakening"
    return "Lagging"


# ---------------------------------------------------------------- main API
def find_sector_rotation(
    universe="sectors",
    benchmark: str = "SPY",
    baskets: dict = None,
    lookback_days: int = 60,
    short_days: int = 20,
    top_n: int = 3,
    price_confirmed_flow: bool = True,
    dedupe_themes: bool = True,
    min_dollar_volume: float = 10e6,
    prices: pd.DataFrame = None,
    volumes: pd.DataFrame = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Rank groups by how strongly money appears to be rotating into them.

    Parameters
    ----------
    universe : "sectors" | "tech" | "industries" | "sectors+tech" | dict {ticker: name} | list of tickers
    benchmark : ticker to measure against. Use a parent (e.g. "XLK") to see
                rotation *within* that sector rather than vs the whole market.
    baskets : optional {name: [tickers]} equal-weight custom themes (e.g. AI).
              They are ranked alongside the universe.
    lookback_days / short_days : baseline and "recent" windows in trading days
    top_n : how many top/bottom groups to print
    price_confirmed_flow : if True (default), the volume/flow signal only counts when price
                agrees. A volume surge while the group is underperforming the benchmark is
                treated as distribution (negative), and a volume drop while it is
                outperforming is treated as neutral, not negative. If False, the raw
                flow_proxy is ranked directly (older behavior).
    dedupe_themes : if True (default), keep only the largest-volume ETF in each theme (see THEMES),
                so e.g. SMH / SOXX / XSD count once. Baskets are never removed.
    min_dollar_volume : drop ETFs whose average daily dollar volume over the lookback + recent
                windows is below this (default $10M/day; set 0 or None to disable). Baskets are exempt.
                Anything removed is listed in the printout and in result.attrs["excluded"].
    prices, volumes : optional DataFrames (columns = tickers, incl. benchmark and
                      basket members) to skip the download

    Returns
    -------
    DataFrame indexed by ticker/basket name, sorted by score (best first).
    Columns: rank, name, group (Sector/Tech/Industry/Basket), theme, rel_ret_long, rel_ret_short, rs_ratio, rs_momentum,
             flow_proxy, flow_adj, quadrant, score, avg_dollar_volume
    """
    names = _resolve_universe(universe)
    names.pop(benchmark, None)              # don't rank the benchmark against itself
    baskets = baskets or {}

    members = sorted({m for ms in baskets.values() for m in ms})
    tickers = sorted(set(names) | {benchmark} | set(members))

    if prices is None or volumes is None:
        prices, volumes = _download(tickers, lookback_days, short_days)

    need = lookback_days + short_days
    # Drop tickers with too little history (recent IPOs, bad symbols) instead of failing
    keep = [t for t in tickers if t in prices.columns and prices[t].count() >= need + 5]
    dropped = sorted(set(tickers) - set(keep))
    if dropped and verbose:
        print(f"Skipped (missing/insufficient data): {', '.join(dropped)}\n")
    if benchmark not in keep:
        raise ValueError(f"No usable data for benchmark {benchmark}.")

    prices = prices[keep].ffill().dropna()
    volumes = volumes[keep].reindex(prices.index).fillna(0)
    if len(prices) < need:
        raise ValueError(f"Not enough data: need {need} rows, got {len(prices)}.")

    groups = {t: n for t, n in names.items() if t in keep}

    # --- one ETF per theme (largest volume wins) and a liquidity floor ---
    adv = (prices * volumes).iloc[-need:].mean()          # avg daily dollar volume
    excluded = {}
    if dedupe_themes:
        by_theme = {}
        for t in groups:
            by_theme.setdefault(THEMES.get(t, groups[t]), []).append(t)
        for theme, ts in by_theme.items():
            if len(ts) > 1:
                best = max(ts, key=lambda x: adv[x])
                for t in ts:
                    if t != best:
                        excluded[t] = (f"smaller than {best} in theme '{theme}' "
                                       f"(${adv[t] / 1e6:,.0f}M/day vs ${adv[best] / 1e6:,.0f}M/day)")
                        del groups[t]
    if min_dollar_volume:
        for t in list(groups):
            if adv[t] < min_dollar_volume:
                excluded[t] = f"low volume (${adv[t] / 1e6:,.1f}M/day)"
                del groups[t]
    if not groups and not baskets:
        raise ValueError("No ETFs left after the volume / theme filters.")

    # Add custom baskets as synthetic price/volume columns
    for bname, ms in baskets.items():
        ms = [m for m in ms if m in keep]
        if not ms:
            continue
        level, vol = _build_basket(prices, volumes, ms)
        prices[bname], volumes[bname] = level, vol
        groups[bname] = f"Basket ({len(ms)} stocks)"

    cols = list(groups)

    # --- relative performance ---
    ret_long = prices.iloc[-1] / prices.iloc[-lookback_days - 1] - 1
    ret_short = prices.iloc[-1] / prices.iloc[-short_days - 1] - 1
    rel_long = ret_long - ret_long[benchmark]
    rel_short = ret_short - ret_short[benchmark]

    # --- RRG-style relative strength ---
    rs_line = prices[cols].div(prices[benchmark], axis=0)
    rs_ratio_s = 100 * rs_line / rs_line.rolling(lookback_days).mean()
    rs_ratio = rs_ratio_s.iloc[-1]
    rs_mom = 100 * rs_ratio_s.iloc[-1] / rs_ratio_s.iloc[-1 - short_days]

    # --- dollar-volume surge relative to benchmark (flow proxy) ---
    dv = prices * volumes
    surge = dv.iloc[-short_days:].mean() / dv.iloc[-(short_days + lookback_days):-short_days].mean()
    flow = surge / surge[benchmark] - 1

    df = pd.DataFrame({
        "name": pd.Series(groups),
        "rel_ret_long": rel_long[cols],
        "rel_ret_short": rel_short[cols],
        "rs_ratio": rs_ratio[cols],
        "rs_momentum": rs_mom[cols],
        "flow_proxy": flow[cols],
    })
    df["quadrant"] = [_quadrant(r, m) for r, m in zip(df["rs_ratio"], df["rs_momentum"])]
    df.insert(1, "group", [
        "Basket" if t in baskets else "Sector" if t in SECTOR_ETFS else "Tech"
        if t in TECH_SUBSECTORS else "Industry" if t in INDUSTRY_ETFS else "Custom"
        for t in df.index
    ])
    df.insert(2, "theme", [
        t if t in baskets else THEMES.get(t, names.get(t, t)) for t in df.index
    ])
    if price_confirmed_flow:
        surge_pos = df["flow_proxy"].clip(lower=0)
        # price agrees (outperforming): a surge adds, no surge is neutral
        # price disagrees (underperforming): a surge is distribution -> negative
        df["flow_adj"] = surge_pos.where(df["rel_ret_short"] > 0, -surge_pos) + 0.0  # +0.0 avoids -0.0
    else:
        df["flow_adj"] = df["flow_proxy"]
    df["score"] = df[["rel_ret_short", "rs_momentum", "flow_adj"]].rank(pct=True).mean(axis=1)
    df["avg_dollar_volume"] = dv.iloc[-need:].mean()[cols]
    df = df.sort_values("score", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.attrs["excluded"] = excluded

    if verbose:
        print(f"Rotation vs {benchmark} (lookback={lookback_days}d, recent={short_days}d)\n")
        for t, why in excluded.items():
            print(f"  excluded {t}: {why}")
        if excluded:
            print()
        def show(title, rows):
            print(title)
            for tkr, r in rows.iterrows():
                print(f"  {tkr:12} {r['name']:26} {r['group']:8} {r['quadrant']:10} score={r['score']:.2f}  "
                      f"rel_ret_short={r['rel_ret_short']:+.1%}  flow={r['flow_proxy']:+.1%}")
        show(f"Top {top_n} by rotation score:", df.head(top_n))
        print()
        show(f"Bottom {top_n} by rotation score:", df.tail(top_n).iloc[::-1])
        inn = df[df["quadrant"].isin(["Leading", "Improving"])]
        print("\nLeading / Improving (actual rotation-in candidates): "
              + (", ".join(f"{t} ({q})" for t, q in inn["quadrant"].items()) or "none"))

    return df


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 200)
    print(find_sector_rotation(universe="tech", benchmark="XLK"))
