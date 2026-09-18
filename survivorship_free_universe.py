"""
survivorship_free_universe.py
=================================
Fixes the survivorship bias baked into fvg_data_pipeline_hourly.load_tickers_by_sector():
that function reads TODAY's S&P 500 sector membership (a live "current constituents"
CSV, Wikipedia as fallback), so an 8-year backtest run today only ever sees stocks
good enough to still be index members right now. Every stock that fell out of the
S&P 500 during the window -- for underperformance, an acquisition, or a straight
delisting -- is invisible to the backtest. For a trend-following strategy this is
close to the worst possible bias: it deletes exactly the downtrend/failure outcomes
the strategy exists to be tested against, leaving only names that turned out fine.

Fix: point-in-time index membership, unioned across every snapshot date in the
backtest window, from a free, community-maintained historical constituents file
(fja05680/sp500 on GitHub -- covers 1996 through within a few weeks of today).
A ticker counts if it was an S&P 500 member on ANY day in [start_date, end_date],
not just today.

Sector tagging (best-effort -- free, reliable GICS history for delisted names
doesn't exist):
  - Tickers still in today's S&P 500 sector CSV: sector comes straight from that
    CSV (the same source load_tickers_by_sector already trusts).
  - Tickers that have since left the index (the whole point of this fix -- fell
    out on price performance, got acquired, delisted, whatever): we have no free,
    reliable way to know their historical GICS sector. An earlier version of this
    file tried a live yfinance lookup per ticker to fill that gap; in practice
    Yahoo's cookie/crumb handshake for the sector-bearing endpoint fails wholesale
    on a well-documented, unpredictable schedule (ranaroussi/yfinance#1729) -- not
    per-symbol, ALL symbols, real ones (BK, a top-20 bank) included -- and hammering
    it with retries across ~150-200 tickers even crashed the run once via file-
    descriptor exhaustion. So: no live lookup. These tickers are included WITHOUT
    sector verification by default (see include_unclassified). Excluding them
    instead would just reintroduce the exact survivorship bias this file exists to
    remove, for the sake of a sector label we can't get from Alpaca's own trading
    data anyway once cache-building fetches prices. This is a documented tradeoff,
    not a silent gap -- report.recovered_unclassified lists exactly which symbols.
  - Set include_unclassified=False to fall back to the stricter (but re-biased)
    behavior of dropping anything not in today's sector CSV.

Residual limitation (still true, still worth saying out loud): a name that was
acquired/delisted outright and has zero surviving price history anywhere (no
free vendor ties data to a retired symbol) still can't be traded in the backtest.
build_cache's existing fetch-failure handling drops those automatically -- no
action needed here, since there's nothing this file could recover for them.

Usage (drop-in replacement for the "loop over sectors, union the results" that
used to call load_tickers_by_sector directly in trend_tsm_backtest.py):

    from survivorship_free_universe import load_survivorship_free_universe
    tickers, report = load_survivorship_free_universe(sectors=["Technology", "Health Care", "Financials"], years=8)
    print(report.summary())
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

MEMBERSHIP_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
CURRENT_SECTOR_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

DEFAULT_CACHE_DIR = "cache/universe"
MEMBERSHIP_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "sp500_membership_history.csv")

# same aliasing load_tickers_by_sector supports, kept in sync so --sectors args behave identically
SECTOR_ALIASES = {
    "tech": "Information Technology",
    "technology": "Information Technology",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "financials": "Financials",
    "financial": "Financials",
}


@dataclass
class UniverseReport:
    sectors: list
    start_date: str
    end_date: str
    current_only_count: int = 0
    point_in_time_total: int = 0
    recovered_via_current_sector_csv: int = 0
    recovered_unclassified: list = field(default_factory=list)
    final_count: int = 0

    def summary(self):
        recovered = self.final_count - self.current_only_count
        lines = [
            f"Universe window: {self.start_date} -> {self.end_date}  (sectors: {', '.join(self.sectors)})",
            f"Old approach (today's sector members only): {self.current_only_count} tickers",
            f"Point-in-time S&P 500 membership, unioned over window: {self.point_in_time_total} tickers",
            f"  -> classified via today's sector CSV: {self.recovered_via_current_sector_csv}",
            f"  -> no longer in today's S&P 500, included unverified "
            f"(the survivorship-bias fix -- see file docstring): {len(self.recovered_unclassified)}",
            f"Final survivorship-free universe: {self.final_count} tickers ({recovered:+d} vs. the old sector-only approach)",
        ]
        if self.recovered_unclassified:
            sample = ", ".join(self.recovered_unclassified[:20])
            more = f", +{len(self.recovered_unclassified) - 20} more" if len(self.recovered_unclassified) > 20 else ""
            lines.append(f"  unclassified tickers added: {sample}{more}")
        return "\n".join(lines)


def _download_membership_history(cache_path=MEMBERSHIP_CACHE_PATH, max_age_days=7):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).days
        if age_days <= max_age_days:
            return pd.read_csv(cache_path, parse_dates=["date"])
    df = pd.read_csv(MEMBERSHIP_CSV_URL, parse_dates=["date"])
    df.to_csv(cache_path, index=False)
    return df


def _point_in_time_union(start_date, end_date, cache_path=MEMBERSHIP_CACHE_PATH):
    """Every ticker that was an S&P 500 member on any snapshot date in
    [start_date, end_date], plus the snapshot immediately before start_date
    (so day-1-of-window membership isn't missed just because that exact date
    isn't itself a change date in the source file)."""
    hist = _download_membership_history(cache_path).sort_values("date")

    before = hist[hist["date"] <= pd.Timestamp(start_date)]
    in_window = hist[(hist["date"] >= pd.Timestamp(start_date)) & (hist["date"] <= pd.Timestamp(end_date))]
    rows = pd.concat([before.tail(1), in_window])

    universe = set()
    for tickers_str in rows["tickers"]:
        universe.update(t.strip() for t in str(tickers_str).split(",") if t.strip())
    return universe


def _load_current_sector_map():
    # Symbols stay in their RAW (dot-class, e.g. "BRK.B") form here -- that's the
    # form the fja05680 point-in-time membership file also uses, and this map's
    # whole job is to be looked up by that file's symbols. Rewriting dots to
    # dashes here (Alpaca's preferred form) would make every dot-class ticker
    # (BRK.B, BF.B) silently miss its own sector row. The rewrite happens once,
    # at the very end, in load_survivorship_free_universe's return.
    df = pd.read_csv(CURRENT_SECTOR_CSV_URL)
    symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    sector_col = next((c for c in df.columns if "sector" in c.lower()), None)
    df[symbol_col] = df[symbol_col].astype(str)
    return dict(zip(df[symbol_col], df[sector_col]))


def load_survivorship_free_universe(sectors, years=8, verbose=True,
                                     include_unclassified=True,
                                     membership_cache_path=MEMBERSHIP_CACHE_PATH):
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365 * years)
    target_sectors = [SECTOR_ALIASES.get(s.strip().lower(), s) for s in sectors]

    current_sector_map = _load_current_sector_map()
    current_only = {
        t for t, sec in current_sector_map.items()
        if any(target.lower() in str(sec).lower() for target in target_sectors)
    }

    point_in_time_all = _point_in_time_union(start_date, end_date, membership_cache_path)

    final, recovered_unclassified = set(), []
    recovered_current_csv = 0

    for symbol in sorted(point_in_time_all):
        sec = current_sector_map.get(symbol)
        if sec is not None:
            if any(target.lower() in str(sec).lower() for target in target_sectors):
                final.add(symbol)
                if symbol not in current_only:
                    recovered_current_csv += 1
            continue

        # Not in today's S&P 500 at all -- this IS the survivorship-bias fix.
        # No reliable free source for its historical sector (see docstring), so
        # include it unverified rather than silently re-excluding it.
        if include_unclassified:
            final.add(symbol)
            recovered_unclassified.append(symbol)

    report = UniverseReport(
        sectors=sectors, start_date=str(start_date), end_date=str(end_date),
        current_only_count=len(current_only),
        point_in_time_total=len(point_in_time_all),
        recovered_via_current_sector_csv=recovered_current_csv,
        recovered_unclassified=recovered_unclassified,
        final_count=len(final),
    )
    if verbose:
        print(report.summary())

    # Alpaca expects dash-class tickers ("BRK-B"), not dot-class ("BRK.B") --
    # same convention load_tickers_by_sector already applied. Every lookup
    # above happens in raw dot-class form to match the source files; this is
    # the one place the rewrite belongs.
    final_alpaca_format = sorted(t.replace(".", "-") for t in final)
    return final_alpaca_format, report
