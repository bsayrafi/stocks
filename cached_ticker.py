"""
cached_ticker.py
----------------
CachedTicker: a drop-in yf.Ticker whose slow, slowly-changing lookups are
cached on disk for the current trading day (US/Eastern calendar date):

    .info / get_info()               -> "info"             (CONFIG["CACHE_INFO_DAILY"])
    .recommendations                 -> "recommendations"  (CONFIG["CACHE_ANALYST_DAILY"])
    .eps_revisions / .eps_trend /
    .earnings_estimate               -> "eps_revisions" / "eps_trend" / "earnings_estimate"
                                                           (CONFIG["CACHE_ANALYST_DAILY"])

The first run of each day fetches fresh data; later runs that day reuse it.

The cache is SKIPPED for a ticker whose earnings report is within
CONFIG["EARNINGS_FRESH_WINDOW_DAYS"] (default 2) days of today -- that is when
EPS, estimates, targets and ratios actually change.

Everything else (price history, options, statements, ...) is plain yf.Ticker.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

import disk_cache
from constants import CONFIG

_ET = "America/New_York"


def _today_et() -> str:
    return pd.Timestamp.now(tz=_ET).strftime("%Y-%m-%d")


def _near_earnings(symbol: str, window_days: float) -> bool:
    """True if a cached earnings date is within window_days of now.
    Uses the earnings-date cache only (no network). Unknown -> False."""
    edates = disk_cache.get("earnings_dates", symbol, ttl_days=30)
    if edates is None:
        return False
    try:
        idx = pd.DatetimeIndex(edates.index)
        now = pd.Timestamp.now(tz=idx.tz) if idx.tz is not None else pd.Timestamp.now()
        return bool((abs(idx - now) <= pd.Timedelta(days=window_days)).any())
    except Exception:
        return True   # can't tell -> be safe, fetch fresh


def _usable(value) -> bool:
    """Only cache real results (a failed/empty lookup is retried next run)."""
    if value is None:
        return False
    if isinstance(value, dict):
        return len(value) > 1          # yfinance returns a near-empty dict on failure
    if isinstance(value, pd.DataFrame):
        return not value.empty
    return True


class CachedTicker(yf.Ticker):
    def __init__(self, ticker, session=None, **kwargs):
        super().__init__(ticker, session=session, **kwargs)
        self._symbol = ticker
        self._mem: dict = {}           # per-object memo (several steps read .info)
        self._bypass = None            # decided lazily on first cached lookup

    # -- core -------------------------------------------------------------
    def _bypass_cache(self) -> bool:
        if self._bypass is None:
            window = CONFIG.get("EARNINGS_FRESH_WINDOW_DAYS", 2)
            self._bypass = _near_earnings(self._symbol, window)
        return self._bypass

    def _daily(self, namespace: str, enabled: bool, fetch):
        if namespace in self._mem:
            return self._mem[namespace]
        today = _today_et()
        use_cache = enabled and not self._bypass_cache()
        if use_cache:
            entry = disk_cache.get(namespace, self._symbol, ttl_days=1.5)
            if entry is not None and entry[0] == today:     # same trading day only
                disk_cache.record(namespace, hit=True)
                self._mem[namespace] = entry[1]
                return entry[1]
            disk_cache.record(namespace, hit=False)
        value = fetch()
        if use_cache and _usable(value):
            disk_cache.put(namespace, self._symbol, (today, value))
        self._mem[namespace] = value
        return value

    # -- cached lookups -----------------------------------------------------
    def get_info(self) -> dict:
        return self._daily("info", CONFIG.get("CACHE_INFO_DAILY", 1) == 1,
                           lambda: yf.Ticker.get_info(self))

    def get_recommendations(self, as_dict=False):
        if as_dict:
            return yf.Ticker.get_recommendations(self, as_dict=True)
        return self._daily("recommendations", CONFIG.get("CACHE_ANALYST_DAILY", 1) == 1,
                           lambda: yf.Ticker.get_recommendations(self))

    def get_eps_revisions(self, as_dict=False):
        if as_dict:
            return yf.Ticker.get_eps_revisions(self, as_dict=True)
        return self._daily("eps_revisions", CONFIG.get("CACHE_ANALYST_DAILY", 1) == 1,
                           lambda: yf.Ticker.get_eps_revisions(self))

    def get_eps_trend(self, as_dict=False):
        if as_dict:
            return yf.Ticker.get_eps_trend(self, as_dict=True)
        return self._daily("eps_trend", CONFIG.get("CACHE_ANALYST_DAILY", 1) == 1,
                           lambda: yf.Ticker.get_eps_trend(self))

    def get_earnings_estimate(self, as_dict=False):
        if as_dict:
            return yf.Ticker.get_earnings_estimate(self, as_dict=True)
        return self._daily("earnings_estimate", CONFIG.get("CACHE_ANALYST_DAILY", 1) == 1,
                           lambda: yf.Ticker.get_earnings_estimate(self))
