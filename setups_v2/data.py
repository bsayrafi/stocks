"""
Everything that fetches data: 1h bars, Alpaca extended hours / pre-market, news, fundamentals, analyst data,
the daily disk cache and the Finnhub rate limit.
Split out of enrich_html_v2.py, which re-exports every name (enrich_html_v2.X still works).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
import datetime as dt
import time
import threading
from datetime import datetime
from collections import deque
from common import cache_store
try:
    import finnhub  # pip install finnhub-python
except ImportError:
    finnhub = None
from common.event_catalysts import get_event_catalysts, _get_sp500_membership, fetch_apewisdom_table, get_earnings_and_ratings
from setups_v2.config import CONFIGH, _SECRETS, _prof_add, _profiled
from setups_v2.indicators import _hhmm_td

# functions imported from event_catalysts are wrapped here so they're timed too
get_event_catalysts = _profiled("event catalysts (event_catalysts.py)")(get_event_catalysts)
_get_sp500_membership = _profiled("S&P 500 list (once per run)")(_get_sp500_membership)
fetch_apewisdom_table = _profiled("ApeWisdom ranking (once per run)")(fetch_apewisdom_table)
get_earnings_and_ratings = _profiled("earnings date + upgrades (yf)")(get_earnings_and_ratings)


# ---------------------------------------------------------------- daily disk cache
# Data that changes at most once a day (yfinance .info, analyst data, earnings date + rating actions, S&P 500 list,
# finished extended-hours windows) is cached on disk by cache_store.py - the SAME cache tickersV2.py uses for the
# Finviz screener. Where it lives: see cache_store.py (CACHE_DIR / FINVIZ_CACHE_DIR environment variable first, then
# CONFIGH["CACHE_DIR"], then constants.CONFIG, then an "htmlv2cache" folder next to the scripts). Old days are deleted.


def _cache_sync() -> None:
    """Pass this script's settings to the shared cache (cheap; called before every cache use)."""
    cache_store.configure(dir=CONFIGH.get("CACHE_DIR"), enabled=CONFIGH.get("CACHE_ENABLED", True),
                          hook=lambda ev, name, key: _prof_add(
                              "  cache hits (read from disk)" if ev == "hit" else "  cache misses (fetched)", key, 0.0))


def _cache_path(name: str, key: str) -> str:
    _cache_sync()
    return cache_store.cache_path(name, key)


def _cache_write(path: str, obj) -> None:
    cache_store.write(path, obj)


def daily_cached(name: str, key: str, fn, is_valid=bool):
    _cache_sync()
    return cache_store.daily_cached(name, key, fn, is_valid)


def ttl_cached(name: str, key: str, fn, ttl_minutes: float, is_valid=lambda v: v is not None):
    _cache_sync()
    return cache_store.ttl_cached(name, key, fn, ttl_minutes, is_valid)


def prune_cache() -> None:
    _cache_sync()
    cache_store.prune(verbose=True)


# ---------------------------------------------------------------- Finnhub rate limit
class _RateLimiter:
    """Thread-safe: allow at most `per_minute` calls in any 60-second window."""
    def __init__(self, per_minute: int):
        self.per_minute = max(1, int(per_minute))
        self.calls = deque()
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                if len(self.calls) < self.per_minute:
                    self.calls.append(now)
                    return
                sleep_for = 60 - (now - self.calls[0]) + 0.05
            time.sleep(sleep_for)

    def try_acquire(self) -> bool:
        """Take a slot if one is free RIGHT NOW; never waits."""
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] >= 60:
                self.calls.popleft()
            if len(self.calls) < self.per_minute:
                self.calls.append(now)
                return True
            return False


_finnhub_limiter = None  # created on first use from CONFIGH["FINNHUB_MAX_PER_MIN"]
_finnhub_limiter_lock = threading.Lock()


def _finnhub_wait(block: bool = True) -> bool:
    """Wait for a free Finnhub slot (block=True), or take one only if free right now (block=False -> False if not)."""
    global _finnhub_limiter
    with _finnhub_limiter_lock:
        if _finnhub_limiter is None:
            _finnhub_limiter = _RateLimiter(CONFIGH.get("FINNHUB_MAX_PER_MIN", 55))
    if not block:
        return _finnhub_limiter.try_acquire()
    _finnhub_limiter.wait()
    return True


_NEWS_BUSY = object()     # fetch_company_news(wait=False): no free Finnhub slot right now


# ---------------------------------------------------------------- data
# 2. Update fetch_hourly_data to flatten MultiIndex columns from yfinance
@_profiled("yfinance 1h download")
def fetch_hourly_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    return df


@_profiled("yfinance 1h batch download (all tickers)")
def prefetch_hourly_data(tickers: list, period: str, interval: str) -> dict:
    """Download 1h bars for ALL tickers in one yf.download call (yfinance
    fetches them concurrently). Returns {ticker: DataFrame}; tickers with no
    data are left out, so run_screen falls back to a single download / error.
    Done up front because yf.download is not safe to call from several threads."""
    if not tickers:
        return {}
    data = yf.download(list(tickers), period=period, interval=interval, auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        df = df.dropna(how="all")
        if not df.empty:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            out[t] = df
    return out


def prefetch_pre_market(tickers: list, cfg: dict) -> dict | None:
    """Pre-market prices for ALL tickers in a few batched Alpaca requests
    (instead of 1-2 per ticker, which would hit Alpaca's 200 requests/minute
    free limit on big lists). Returns {ticker: details} or None if disabled."""
    if not cfg.get("PREMARKET_ENABLED", True):
        return None
    feed = cfg.get("ALPACA_FEED", "auto")
    kw = dict(headers=cfg.get("ALPACA_HEADERS"), tz=cfg.get("MARKET_TZ", "America/New_York"),
              start_hhmm=cfg.get("PREMARKET_START", "04:00"), open_hhmm=cfg.get("MARKET_OPEN", "09:30"))
    if feed == "auto":
        return get_best_pre_market_prices(
            tickers, sip_delay_minutes=cfg.get("ALPACA_SIP_DELAY_MIN", 16),
            move_pct=cfg.get("PREMARKET_MOVE_PCT", 0.3),
            max_iex_gap_pct=cfg.get("PREMARKET_IEX_MAX_GAP_PCT", 0.5), **kw)
    return get_pre_market_prices(
        tickers, feed=feed, delay_minutes=cfg.get("ALPACA_SIP_DELAY_MIN", 16) if feed == "sip" else 0, **kw)


ALPACA_DATA_URL = "https://data.alpaca.markets/v2"


def _alpaca_headers(headers: dict | None) -> dict | None:
    """Explicit headers -> constants.CONFIG -> APCA_* environment variables."""
    if headers is None:
        headers = _SECRETS.get("ALPACA_HEADERS")
    if headers is None:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if key and secret:
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    return headers


def _alpaca_bars_window_multi(tickers: list, headers: dict | None, feed: str,
                              start: pd.Timestamp, end: pd.Timestamp, delay_minutes: int = 0,
                              tz: str = "America/New_York", timeout: float = 15.0,
                              chunk: int = 100) -> dict | None:
    """1-minute bars for MANY tickers between `start` and `end` on one Alpaca feed,
    with `end` clipped to (now - delay_minutes) - e.g. the free plan's 15-min SIP
    and overnight (boats) delay. Multi-symbol endpoint: one request per `chunk`
    tickers (plus extra pages when there are more than 10,000 bars).
    Returns {ticker: [{"t": Timestamp (exchange tz), "c", "v", "n"}, ...]} ([] = no
    trades / window not reached yet), or None on error / missing keys."""
    headers = _alpaca_headers(headers)
    if headers is None:
        return None
    now = pd.Timestamp.now(tz=tz)
    end = min(end, (now - pd.Timedelta(minutes=delay_minutes)).floor("min"))
    out = {t: [] for t in tickers}
    if end <= start:
        return out

    # Yahoo writes share classes as BRK-B, Alpaca as BRK.B
    to_alpaca = {t: t.replace("-", ".") for t in tickers}
    from_alpaca = {v: k for k, v in to_alpaca.items()}
    symbols = list(to_alpaca.values())
    try:
        for i in range(0, len(symbols), chunk):
            params = {"symbols": ",".join(symbols[i:i + chunk]), "timeframe": "1Min",
                      "start": start.isoformat(), "end": end.isoformat(), "feed": feed,
                      "adjustment": "raw", "limit": 10000}
            while True:
                resp = requests.get(f"{ALPACA_DATA_URL}/stocks/bars", headers=headers,
                                    params=params, timeout=timeout)
                resp.raise_for_status()
                body = resp.json()
                for sym, bars in (body.get("bars") or {}).items():
                    t = from_alpaca.get(sym, sym)
                    out.setdefault(t, []).extend(
                        {"t": pd.Timestamp(b["t"]).tz_convert(tz), "c": float(b["c"]),
                         "v": float(b.get("v", 0)), "n": int(b.get("n", 0) or 0)}
                        for b in bars)
                token = body.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
    except (requests.RequestException, ValueError) as e:
        resp_ = getattr(e, "response", None)
        detail = ""
        if resp_ is not None:
            try:
                detail = f" | Alpaca says: {resp_.text.strip()[:300]}"
            except Exception:
                pass
        hint = ""
        if feed == "boats" and getattr(resp_, "status_code", None) == 403:
            hint = (" -> the overnight (boats) feed needs an Alpaca overnight market-data subscription; "
                    "check the account's data plan, or set OVERNIGHT_ENABLED = False to skip it")
        print(f"  Alpaca bars unavailable ({feed}, {start:%a %H:%M}-{end:%a %H:%M} ET): {e}{detail}{hint}")
        return None
    for t in out:
        out[t].sort(key=lambda b: b["t"])
    return out


def _alpaca_premarket_bars_multi(tickers: list, headers: dict | None, feed: str, delay_minutes: int = 0,
                                 tz: str = "America/New_York", start_hhmm: str = "04:00",
                                 open_hhmm: str = "09:30", timeout: float = 15.0,
                                 chunk: int = 100) -> dict | None:
    """1-minute bars for TODAY's pre-market session (start_hhmm to open_hhmm) for
    many tickers - see _alpaca_bars_window_multi."""
    now = pd.Timestamp.now(tz=tz)
    sh, sm = map(int, start_hhmm.split(":"))
    oh, om = map(int, open_hhmm.split(":"))
    return _alpaca_bars_window_multi(
        tickers, headers, feed, now.normalize() + pd.Timedelta(hours=sh, minutes=sm),
        now.normalize() + pd.Timedelta(hours=oh, minutes=om), delay_minutes, tz, timeout, chunk)


def session_calendar(ref_hourly: pd.DataFrame | None, cfg: dict, now: pd.Timestamp | None = None) -> dict:
    """Work out which sessions matter right now.
      close_date = last COMPLETED regular session (from the hourly data)
      next_date  = the session the pre-market belongs to (today, or the next weekday)
      market_open = regular hours are running right now
    plus the after-hours / overnight / pre-market windows as exchange-time Timestamps."""
    tz = cfg.get("MARKET_TZ", "America/New_York")
    now = now if now is not None else pd.Timestamp.now(tz=tz)
    today = now.date()
    day0 = now.normalize()
    open_td, close_td = _hhmm_td(cfg.get("MARKET_OPEN", "09:30")), _hhmm_td(cfg.get("MARKET_CLOSE", "16:00"))
    ah_end_td = _hhmm_td(cfg.get("AFTERHOURS_END", "20:00"))
    pm_start_td = _hhmm_td(cfg.get("PREMARKET_START", "04:00"))

    dates = []
    if ref_hourly is not None and not ref_hourly.empty:
        ix = ref_hourly.index.tz_convert(tz) if ref_hourly.index.tz is not None else ref_hourly.index
        dates = sorted(set(ix.date))
    in_regular = now.weekday() < 5 and day0 + open_td <= now < day0 + close_td
    market_open = bool(in_regular and dates and dates[-1] == today)

    if dates:
        close_date = dates[-1]
        if close_date == today and now < day0 + close_td:     # today's session not finished
            close_date = dates[-2] if len(dates) > 1 else close_date
    else:                                                      # no data: previous weekday
        d = today - dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= dt.timedelta(days=1)
        close_date = d

    if today > close_date and today.weekday() < 5:
        next_date = today
    else:
        next_date = max(close_date, today) + dt.timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += dt.timedelta(days=1)

    c0 = pd.Timestamp(close_date).tz_localize(tz)
    n0 = pd.Timestamp(next_date).tz_localize(tz)
    return {
        "now": now, "market_open": market_open,
        "close_date": close_date, "next_date": next_date,
        "ah": (c0 + close_td, c0 + ah_end_td),
        "on": (c0 + ah_end_td, n0 + pm_start_td),
        "pm": (n0 + pm_start_td, n0 + open_td),
    }


def _single_feed_details(bars: list, feed: str, reason: str) -> dict:
    return {"price": round(bars[-1]["c"], 2) if bars else None,
            "time": bars[-1]["t"].strftime("%H:%M") if bars else None,
            "day": bars[-1]["t"].strftime("%a") if bars else None,
            "source": feed if bars else None, "reason": reason,
            "trades": sum(b.get("n", 0) for b in bars), "volume": sum(b.get("v", 0) for b in bars)}


def _ext_window_cached(name: str, tickers: list, fetch, win: tuple, delay_min: int, now: pd.Timestamp):
    """Alpaca bars for one extended-hours window. Once the window is over (plus the feed's delay) its bars can no
    longer change, so they are kept in the daily cache (one file per window and feed) and only tickers not yet in
    that file are requested. Windows still in progress are always fetched live."""
    done = win[1] + pd.Timedelta(minutes=delay_min or 0) <= now
    if not done or not CONFIGH.get("CACHE_ENABLED", True):
        return fetch(tickers)
    path = _cache_path("ext", f"{name}_{win[0]:%Y%m%d%H%M}")
    stored = cache_store.read(path, {})
    if not isinstance(stored, dict):
        stored = {}
    missing = [t for t in tickers if t not in stored]
    if missing:
        got = fetch(missing)
        if got is None:                           # request failed: use what is cached, if anything
            return {t: stored.get(t, []) for t in tickers} if stored else None
        stored.update(got)
        _cache_write(path, stored)
    if not missing:
        _prof_add("  ext-hours windows from cache", name, 0.0)
    return {t: stored.get(t, []) for t in tickers}


@_profiled("extended hours (Alpaca batch, all tickers)")
def prefetch_extended_hours(tickers: list, cfg: dict, hourly_by_ticker: dict | None = None) -> dict | None:
    """After-hours, overnight and pre-market prices for ALL tickers in a few
    batched Alpaca requests (about 5 per 100 tickers, plus pages).
      after-hours / pre-market: SIP (all exchanges, ~16 min delayed on the free
        plan) vs live IEX, chosen as in get_best_pre_market_price (ALPACA_FEED="auto"),
        or one fixed feed ("sip"/"iex").
      overnight: OVERNIGHT_FEED (Blue Ocean ATS "boats"), 16 min delayed.
    Returns {"calendar": session_calendar(...), "by_ticker": {t: {"ah": d, "on": d,
    "pm": d}}} where d = {"price", "time", "source", "reason", "status", ...}, or
    None if disabled / no Alpaca keys."""
    if not cfg.get("PREMARKET_ENABLED", True) or _alpaca_headers(cfg.get("ALPACA_HEADERS")) is None:
        return None
    tz = cfg.get("MARKET_TZ", "America/New_York")
    ref = None
    for df in (hourly_by_ticker or {}).values():   # the ticker with the most recent data
        if df is not None and not df.empty and (ref is None or df.index[-1] > ref.index[-1]):
            ref = df
    cal = session_calendar(ref, cfg)
    now = cal["now"]
    headers, delay = cfg.get("ALPACA_HEADERS"), cfg.get("ALPACA_SIP_DELAY_MIN", 16)
    feed = cfg.get("ALPACA_FEED", "auto")
    move, gap = cfg.get("PREMARKET_MOVE_PCT", 0.3), cfg.get("PREMARKET_IEX_MAX_GAP_PCT", 0.5)

    def status(win, bars):
        if win[0] > now:
            return "not started yet"
        return "ok" if bars else "no trades"

    def one(key, win, f, dly):
        return _ext_window_cached(f"{key}_{f}", tickers,
                                  lambda ts: _alpaca_bars_window_multi(ts, headers, f, *win, dly, tz), win, dly, now)

    def two_feed(key, win):
        if win[0] > now:
            return None, None
        if feed == "auto":
            return one(key, win, "sip", delay), one(key, win, "iex", 0)
        return one(key, win, feed, delay if feed == "sip" else 0), None

    by_ticker = {t: {} for t in tickers}
    for key in ("ah", "pm"):
        win = cal[key]
        a, b = two_feed(key, win)
        for t in tickers:
            if feed == "auto":
                d = _choose_pre_market((a or {}).get(t), (b or {}).get(t), move, gap)
            else:
                d = _single_feed_details((a or {}).get(t) or [], feed, "fixed feed (ALPACA_FEED)")
            d["status"] = ("unavailable" if win[0] <= now and a is None and b is None
                           else status(win, d.get("price")))
            by_ticker[t][key] = d

    on_win = cal["on"]
    on_feed = cfg.get("OVERNIGHT_FEED", "boats")
    on_bars = None
    if cfg.get("OVERNIGHT_ENABLED", True) and on_win[0] <= now:
        for f in [on_feed] + [x for x in cfg.get("OVERNIGHT_FEED_FALLBACKS", ()) if x != on_feed]:
            on_bars = _ext_window_cached(f"on_{f}", tickers,
                                         lambda ts, f=f: _alpaca_bars_window_multi(ts, headers, f, *on_win, delay, tz),
                                         on_win, delay, now)
            if on_bars is not None:               # None = the request failed (e.g. 403): try the next feed
                if f != on_feed:
                    print(f"  overnight: '{on_feed}' failed, using the fallback feed '{f}'")
                on_feed = f
                break
    why = {"boats": "Blue Ocean ATS (thin, ~16 min delayed)",
           "overnight": "Alpaca derived overnight feed (approximate)"}.get(on_feed, on_feed)
    for t in tickers:
        bars = (on_bars or {}).get(t) or []
        d = _single_feed_details(bars, on_feed, why)
        d["status"] = ("disabled" if not cfg.get("OVERNIGHT_ENABLED", True)
                       else "not started yet" if on_win[0] > now
                       else "unavailable" if on_bars is None else status(on_win, bars))
        by_ticker[t]["on"] = d
    return {"calendar": cal, "by_ticker": by_ticker}


def extended_hours_for_ticker(ext_batch: dict | None, ticker: str, hourly: pd.DataFrame, cfg: dict) -> dict | None:
    """Per-ticker view: the regular close it's all measured against, each
    session's price and % vs that close, and the "current" price (latest of
    after-hours / overnight / pre-market) - None while regular hours are open."""
    if not ext_batch:
        return None
    cal = ext_batch["calendar"]
    tz = cfg.get("MARKET_TZ", "America/New_York")
    ix = hourly.index.tz_convert(tz) if hourly.index.tz is not None else hourly.index
    upto = hourly[np.asarray(ix.date) <= cal["close_date"]]
    if upto.empty:
        return None
    close_px = round(float(upto["Close"].iloc[-1]), 2)
    close_day = (upto.index[-1].tz_convert(tz) if upto.index.tz is not None else upto.index[-1]).date()
    sessions = {}
    for key in ("ah", "on", "pm"):
        d = dict((ext_batch["by_ticker"].get(ticker) or {}).get(key) or {})
        if d.get("price") is not None and close_px:
            d["pct"] = round((d["price"] / close_px - 1) * 100, 2)
        sessions[key] = d
    current = None
    if not cal["market_open"]:
        for key in ("pm", "on", "ah"):           # most recent session with a price
            if sessions[key].get("price") is not None:
                current = {"price": sessions[key]["price"], "session": key,
                           "time": sessions[key].get("time"), "pct": sessions[key].get("pct")}
                break
    return {"close": close_px, "close_date": close_day, "market_open": cal["market_open"],
            "sessions": sessions, "current": current}


_SESSION_NAMES = {"ah": "after-hours", "on": "overnight", "pm": "pre-market"}


def _alpaca_premarket_bars(ticker: str, headers: dict | None, feed: str, delay_minutes: int = 0,
                           tz: str = "America/New_York", start_hhmm: str = "04:00",
                           open_hhmm: str = "09:30", timeout: float = 10.0) -> list | None:
    """Single-ticker version of _alpaca_premarket_bars_multi: the bars list, []
    if no trades yet, None on error / missing keys."""
    bars = _alpaca_premarket_bars_multi([ticker], headers, feed, delay_minutes, tz,
                                        start_hhmm, open_hhmm, timeout)
    return None if bars is None else bars.get(ticker, [])


@_profiled("pre-market (Alpaca, 1 feed)")
def get_pre_market_price(ticker: str, headers: dict | None = None, feed: str = "sip",
                         delay_minutes: int = 16, tz: str = "America/New_York",
                         start_hhmm: str = "04:00", open_hhmm: str = "09:30",
                         timeout: float = 10.0, with_time: bool = False):
    """Latest pre-market price for `ticker` from ONE Alpaca feed (today's session).

    feed="sip": all exchanges; on the free plan it must be >=15 min old, so
    delay_minutes=16 keeps the query out of the restricted window.
    feed="iex": live (pass delay_minutes=0) but only IEX-exchange trades.

    Returns the price (float) or None. with_time=True returns (price, "HH:MM").
    See get_best_pre_market_price() to combine both feeds automatically, and
    get_pre_market_prices() / get_best_pre_market_prices() for many tickers at once.
    """
    bars = _alpaca_premarket_bars(ticker, headers, feed, delay_minutes, tz, start_hhmm, open_hhmm, timeout)
    if not bars:
        return (None, None) if with_time else None
    price = round(bars[-1]["c"], 2)
    return (price, bars[-1]["t"].strftime("%H:%M")) if with_time else price


def _choose_pre_market(sip: list, iex: list, move_pct: float = 0.3, max_iex_gap_pct: float = 0.5) -> dict:
    """Decide between delayed SIP bars and live IEX bars (see get_best_pre_market_price)."""
    sip, iex = sip or [], iex or []
    d = {"price": None, "time": None, "source": None, "reason": "no pre-market data",
         "sip_price": None, "sip_time": None, "iex_price": None, "iex_time": None, "move_pct": None}
    if sip:
        d["sip_price"], d["sip_time"] = round(sip[-1]["c"], 2), sip[-1]["t"]
    if iex:
        d["iex_price"], d["iex_time"] = round(iex[-1]["c"], 2), iex[-1]["t"]

    def pick(src, reason):
        d["source"], d["reason"] = src, reason
        d["price"], d["time"] = d[f"{src}_price"], d[f"{src}_time"]

    if sip and iex:
        d["move_pct"] = round((d["iex_price"] / d["sip_price"] - 1) * 100, 2)
        # what IEX said at (or just before) the SIP bar's time - a like-for-like check
        iex_then = [b for b in iex if b["t"] <= d["sip_time"]]
        gap = abs(iex_then[-1]["c"] / d["sip_price"] - 1) * 100 if iex_then else None

        if d["iex_time"] <= d["sip_time"]:
            pick("sip", "no IEX trade newer than the SIP snapshot")
        elif gap is not None and gap > max_iex_gap_pct:
            pick("sip", f"IEX off by {gap:.2f}% vs SIP at the same time (thin IEX trading)")
        elif abs(d["move_pct"]) >= move_pct:
            pick("iex", f"moved {d['move_pct']:+.2f}% since the SIP snapshot")
        else:
            pick("sip", f"only {d['move_pct']:+.2f}% move since the SIP snapshot")
    elif sip:
        pick("sip", "no IEX pre-market trades")
    elif iex:
        pick("iex", "no SIP data yet (within the 15-min delay)")

    for k in ("time", "sip_time", "iex_time"):
        if d[k] is not None:
            d[k] = d[k].strftime("%H:%M")
    return d


@_profiled("pre-market (Alpaca auto: 2 calls)")
def get_best_pre_market_price(ticker: str, headers: dict | None = None, sip_delay_minutes: int = 16,
                              move_pct: float = 0.3, max_iex_gap_pct: float = 0.5,
                              tz: str = "America/New_York", start_hhmm: str = "04:00",
                              open_hhmm: str = "09:30", timeout: float = 10.0,
                              with_details: bool = False):
    """Pick between delayed SIP (complete, ~16 min old) and live IEX (fresh,
    but only IEX-exchange trades) for the pre-market price.

    Decision, in order:
      1. Only one feed has data            -> use that one.
      2. IEX has no trade newer than SIP   -> SIP (IEX adds nothing fresher).
      3. IEX disagreed with SIP at SIP's own timestamp by more than
         max_iex_gap_pct                   -> SIP (IEX too thin/unreliable today).
      4. Price moved >= move_pct from the SIP price to the latest IEX price
                                           -> IEX (the market has moved since the
                                              SIP snapshot, so the live price matters).
      5. Otherwise (little movement)       -> SIP (full-market price, still accurate).

    Returns the price (float) or None. with_details=True returns a dict:
    {"price", "time", "source" ("sip"/"iex"), "reason", "sip_price", "sip_time",
     "iex_price", "iex_time", "move_pct"} (price None if neither feed has data).
    For many tickers use get_best_pre_market_prices() - same logic, 2 batched requests.
    """
    kw = dict(tz=tz, start_hhmm=start_hhmm, open_hhmm=open_hhmm, timeout=timeout)
    sip = _alpaca_premarket_bars(ticker, headers, "sip", sip_delay_minutes, **kw)
    iex = _alpaca_premarket_bars(ticker, headers, "iex", 0, **kw)
    d = _choose_pre_market(sip, iex, move_pct, max_iex_gap_pct)
    return d if with_details else d["price"]


@_profiled("pre-market (Alpaca batch, all tickers)")
def get_best_pre_market_prices(tickers: list, headers: dict | None = None, sip_delay_minutes: int = 16,
                               move_pct: float = 0.3, max_iex_gap_pct: float = 0.5,
                               tz: str = "America/New_York", start_hhmm: str = "04:00",
                               open_hhmm: str = "09:30") -> dict:
    """get_best_pre_market_price(..., with_details=True) for MANY tickers, using
    one batched SIP request and one batched IEX request (per 100 tickers).
    Returns {ticker: details dict}; {} if both feeds failed."""
    kw = dict(tz=tz, start_hhmm=start_hhmm, open_hhmm=open_hhmm)
    sip = _alpaca_premarket_bars_multi(tickers, headers, "sip", sip_delay_minutes, **kw)
    iex = _alpaca_premarket_bars_multi(tickers, headers, "iex", 0, **kw)
    if sip is None and iex is None:
        return {}
    return {t: _choose_pre_market((sip or {}).get(t), (iex or {}).get(t), move_pct, max_iex_gap_pct)
            for t in tickers}


@_profiled("pre-market (Alpaca batch, all tickers)")
def get_pre_market_prices(tickers: list, headers: dict | None = None, feed: str = "sip",
                          delay_minutes: int = 16, tz: str = "America/New_York",
                          start_hhmm: str = "04:00", open_hhmm: str = "09:30") -> dict:
    """Latest pre-market price for MANY tickers from ONE feed, batched.
    Returns {ticker: {"price", "time", "source", "reason"}}; {} on failure."""
    bars = _alpaca_premarket_bars_multi(tickers, headers, feed, delay_minutes, tz, start_hhmm, open_hhmm)
    if bars is None:
        return {}
    out = {}
    for t in tickers:
        b = bars.get(t) or []
        out[t] = {"price": round(b[-1]["c"], 2) if b else None,
                  "time": b[-1]["t"].strftime("%H:%M") if b else None,
                  "source": feed if b else None, "reason": "fixed feed (ALPACA_FEED)"}
    return out


@_profiled("news (Finnhub)")
def fetch_company_news(ticker: str, api_key: str | None = None, days: int = 0,
                       max_items: int | None = 25, tz: str = "America/New_York", wait: bool = True):
    """Company news from Finnhub for today (and the previous `days` days).
    Returns a list of {"time": "YYYY-MM-DD HH:MM", "date": date, "headline": str,
    "url": str}, newest first, duplicates (same headline) removed, at most
    max_items (None = no limit). Returns [] if there's no news, and None if the
    finnhub package or API key is missing or the request fails.
    Calls are rate-limited to CONFIGH["FINNHUB_MAX_PER_MIN"] across threads. wait=False: if no call is free
    right now, return _NEWS_BUSY instead of waiting (used by refresh(), which must stay fast)."""
    api_key = api_key or _SECRETS.get("FINNHUB_API_KEY") or os.environ.get("FINNHUB_API_KEY")
    if finnhub is None or not api_key:
        return None
    today = dt.date.today()
    try:
        if not _finnhub_wait(block=wait):
            return _NEWS_BUSY
        client = finnhub.Client(api_key=api_key)
        news = client.company_news(ticker, _from=str(today - dt.timedelta(days=days)), to=str(today)) or []
    except Exception as e:
        print(f"  [{ticker}] news unavailable: {e}")
        return None

    items, seen = [], set()
    for n in sorted(news, key=lambda n: n.get("datetime", 0), reverse=True):
        headline, url = (n.get("headline") or "").strip(), (n.get("url") or "").strip()
        if not headline or headline in seen:
            continue
        seen.add(headline)
        when = pd.Timestamp(n.get("datetime", 0), unit="s", tz="UTC").tz_convert(tz)
        items.append({"time": when.strftime("%Y-%m-%d %H:%M"), "date": when.date(),
                      "headline": headline, "url": url})
        if max_items and len(items) >= max_items:
            break
    return items


# ---------------------------------------------------------------- fundamentals
# Informational only - NOT part of the hard/soft buy-signal logic. Pulled from
# yfinance's .info, which is sometimes incomplete or slow, so every field is
# fetched defensively and missing data shows as "N/A" rather than crashing the run.

def _fmt_billions(v) -> str:
    v = float(v)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v) / 1e9:,.2f}B"


def dividend_yield_fraction(info: dict):
    # Dividend yield as a FRACTION (0.0102 = 1.02%), whatever unit yfinance uses for 'dividendYield'. yfinance moved that
    # field from a fraction to a percent, so the old "x 100" showed 102% for a 1.02% yield. Prefer values whose unit never
    # changed: annual dividend rate / price, then trailingAnnualDividendYield (always a fraction); only when neither exists
    # read dividendYield as a percent (the current yfinance convention). None when nothing is available.
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    rate = num(info.get("dividendRate") or info.get("trailingAnnualDividendRate"))
    px = num(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    if rate and px and px > 0:
        return rate / px
    t = num(info.get("trailingAnnualDividendYield"))
    if t is not None:
        return t
    v = num(info.get("dividendYield"))
    return None if v is None else v / 100.0


FUNDAMENTAL_FIELDS = [
    # (label,                      info_key,                       formatter)
    ("Name",                       "longName",                     lambda v: str(v)),
    ("Sector",                     "sector",                        lambda v: str(v)),
    ("Industry",                   "industry",                      lambda v: str(v)),
    ("Market Cap",                 "marketCap",                     _fmt_billions),
    ("Trailing P/E",               "trailingPE",                    lambda v: f"{v:.2f}"),
    ("Forward P/E",                "forwardPE",                     lambda v: f"{v:.2f}"),
    ("PEG Ratio",                  "pegRatio",                      lambda v: f"{v:.2f}"),
    ("Price/Book",                 "priceToBook",                   lambda v: f"{v:.2f}"),
    ("Price/Sales (TTM)",          "priceToSalesTrailing12Months",  lambda v: f"{v:.2f}"),
    ("EPS (TTM)",                  "trailingEps",                   lambda v: f"{v:.2f}"),
    ("Dividend Yield",             "dividendYield",                 lambda v: f"{v * 100:.2f}%"),
    ("Profit Margin",              "profitMargins",                 lambda v: f"{v * 100:.2f}%"),
    ("Operating Margin",           "operatingMargins",              lambda v: f"{v * 100:.2f}%"),
    ("Return on Equity",           "returnOnEquity",                lambda v: f"{v * 100:.2f}%"),
    ("Revenue Growth (YoY)",       "revenueGrowth",                 lambda v: f"{v * 100:.2f}%"),
    ("Free Cash Flow",             "freeCashflow",                  _fmt_billions),
    ("Debt/Equity",                "debtToEquity",                  lambda v: f"{v:.2f}"),
    ("Current Ratio",              "currentRatio",                  lambda v: f"{v:.2f}"),
    ("Institutional Ownership",    "heldPercentInstitutions",       lambda v: f"{v * 100:.2f}%"),
    ("Short % of Float",           "shortPercentOfFloat",           lambda v: f"{v * 100:.2f}%"),
    ("Beta",                       "beta",                          lambda v: f"{v:.2f}"),
    ("52-Week High",               "fiftyTwoWeekHigh",              lambda v: f"{v:.2f}"),
    ("52-Week Low",                "fiftyTwoWeekLow",               lambda v: f"{v:.2f}"),
    ("Analyst Target (Mean)",      "targetMeanPrice",               lambda v: f"{v:.2f}"),
]


@_profiled("yf .info (shared, once per ticker)")
def fetch_info(ticker: str, tkr: "yf.Ticker | None" = None) -> dict:
    """yfinance .info for a ticker ({} on failure). Fetched ONCE per ticker and
    shared by fundamentals, analyst data and the earnings date."""
    try:
        return (tkr if tkr is not None else yf.Ticker(ticker)).info or {}
    except Exception:
        return {}


def fetch_fundamentals(ticker: str, info: dict | None = None) -> dict:
    """Pull a handful of fundamental data points for a ticker. Returns a dict of
    label -> formatted string. Any field yfinance doesn't have is shown as "N/A";
    if there's no .info at all (network hiccup, delisted ticker, etc.) an empty
    dict is returned so the technical screen can still run.
    Pass `info` (from fetch_info) to avoid downloading it again."""
    if info is None:
        info = fetch_info(ticker)
    if not info:
        return {}

    fundamentals = {}
    for label, key, fmt in FUNDAMENTAL_FIELDS:
        raw = dividend_yield_fraction(info) if key == "dividendYield" else info.get(key)
        if raw is None:
            fundamentals[label] = "N/A"
            continue
        try:
            fundamentals[label] = fmt(raw)
        except (TypeError, ValueError):
            fundamentals[label] = "N/A"
    return fundamentals


# ---------------------------------------------------------------- analyst data
# Also informational only - not part of the buy/sell signal logic. Two pieces:
#   1. Recommendation counts (strong buy / buy / hold / sell / strong sell) for
#      the most recent period, plus the consensus key/mean and analyst count.
#   2. EPS estimate revision trend - current-quarter consensus EPS now vs. what
#      it was 30 days ago, and whether analysts are revising it up or down.
# Analyst coverage varies a lot by ticker (some have none), so every piece is
# fetched independently and defensively - a missing piece never blocks the rest.

@_profiled("analyst data (yf, 3 calls)")
def fetch_analyst_data(ticker: str, tkr: "yf.Ticker | None" = None, info: dict | None = None) -> dict:
    """Analyst consensus + EPS revisions. Pass the shared `tkr` and `info` to
    avoid creating another yf.Ticker and downloading .info a second time."""
    data = {
        "recommendation_key": "N/A",
        "recommendation_mean": "N/A",
        "num_analyst_opinions": "N/A",
        "recommendation_counts": {},
        "eps_current": "N/A",
        "eps_30d_ago": "N/A",
        "eps_improving": None,
        "eps_num_analysts": "N/A",
    }

    if tkr is None:
        try:
            tkr = yf.Ticker(ticker)
        except Exception:
            return data

    # --- Consensus key ("buy", "hold", ...), mean score, and analyst count
    try:
        if info is None:
            info = fetch_info(ticker, tkr)
        if info.get("recommendationKey") is not None:
            data["recommendation_key"] = str(info["recommendationKey"]).replace("_", " ").title()
        if info.get("recommendationMean") is not None:
            data["recommendation_mean"] = f"{float(info['recommendationMean']):.2f}"
        if info.get("numberOfAnalystOpinions") is not None:
            data["num_analyst_opinions"] = str(int(info["numberOfAnalystOpinions"]))
    except Exception:
        pass


    # --- Buy/hold/sell breakdown, most recent period ("0m" = current month)
    t0 = time.perf_counter()
    try:
        rec = tkr.recommendations
        if rec is not None and not rec.empty:
            latest = rec.iloc[0]
            data["recommendation_counts"] = {
                "Strong Buy": int(latest.get("strongBuy", 0) or 0),
                "Buy": int(latest.get("buy", 0) or 0),
                "Hold": int(latest.get("hold", 0) or 0),
                "Sell": int(latest.get("sell", 0) or 0),
                "Strong Sell": int(latest.get("strongSell", 0) or 0),
            }
    except Exception:
        pass

    _prof_add("  analyst: recommendations", ticker, time.perf_counter() - t0)

    # --- EPS estimate revision trend, current quarter ("0q")
    t0 = time.perf_counter()
    try:
        trend = tkr.eps_trend
        if trend is not None and not trend.empty and "0q" in trend.index:
            row = trend.loc["0q"]
            current = row.get("current")
            ago_30d = row.get("30daysAgo")
            if current is not None and pd.notna(current):
                data["eps_current"] = f"{float(current):.2f}"
            if ago_30d is not None and pd.notna(ago_30d):
                data["eps_30d_ago"] = f"{float(ago_30d):.2f}"
            if data["eps_current"] != "N/A" and data["eps_30d_ago"] != "N/A":
                data["eps_improving"] = float(data["eps_current"]) >= float(data["eps_30d_ago"])
    except Exception:
        pass

    _prof_add("  analyst: eps_trend", ticker, time.perf_counter() - t0)

    # --- Number of analysts contributing to the current-quarter EPS estimate
    t0 = time.perf_counter()
    try:
        est = tkr.earnings_estimate
        if est is not None and not est.empty and "0q" in est.index:
            n = est.loc["0q"].get("numberOfAnalysts")
            if n is not None and pd.notna(n):
                data["eps_num_analysts"] = str(int(n))
    except Exception:
        pass
    _prof_add("  analyst: earnings_estimate", ticker, time.perf_counter() - t0)

    return data
