"""
pre_market.py
-------------
Pre-market prices from Alpaca's free data feeds, choosing per ticker between:

  * SIP  - all US exchanges, but on the free plan it must be >= 15 min old
           (queried with a 16-minute delay), and
  * IEX  - live, but only trades printed on the IEX exchange.

get_*pre_market* functions: only TODAY's pre-market session (default
04:00-09:30 America/New_York), using 1-minute bars. See
get_best_pre_market_price() for the decision rules.

prefetch_extended_hours(): after-hours (16:00-20:00), overnight (20:00-04:00,
Blue Ocean ATS "boats" feed) AND pre-market, with the session that matters
right now worked out from the clock + the dates in the price data (same logic
as setups_v2/data.py). extended_hours_for_ticker() then gives each ticker's
"current" price: the latest of pre-market / overnight / after-hours.

For a screener run, call prefetch_extended_hours(tickers, CONFIG, data) ONCE
from the main thread; it batches every ticker into a few requests per feed.
"""

from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

from common import constants
from common.profiling import _profiled

ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

# Where _alpaca_headers looks for keys when none are passed in.
_SECRETS = constants.CONFIG


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _alpaca_headers(headers: dict | None) -> dict | None:
    """Explicit headers -> constants.CONFIG -> APCA_* environment variables."""
    if headers is None:
        headers = _SECRETS.get("ALPACA_HEADERS")
    if headers is None:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if key and secret:
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    return headers


# --------------------------------------------------------------------------
# Bar fetching
# --------------------------------------------------------------------------

def _hhmm_td(hhmm: str) -> pd.Timedelta:
    h, m = map(int, hhmm.split(":"))
    return pd.Timedelta(hours=h, minutes=m)


def _alpaca_bars_window_multi(tickers: list, headers: dict | None, feed: str,
                              start: pd.Timestamp, end: pd.Timestamp, delay_minutes: int = 0,
                              tz: str = "America/New_York", timeout: float = 15.0,
                              chunk: int = 100) -> dict | None:
    """1-minute bars for MANY tickers between `start` and `end` on one Alpaca feed,
    with `end` clipped to (now - delay_minutes) - e.g. the free plan's 15-min SIP
    and overnight (boats) delay. Uses Alpaca's multi-symbol endpoint: one request
    per `chunk` tickers (plus extra pages when there are more than 10,000 bars).
    Returns {ticker: [{"t": Timestamp (exchange tz), "c": close, "v": volume,
    "n": trades}, ...]} (a ticker with no trades in the window gets []), or None
    on missing keys / every chunk failed.
    If one chunk fails, only that chunk's tickers are left out (absent from the
    result); the other chunks are kept."""
    headers = _alpaca_headers(headers)
    if headers is None:
        return None
    now = pd.Timestamp.now(tz=tz)
    end = min(end, (now - pd.Timedelta(minutes=delay_minutes)).floor("min"))
    out = {t: [] for t in tickers}
    if end <= start:
        return out  # window hasn't started yet (or is still inside the delay)

    # Yahoo writes share classes as BRK-B, Alpaca as BRK.B
    to_alpaca = {t: t.replace("-", ".") for t in tickers}
    from_alpaca = {v: k for k, v in to_alpaca.items()}
    symbols = list(to_alpaca.values())

    failed_chunks = 0
    for i in range(0, len(symbols), chunk):
        chunk_syms = symbols[i:i + chunk]
        params = {"symbols": ",".join(chunk_syms), "timeframe": "1Min",
                  "start": start.isoformat(), "end": end.isoformat(), "feed": feed,
                  "adjustment": "raw", "limit": 10000}
        chunk_bars: dict[str, list] = {}
        try:
            while True:
                # Retry rate-limit responses (free plan: 200 requests/minute).
                for attempt in range(3):
                    resp = requests.get(f"{ALPACA_DATA_URL}/stocks/bars", headers=headers,
                                        params=params, timeout=timeout)
                    if resp.status_code != 429:
                        break
                    time.sleep(2 * (attempt + 1))
                resp.raise_for_status()
                body = resp.json()
                for sym, bars in (body.get("bars") or {}).items():
                    t = from_alpaca.get(sym, sym)
                    chunk_bars.setdefault(t, []).extend(
                        {"t": pd.Timestamp(b["t"]).tz_convert(tz), "c": float(b["c"]),
                         "v": float(b.get("v", 0)), "n": int(b.get("n", 0) or 0)}
                        for b in bars)
                token = body.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            # Drop only this chunk's tickers ("unknown"), keep everything else.
            failed_chunks += 1
            resp_ = getattr(e, "response", None)
            hint = ""
            if feed == "boats" and getattr(resp_, "status_code", None) == 403:
                hint = (" -> the overnight (boats) feed needs an Alpaca overnight market-data subscription; "
                        "set OVERNIGHT_ENABLED = False in CONFIG to skip it")
            print(f"  Alpaca bars chunk {i // chunk + 1} failed ({feed}, "
                  f"{start:%a %H:%M}-{end:%a %H:%M} ET): {e}{hint}")
            for s in chunk_syms:
                out.pop(from_alpaca.get(s, s), None)
            if hint:          # every chunk would get the same 403 - don't repeat it
                return None
            continue
        for t, bars in chunk_bars.items():
            out.setdefault(t, []).extend(bars)

    n_chunks = -(-len(symbols) // chunk) if symbols else 0
    if n_chunks and failed_chunks == n_chunks:
        return None  # nothing at all came back on this feed
    for t in out:
        out[t].sort(key=lambda b: b["t"])
    return out


def _alpaca_premarket_bars_multi(tickers: list, headers: dict | None, feed: str, delay_minutes: int = 0,
                                 tz: str = "America/New_York", start_hhmm: str = "04:00",
                                 open_hhmm: str = "09:30", timeout: float = 15.0,
                                 chunk: int = 100) -> dict | None:
    """1-minute bars for TODAY's pre-market session (start_hhmm to open_hhmm) for
    many tickers - see _alpaca_bars_window_multi."""
    day0 = pd.Timestamp.now(tz=tz).normalize()
    return _alpaca_bars_window_multi(tickers, headers, feed, day0 + _hhmm_td(start_hhmm),
                                     day0 + _hhmm_td(open_hhmm), delay_minutes, tz, timeout, chunk)


def _alpaca_premarket_bars(ticker: str, headers: dict | None, feed: str, delay_minutes: int = 0,
                           tz: str = "America/New_York", start_hhmm: str = "04:00",
                           open_hhmm: str = "09:30", timeout: float = 10.0) -> list | None:
    """Single-ticker version of _alpaca_premarket_bars_multi: the bars list, []
    if no trades yet, None on error / missing keys."""
    bars = _alpaca_premarket_bars_multi([ticker], headers, feed, delay_minutes, tz,
                                        start_hhmm, open_hhmm, timeout)
    return None if bars is None else bars.get(ticker)


# --------------------------------------------------------------------------
# One feed
# --------------------------------------------------------------------------

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


@_profiled("pre-market (Alpaca batch, 1 feed)")
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


# --------------------------------------------------------------------------
# Both feeds (auto)
# --------------------------------------------------------------------------

# How close (minutes) an IEX bar must be to the SIP bar's time to be used for
# the like-for-like IEX-vs-SIP sanity check.
IEX_COMPARE_WINDOW_MIN = 5


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
        # What IEX said at (or just before) the SIP bar's time - a like-for-like
        # check. Only IEX bars within IEX_COMPARE_WINDOW_MIN of the SIP time count,
        # so a stale IEX print from hours earlier can't create a false "gap".
        window_start = d["sip_time"] - pd.Timedelta(minutes=IEX_COMPARE_WINDOW_MIN)
        iex_then = [b for b in iex if window_start <= b["t"] <= d["sip_time"]]
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
    Returns {ticker: details dict}; {} if both feeds failed.
    A ticker whose chunk failed on one feed is decided from the other feed."""
    kw = dict(tz=tz, start_hhmm=start_hhmm, open_hhmm=open_hhmm)
    sip = _alpaca_premarket_bars_multi(tickers, headers, "sip", sip_delay_minutes, **kw)
    iex = _alpaca_premarket_bars_multi(tickers, headers, "iex", 0, **kw)
    if sip is None and iex is None:
        return {}
    return {t: _choose_pre_market((sip or {}).get(t), (iex or {}).get(t), move_pct, max_iex_gap_pct)
            for t in tickers}


# --------------------------------------------------------------------------
# Entry point for the screener
# --------------------------------------------------------------------------

def prefetch_pre_market(tickers: list, cfg: dict) -> dict | None:
    """Pre-market prices for ALL tickers in a few batched Alpaca requests
    (instead of 1-2 per ticker, which would hit Alpaca's 200 requests/minute
    free limit on big lists). Returns {ticker: details} or None if disabled."""
    if not cfg.get("PREMARKET_ENABLED", True):
        return None
    if _alpaca_headers(cfg.get("ALPACA_HEADERS")) is None:
        print("Pre-market skipped: Alpaca keys not set "
              "(APCA_API_KEY_ID / APCA_API_SECRET_KEY secrets or environment variables)")
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


# --------------------------------------------------------------------------
# Extended hours: after-hours / overnight / pre-market, picked automatically
# (same logic as setups_v2/data.py, used by enrich_html_v2)
# --------------------------------------------------------------------------

SESSION_NAMES = {"ah": "after-hours", "on": "overnight", "pm": "pre-market"}
SESSION_LABELS = {"ah": "AH", "on": "ON", "pm": "PM"}


def session_calendar(ref, cfg: dict, now: pd.Timestamp | None = None) -> dict:
    """Work out which sessions matter right now.
      close_date  = last COMPLETED regular session (from the dates in `ref`)
      next_date   = the session the pre-market belongs to (today, or the next weekday)
      market_open = regular hours are running right now
    plus the after-hours / overnight / pre-market windows as exchange-time Timestamps.
    `ref` is a DataFrame or DatetimeIndex of bars (daily or hourly) - only its dates
    are used, so market holidays are handled by the data itself."""
    tz = cfg.get("MARKET_TZ", "America/New_York")
    now = now if now is not None else pd.Timestamp.now(tz=tz)
    today = now.date()
    day0 = now.normalize()
    open_td, close_td = _hhmm_td(cfg.get("MARKET_OPEN", "09:30")), _hhmm_td(cfg.get("MARKET_CLOSE", "16:00"))
    ah_end_td = _hhmm_td(cfg.get("AFTERHOURS_END", "20:00"))
    pm_start_td = _hhmm_td(cfg.get("PREMARKET_START", "04:00"))

    dates = []
    ix = getattr(ref, "index", ref)
    if ix is not None and len(ix):
        ix = pd.DatetimeIndex(ix)
        ix = ix.tz_convert(tz) if ix.tz is not None else ix
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


@_profiled("extended hours (Alpaca batch, all tickers)")
def prefetch_extended_hours(tickers: list, cfg: dict, ref=None) -> dict | None:
    """After-hours, overnight and pre-market prices for ALL tickers in a few
    batched Alpaca requests (about 5 per 100 tickers, plus pages).
      after-hours / pre-market: SIP (all exchanges, ~16 min delayed on the free
        plan) vs live IEX, chosen as in get_best_pre_market_price (ALPACA_FEED="auto"),
        or one fixed feed ("sip"/"iex").
      overnight: OVERNIGHT_FEED (Blue Ocean ATS "boats"), 16 min delayed.
    `ref` = the price bars (DataFrame / DatetimeIndex) whose dates decide the last
    completed session - see session_calendar.
    Returns {"calendar": session_calendar(...), "by_ticker": {t: {"ah": d, "on": d,
    "pm": d}}} where d = {"price", "time", "source", "reason", "status", ...}, or
    None if disabled / no Alpaca keys."""
    if not cfg.get("PREMARKET_ENABLED", True):
        return None
    if _alpaca_headers(cfg.get("ALPACA_HEADERS")) is None:
        print("Extended hours skipped: Alpaca keys not set "
              "(APCA_API_KEY_ID / APCA_API_SECRET_KEY secrets or environment variables)")
        return None
    tz = cfg.get("MARKET_TZ", "America/New_York")
    cal = session_calendar(ref, cfg)
    now = cal["now"]
    headers, delay = cfg.get("ALPACA_HEADERS"), cfg.get("ALPACA_SIP_DELAY_MIN", 16)
    feed = cfg.get("ALPACA_FEED", "auto")
    move, gap = cfg.get("PREMARKET_MOVE_PCT", 0.3), cfg.get("PREMARKET_IEX_MAX_GAP_PCT", 0.5)

    def status(win, price):
        if win[0] > now:
            return "not started yet"
        return "ok" if price is not None else "no trades"

    def fetch(win, f, dly):
        return _alpaca_bars_window_multi(tickers, headers, f, *win, dly, tz)

    by_ticker = {t: {} for t in tickers}
    for key in ("ah", "pm"):
        win = cal[key]
        a = b = None
        if win[0] <= now:
            if feed == "auto":
                a, b = fetch(win, "sip", delay), fetch(win, "iex", 0)
            else:
                a = fetch(win, feed, delay if feed == "sip" else 0)
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
            on_bars = fetch(on_win, f, delay)
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
                       else "unavailable" if on_bars is None else status(on_win, d["price"]))
        by_ticker[t]["on"] = d
    return {"calendar": cal, "by_ticker": by_ticker}


def extended_hours_for_ticker(ext_batch: dict | None, ticker: str, bars: pd.DataFrame, cfg: dict) -> dict | None:
    """Per-ticker view: the regular close it's all measured against (from `bars`,
    daily or hourly, needs a "Close" column), each session's price and % vs that
    close, and the "current" price (latest of after-hours / overnight /
    pre-market) - None while regular hours are open."""
    if not ext_batch or bars is None or bars.empty:
        return None
    cal = ext_batch["calendar"]
    tz = cfg.get("MARKET_TZ", "America/New_York")
    ix = bars.index.tz_convert(tz) if bars.index.tz is not None else bars.index
    upto = bars[np.asarray(ix.date) <= cal["close_date"]]
    upto = upto[upto["Close"].notna()]
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
