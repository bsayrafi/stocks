"""
Settings for the v2 setup screener (CONFIGH, incl. the setup engine's defaults), API keys, and run profiling.
Split out of enrich_html_v2.py, which re-exports every name (enrich_html_v2.X still works).
"""

import time
import functools
from common import constants

# Load a local .env file (VS Code / your own machine) into the environment, so
# NTFY_TOPIC, APCA_*, FINNHUB_API_KEY... are found there too. Variables that are
# already set (e.g. GitHub Actions secrets) are NOT overridden.
# Needs: pip install python-dotenv  (optional - skipped if not installed)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))   # .env in the folder you run from (or a parent)
    load_dotenv()                            # .env next to this file (or a parent)
except ImportError:
    pass

# API keys live in constants.CONFIG (kept out of this file):
#   CONFIG = {"ALPACA_HEADERS": {"APCA-API-KEY-ID": ..., "APCA-API-SECRET-KEY": ...},
#             "FINNHUB_API_KEY": ...}
# API keys live in constants.CONFIG (kept out of this file):
#   CONFIG = {"ALPACA_HEADERS": {"APCA-API-KEY-ID": ..., "APCA-API-SECRET-KEY": ...},
#             "FINNHUB_API_KEY": ...}
_SECRETS = getattr(constants, "CONFIG", {}) or {}

# ---------------------------------------------------------------- profiling
# Records how long each network call / step takes per ticker, and main()
# prints a summary at the end. Turn off with CONFIGH["PROFILE"] = False.
PROFILE: dict = {}   # step name -> list of (ticker, seconds)


def _prof_add(step: str, ticker: str, seconds: float) -> None:
    PROFILE.setdefault(step, []).append((ticker, seconds))


def _profiled(step: str):
    """Decorator: time every call of the function under `step`. The ticker is
    taken from the first argument (a ticker string, or a report dict)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                first = args[0] if args else kwargs.get("ticker", "")
                ticker = (first.get("ticker", "") if isinstance(first, dict)
                          else "ALL" if isinstance(first, (list, tuple)) else str(first))
                _prof_add(step, ticker, time.perf_counter() - t0)
        return wrapper
    return deco


def print_profile_summary(wall_seconds: float, n_tickers: int) -> None:
    """Print where the time went, biggest step first."""
    if not PROFILE:
        return
    rows = []
    for step, calls in PROFILE.items():
        total = sum(sec for _, sec in calls)
        slow_t, slow_s = max(calls, key=lambda c: c[1])
        rows.append((step, total, len(calls), total / len(calls), slow_t, slow_s))
    rows.sort(key=lambda r: -r[1])
    work = sum(r[1] for r in rows if not r[0].startswith("  ") and r[0] != "whole ticker (run_screen)")

    print(f"\n=== Timing summary: {n_tickers} tickers, {wall_seconds:.1f}s wall time "
          f"({wall_seconds / max(n_tickers, 1):.1f}s per ticker) ===")
    print(f"total work {work:.1f}s across {CONFIGH.get('MAX_WORKERS', 1)} parallel workers "
          f"(steps overlap, so work > wall time)")
    print(f"{'step':<40}{'total s':>9}{'% work':>8}{'calls':>7}{'avg s':>8}   slowest")
    for step, total, n, avg, slow_t, slow_s in rows:
        if step == "whole ticker (run_screen)":
            continue
        print(f"{step:<40}{total:>9.1f}{total / max(work, 1e-9) * 100:>7.1f}%{n:>7}{avg:>8.2f}"
              f"   {slow_t} {slow_s:.1f}s")
    per_ticker = sorted(PROFILE.get("whole ticker (run_screen)", []), key=lambda c: -c[1])[:5]
    if per_ticker:
        print("slowest tickers: " + ", ".join(f"{t} {sec:.1f}s" for t, sec in per_ticker))
    print("(sub-steps marked '  analyst:' are included in 'analyst data')\n")


CONFIGH = {
    "TICKER": "ORCL",
    "PERIOD": "200d",              # max for 1h interval on yfinance is 730d, 60d is plenty here
    "REFRESH_PERIOD": "5d",        # refresh(): download only this much 1h history and attach it to the bars the full scan
                                   # saved (same day) - measured 2.7 s vs 4.2 s for 93 tickers. None = always download PERIOD
    "INTERVAL": "1h",

    # Stochastic
    "STOCH_K_PERIOD": 14,
    "STOCH_D_PERIOD": 3,
    "STOCH_SMOOTH": 3,
    "OVERSOLD": 20,
    "OVERBOUGHT": 80,

    # Moving average trend filter (applied per timeframe)
    "MA_PERIOD": 50,
    "MA_TYPE": "ema",              # "ema" or "sma"

    # MACD (applied on the 1h entry timeframe)
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,

    # Volume confirmation (applied on 1h)
    "VOLUME_MA_PERIOD": 20,
    "VOLUME_MULTIPLIER": 1.0,      # current volume must be >= MULTIPLIER * its rolling avg

    # ATR period (daily ATR = the unit for stops and level distances)
    "ATR_PERIOD": 14,

    # Candlestick pattern detection
    "CANDLE_BODY_RATIO": 0.3,      # body must be <= this fraction of range to count as "small body"
    "CANDLE_WICK_RATIO": 2.0,      # wick must be >= this multiple of body to count as "long wick"
    "CANDLE_BIG_BODY_RATIO": 0.5,  # body must be >= this fraction of range to count as "big body"

    # Support/resistance swing lookback, per timeframe
    "SR_LOOKBACK": {"1D": 10, "4h": 20, "1h": 20},

    # ---- phone alerts (ntfy) - every setting for what reaches your phone is here (NTFY_TOPIC / NTFY_TOKEN /
    #      NTFY_SERVER come from the environment or .env)
    "NTFY_ENABLED": True,          # push the report(s) to your phone via ntfy (needs NTFY_TOPIC)
    "NTFY_FILES": ["up"],          # which reports to send: any of "up", "down", "csv"
    "NTFY_UP_VERSION": "lite",     # which "up" report the phone gets: "lite" = the BUY / WATCH list + their cards only
                                   # (<name>_up_lite.html, always written next to the full report); "full" = the whole _up report
    "NTFY_ON_REFRESH": "changes",  # refresh(): send "changes" (only when a signal changed), "always" or "never"
    "MOBILE_CHART_DAYS": 40,       # daily candles on the lite report's chart
    "NTFY_PDF": "auto",            # send a PDF instead of the .html: "auto" = if Playwright is installed, True = warn if it is not, False = never
    "NTFY_LINK": True,             # upload to a storage topic and send the alert with the file's URL in Click + a button (opens in the phone's browser)
    "NTFY_STORAGE_TOPIC": None,    # default "<NTFY_TOPIC>-files"; nobody should subscribe to it
    "NTFY_COMPRESS": "auto",       # "auto": zip attachments above NTFY_COMPRESS_ABOVE_MB; True: always; False: never
    "NTFY_COMPRESS_ABOVE_MB": 1.5,
    "NTFY_MAX_MB": 15,             # ntfy.sh attachment limit; bigger files are announced without the file

    "CSV_REPORT": True,            # also write reports/<name>_signal_report_<timestamp>.csv (1 row per ticker)
    "PROFILE": True,               # print a timing breakdown at the end of main()
    "MAX_WORKERS": 4,              # tickers screened in parallel (1 = one at a time)
    "CACHE_ENABLED": True,         # cache once-a-day data (.info, analyst, earnings...) on disk
    "CACHE_DIR": None,             # None = shared default (see cache_store.py); a CACHE_DIR / FINVIZ_CACHE_DIR env var wins
    "FINNHUB_MAX_PER_MIN": 55,     # Finnhub free tier allows 60 calls/minute
    "CATALYST_NEWS_DAYS": 14,      # headlines scanned for buyback/guidance keywords
    "HHHL_DAYS": 20,               # window for the higher-high/higher-low day count (completed daily bars)
    "DAILY_CHART_DAYS": 50,        # daily candles in the second (daily) chart
    # Daily support/resistance = industry-standard swing levels (see swing_sr_levels)
    "SR_DAYS": 60,                 # daily bars searched for swing support/resistance (~3 months; max ~130)
    "NEAR_SR_ATR": 0.5,            # "Near S1/S2" = price within this many daily ATRs of the level
    "SR_PIVOT": 2,                 # a swing low/high must be the lowest/highest of this many days on each side
    "SR_MERGE_ATR": 0.5,           # swing points within this many daily ATRs are merged into one level
    "TARGET_MIN_R": 2.0,           # targets below this reward:risk are flagged; fallback target when no resistance
    "AVWAP_ANCHOR": "low",         # anchored VWAP starts at the daily chart's lowest low ("low") or highest high ("high")
    "DAILY_CHART_OPEN": False,     # daily chart collapsed by default (its own show/hide link)
    "CHART_OPEN": False,           # chart collapsed by default (each card has a show/hide link)
    "CHART_DAYS": 7,               # number of recent trading days of 1h candles to plot
    "CHART_MAS": [("ema", 9), ("ema", 50)],  # MA overlays drawn on the chart
    "POC_BINS": 50,                # price buckets for the volume profile / POC
    "MARKET_TZ": "America/New_York",
    "MARKET_OPEN": "09:30",        # regular session, local to MARKET_TZ
    "MARKET_CLOSE": "16:00",

    # Linear regression channel on the chart (display only, not part of the signal)
    "LRC_ENABLED": True,
    "LRC_LENGTH": None,            # bars to fit (1h bars); None = the whole chart window
    "LRC_DEV": 2.0,                # channel half-width, in std devs of the residuals
    "LRC_MIN_R2_UP": 0.5,          # rising channels with R-squared below this go to the _down report
    "LRC_SOURCE": "Close",         # price column to fit: "Close", "High", "Low", "Open"

    # Pre-market price from Alpaca (display only). Keys come from the `headers`
    # passed to get_pre_market_price(), or the APCA_API_KEY_ID /
    # APCA_API_SECRET_KEY environment variables.
    "PREMARKET_ENABLED": True,
    "PREMARKET_START": "04:00",    # pre-market session start, MARKET_TZ
    "AFTERHOURS_END": "20:00",     # after-hours session end (MARKET_TZ); overnight runs from here to PREMARKET_START
    "OVERNIGHT_ENABLED": True,     # overnight (20:00-04:00 ET) price from Blue Ocean ATS via Alpaca
    "OVERNIGHT_FEED": "boats",     # Alpaca feed for overnight bars (historical must be "boats")
    "ALPACA_FEED": "auto",         # "auto" = pick SIP or IEX (see get_best_pre_market_price),
                                   # "sip" = all exchanges, delayed; "iex" = live but IEX-only
    "ALPACA_SIP_DELAY_MIN": 16,    # free plan: SIP must be >=15 min old; 0 if you pay for live SIP
    "PREMARKET_MOVE_PCT": 0.3,     # auto: use live IEX if price moved >= this % since the SIP snapshot
    "PREMARKET_IEX_MAX_GAP_PCT": 0.5,  # auto: distrust IEX if it differs from SIP by > this % at the same time
    "ALPACA_HEADERS": _SECRETS.get("ALPACA_HEADERS"),   # from constants.CONFIG

    # Company news (Finnhub) in the Event Catalysts section, collapsed by default.
    # Key: here, or the FINNHUB_API_KEY environment variable.
    "FINNHUB_API_KEY": _SECRETS.get("FINNHUB_API_KEY"),  # from constants.CONFIG
    "NEWS_DAYS": 1,                # 0 = today only, 1 = today + yesterday, ...
    "NEWS_MAX": 50,                # max articles shown per ticker (newest first)
    "NEWS_TZ": "America/New_York", # time zone for the article times shown
    "NEWS_FOR": "up",              # Finnhub news for: "up" = only tickers shown in the _up report,
                                   # "all" = every ticker shown, "none" = no news at all
    "NEWS_CACHE_MIN": 60,          # reuse a ticker's news for this many minutes (0 = always fetch)
    "NEWS_CACHE_MIN_REFRESH": 360, # refresh(): reuse news up to this old, and never wait on Finnhub's per-minute limit
}

SETUP_DEFAULTS = {
    # ---- strategy
    "TRADEABLE_SETUPS": ("PULLBACK",),   # the only setup with a measurable edge; add "MA_BOUNCE" / "SUPPORT_BOUNCE" / "MA_RECLAIM" / "FRESH_BREAKOUT" to trade them
    "OVERNIGHT_FEED_FALLBACKS": (),    # e.g. ("overnight",): feeds tried in order if OVERNIGHT_FEED is refused (403); see alpaca_check.py
    "REPORT_FILTER_MIN_TICKERS": 20,   # below this many tickers the report keeps EVERY ticker (no 4-check filter)
    "REPORT_SERVER": True,             # "Refresh now" / "Full scan" buttons in the _up report (only shown while report_server.py runs on this Mac)
    "REPORT_SERVER_PORT": 8765,        # must match report_server.py (REPORT_SERVER_PORT environment variable there)
    "TURN_WATCH": True,                # flag tickers with no setup that close at most TURN_WATCH_MAX_ATR under the EMA50 with the
    "TURN_WATCH_MAX_ATR": 1.5,         # daily AND 1h turn passing: never filtered out, sent to _up, listed (watch only, NOT traded, untested)
    "TRAIL_ARM_R": 2.0,            # the trailing stop arms once price has gained this many R
    "TRAIL_MA": "EMA20",           # ... and then trails under this daily moving average (raise only)
    "EXIT_MAX_HOLD_DAYS": 90,      # time stop for the trailing exit
    # ---- risk / reward
    "SETUP_MIN_RR": 1.5,           # BUY needs reward:risk >= this
    "TARGET_MIN_R": 1.5,           # (report flag) targets below this R are flagged
    "MIN_RISK_ATR": 0.8,           # stop is never closer than this many daily ATRs (stops R:R inflation)
    "MIN_TARGET_ATR": 1.0,         # a real resistance closer than this many ATRs is "blocking" (boxed in); minor pivots inside it are listed only
    "TARGET_FALLBACK_R": 2.0,      # no resistance far enough above -> target = this many R
    "TARGET_DAYS": 120,            # daily bars searched for resistance targets (longer than SR_DAYS)
    "DIP_TAIL_DAYS": None,         # volume-fade tail: None = 2 days on short dips (<= 4 days) else 3; 2 or 3 forces that length
    "RESET_INDICATOR": "stoch",    # "stoch" (daily %K, live) or "rsi" (daily RSI14) for the 'reset' confirmation
    "RESET_BARS": 5,               # ... must have been at/below RESET_LEVEL at least once in this many recent daily bars
    "RESET_LEVEL": 40,
    "VALUE_ZONE_ATR": 0.5,         # pullback confirmation: the dip low sat within this many ATRs of EMA20/EMA50/SMA50/a swing level
    "DIP_TAIL_MAX_RVOL": 1.0,      # volume must be fading AND the last dip days at/below this multiple of normal volume
    "OVERHEAD_MIN_TOUCHES": 2,     # a swing level inside MIN_TARGET_ATR blocks the target when it has >= this many touches
    "WATCH_CONFIRM_GAP": 1,        # WATCH = required checks pass and confirmations are within this many of the minimum
    "GROUP_BY_SETUP": True,        # any traded setup (BUY/WATCH/WAIT) goes to the _up report even if the 1h channel slopes down
    "MAX_EXT_ATR": 3.0,            # non-breakout setups are rejected if price is this far above EMA50
    # ---- context
    "RECLAIM_MAX_DAYS": 6,         # EARLY_UPTREND = price closed above EMA50 for 1..N days after being below
    "CORRECTION_MAX_BELOW_ATR": 0.75,  # CORRECTION = closed under EMA50 by <= this many ATRs with a higher low intact
    "RECLAIM_MIN_BELOW": 4,        # ...and closed below it on >= this many of the 15 days before
    # ---- fresh breakout
    "BREAKOUT_BASE_BARS": 20,      # the level = highest high of this many bars before the breakout window
    "BREAKOUT_FRESH_BARS": 3,      # the first close above the level happened within this many bars
    "BREAKOUT_MAX_EXT_ATR": 1.0,   # not more than this many ATRs above the broken level (no chasing)
    "BREAKOUT_RVOL": 1.5,          # breakout volume vs 20-day average (REQUIRED)
    "BREAKOUT_BASE_MAX_ATR": 4.0,  # tight base: the 10 bars before the breakout span <= this many ATRs
    "BREAKOUT_STOP_ATR": 0.5,      # stop = broken level - this many ATRs
    "BREAKOUT_MIN_TOUCHES": 2,     # a swing level counts as resistance only with >= this many touches
    # ---- support bounce
    "SUPPORT_MIN_STRENGTH": 2,     # touches (+1 if the level was both a swing high and a swing low)
    "SUPPORT_TEST_ATR": 0.35,      # low within this many ATRs above the level = "tested"
    "SUPPORT_MAX_PIERCE_ATR": 0.75,  # wick may undercut the level by up to this many ATRs (stop-run), closes may not
    "SUPPORT_MAX_AWAY_ATR": 1.0,   # price no more than this many ATRs above the level
    # ---- MA bounce
    "MA_TEST_ATR": 0.3,            # low within this many ATRs above the MA = "tested"
    "MA_MAX_PIERCE_ATR": 1.0,
    "MA_MAX_AWAY_ATR": 1.0,
    # ---- pullback
    "PULLBACK_LOOKBACK": 30,       # bars searched for the last impulse high
    "PULLBACK_MIN_LEG_ATR": 3.0,   # impulse leg must span >= this many ATRs
    "PULLBACK_LEG_BARS": 30,       # the impulse leg starts at the lowest low of this many bars before the high
    "STRUCT_PIVOT": 5,             # trend structure (higher highs / higher lows) uses major swings: N daily bars each side
    "STRUCT_DAYS": 90,             # daily bars searched for those swings
    "PULLBACK_STRUCT_BARS": 5,     # launch low = lowest low of this many daily bars on each side (before the high)
    "PULLBACK_LOW_TOL_ATR": 0.25,  # the pullback low may undercut the launch low by this many ATRs (stop-run wick)
    "PULLBACK_MIN_DROP_ATR": 1.5,  # the pullback must have dropped at least this many ATRs from the high
    "PULLBACK_MIN_DEPTH": 0.20,    # retracement of the leg (0.236-0.618 Fib zone, with slack)
    "PULLBACK_MAX_DEPTH": 0.65,
    # ---- MA reclaim
    "RECLAIM_MIN_HOLD_DAYS": 2,    # consecutive closes above EMA50 before it counts as a reclaim
    "RECLAIM_MAX_EXT_ATR": 2.0,
    # ---- confirmations needed per setup. Each detector lists 4 (MA_RECLAIM: 5); the traded PULLBACK needs 3 of 4
    "MIN_CONFIRMS": {"FRESH_BREAKOUT": 3, "SUPPORT_BOUNCE": 3, "MA_BOUNCE": 3, "PULLBACK": 3, "MA_RECLAIM": 3},
}
CONFIGH.update(SETUP_DEFAULTS)
