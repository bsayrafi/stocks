"""
 Multi-Timeframe Setup Screener (v2)
------------------------------------------------------
Fetches hourly price data, builds 4h and daily bars from it, and classifies each
ticker into a trade SETUP instead of running one generic gate:

  1. CONTEXT  (daily)  UPTREND / EARLY_UPTREND / UPTREND_WEAK / RANGE / DOWNTREND
                       from price vs EMA50, EMA50 slope, swing higher-lows/highs,
                       MA reclaim history and extension above EMA50 (in ATRs)
  2. SETUP             traded: PULLBACK only  (MA_BOUNCE, SUPPORT_BOUNCE, MA_RECLAIM, FRESH_BREAKOUT are detected, not traded)
  3. CONFIRMATION      setup-specific: volume, stochastic, candle, structure, 1h timing
  4. RISK              setup-specific stop; nearest real resistance only FILTERS entries (R:R >= 1.5)
  5. VERDICT           BUY (required checks + enough confirmations + R:R)
                       WATCH (setup present but blocked)  /  WAIT (no setup)
  6. EXIT              initial stop, no profit target; after +2R trail the stop under the daily EMA20 (raise only);
                       leave after 90 trading days. (Backtested over 2 years; see backtest_v2.py for the evidence.)

WHERE THINGS LIVE (setups_v2/)
  config.py         settings (CONFIGH, incl. the setup engine's defaults and the phone-alert block), API keys, profiling
  indicators.py     indicators, candle patterns, levels, session VWAP, 4h/daily bars - pure pandas / numpy
  engine.py         the setup engine (layers 1-5 above, evaluate_setups) - pure pandas / numpy
  data.py           1h bars, Alpaca extended hours / pre-market, news, fundamentals, analyst data, disk cache
  report_html.py    the desktop report;   report_mobile.py   the phone (lite) report
  enrich_html_v2.py (this file) run_screen, main(), refresh(), CSV, ntfy. It re-exports every name of the modules
                    above, so enrich_html_v2.<name> keeps working for run_trend_entry.py and backtest_v2.py.
"""

import yfinance as yf
import pandas as pd
import html
import os
import requests
import datetime as dt
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from common import constants
from common import cache_store
try:
    import finnhub  # pip install finnhub-python
except ImportError:
    finnhub = None

# ---- re-exports: everything that used to live in this file is still reachable as enrich_html_v2.<name>
from setups_v2.config import (  # noqa: F401
    PROFILE,
    _prof_add,
    _profiled,
    print_profile_summary,
    CONFIGH,
    SETUP_DEFAULTS,
    _SECRETS,
)
from setups_v2.indicators import (  # noqa: F401
    resample_ohlc,
    stochastic_oscillator,
    moving_average,
    macd,
    average_true_range,
    is_bullish_engulfing,
    is_bullish_harami,
    is_piercing_line,
    is_hammer,
    is_morning_star,
    BULLISH_PATTERNS,
    detect_candlestick_patterns,
    evaluate_timeframe,
    volume_confirmed,
    macd_bullish,
    swing_support_resistance,
    day_range,
    chart_window,
    volume_poc,
    _hhmm_td,
    swing_sr_levels,
    anchored_vwap,
    hhhl_stats,
    linear_regression_channel,
    session_vwap,
    prev_session,
)
from setups_v2.data import (  # noqa: F401
    get_event_catalysts,
    _get_sp500_membership,
    fetch_apewisdom_table,
    get_earnings_and_ratings,
    _cache_sync,
    _cache_path,
    _cache_write,
    daily_cached,
    ttl_cached,
    prune_cache,
    _RateLimiter,
    _finnhub_limiter,
    _finnhub_limiter_lock,
    _finnhub_wait,
    _NEWS_BUSY,
    fetch_hourly_data,
    prefetch_hourly_data,
    prefetch_pre_market,
    ALPACA_DATA_URL,
    _alpaca_headers,
    _alpaca_bars_window_multi,
    _alpaca_premarket_bars_multi,
    session_calendar,
    _single_feed_details,
    _ext_window_cached,
    prefetch_extended_hours,
    extended_hours_for_ticker,
    _SESSION_NAMES,
    _alpaca_premarket_bars,
    get_pre_market_price,
    _choose_pre_market,
    get_best_pre_market_price,
    get_best_pre_market_prices,
    get_pre_market_prices,
    fetch_company_news,
    _fmt_billions,
    dividend_yield_fraction,
    FUNDAMENTAL_FIELDS,
    fetch_info,
    fetch_fundamentals,
    fetch_analyst_data,
)
from setups_v2.engine import (  # noqa: F401
    SETUP_LABELS,
    SETUP_EVIDENCE,
    NOT_TRADED_WHY,
    _ema,
    _pct_slope,
    _pivots,
    swing_levels_all,
    intraday_rvol,
    daily_context,
    _candle_hits,
    _rejection,
    _rsi,
    _dip_volume_contracting,
    _stoch_reset_turn,
    _reset_label,
    detect_breakout,
    detect_support_bounce,
    detect_ma_bounce,
    _launch_low,
    detect_pullback,
    detect_reclaim,
    trade_plan,
    evaluate_setups,
)
from setups_v2.report_html import (  # noqa: F401
    check_key,
    signal_badge_text,
    print_signal_block,
    render_signal_section,
    _badge,
    _four_col_table,
    _hhmm_to_min,
    _series_xy,
    render_price_chart_svg,
    render_daily_chart_svg,
    _streak_text,
    render_chart_toggles,
    _sr_text,
    near_level,
    premarket_near_support,
    _premarket_near_row,
    _near_sr_rows,
    _extended_hours_rows,
    _target_r_html,
    render_ticker_html,
    build_html_report,
    _snake,
    report_filter_checks,
    _skipped_card,
    _EXT_ABBR,
    _ext_line,
    _summary_price,
    _TIP_SESSIONS,
    _ticker_tip,
    _TIP_SCRIPT,
    _card_id,
    _SUMMARY_COLS,
    _TURN_COLS,
    _summary_head_html,
    _setup_summary_card,
    _turning_up_card,
    _sector_filter_bar,
)
from setups_v2.report_mobile import (  # noqa: F401
    _M_CSS,
    build_mobile_report,
    _m_num,
    _m_sessions,
    _m_checks,
    _m_chart_svg,
    _m_swatch,
    _m_charts,
    _m_kv,
    _m_details,
    _m_sec_timeframes,
    _m_sec_levels,
    _m_sec_catalysts,
    _m_sec_analysts,
    _m_sec_fundamentals,
    render_mobile_card,
    mobile_summary,
)

# ---------------------------------------------------------------- main

def run_screen(ticker: str, cfg: dict, sp500_members: dict | None = None,
               hourly: pd.DataFrame | None = None, social_table: dict | None = None,
               ext_batch: dict | None = None) -> dict:
    """Fetch data and compute everything needed for one ticker's report.
    Pure computation - no printing - so the same result can feed both the
    console output and the HTML report."""
    cfg = dict(cfg)
    cfg["TICKER"] = ticker
    if hourly is None or hourly.empty:   # not prefetched -> download just this one
        hourly = fetch_hourly_data(cfg["TICKER"], cfg["PERIOD"], cfg["INTERVAL"])

    tf_data = {
        "1h": hourly,
        "4h": resample_ohlc(hourly, "4h"),
        "1D": resample_ohlc(hourly, "1D"),
    }

    stoch = {tf: stochastic_oscillator(df, cfg["STOCH_K_PERIOD"], cfg["STOCH_D_PERIOD"], cfg["STOCH_SMOOTH"])
              for tf, df in tf_data.items()}
    ma = {tf: moving_average(df, cfg["MA_PERIOD"], cfg["MA_TYPE"]) for tf, df in tf_data.items()}

    results = {tf: evaluate_timeframe(tf_data[tf], stoch[tf], ma[tf], cfg) for tf in tf_data}

    macd_1h = macd(tf_data["1h"], cfg["MACD_FAST"], cfg["MACD_SLOW"], cfg["MACD_SIGNAL"])
    h1_macd_ok = macd_bullish(macd_1h)

    vol_ok = volume_confirmed(tf_data["1h"], cfg["VOLUME_MA_PERIOD"], cfg["VOLUME_MULTIPLIER"])

    current_price = results["1h"]["close"]
    # daily ATR: the unit for every stop / level distance in the setup engine
    daily_atr_val = float(average_true_range(tf_data["1D"], cfg["ATR_PERIOD"]).iloc[-1])

    # swing support / resistance per timeframe and today's range - context only.
    # The stop and target come from the setup (evaluate_setups); a ticker with no setup has neither.
    sr_levels = {tf: swing_support_resistance(tf_data[tf], cfg["SR_LOOKBACK"][tf]) for tf in tf_data}
    entry_support = sr_levels["1h"]["support"]
    today_range = day_range(tf_data["1h"])

    # Chart data (display only): MA overlays on 1h + 7-day volume POC
    chart_mas = {}
    for ma_type, period in cfg.get("CHART_MAS", [(cfg["MA_TYPE"], cfg["MA_PERIOD"])]):
        chart_mas[f"{ma_type.upper()}{period}"] = moving_average(tf_data["1h"], period, ma_type)
    chart = chart_window(tf_data["1h"], chart_mas, cfg.get("CHART_DAYS", 7))
    poc = volume_poc(chart["ohlcv"], cfg.get("POC_BINS", 50))

    vwap_1h = session_vwap(tf_data["1h"], cfg.get("MARKET_TZ", "America/New_York"),
                           cfg.get("MARKET_OPEN", "09:30"), cfg.get("MARKET_CLOSE", "16:00"))
    chart["vwap"] = vwap_1h.loc[chart["ohlcv"].index]

    # ---- daily structure (HH/HL) + daily chart - built from the daily bars we
    # already resampled from the hourly download, so no extra data is fetched
    hhhl = hhhl_stats(tf_data["1D"], cfg.get("HHHL_DAYS", 20), cfg.get("MARKET_TZ", "America/New_York"),
                      cfg.get("MARKET_CLOSE", "16:00"))
    chart["hhhl_by_date"] = hhhl["by_date"]
    n_daily = cfg.get("DAILY_CHART_DAYS", 20)
    daily_mas = {}
    for ma_type, period in cfg.get("CHART_MAS", [(cfg["MA_TYPE"], cfg["MA_PERIOD"])]):
        daily_mas[f"{ma_type.upper()}{period}"] = moving_average(tf_data["1D"], period, ma_type).iloc[-n_daily:]
    chart["daily"] = {"ohlcv": tf_data["1D"].iloc[-n_daily:], "overlays": daily_mas,
                      "today_in_progress": hhhl["today_in_progress"]}
    avwap = anchored_vwap(tf_data["1h"], chart["daily"]["ohlcv"], cfg.get("AVWAP_ANCHOR", "low"),
                          cfg.get("MARKET_TZ", "America/New_York"))
    chart["daily"]["avwap"] = avwap
    _nm = cfg.get("MOBILE_CHART_DAYS", 40)
    _d1 = tf_data["1D"]
    chart["daily_m"] = {"ohlcv": _d1.iloc[-_nm:],
                        "ema20": _d1["Close"].ewm(span=20, adjust=False).mean().iloc[-_nm:],
                        "ema50": _d1["Close"].ewm(span=50, adjust=False).mean().iloc[-_nm:],
                        "today_in_progress": hhhl["today_in_progress"]}
    vwap_valid = vwap_1h.dropna()
    vwap_now = round(float(vwap_valid.iloc[-1]), 2) if not vwap_valid.empty else None
    above_vwap = (current_price > vwap_now) if vwap_now is not None else None

    # ---- close / after-hours / overnight / pre-market (Alpaca). main() fetches all
    # tickers in one batch; a single run_screen() call fetches just this ticker.
    ext = None
    if cfg.get("PREMARKET_ENABLED", True):
        if ext_batch is None:   # standalone call (not from main) - fetch just this ticker
            ext_batch = prefetch_extended_hours([cfg["TICKER"]], cfg, {cfg["TICKER"]: hourly})
        ext = extended_hours_for_ticker(ext_batch, cfg["TICKER"], hourly, cfg)
    pm = (ext or {}).get("sessions", {}).get("pm") or {}
    pre_market = pm.get("price")
    pre_market_time = pm.get("time")
    pre_market_info = pm if pre_market is not None else None

    # The channel is always computed because its slope decides whether the ticker
    # goes to the _up or _down report; LRC_ENABLED only controls whether it's shown.
    lrc_fit = linear_regression_channel(chart["ohlcv"], cfg.get("LRC_LENGTH"),
                                        cfg.get("LRC_DEV", 2.0), cfg.get("LRC_SOURCE", "Close"))
    # "up" needs a rising channel AND a decent fit: a rising line through choppy
    # prices (low R-squared) isn't a real uptrend, so it goes to the _down report.
    min_r2 = cfg.get("LRC_MIN_R2_UP", 0.5)
    if lrc_fit is not None:
        if lrc_fit["slope_per_bar"] < 0:
            trend_dir, trend_reason = "down", "channel sloping down"
        elif lrc_fit["r2"] < min_r2:
            trend_dir, trend_reason = "down", f"channel rising but weak fit (R2 {lrc_fit['r2']} < {min_r2})"
        else:
            trend_dir, trend_reason = "up", f"channel rising (R2 {lrc_fit['r2']})"
    else:  # too few bars to fit - fall back to first vs last close in the window
        w = chart["ohlcv"]["Close"]
        trend_dir = "up" if float(w.iloc[-1]) >= float(w.iloc[0]) else "down"
        trend_reason = "too few bars for a channel - first vs last close"
    lrc = lrc_fit if cfg.get("LRC_ENABLED", True) else None
    chart["lrc"] = lrc

    # ---- fundamentals / analyst / catalysts: one shared yf.Ticker and one .info,
    # with the once-a-day parts cached on disk (see daily_cached)
    tk = cfg["TICKER"]
    tkr = yf.Ticker(tk)
    info = daily_cached("info", tk, lambda: fetch_info(tk, tkr))

    # ---- setup-based signal (context -> setup -> confirmations -> risk -> verdict).
    # Runs after .info so the 52-week high can count as resistance.
    signal = evaluate_setups(tf_data["1D"], daily_atr_val, results, stoch["1D"], tf_data["1h"],
                             h1_macd_ok, above_vwap, hhhl["today_in_progress"], cfg,
                             high_52w=(info or {}).get("fiftyTwoWeekHigh"), vwap=vwap_now)
    # Trade plan: only a setup has one (its structural stop + the nearest real resistance, used as the R:R filter).
    # With no setup, stop / target stay None and the daily S/R levels are listed without R multiples.
    stop, take_profit_target = signal["stop"], signal["target"]
    # levels are searched over SR_DAYS (independent of how many days the chart shows)
    day_sr = swing_sr_levels(tf_data["1D"].iloc[-cfg.get("SR_DAYS", 60):], current_price, daily_atr_val, stop,
                             pivot=cfg.get("SR_PIVOT", 2), merge_atr=cfg.get("SR_MERGE_ATR", 0.5),
                             min_r=cfg.get("TARGET_MIN_R", 1.5))
    day_sr["target"] = None if stop is None else {
        "price": signal["target"], "r": signal["rr"], "source": signal["target_src"],
        "below_min_r": signal["rr"] < cfg["SETUP_MIN_RR"]}
    chart["daily"]["sr"] = day_sr
    # A pullback pushes the 1h channel down almost by definition, so the channel would
    # hide exactly the setups we trade. Any traded setup (BUY, WATCH or a still-forming
    # WAIT) goes to the _up report; the channel is only used for tickers with no setup.
    if cfg.get("GROUP_BY_SETUP", True) and signal["setup"] is not None and trend_dir != "up":
        trend_reason = f"{signal['setup_label']} ({signal['signal']}) overrides the 1h channel ({trend_reason})"
        trend_dir = "up"
    elif signal.get("turning_up") and trend_dir != "up":      # watch-only: shown with the setups, not hidden in _down
        trend_reason = f"turning up near the EMA50 overrides the 1h channel ({trend_reason})"
        trend_dir = "up"
    fundamentals = fetch_fundamentals(tk, info=info)
    analyst = daily_cached(
        "analyst", tk, lambda: fetch_analyst_data(tk, tkr=tkr, info=info),
        is_valid=lambda a: bool(info) and (a.get("recommendation_key") != "N/A" or a.get("eps_current") != "N/A"))
    earnings_ratings = daily_cached(
        "earnings_ratings", tk, lambda: get_earnings_and_ratings(tk, yf_ticker=tkr, info=info),
        is_valid=lambda er: bool(info))

    # One Finnhub call covers both the News row (last NEWS_DAYS) and the
    # buyback/guidance keyword scan (last CATALYST_NEWS_DAYS) - no Yahoo news call.
    # Finnhub's free plan allows 60 calls/minute, so news is only fetched for the
    # tickers that NEWS_FOR selects (default: shown in the _up report), and each
    # ticker's result is reused for NEWS_CACHE_MIN minutes.
    news_days = cfg.get("NEWS_DAYS", 0)
    catalyst_days = cfg.get("CATALYST_NEWS_DAYS", 14)
    news_for = cfg.get("NEWS_FOR", "up")
    will_be_shown = (not cfg.get("_filter_active", True)) or not all(
        report_filter_checks({"signal": signal, "analyst": analyst}).values())
    if news_for == "none":
        news_skipped = "news turned off (NEWS_FOR)"
    elif not will_be_shown:
        news_skipped = "ticker filtered out"
    elif news_for == "up" and trend_dir != "up":
        news_skipped = "not fetched for down-trend tickers"
    else:
        news_skipped = None

    news_all = None
    if news_skipped is None:
        fetch_days = max(news_days, catalyst_days)
        if cfg.get("_refresh_mode"):
            news_all = _news_for_refresh(tk, fetch_days, cfg)
        else:
            news_all = ttl_cached(
                "news", f"{tk}_{fetch_days}",
                lambda: fetch_company_news(tk, api_key=cfg.get("FINNHUB_API_KEY"), days=fetch_days,
                                           max_items=None, tz=cfg.get("NEWS_TZ", "America/New_York")),
                ttl_minutes=cfg.get("NEWS_CACHE_MIN", 60))
    # skipped -> [] so event_catalysts doesn't fall back to a Yahoo news call
    headlines = [] if news_skipped else (None if news_all is None else [(n["date"], n["headline"]) for n in news_all])

    catalysts = get_event_catalysts(tk, news_lookback_days=catalyst_days, sp500_members=sp500_members,
                                    yf_ticker=tkr, info=info, headlines=headlines,
                                    social_table=social_table, earnings_ratings=earnings_ratings).to_dict()

    news_cutoff = dt.date.today() - dt.timedelta(days=news_days)
    news = [n for n in (news_all or []) if n["date"] >= news_cutoff][:cfg.get("NEWS_MAX", 25)]
    catalysts["news_skipped"] = news_skipped

    return {
        "ticker": cfg["TICKER"],
        "cfg": cfg,
        "current_price": current_price,
        "results": results,
        "sr_levels": sr_levels,
        "today_range": today_range,
        "daily_atr": round(daily_atr_val, 2),
        "h1_macd_ok": h1_macd_ok,
        "vol_ok": vol_ok,
        "stop": stop,
        "take_profit": take_profit_target,
        "entry_support": entry_support,
        "signal": signal,
        "fundamentals": fundamentals,
        "analyst": analyst,
        "catalysts": catalysts,
        "chart": chart,
        "hhhl": {k: v for k, v in hhhl.items() if k != "by_date"},
        "day_sr": day_sr,
        "avwap": None if not avwap else {
            "value": avwap["value"], "anchor_date": avwap["anchor_date"].isoformat(),
            "anchor_type": avwap["anchor_type"], "anchor_price": round(avwap["anchor_price"], 2),
            "above": (current_price > avwap["value"]) if avwap["value"] is not None else None},
        "poc": poc,
        "vwap": vwap_now,
        "above_vwap": above_vwap,
        "lrc": lrc,
        "trend_dir": trend_dir,
        "trend_reason": trend_reason,
        "pre_market": pre_market,
        "ext": ext,
        "pre_market_time": pre_market_time,
        "pre_market_info": pre_market_info,
        "news": news,
        # raw (unformatted) yfinance values behind the Fundamentals table, for the CSV
        "fund_raw": ({key: (dividend_yield_fraction(info) if key == "dividendYield" else info.get(key))
                      for _, key, _ in FUNDAMENTAL_FIELDS} if info else {}),
        "prev_day": prev_session(tf_data["1D"], hhhl["today_in_progress"]),
    }


def print_report(report: dict) -> None:
    """Console rendering of a run_screen() report - unchanged formatting from before."""
    cfg = report["cfg"]
    results = report["results"]
    sr_levels = report["sr_levels"]
    today_range = report["today_range"]
    signal = report["signal"]
    current_price = report["current_price"]
    entry_support = report["entry_support"]

    print(f"\n=== {cfg['TICKER']} Multi-Indicator Signal Check ===")
    print(f"Current price: {current_price:.2f}\n")
    for tf in ["1D", "4h", "1h"]:
        r = results[tf]
        sr = sr_levels[tf]
        patterns = ", ".join(r["patterns_detected"]) if r["patterns_detected"] else "none"
        print(f"[{tf}] close={r['close']} {cfg['MA_TYPE'].upper()}{cfg['MA_PERIOD']}={r['MA']} trend_up={r['trend_up']!s:<5} "
              f"%K={r['%K']:>6} %D={r['%D']:>6} K>D={r['k_above_d']!s:<5} "
              f"stoch_bullish={r['stoch_bullish']!s:<5} overbought={r['overbought']!s:<5} "
              f"candles=[{patterns}]")
        print(f"      support={sr['support']}  resistance={sr['resistance']}  "
              f"(last {cfg['SR_LOOKBACK'][tf]} bars)")

    print(f"\nToday's range: {today_range['day_low']} - {today_range['day_high']} "
          f"(range: {today_range['day_range']})")

    if report.get("poc") is not None:
        print(f"POC ({cfg.get('CHART_DAYS', 7)}d volume profile): {report['poc']}")
    if report.get("vwap") is not None:
        side = "above" if report["above_vwap"] else "below"
        print(f"Session VWAP: {report['vwap']} (price {side} VWAP)")
    ext = report.get("ext")
    if ext:
        line = f"Close {ext['close']:.2f} ({ext['close_date']})"
        for key, label in (("ah", "after-hours"), ("on", "overnight"), ("pm", "pre-market")):
            d = ext["sessions"].get(key) or {}
            line += (f" | {label} {d['price']:.2f} ({d['pct']:+.2f}%)" if d.get("price") is not None
                     else f" | {label} {d.get('status', 'N/A')}")
        print(line + ("" if not ext.get("current") else f" | current = {_SESSION_NAMES[ext['current']['session']]}"))
    st = report.get("hhhl")
    if st:
        print(f"HH/HL days (last {st['window']} completed): {st['up_days']} up / {st['down_days']} down / "
              f"{st['mixed_days']} mixed (net {st['net']:+d}); current streak {_streak_text(st['current_streak'])}, "
              f"previous {_streak_text(st['previous_streak'])}")
    lrc = report.get("lrc")
    if lrc:
        print(f"Regression channel ({lrc['bars']} bars, {lrc['dev']}σ): {lrc['last_lower']} / "
              f"{lrc['last_mid']} / {lrc['last_upper']}  slope {lrc['slope_pct_per_day']:+.2f}%/day  "
              f"R² {lrc['r2']}  price at {lrc['position_pct']:.0f}% of channel")

    print(f"\n1h MACD bullish: {report['h1_macd_ok']}")
    print(f"1h volume confirmed (>= {cfg['VOLUME_MULTIPLIER']}x {cfg['VOLUME_MA_PERIOD']}-period avg): {report['vol_ok']}")
    if report["stop"] is not None:
        risk = report['current_price'] - report['stop']
        tgt = (report.get("day_sr") or {}).get("target") or {}
        print(f"Initial stop (from the setup): {report['stop']} (risking {risk:.2f} per share)")
        print(f"Nearest real resistance (R:R filter, no profit target): {report['take_profit']} "
              f"({tgt.get('source')}, {tgt.get('r')}R)")
    else:
        print("No setup - no stop or target")

    print_signal_block(signal)

    fundamentals = report.get("fundamentals") or {}
    if fundamentals:
        print("Fundamentals:")
        for label, value in fundamentals.items():
            print(f"  {label:<24} {value}")
        print()

    analyst = report.get("analyst") or {}
    if analyst:
        print("Analyst Recommendations:")
        print(f"  Consensus: {analyst['recommendation_key']}  "
              f"(mean score {analyst['recommendation_mean']}, "
              f"{analyst['num_analyst_opinions']} analysts)")
        counts = analyst.get("recommendation_counts") or {}
        if counts:
            counts_str = "  ".join(f"{label}: {n}" for label, n in counts.items())
            print(f"  {counts_str}")
        print()

        print("EPS Estimate Revisions (current quarter):")
        print(f"  Current consensus EPS: {analyst['eps_current']}   "
              f"30 days ago: {analyst['eps_30d_ago']}   "
              f"# analysts: {analyst['eps_num_analysts']}")
        if analyst["eps_improving"] is not None:
            trend_str = "IMPROVING (revised up)" if analyst["eps_improving"] else "DETERIORATING (revised down)"
            print(f"  EPS trend: {trend_str}")
        else:
            print("  EPS trend: N/A")
        print()



def screen_ticker(ticker):
    """Run the screen for a single ticker and print its console report."""
    report = run_screen(ticker, CONFIGH)
    #print_report(report)
    return report


def _ascii(text: str) -> str:
    """HTTP header values must be plain ASCII."""
    return str(text).encode("ascii", "replace").decode("ascii")


def html_to_pdf(html_path: str, width_px: int = 1100, max_page_px: int = 19000) -> str | None:
    '''Render a report to ONE long PDF page (no page breaks through the cards), the format every phone can open.
    Returns the PDF path, or None when Playwright/Chromium is not installed or rendering fails (the caller then
    sends the HTML). Install once:  pip install playwright   and   playwright install chromium'''
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        if CONFIGH.get("NTFY_PDF", "auto") is True:
            print("ntfy: NTFY_PDF is on but Playwright is not installed - sending the HTML. "
                  "Install it with:  pip install playwright  &&  playwright install chromium")
        return None
    pdf_path = os.path.splitext(html_path)[0] + ".pdf"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": width_px, "height": 900})
            page.goto("file://" + os.path.abspath(html_path))
            page.wait_for_timeout(300)
            page.emulate_media(media="screen")
            page.add_style_tag(content=".table-wrap{overflow:visible !important} html,body{background:#0f1115 !important}")
            height = int(page.evaluate("Math.ceil(document.documentElement.scrollHeight)"))
            page.pdf(path=pdf_path, width=f"{width_px}px", height=f"{min(height + 4, max_page_px)}px", print_background=True,
                     margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
            browser.close()
        print(f"PDF written to {pdf_path} ({os.path.getsize(pdf_path) / 1024 / 1024:.1f} MB)")
        return pdf_path
    except Exception as e:
        print(f"PDF conversion failed ({type(e).__name__}: {e}) - sending the HTML instead")
        return None


def _ntfy_send_link(path, server, topic, token, title, message, tags, priority, timeout):
    '''Upload the report to a separate STORAGE topic (nobody subscribes to it), read the attachment URL from ntfy's reply,
    then publish the real alert on the main topic with that URL in the Click header and a "view" button, so tapping opens
    it in the phone's browser. ntfy.sh deletes attachments after about 3 hours. Raises on any failure.'''
    store = (CONFIGH.get("NTFY_STORAGE_TOPIC") or _SECRETS.get("NTFY_STORAGE_TOPIC")
             or os.environ.get("NTFY_STORAGE_TOPIC") or f"{topic}-files")[:64]
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    with open(path, "rb") as f:
        r = requests.put(f"{server}/{store}", data=f, timeout=timeout,
                         headers={**auth, "Filename": _ascii(os.path.basename(path)), "Title": "report file (storage)",
                                  "Priority": "min", "Tags": "file_folder"})
    r.raise_for_status()
    att = (r.json() or {}).get("attachment") or {}
    url = att.get("url")
    if not url:
        raise ValueError("ntfy's reply has no attachment URL")
    body = message or ""
    if att.get("expires"):
        body += f"\n(link valid until {datetime.fromtimestamp(att['expires']).strftime('%H:%M')})"
    hdr = {**auth, "Title": _ascii(title or os.path.basename(path)), "Click": url, "Actions": f"view, Open report, {url}"}
    if tags:
        hdr["Tags"] = _ascii(tags)
    if priority:
        hdr["Priority"] = _ascii(priority)
    r2 = requests.post(f"{server}/{topic}", data=body.encode("utf-8"), headers=hdr, timeout=timeout)
    r2.raise_for_status()
    return url


def send_ntfy_file(path: str, topic: str | None = None, server: str | None = None,
                   token: str | None = None, title: str | None = None, message: str | None = None,
                   tags: str | None = None, priority: str | None = None, timeout: float = 60) -> bool:
    """Send a file (e.g. the HTML report) as an ntfy attachment; the phone app
    shows a notification you tap to open the file.
    topic / server / token come from the arguments, else constants.CONFIG
    ("NTFY_TOPIC", "NTFY_SERVER", "NTFY_TOKEN"), else the environment variables
    of the same names. Server defaults to https://ntfy.sh.
    Files larger than CONFIGH["NTFY_MAX_MB"] are announced without the file.
    Returns True if ntfy accepted it."""
    topic = topic or _SECRETS.get("NTFY_TOPIC") or os.environ.get("NTFY_TOPIC")
    if not topic:
        print("ntfy: no NTFY_TOPIC set - skipped")
        return False
    server = (server or _SECRETS.get("NTFY_SERVER") or os.environ.get("NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
    token = token or _SECRETS.get("NTFY_TOKEN") or os.environ.get("NTFY_TOKEN")

    size_mb = os.path.getsize(path) / 1024 / 1024
    headers = {"Title": _ascii(title or os.path.basename(path))}
    if tags:
        headers["Tags"] = _ascii(tags)
    if priority:
        headers["Priority"] = _ascii(priority)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    def _reason(e):
        r = getattr(e, "response", None)
        try:
            return f" | ntfy says: {r.text.strip()[:250]}" if r is not None and r.text else ""
        except Exception:
            return ""

    def _put(file_path, base_headers):
        h = dict(base_headers)
        h["Filename"] = _ascii(os.path.basename(file_path))
        if message:
            h["Message"] = _ascii(message)
        with open(file_path, "rb") as f:
            r_ = requests.put(f"{server}/{topic}", data=f, headers=h, timeout=timeout)
        r_.raise_for_status()

    def _text_only(why):
        body = (f"{message or ''}\n({os.path.basename(path)}, {size_mb:.1f} MB, not attached: {why})").strip()
        r_ = requests.post(f"{server}/{topic}", data=body.encode("utf-8"), headers=headers, timeout=timeout)
        r_.raise_for_status()

    if CONFIGH.get("NTFY_LINK", True) and size_mb <= CONFIGH.get("NTFY_MAX_MB", 15):
        try:
            url = _ntfy_send_link(path, server, topic, token, title, message, tags, priority, timeout)
            print(f"ntfy: uploaded {os.path.basename(path)} ({size_mb:.1f} MB) to the storage topic, alert sent with a link: {url}")
            return True
        except Exception as e:                                         # RequestException, ValueError, ...
            print(f"ntfy: the link method failed ({type(e).__name__}: {e}{_reason(e)}) - falling back to an attachment")
    if CONFIGH.get("NTFY_PDF", "auto") and path.lower().endswith(".html"):
        pdf_ = html_to_pdf(path)                                       # a PDF opens on any phone
        if pdf_:
            path, size_mb = pdf_, os.path.getsize(pdf_) / 1024 / 1024
    mode = CONFIGH.get("NTFY_COMPRESS", "auto")                      # "auto" | True | False
    zip_above = CONFIGH.get("NTFY_COMPRESS_ABOVE_MB", 1.5)
    use_zip = bool((mode is True or (mode == "auto" and size_mb > zip_above)) and not path.lower().endswith(".pdf"))
    to_send, sent_mb = path, size_mb
    tmp_zip = None
    try:
        if size_mb > CONFIGH.get("NTFY_MAX_MB", 15) and not use_zip:
            _text_only("too big to attach")
            print(f"ntfy: {os.path.basename(path)} is {size_mb:.1f} MB - sent the text summary only")
            return True
        if use_zip:
            import zipfile
            tmp_zip = path + ".zip"
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                z.write(path, arcname=os.path.basename(path))
            to_send, sent_mb = tmp_zip, os.path.getsize(tmp_zip) / 1024 / 1024
        try:
            _put(to_send, headers)
            note = f", zipped from {size_mb:.1f} MB" if use_zip else ""
            print(f"ntfy: sent {os.path.basename(to_send)} ({sent_mb:.1f} MB{note}) to {server}/<topic>")
            return True
        except requests.RequestException as e:
            print(f"ntfy: attachment {os.path.basename(to_send)} ({sent_mb:.1f} MB) refused: {e}{_reason(e)}")
            if not use_zip and not to_send.lower().endswith(".pdf"):   # retry once, zipped (a PDF is already compressed)
                import zipfile
                tmp_zip = path + ".zip"
                with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                    z.write(path, arcname=os.path.basename(path))
                try:
                    _put(tmp_zip, headers)
                    print(f"ntfy: sent the zipped report ({os.path.getsize(tmp_zip) / 1024 / 1024:.1f} MB) on the retry")
                    return True
                except requests.RequestException as e2:
                    print(f"ntfy: the zipped retry was refused too: {e2}{_reason(e2)}")
            _text_only("attachment refused by the server")
            print("ntfy: sent the text summary only (BUY / WATCH list)")
            return True
    except requests.RequestException as e:
        print(f"ntfy: failed to send {os.path.basename(path)}: {e}{_reason(e)}")
        return False
    finally:
        if tmp_zip and os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass


def report_to_row(report: dict, report_group: str) -> dict:
    """Flatten one ticker's report into a single CSV row (column -> value).
    Lists of text (news, buyback/guidance headlines, rating actions) keep only
    the LATEST item plus a count. report_group is "up", "down" or "filtered"."""
    cfg = report["cfg"]
    sig = report["signal"]
    row = {
        "ticker": report["ticker"],
        "report": report_group,
        "signal": sig["signal"],
        "current_price": report["current_price"],
    }

    # --- close + extended hours (each % is vs the close)
    ext = report.get("ext") or {}
    ses = ext.get("sessions") or {}
    cur = ext.get("current") or {}
    row.update({
        "close_price": ext.get("close"),
        "close_date": ext["close_date"].isoformat() if ext.get("close_date") else None,
        "market_open": ext.get("market_open"),
    })
    for key, pre in (("ah", "after_hours"), ("on", "overnight"), ("pm", "pre_market")):
        d = ses.get(key) or {}
        row[pre] = d.get("price")
        row[f"{pre}_time_et"] = d.get("time")
        row[f"{pre}_pct"] = d.get("pct")
        row[f"{pre}_source"] = d.get("source")
        if key == "on":
            row["overnight_trades"] = d.get("trades")
    row["current_ext_price"] = cur.get("price")
    row["current_ext_session"] = _SESSION_NAMES.get(cur.get("session")) if cur else None
    row["current_ext_pct"] = cur.get("pct") if cur else None
    pdy = report.get("prev_day") or {}
    row.update({"prev_day_date": pdy.get("date"), "prev_day_open": pdy.get("open"), "prev_day_high": pdy.get("high"),
                "prev_day_low": pdy.get("low"), "prev_day_close": pdy.get("close"),
                "prev_day_prior_close": pdy.get("prior_close")})

    # --- setup signal breakdown
    ctx = sig["context"]
    row.update({
        "setup": sig["setup"], "setup_label": sig["setup_label"], "grade": sig["grade"],
        "turning_up": sig.get("turning_up"), "day_chg_pct": sig.get("day_chg_pct"), "ctx_ema50": ctx.get("ema50"),
        "ctx_state": ctx["state"], "ctx_ext_ema50_atr": ctx["ext_ema50_atr"],
        "ctx_ema50_slope_pct": ctx["ema50_slope_pct"], "ctx_higher_lows": ctx["higher_lows"],
        "ctx_higher_highs": ctx["higher_highs"], "ctx_days_above_ema50": ctx["days_above_ema50"],
        "setup_level": sig["level"], "setup_stop": sig["stop"], "setup_target": sig["target"],
        "setup_rr": sig["rr"], "setup_buy_below": sig.get("buy_below"), "setup_buy_up_to": sig.get("buy_up_to"),
        "setup_watch_kind": sig.get("watch_kind"), "setup_breakout_above": sig.get("breakout_above"), "setup_turn_above": sig.get("turn_above"),
        "setup_arm_level": sig.get("arm_level"), "setup_risk": sig.get("risk"), "setup_trail_ema20": sig.get("trail_now"),
        "setup_not_traded": ", ".join(sig.get("not_traded") or []) or None,
        "setup_buy_when": " | ".join(sig.get("buy_when") or []) or None,
        "setup_distance": sig.get("distance"), "setup_target_src": sig.get("target_src") or None,
        "setup_overhead": ", ".join(str(x) for x in sig.get("overhead", [])) or None, "setup_confirms": sig["n_confirms"] if sig["setup"] else None,
        "setup_fails": "; ".join(sig["fails"]) or None,
        "wait_reason": sig["wait_reason"] or None,
        "other_setups": "; ".join(f"{s['type']}({'BUY' if s['buy'] else 'watch' if s['near'] else 'weak'})" for s in sig["all_setups"]
                                  if s["type"] != sig["setup"]) or None,
    })
    for lbl, ok in sig["required"]:
        row[f"req_{check_key(lbl)}"] = ok
    for lbl, ok in sig["confirms"]:
        row[f"conf_{check_key(lbl)}"] = ok
    for lbl, ok in sig["turn"]:
        row[f"turn_{check_key(lbl)[:12]}"] = ok
    fails = report_filter_checks(report)
    row["filter_checks_failed"] = sum(fails.values())
    for k, v in fails.items():
        row[f"fails_{k}"] = v

    # --- per timeframe (d1_ = daily, h4_ = 4-hour, h1_ = 1-hour)
    ma_lbl = f"{cfg['MA_TYPE']}{cfg['MA_PERIOD']}"
    for tf, pre_tf in (("1D", "d1"), ("4h", "h4"), ("1h", "h1")):
        r, sr = report["results"][tf], report["sr_levels"][tf]
        row.update({
            f"{pre_tf}_close": r["close"], f"{pre_tf}_{ma_lbl}": r["MA"], f"{pre_tf}_trend_up": r["trend_up"],
            f"{pre_tf}_stoch_k": r["%K"], f"{pre_tf}_stoch_d": r["%D"], f"{pre_tf}_k_above_d": r["k_above_d"],
            f"{pre_tf}_stoch_bullish": r["stoch_bullish"], f"{pre_tf}_overbought": r["overbought"],
            f"{pre_tf}_patterns": ";".join(r["patterns_detected"]),
            f"{pre_tf}_support": sr["support"], f"{pre_tf}_resistance": sr["resistance"],
        })

    # --- levels
    tr = report["today_range"]
    atr = report.get("daily_atr")
    row.update({
        "today_low": tr["day_low"], "today_high": tr["day_high"], "today_range": tr["day_range"],
        "daily_atr": atr,
        "atr_used_today_pct": round(tr["day_range"] / atr * 100, 1) if atr else None,
        "atr_projected_low": round(tr["day_high"] - atr, 2) if atr else None,
        "atr_projected_high": round(tr["day_low"] + atr, 2) if atr else None,
        "entry_support_1h": report["entry_support"],
        "stop": report["stop"], "take_profit": report["take_profit"],
        "target_source": ((report.get("day_sr") or {}).get("target") or {}).get("source"),
        "target_r": ((report.get("day_sr") or {}).get("target") or {}).get("r"),
        "target_below_min_r": ((report.get("day_sr") or {}).get("target") or {}).get("below_min_r"),
        "poc": report.get("poc"), "vwap": report.get("vwap"), "above_vwap": report.get("above_vwap"),
    })
    lrc = report.get("lrc") or {}
    row.update({
        "trend_dir": report.get("trend_dir"),
        "trend_reason": report.get("trend_reason"),
        "avwap": (report.get("avwap") or {}).get("value"),
        "avwap_anchor_type": (report.get("avwap") or {}).get("anchor_type"),
        "avwap_anchor_date": (report.get("avwap") or {}).get("anchor_date"),
        "above_avwap": (report.get("avwap") or {}).get("above"),
        "hhhl_window_days": (report.get("hhhl") or {}).get("window"),
        "hhhl_up_days": (report.get("hhhl") or {}).get("up_days"),
        "hhhl_down_days": (report.get("hhhl") or {}).get("down_days"),
        "hhhl_mixed_days": (report.get("hhhl") or {}).get("mixed_days"),
        "hhhl_net": (report.get("hhhl") or {}).get("net"),
        "streak_current_type": ((report.get("hhhl") or {}).get("current_streak") or (None, None))[0],
        "streak_current_days": ((report.get("hhhl") or {}).get("current_streak") or (None, None))[1],
        "streak_previous_type": ((report.get("hhhl") or {}).get("previous_streak") or (None, None))[0],
        "streak_previous_days": ((report.get("hhhl") or {}).get("previous_streak") or (None, None))[1],
        "channel_lower": lrc.get("last_lower"), "channel_mid": lrc.get("last_mid"),
        "channel_upper": lrc.get("last_upper"), "channel_slope_pct_day": lrc.get("slope_pct_per_day"),
        "channel_r2": lrc.get("r2"), "channel_position_pct": lrc.get("position_pct"),
    })

    pns = premarket_near_support(report)
    row["premarket_near_support"] = pns["near"] if pns else None
    row["premarket_near_level"] = pns["level"] if pns else None
    row["premarket_near_atr"] = pns["dist_atr"] if pns else None
    dsr = report.get("day_sr") or {}
    for k in range(2):
        sups = dsr.get("supports") or []
        nr = near_level(report["current_price"], sups[k]["price"] if len(sups) > k else None,
                        report.get("daily_atr"), cfg.get("NEAR_SR_ATR", 0.5))
        row[f"near_s{k + 1}"] = nr["near"] if nr else None
        row[f"near_s{k + 1}_atr"] = nr["dist_atr"] if nr else None
    for pre, key in (("s", "supports"), ("t", "resistances")):
        for k in range(2):
            lvl = (dsr.get(key) or [])[k] if len(dsr.get(key) or []) > k else {}
            row[f"day_{pre}{k + 1}"] = lvl.get("price")
            row[f"day_{pre}{k + 1}_pct"] = lvl.get("dist_pct")
            row[f"day_{pre}{k + 1}_touches"] = lvl.get("touches")
            if pre == "t":
                row[f"day_{pre}{k + 1}_r"] = lvl.get("r")

    # --- fundamentals (raw yfinance values: ratios as fractions, e.g. 0.25 = 25%)
    raw = report.get("fund_raw") or {}
    for label, key, _ in FUNDAMENTAL_FIELDS:
        row[_snake(label)] = raw.get(key)
    lo52, hi52 = raw.get("fiftyTwoWeekLow"), raw.get("fiftyTwoWeekHigh")
    row["pos_in_52w_range_pct"] = (round((report["current_price"] - lo52) / (hi52 - lo52) * 100, 1)
                                  if isinstance(lo52, (int, float)) and isinstance(hi52, (int, float)) and hi52 > lo52
                                  else None)

    # --- analyst
    an = report.get("analyst") or {}
    row.update({
        "analyst_consensus": an.get("recommendation_key"), "analyst_mean_score": an.get("recommendation_mean"),
        "analyst_opinions": an.get("num_analyst_opinions"),
    })
    for label, n in (an.get("recommendation_counts") or {}).items():
        row[f"analyst_{_snake(label)}"] = n
    row.update({
        "eps_current": an.get("eps_current"), "eps_30d_ago": an.get("eps_30d_ago"),
        "eps_improving": an.get("eps_improving"), "eps_num_analysts": an.get("eps_num_analysts"),
    })

    # --- catalysts (latest item only for text lists)
    cat = report.get("catalysts") or {}
    actions = sorted(cat.get("rating_actions") or [], reverse=True)   # strings start with the ISO date
    buy, guid = cat.get("buyback_headlines") or [], cat.get("guidance_headlines") or []
    row.update({
        "next_earnings_date": cat.get("next_earnings_date"), "days_to_earnings": cat.get("days_to_earnings"),
        "in_earnings_window": cat.get("in_earnings_window"),
        "in_sp500": cat.get("in_sp500"), "sp500_added_date": cat.get("sp500_added_date"),
        "upgrades": cat.get("upgrades"), "downgrades": cat.get("downgrades"),
        "latest_rating_action": actions[0] if actions else None,
        "buyback_headlines_count": len(buy), "latest_buyback_headline": buy[0] if buy else None,
        "guidance_headlines_count": len(guid), "latest_guidance_headline": guid[0] if guid else None,
        "reddit_rank": cat.get("social_rank"), "reddit_mentions": cat.get("social_mentions"),
        "reddit_momentum_pct_24h": cat.get("social_momentum_pct"), "reddit_upvotes": cat.get("social_upvotes"),
    })

    # --- news (latest article only)
    news = report.get("news") or []
    row.update({
        "news_count": len(news),
        "latest_news_time": news[0]["time"] if news else None,
        "latest_news_headline": news[0]["headline"] if news else None,
        "latest_news_url": news[0]["url"] if news else None,
        "news_not_fetched_reason": cat.get("news_skipped"),
        "error": None,
    })
    return row


def write_csv_report(rows: list, out_path: str) -> None:
    """One row per ticker. utf-8-sig so Excel shows non-ASCII headlines correctly."""
    if not rows:
        return
    full = next((r for r in rows if r.get("error") is None), rows[0])
    cols = list(full.keys())
    cols += [c for r in rows for c in r if c not in cols]   # any extra columns (e.g. error-only rows)
    pd.DataFrame(rows, columns=list(dict.fromkeys(cols))).to_csv(out_path, index=False, encoding="utf-8-sig")



def main(tickers, fileapp):
    """Screen a list of tickers and write two HTML reports:
      <fileapp>_signal_report_<timestamp>_up.html   - any traded setup, or no setup but a rising 1h channel
      <fileapp>_signal_report_<timestamp>_down.html - no traded setup and a falling or weak 1h channel
    A ticker is left out entirely only if it fails ALL four checks in
    report_filter_checks() - and only when there are at least REPORT_FILTER_MIN_TICKERS
    (default 20) tickers; a shorter list gets a card for every ticker.
    Every full run remembers its _up report (tickers, signals, file names) so refresh(fileapp) can update it later.
    Call as main(TICKERS, "name")."""
    return _run(list(tickers), fileapp, state=None)


def refresh_blocked_reason(fileapp):
    """None if refresh(fileapp) can run; otherwise why a FULL scan is needed instead. A refresh only updates a report
    whose full scan ran TODAY (local date): the ticker list must be rebuilt by a full scan at least once a day."""
    state = _load_watchlist(fileapp)
    if not state or not state.get("tickers"):
        return "no saved report from a full scan"
    day = state.get("full_run_date")
    if day != dt.date.today().isoformat():
        return f"the last full scan is not from today ({day or 'unknown date'})"
    return None


def _is_setup_row(r: dict) -> bool:
    """'Setup' tickers for refresh(scope="setups"): BUY / WATCH, a forming setup (WAIT with a setup) or turning up."""
    return (r.get("signal") in ("BUY", "WATCH") or (r.get("signal") == "WAIT" and r.get("setup") is not None)
            or bool(r.get("turning_up")))


def refresh(fileapp, scope="all"):
    """Quick refresh of the LAST full run's _up report for `fileapp`, written over the SAME files
    (<...>_up.html, <...>_up_lite.html; the matching rows of the CSV are updated in place).
    Only the tickers that were in that _up report are re-screened - with today's cache this takes seconds, not minutes.
    The _down report is not touched, and tickers that were not in _up are not checked: run main() for that.
    Signal changes since the previous run (full or refresh) are listed at the top of the report; ntfy is sent
    according to NTFY_ON_REFRESH ("changes" = only when a signal changed, "always", "never").
    scope="setups": re-screen only the setup tickers (BUY / WATCH / forming / turning up) - about twice as fast; the
    other cards are taken over unchanged from the previous run (the report says how old each part is). A ticker that
    newly forms a setup is only seen by scope="all" (or a full scan)."""
    why = refresh_blocked_reason(fileapp)
    if why:
        print(f"refresh({fileapp!r}): not possible - {why} ({_watchlist_path(fileapp)}). Run main() (a full scan) first.")
        return None
    state = _load_watchlist(fileapp)
    tickers = list(state["tickers"])
    if scope == "setups":
        saved = cache_store.read(_cache_path("cards", fileapp), {}) or {}
        if not saved:
            print(f"refresh({fileapp!r}, setups): no saved cards from today's runs - refreshing ALL tickers instead")
            scope = "all"
        else:
            tickers = [t for t in tickers if t not in saved or _is_setup_row(saved[t]["row"])]
            if not tickers:
                print(f"refresh({fileapp!r}, setups): no setup tickers in this report - nothing to refresh")
                return None
    print(f"refresh({fileapp!r}, {scope}): {len(tickers)} of {len(state['tickers'])} tickers, previous run "
          f"{state.get('last_run_at')} -> {state['paths'].get('up')}")
    return _run(tickers, fileapp, state=state, scope=scope)


# ---------------------------------------------------------------- refresh state ("watchlist")
# One small JSON per fileapp in the shared cache folder. It has no date in its name, so the daily cache clean-up
# keeps it: a refresh on Monday morning still finds Friday's report.

def _watchlist_path(fileapp: str) -> str:
    _cache_sync()
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(fileapp))
    return os.path.join(cache_store.cache_dir(), f"watchlist_{safe}.json")


def _load_watchlist(fileapp: str):
    try:
        with open(_watchlist_path(fileapp), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save_watchlist(fileapp: str, state: dict) -> None:
    path = _watchlist_path(fileapp)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        print(f"WARNING: could not save the refresh list ({type(e).__name__}: {e}) - refresh({fileapp!r}) will not work")


def _update_csv_rows(csv_path: str, new_rows: list) -> None:
    """Replace the refreshed tickers' rows in an existing CSV (other rows and the row order are kept);
    write a new CSV if there is none (e.g. a fresh GitHub Actions runner)."""
    if not os.path.exists(csv_path):
        write_csv_report(new_rows, csv_path)
        return
    old = pd.read_csv(csv_path, encoding="utf-8-sig")
    fresh = {r["ticker"]: r for r in new_rows}
    rows = [fresh.pop(r["ticker"], r) for r in old.to_dict(orient="records")]    # same order, refreshed rows swapped in
    rows += list(fresh.values())                                                # tickers the old CSV did not have
    cols = list(dict.fromkeys(list(old.columns) + [c for r in new_rows for c in r]))
    pd.DataFrame(rows, columns=cols).to_csv(csv_path, index=False, encoding="utf-8-sig")


def _hhmm(txt) -> str:
    """'Sat 03 Oct 21:30' -> '21:30' (the saved run times are written as '%a %d %b %H:%M')."""
    t = str(txt or "")
    return t[-5:] if len(t) >= 5 and t[-3] == ":" else t


def _run_info(state, csv_rows: list, times: dict) -> str:
    """One small line under the 'Setups at a glance' header: when each part of this report was made (times in green)
    and, after a refresh, which signals changed since the previous run."""
    g = lambda k: f'<span class="t">{html.escape(_hhmm(times.get(k)))}</span>'
    if state is None:                                             # full scan
        return f'<div class="runinfo">Full scan {g("full")}</div>'
    changes = [(r["ticker"], state["signals"].get(r["ticker"]), r.get("signal")) for r in csv_rows
               if r.get("signal") and state.get("signals", {}).get(r["ticker"]) not in (None, r.get("signal"))]
    since = html.escape(_hhmm(state.get("last_run_at")))
    ch = (f"Signal changes since {since}: " + ", ".join(
        f'<a class="tk" href="#{_card_id(t)}">{html.escape(t)}</a> {html.escape(a)} &rarr; '
        f'<span class="badge {"pass" if b == "BUY" else "watch" if b == "WATCH" else "wait"}">{html.escape(b)}</span>'
        for t, a, b in changes)) if changes else f"no signal changes since {since}"
    when = (f'Setups refreshed {g("setups")} &middot; others {g("all")}' if times.get("setups") != times.get("all")
            else f'Refreshed {g("all")}')
    return (f'<div class="runinfo">{when} &middot; full scan {g("full")} &middot; {len(csv_rows)} tickers &middot; {ch}'
            '<span class="muted-small"> &middot; the _down report and tickers outside this report are only updated '
            'by a full scan</span></div>')


def _news_for_refresh(tk: str, fetch_days: int, cfg: dict):
    """News in a refresh: reuse what is cached if younger than NEWS_CACHE_MIN_REFRESH minutes; otherwise fetch -
    but never WAIT for Finnhub's per-minute limit (93 tickers > 55 calls/minute would stall a refresh for a minute).
    No free call right now -> keep the older news (or none if there never was any)."""
    path = _cache_path("news", f"{tk}_{fetch_days}")
    got = cache_store.read(path)
    old = got[1] if isinstance(got, tuple) and len(got) == 2 else None
    if old is not None and time.time() - got[0] < cfg.get("NEWS_CACHE_MIN_REFRESH", 360) * 60:
        _prof_add("  news reused (refresh)", tk, 0.0)
        return old
    val = fetch_company_news(tk, api_key=cfg.get("FINNHUB_API_KEY"), days=fetch_days, max_items=None,
                             tz=cfg.get("NEWS_TZ", "America/New_York"), wait=False)
    if val is _NEWS_BUSY:
        _prof_add("  news kept (Finnhub limit, no waiting in a refresh)", tk, 0.0)
        return old
    if val is not None:
        _cache_write(path, (time.time(), val))
        return val
    return old


def _merge_hourly(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """The saved history up to where the new download starts, then the new bars (they replace any overlap,
    including the hour that was still in progress when the full scan ran)."""
    if new is None or new.empty:
        return old
    if old.index.tz is not None and new.index.tz is not None and str(old.index.tz) != str(new.index.tz):
        new = new.tz_convert(old.index.tz)
    cut = new.index.min()
    return pd.concat([old[old.index < cut], new[[c for c in old.columns if c in new.columns]]]).sort_index()


def _refresh_hourly(fileapp: str, tickers: list) -> dict:
    """1h bars for a refresh: the bars the same day's full scan saved + only the last REFRESH_PERIOD downloaded.
    Tickers without saved bars are downloaded in full; a ticker whose short download fails is left out, so
    run_screen downloads it on its own (exactly as in a full scan)."""
    period, full = CONFIGH.get("REFRESH_PERIOD", "5d"), CONFIGH["PERIOD"]
    old = cache_store.read(_cache_path("hourly", fileapp), {}) or {}
    have = [t for t in tickers if t in old]
    miss = [t for t in tickers if t not in old]
    out = {}
    if have:
        new = prefetch_hourly_data(have, period, CONFIGH["INTERVAL"])
        out = {t: _merge_hourly(old[t], new[t]) for t in have if t in new}
    if miss:
        out.update(prefetch_hourly_data(miss, full, CONFIGH["INTERVAL"]))
    print(f"1h bars: {len(have)} ticker(s) = saved history + last {period}"
          + (f", {len(miss)} downloaded in full ({full})" if miss else ""))
    return out


def _report_server_token():
    """The secret shared with report_server.py (file .report_server_token next to the scripts; created if missing)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".report_server_token")
    try:
        with open(path, encoding="utf-8") as f:
            tok = f.read().strip()
        if tok:
            return tok
    except OSError:
        pass
    import secrets
    tok = secrets.token_urlsafe(18)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(tok)
        os.chmod(path, 0o600)
    except OSError:
        return None
    return tok


def _run_bar(report_file: str) -> str:
    """Buttons that ask the local report_server.py to run `run_trend_entry.py refresh|full`. Hidden unless that server
    answers on 127.0.0.1 - so they never show on the phone, on GitHub, or when the server is not running.
    Not written at all when the report is built on GitHub Actions or REPORT_SERVER is off."""
    if not CONFIGH.get("REPORT_SERVER", True) or os.environ.get("GITHUB_ACTIONS"):
        return ""
    tok = _report_server_token()
    if not tok:
        return ""
    base = f"http://127.0.0.1:{int(CONFIGH.get('REPORT_SERVER_PORT', 8765))}"
    from urllib.parse import urlencode

    def link(mode):
        return html.escape(f"{base}/run?" + urlencode({"mode": mode, "token": tok, "back": report_file}), quote=True)

    tip_s = ("Fast: update only the setup tickers (WATCH, forming, turning up); the other cards stay as they are. "
             "New setups are only found by Refresh all")
    tip_r = ("Update every ticker in this report with fresh data (a full scan of this list instead if its last full scan "
             "is not from today)")
    tip_f = "New Finviz list and a full scan of every list (Small, Med, Large) - a few minutes"
    return (f'<span class="runbar" id="runbar">'
            f'<a class="runbtn" href="{link("refresh-setups")}" title="{tip_s}">&#x21bb; Refresh setups</a>'
            f'<a class="runbtn" href="{link("refresh")}" title="{tip_r}">&#x21bb; Refresh all</a>'
            f'<a class="runbtn" href="{link("full")}" title="{tip_f}" onclick="return confirm(\'Full scan: new Finviz '
            f'list and every ticker - this takes a few minutes. Start?\')">Full scan</a></span>'
            f"<script>(function(){{var i=new Image();i.onload=function(){{document.getElementById('runbar')"
            f".style.display='inline-flex';}};i.src='{base}/ping.gif?'+Date.now();}})();</script>")


def _run(tickers, fileapp, state=None, scope="all"):
    """The full run (state=None) and the quick refresh (state = the saved _up report) share this code."""
    is_refresh = state is not None
    sections = {"up": [], "down": []}
    skipped = []  # (ticker, reason) - listed at the bottom of both reports
    # the 4-check filter only makes sense for a long list; with a short list every ticker gets a card.
    # A refresh never filters: every ticker of the report keeps its card, so you can see one weaken.
    filter_on = (not is_refresh) and len(tickers) >= CONFIGH.get("REPORT_FILTER_MIN_TICKERS", 20)
    CONFIGH["_refresh_mode"] = is_refresh            # news: no waiting on Finnhub's limit in a refresh
    CONFIGH["_filter_active"] = filter_on
    if not filter_on and not is_refresh:
        print(f"{len(tickers)} tickers (fewer than {CONFIGH.get('REPORT_FILTER_MIN_TICKERS', 20)}): "
              "the report filter is OFF - every ticker gets a card.")
    PROFILE.clear()
    run_t0 = time.perf_counter()
    prune_cache()
    sp500_members = daily_cached("sp500", "all", lambda: _get_sp500_membership(verbose=False))
    social_table = fetch_apewisdom_table()   # once per run (mentions change intraday, so not cached)

    try:
        if is_refresh and CONFIGH.get("REFRESH_PERIOD"):
            prefetched = _refresh_hourly(fileapp, tickers)
        else:
            prefetched = prefetch_hourly_data(tickers, CONFIGH["PERIOD"], CONFIGH["INTERVAL"])
    except Exception as e:
        print(f"batch download failed ({type(e).__name__}: {e}) - falling back to one download per ticker")
        prefetched = {}
    # after-hours + overnight + pre-market for all tickers: ~5 batched Alpaca requests
    ext_batch = prefetch_extended_hours(tickers, CONFIGH, prefetched)

    def screen_one(ticker):
        """Runs in a worker thread. Returns (status, direction_or_reason, html, csv_row)."""
        try:
            t0 = time.perf_counter()
            report = run_screen(ticker, CONFIGH, sp500_members=sp500_members,
                                hourly=prefetched.get(ticker), social_table=social_table,
                                ext_batch=ext_batch if ext_batch is not None else {})
            elapsed = time.perf_counter() - t0
            _prof_add("whole ticker (run_screen)", ticker, elapsed)
            fetched = sum(sec for step, calls in list(PROFILE.items())
                          if not step.startswith("  ") and step != "whole ticker (run_screen)"
                          for t, sec in list(calls) if t == ticker)
            _prof_add("indicators & signal (CPU, no network)", ticker, max(elapsed - fetched, 0.0))
            #print_report(report)
            fails = report_filter_checks(report)
            if filter_on and all(fails.values()):
                print(f"=== {ticker}: filtered out (fails all 4 checks) ===")
                return ("skip", "filtered out - fails all 4 checks (no setup, downtrend/range, "
                                "WAIT, EPS deteriorating)", None, report_to_row(report, "filtered"))
            direction = "up" if is_refresh else report["trend_dir"]     # a refresh rewrites the _up report only
            row_ = report_to_row(report, direction)
            mobile_ = None
            if report["signal"]["signal"] in ("BUY", "WATCH"):
                try:
                    mobile_ = render_mobile_card(report, row_)
                except Exception as e_:                       # never let the phone card break the run
                    print(f"  {ticker}: mobile card failed ({type(e_).__name__}: {e_})")
            return ("ok", direction, render_ticker_html(report), row_, mobile_)
        except Exception as e:
            print(f"\n=== {ticker}: skipped due to error ===")
            print(f"  {type(e).__name__}: {e}")
            return ("skip", f"error - {type(e).__name__}: {e}", None,
                    {"ticker": ticker, "report": "error", "error": f"{type(e).__name__}: {e}"})

    # executor.map keeps results in the same order as `tickers`
    with ThreadPoolExecutor(max_workers=max(1, int(CONFIGH.get("MAX_WORKERS", 4)))) as ex:
        results = list(ex.map(screen_one, tickers))

    n_screened = len(tickers)
    if is_refresh and scope == "setups":
        # partial refresh: the other tickers keep the cards / rows of the previous run, in the report's order
        saved = cache_store.read(_cache_path("cards", fileapp), {}) or {}
        fresh = dict(zip(tickers, results))
        tickers = [t for t in state["tickers"] if t in fresh or t in saved]
        results = [fresh[t] if t in fresh else ("ok", "up", saved[t]["html"], dict(saved[t]["row"]), saved[t].get("mobile"))
                   for t in tickers]
    csv_rows = [res[3] for res in results if res[3] is not None]
    for r in csv_rows:
        r.pop("prev_signal", None)                    # a saved row may still carry the marker of an older refresh
    lite_cards = []                                   # cards of BUY / WATCH tickers in the up report (for the small ntfy file)
    for ticker, res in zip(tickers, results):
        if res[0] == "ok":
            sections[res[1]].append(res[2])
            if res[1] == "up" and (res[3] or {}).get("signal") in ("BUY", "WATCH"):
                lite_cards.append((len(res) > 4 and res[4]) or res[2])      # phone card (desktop card as a fallback)
        else:
            skipped.append((ticker, res[1]))

    now_txt = datetime.now().strftime("%a %d %b %H:%M")
    if not is_refresh:
        times = {"full": now_txt, "all": now_txt, "setups": now_txt}
    elif scope == "setups":
        times = {"full": state.get("full_run_at"), "all": state.get("all_at") or state.get("last_run_at"), "setups": now_txt}
    else:
        times = {"full": state.get("full_run_at"), "all": now_txt, "setups": now_txt}
    if is_refresh:
        timestamp = state["timestamp"]                # same report, same name
        out_paths = dict(state["paths"])
        prev = state.get("signals") or {}
        for r in csv_rows:                            # "was WATCH" markers in the setups table
            if r.get("signal") and prev.get(r["ticker"]) not in (None, r.get("signal")):
                r["prev_signal"] = prev[r["ticker"]]
    else:
        timestamp = constants.get_dayprefix() + "_" + constants.get_timeprefix()
        out_paths = {d: os.path.join("reports", f"{fileapp}_signal_report_{timestamp}_{d}.html") for d in ("up", "down")}
        out_paths["up_lite"] = out_paths["up"].replace("_up.html", "_up_lite.html")
        out_paths["csv"] = os.path.join("reports", f"{fileapp}_signal_report_{timestamp}.csv")
    down_name = os.path.basename(out_paths.get("down") or "")

    for direction in (("up",) if is_refresh else ("up", "down")):
        out_path = out_paths[direction]
        report_title = f"{fileapp}_Signal Report {timestamp}_{direction}"
        body = "\n".join(sections[direction]) or (
            f'<section class="card"><p class="note">No tickers in the {direction} report.</p></section>')
        if direction == "up":
            body = (_sector_filter_bar(csv_rows)
                    + _setup_summary_card(csv_rows, down_name, header_right=_run_bar(os.path.basename(out_path)),
                                        info=_run_info(state, csv_rows, times))
                    + _turning_up_card(csv_rows) + body)
        else:
            body = _sector_filter_bar([r for r in csv_rows if r.get("report") == direction]) + body
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(build_html_report(report_title, body + _skipped_card(skipped)))
        print(f"HTML report ({direction}, {len(sections[direction])} tickers) "
              f"{'refreshed' if is_refresh else 'written'}: {out_path}")
        if direction == "up":                         # the lite (phone) version is always written; NTFY_UP_VERSION picks what is sent
            lite_path = out_paths.setdefault("up_lite", out_paths["up"].replace("_up.html", "_up_lite.html"))
            lite_body = mobile_summary(csv_rows) + (
                "\n".join(lite_cards) if lite_cards else "")
            with open(lite_path, "w", encoding="utf-8") as f:
                f.write(build_mobile_report(f"{fileapp} {timestamp} - BUY / WATCH"
                                            + (f" (refreshed {now_txt})" if is_refresh else ""), lite_body))
            print(f"Lite report ({len(lite_cards)} BUY/WATCH cards) {'refreshed' if is_refresh else 'written'}: {lite_path}")

    if CONFIGH.get("CSV_REPORT", True) and out_paths.get("csv"):
        csv_path = out_paths["csv"]
        try:
            if is_refresh:
                _update_csv_rows(csv_path, csv_rows)
                print(f"CSV report: {len(csv_rows)} rows updated in {csv_path}")
            else:
                write_csv_report(csv_rows, csv_path)
                print(f"CSV report ({len(csv_rows)} rows) written to {csv_path}")
        except Exception as e:
            print(f"CSV report failed: {type(e).__name__}: {e}")

    # remember this _up report so refresh(fileapp) can update it (the signals let the next refresh show what changed)
    up_rows = [r for r in csv_rows if r.get("report") == "up"]
    if not is_refresh and CONFIGH.get("REFRESH_PERIOD"):
        # keep the _up tickers' 1h bars, so a refresh later today only downloads the last few days
        _cache_write(_cache_path("hourly", fileapp), {r["ticker"]: prefetched[r["ticker"]]
                                                     for r in up_rows if r["ticker"] in prefetched})
    # every _up card of this run, so refresh(scope="setups") can rebuild the page from new + saved cards
    _cache_write(_cache_path("cards", fileapp),
                 {t: {"html": res[2], "row": res[3], "mobile": res[4] if len(res) > 4 else None}
                  for t, res in zip(tickers, results) if res[0] == "ok" and res[1] == "up"})
    changed = []
    if is_refresh:
        changed = [f'{r["ticker"]} {r["prev_signal"]}->{r["signal"]}' for r in csv_rows if r.get("prev_signal")]
    _save_watchlist(fileapp, {
        "fileapp": fileapp, "timestamp": timestamp, "paths": out_paths,
        "tickers": [r["ticker"] for r in up_rows] if not is_refresh else list(tickers),
        "signals": {r["ticker"]: r.get("signal") for r in up_rows},
        "full_run_at": state.get("full_run_at") if is_refresh else now_txt,
        "full_run_date": state.get("full_run_date") if is_refresh else dt.date.today().isoformat(),
        "last_run_at": now_txt, "all_at": times["all"], "setups_at": times["setups"],
    })

    notify = CONFIGH.get("NTFY_ENABLED", True)
    if notify and is_refresh:
        mode = str(CONFIGH.get("NTFY_ON_REFRESH", "changes")).lower()
        notify = mode == "always" or (mode == "changes" and bool(changed))
        if not notify:
            print(f"ntfy: not sent (NTFY_ON_REFRESH={mode!r}, {len(changed)} signal change(s))")
    if notify:
        buys = {g: [f'{r["ticker"]}({r.get("setup_label")})' for r in csv_rows if r.get("report") == g and r.get("signal") == "BUY"]
                for g in ("up", "down")}
        counts = {g: sum(1 for r in csv_rows if r.get("report") == g) for g in ("up", "down")}
        for kind in CONFIGH.get("NTFY_FILES", ["up"]):
            if is_refresh and kind != "up":
                continue                                  # only the _up report is refreshed
            path = out_paths.get(kind)
            if (kind == "up" and str(CONFIGH["NTFY_UP_VERSION"]).lower() != "full"
                    and out_paths.get("up_lite") and os.path.exists(out_paths["up_lite"])):
                path = out_paths["up_lite"]               # the phone gets the BUY / WATCH-only version
            if not path or not os.path.exists(path):
                continue
            if kind in buys:
                watch = [f'{r["ticker"]}<={r.get("setup_buy_below")}' if r.get("setup_buy_below") else r["ticker"]
                         for r in csv_rows if r.get("report") == kind and r.get("signal") == "WATCH"]
                msg = (f"{counts[kind]} tickers, BUY_SIGNAL: {', '.join(buys[kind]) or 'none'}"
                       f" | WATCH ({len(watch)}): {', '.join(watch[:12])}{' ...' if len(watch) > 12 else ''}")
                if changed:
                    msg = "CHANGED: " + ", ".join(changed) + " | " + msg
                tags = "chart_with_upwards_trend" if kind == "up" else "chart_with_downwards_trend"
                prio = "high" if (kind == "up" and (buys["up"] or any(c.endswith("->BUY") for c in changed))) else "default"
            else:
                msg, tags, prio = f"{len(csv_rows)} rows", "page_facing_up", "default"
            send_ntfy_file(path, title=f"{fileapp} {kind} report {timestamp}" + (f" - refreshed {now_txt}" if is_refresh else ""),
                           message=msg, tags=tags, priority=prio)

    if CONFIGH.get("PROFILE", True):
        print_profile_summary(time.perf_counter() - run_t0, n_screened)
    return out_paths
