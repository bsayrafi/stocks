
import pandas as pd
import yfinance as yf

from constants import CONFIG, DAYm1_IDX, DAYm2_IDX, DAYm3_IDX, DAY_IDXS, DAY_LABELS, DAY_ORDER
from indicators import *
from external_data import *
from strategy import *
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from boxfilters import run_pipeline
from intraday_strategy import * 
from pre_market import prefetch_pre_market, get_best_pre_market_price
from profiling import profile_step
import disk_cache
from cached_ticker import CachedTicker


# 1. Create the session
yf_session = Session()

# --- THE STEALTH FIX: Mimic a real web browser ---
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"
})

# 2. Configure the retry rules (Notice I added 403 to the blocklist!)
retry = Retry(
    total=2,              # was 3 -- long retry chains stalled single tickers for 13-17s
    backoff_factor=0.5,   # was 2 -- short pauses between retries
    status_forcelist=[403, 429, 500, 502, 503, 504] 
)

adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)
yf_session.mount('https://', adapter)
yf_session.mount('http://', adapter)



# Sentinel: pre-market prices are being fetched in the background and will be
# filled into the rows after the worker pool finishes (see evaluate_tickers_parallel).
PREMARKET_DEFERRED = object()
_D1_CLOSE_KEY = "__d1_close"   # temporary row key used to compute Pre-Market%; removed before output
# Temporary row key marking where the intraday columns go when a ticker's
# intraday analysis is deferred until the background download finishes.
# Holds the ticker's sector; replaced by the intraday columns before output.
_INTRADAY_KEY = "__intraday_deferred"


def _premarket_columns(pinfo, d1_close):
    """Build the pre-market columns from a pre_market.py details dict.
    Returns (columns_dict, latest_price_or_None)."""
    cols = {"Pre-Market": "N/A", "Pre-Market%": "N/A", "Pre-Mkt Src": "N/A",
            "Pre-Mkt Time": "N/A", "Pre-Mkt Reason": "N/A"}
    latest_price = None
    if pinfo:
        cols["Pre-Mkt Reason"] = pinfo.get("reason") or "N/A"   # why that feed was chosen
        if pinfo.get("price") is not None:
            latest_price = float(pinfo["price"])
            cols["Pre-Market"] = f"{latest_price:.2f} "
            if d1_close:
                cols["Pre-Market%"] = f"{(latest_price - d1_close) / d1_close * 100:.1f} "
            cols["Pre-Mkt Src"] = (pinfo.get("source") or "N/A").upper()  # SIP (~16 min delayed) or IEX (live)
            cols["Pre-Mkt Time"] = pinfo.get("time") or "N/A"             # HH:MM ET of the chosen 1-min bar
    return cols, latest_price


def evaluate_tickers(sp500_tickers, data, verbose_errors=True, finbert_pipeline=None, extra_data_store=None,tv_exchange_map=None,
                     premarket_prices=None, intraday_ready=None):
    """
    Evaluate each ticker across the last several completed sessions.
    Computes SMA, RSI, StochRSI, MACD, ADX, Bollinger Bands, OBV, volume,
    candle body/wick quality, 52-week high/low, and (gated by the ENABLE_*
    flags in the constants cell) analyst ratings, TradingView rating, Finviz
    target price, short interest, EPS revisions/trend/estimate, next earnings
    date, Altman Z-score, and FinBERT news sentiment.

    Heavy/DataFrame-shaped data (recommendations_summary, upgrades_downgrades,
    financials, balance_sheet, cashflow, option chains) is stored per-symbol
    in `extra_data_store` rather than flattened into the result row.

    Pass a prebuilt `finbert_pipeline` (see build_finbert_pipeline()) if
    ENABLE_NEWS_SENTIMENT=1, so the model is loaded once, not per ticker.
    """
    results2 = []
    spy_df = None   # fetched on first use (from the prefetched bars when available)
    sector_cache: dict[str, pd.DataFrame] = {}

    for symbol in sp500_tickers:
        try:
            df = data.xs(symbol, level=1, axis=1).dropna()
            if len(df) < 200:
                continue



            with profile_step("daily indicators + S/R", symbol):
                week52_high = df['Close'].max()
                week52_low = df['Close'].min()

                df['SMA9'], df['SMA50'] = calculate_sma(df['Close'])
                df['RSI'] = calculate_rsi(df['Close'])
                df['StochK'], df['StochD'] = calculate_stoch_rsi(df['Close'])
                df['MACD'], df['MACDSignal'], df['MACDHist'] = calculate_macd(df['Close'])
                df['ADX'] = calculate_adx(df)
                df['BBUpper'], df['BBMid'], df['BBLower'] = calculate_bollinger_bands(df['Close'])
                df['OBV'] = calculate_obv(df['Close'], df['Volume'])

                # --- NEW: ATR, Stop Loss, and Take Profit Calculations ---
                # 1. Calculate 14-period Daily ATR
                high_low = df['High'] - df['Low']
                high_close = (df['High'] - df['Close'].shift()).abs()
                low_close = (df['Low'] - df['Close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['ATR'] = tr.rolling(14).mean()
                daily_atr_val = float(df['ATR'].iloc[DAYm1_IDX])
            
                days = {i: df.iloc[i] for i in DAY_IDXS}


                # 2. Get current price and nearest support
                current_price = days[DAYm1_IDX]['Close']


                # --- NEW UPDATED BLOCK ---
                sr = calculate_support_resistance(df)
                res_levels = sr["resistance"] + [None] * (3 - len(sr["resistance"]))
                sup_levels = sr["support"] + [None] * (3 - len(sr["support"]))

                sr_block = {
                    "Support1": sup_levels[0],
                    "Support2": sup_levels[1],
                    "Support3": sup_levels[2],
                    "Resistance1": res_levels[0],
                    "Resistance2": res_levels[1],
                    "Resistance3": res_levels[2],
                }

                # --- NEW: Structural Stop and 1.5x Take Profit ---
                # Use Support1 as the baseline. If no support is found, fallback to current price.
                entry_support = sup_levels[0] if sup_levels[0] is not None else current_price
            
                structural_stop = round(entry_support - (0.5 * daily_atr_val), 2)
                risk_per_share = current_price - structural_stop
                take_profit_target = round(current_price + (1.5 * risk_per_share), 2)

            vol_block = {f"D{DAY_LABELS[i]}Vol ": f"{days[i]['Volume']:.0f} " for i in DAY_ORDER}
            # 2. Calculate the average volume across the selected days
            avg_vol = sum(days[i]['Volume'] for i in DAY_ORDER) / len(DAY_ORDER)

            # 3. Add the average to the dictionary
            vol_block["DAvgVol "] = f"{avg_vol:.0f} "

            risingVol = (days[DAYm1_IDX]['Volume'] > days[DAYm2_IDX]['Volume']) and (days[DAYm2_IDX]['Volume'] > days[DAYm3_IDX]['Volume'])
            d1_close = float(days[DAYm1_IDX]['Close'])
            pm_cols, latest_price = _premarket_columns(None, d1_close)   # all "N/A" placeholders

            if CONFIG["SHOW_PREMARKET_PRICE"] == 1 and premarket_prices is not PREMARKET_DEFERRED:
                try:
                    if premarket_prices is not None:
                        pinfo = premarket_prices.get(symbol)                      # batched lookup
                    else:
                        pinfo = get_best_pre_market_price(                        # single-ticker fallback
                            symbol, CONFIG.get("ALPACA_HEADERS"), with_details=True)
                    pm_cols, latest_price = _premarket_columns(pinfo, d1_close)
                except Exception:
                    latest_price = None


            passes_gap_filter = (
               CONFIG["FILTER_PREMARKET_GAP_UP"] == 0
               or (latest_price is not None )
            )

            with profile_step("daily signals", symbol):
                gflags = {i: (1 if days[i]['Close'] > days[i]['Open'] else 0) for i in DAY_IDXS}
                #cflags = {i: is_nice_green_candle(days[i]) for i in DAY_IDXS}

                rsig = 1 if (days[DAYm3_IDX]['RSI'] < days[DAYm2_IDX]['RSI'] and days[DAYm2_IDX]['RSI'] < days[DAYm1_IDX]['RSI']) else 0

                stoch_buy_signal = is_stoch_rsi_buy_signal(df)
                golden_cross = is_golden_cross(df)
                macd_buy_signal = is_macd_buy_signal(df)
                trending = is_trending(df)
                bollinger_bounce = is_bollinger_bounce(df)
                obv_rising = is_obv_rising(df)
                volume_spike = is_volume_spike(df)
                near_low_bounce = is_near_52week_low_bounce(days[DAYm1_IDX]['Close'], week52_low)
                pct_from_high = (week52_high - days[DAYm1_IDX]['Close']) / (week52_high - week52_low)
                pct_from_low = (days[DAYm1_IDX]['Close'] - week52_low) / (week52_high - week52_low)

                cmf_buy = is_cmf_buy_signal(df)
                bull_div = is_bullish_divergence(df)
                trend_pullback = is_trend_pullback_buy(df)
                bear_div = is_bearish_divergence(df)
                death_cross = is_death_cross(df)
                exhaustion_sell = is_exhaustion_sell(df)

                sig = compute_signals(df, days, DAYm1_IDX, week52_low)
                sig["BuyScore"] = buy_score(sig)
                sig["SellScore"] = sell_score(sig)
                sig["HighQualityBuy"] = is_high_quality_buy(sig)

            # --- Build repeating per-day field blocks ---
            oc_block = {}
            for i in DAY_ORDER:
                label = DAY_LABELS[i]
                oc_block[f"D{label}Open "] = f"{days[i]['Open']:.2f} "
                oc_block[f"D{label}Close "] = f"{days[i]['Close']:.2f} "

            g_block = {f"G{DAY_LABELS[i]} ": f"{gflags[i]:.0f} " for i in DAY_ORDER}
            #c_block = {f"C{DAY_LABELS[i]}": cflags[i] for i in DAY_ORDER}
            rsi_block = {f"RSI{DAY_LABELS[i]} ": f"{round(float(days[i]['RSI']), 1)} " for i in DAY_ORDER}

            # --- Shared Ticker object for all yfinance-based extras below ---
            needs_yf_ticker = any([
                CONFIG["ENABLE_ANALYST_DATA"], CONFIG["ENABLE_SHORT_INTEREST"],
                CONFIG["ENABLE_EPS_DATA"], CONFIG["ENABLE_EARNINGS_DATES"],
                CONFIG["ENABLE_NEWS_SENTIMENT"], CONFIG["ENABLE_RAW_STATEMENTS"],
                CONFIG["ENABLE_ALTMAN_ZSCORE"], CONFIG["ENABLE_CASH_METRICS"],
                CONFIG["ENABLE_COMPANY_INFO"],
                CONFIG["ENABLE_VALUATION"],


            ])
            
            with profile_step("company info (first .info fetch)", symbol, enabled=CONFIG["ENABLE_COMPANY_INFO"] == 1):
                # CachedTicker = yf.Ticker with same-day disk caching of .info and analyst
                # data (skipped near earnings) -- see cached_ticker.py
                yf_ticker = CachedTicker(symbol, session=yf_session) if needs_yf_ticker else None

                company_block = {}
                if CONFIG["ENABLE_COMPANY_INFO"] == 1:
                    company_info = get_company_info(symbol, yf_ticker)
                    company_block["Name"] = company_info["Name"]
                    company_block["Sector"] = company_info["Sector"]
                    company_block["Industry"] = company_info["Industry"]
                    company_block["MarketCap"] = company_info["MarketCap"]

            # --- Analyst ratings / price targets ---
            with profile_step("analyst data", symbol, enabled=CONFIG["ENABLE_ANALYST_DATA"] == 1):
                analyst_block = {}
                if CONFIG["ENABLE_ANALYST_DATA"] == 1:
                
                    with profile_step("  analyst: rec mean", symbol):
                        rec_mean = get_recommendation_mean(yf_ticker)
                    with profile_step("  analyst: price targets", symbol):
                        targets = get_analyst_price_targets(yf_ticker)
                    analyst_block["RecMean"] = rec_mean
                    analyst_block["TargetMean"] = targets.get("mean") if isinstance(targets, dict) else None
                    analyst_block["TargetHigh"] = targets.get("high") if isinstance(targets, dict) else None
                    analyst_block["TargetLow"] = targets.get("low") if isinstance(targets, dict) else None
                
                    with profile_step("  analyst: rec breakdown", symbol):
                        rec_breakdown = get_yahoo_recommendation_breakdown(symbol, yf_ticker)
                    analyst_block.update(rec_breakdown)

                    # NEW: derived smart-signal block, reuses same yf_ticker
                    with profile_step("  analyst: smart signals", symbol):
                        smart_signals = extract_smart_analyst_signals(
                            yf_ticker, current_price=days[DAYm1_IDX]['Close']
                        )
                    analyst_block.update(smart_signals)


            # --- Finviz target price / short float ---
            with profile_step("finviz", symbol, enabled=CONFIG["ENABLE_FINVIZ"] == 1):
                finviz_block = {}
                if CONFIG["ENABLE_FINVIZ"] == 1:
                    fz_target, fz_short_float = get_finviz_fundamentals(symbol)
                    finviz_block["FinvizTarget"] = fz_target
                    finviz_block["FinvizShortFloat"] = fz_short_float

            # --- FinBERT news sentiment ---
            with profile_step("news sentiment", symbol, enabled=CONFIG["ENABLE_NEWS_SENTIMENT"] == 1):
                sentiment_block = {}
                if CONFIG["ENABLE_NEWS_SENTIMENT"] == 1:
                    sentiment_results = get_news_sentiment(yf_ticker, nlp_pipeline=finbert_pipeline)
                    if sentiment_results:
                        top_label = sentiment_results[0][1][0]['label']
                        top_score = sentiment_results[0][1][0]['score']
                        sentiment_block["NewsSentiment"] = top_label
                        sentiment_block["NewsSentimentScore"] = round(float(top_score), 3)
                    else:
                        sentiment_block["NewsSentiment"] = None
                        sentiment_block["NewsSentimentScore"] = None



            # --- EPS revisions / trend / estimate (current-quarter row, if present) ---
            with profile_step("EPS data", symbol, enabled=CONFIG["ENABLE_EPS_DATA"] == 1):
                eps_block = {}
                if CONFIG["ENABLE_EPS_DATA"] == 1:
                    try:
                        eps_rev = get_eps_revisions(yf_ticker)
                        if eps_rev is not None and not eps_rev.empty and '0q' in eps_rev.index:
                            row = eps_rev.loc['0q']
                            eps_block["EPSRevUp7d"] = row.get('upLast7days')
                            eps_block["EPSRevDown7d"] = row.get('downLast7days')
                    except Exception:
                        pass
                    try:
                        eps_trend = get_eps_trend(yf_ticker)
                        if eps_trend is not None and not eps_trend.empty and '0q' in eps_trend.index:
                            row = eps_trend.loc['0q']
                            eps_block["EPSTrendCurrent"] = row.get('current')
                            eps_block["EPSTrend30dAgo"] = row.get('30daysAgo')
                            eps_block["EPSDiff"] = eps_block["EPSTrendCurrent"] - eps_block["EPSTrend30dAgo"]

                    except Exception:
                        pass
                    est = None
                    try:
                        est = get_earnings_estimate(yf_ticker)
                        if est is not None and not est.empty and '0q' in est.index:
                            row = est.loc['0q']
                            eps_block["EPSEstAvg"] = row.get('avg')
                            eps_block["EPSEstNumAnalysts"] = row.get('numberOfAnalysts')
                            eps_block["EPSEstGrowth"] = row.get('growth')  # forward growth estimate, current quarter

                    except Exception:
                        pass

                    # Also pull the full-year growth estimate ('0y') for a longer-horizon view
                    # (reuses the estimate frame fetched above -- no second request)
                    try:
                        est_full = est
                        if est_full is not None and not est_full.empty and '0y' in est_full.index:
                            row_y = est_full.loc['0y']
                            eps_block["EPSEstGrowthFY"] = row_y.get('growth')
                    except Exception:
                        pass

                    try:
                        TPE = get_PE(yf_ticker)
                        if TPE is not None:
                            eps_block["TPE"] = f"{TPE:.2f} "
                    except Exception:
                        pass
                    try:
                        FPE = get_FPE(yf_ticker)
                        if FPE is not None:
                            eps_block["FPE"] = f"{FPE:.2f} "
                    except Exception:
                        pass

            # --- Next earnings date ---
            with profile_step("earnings dates", symbol, enabled=CONFIG["ENABLE_EARNINGS_DATES"] == 1):
                earnings_block = {}
                edates = None   # fetched once here, reused by the dip block below
                if CONFIG["ENABLE_EARNINGS_DATES"] == 1:
                    try:
                        edates = get_earnings_dates_cached(symbol, yf_ticker)   # disk-cached
                        if edates is not None and not edates.empty:
                            earnings_block["NextEarningsDate"] = str(edates.index[0].date())
                    except Exception:
                        pass

            # --- Altman Z-score (Financial Modeling Prep) ---
            with profile_step("altman z-score", symbol, enabled=CONFIG["ENABLE_ALTMAN_ZSCORE"] == 1):
                altman_block = {}
                if CONFIG["ENABLE_ALTMAN_ZSCORE"] == 1:
                    altman_block["AltmanZ"] = get_altman_zscore_exact(symbol,yf_ticker)

            with profile_step("cash metrics", symbol, enabled=CONFIG["ENABLE_CASH_METRICS"] == 1):
                cash_block = {}
                if CONFIG["ENABLE_CASH_METRICS"] == 1:
                    cash_metrics = get_cash_metrics(symbol, yf_ticker)
                    cash_block["LatestFCF"] = cash_metrics["LatestFCF"]
                    cash_block["FCFYield"] = cash_metrics["FCFYield"]
                    cash_block["FCFMargin"] = cash_metrics["FCFMargin"]
                    cash_block["OperatingCashFlow"] = cash_metrics["OperatingCashFlow"]
                    cash_block["CapEx"] = cash_metrics["CapEx"]
                    cash_block["TotalCash"] = cash_metrics["TotalCash"]
                    cash_block["TotalDebt"] = cash_metrics["TotalDebt"]
                    cash_block["NetCashPosition"] = cash_metrics["NetCashPosition"]
                    cash_block["CashToDebt"] = cash_metrics["CashToDebt"]
                    cash_block["CurrentRatio"] = cash_metrics["CurrentRatio"]
                    cash_block["QuickRatio"] = cash_metrics["QuickRatio"]
                    cash_block["pegRatio"] = cash_metrics["pegRatio"]



            # --- Heavy/DataFrame-shaped data: stored separately, not as row columns ---
            if CONFIG["ENABLE_RAW_STATEMENTS"] == 1 and extra_data_store is not None:
                fin, bs, cf = get_financial_statements(yf_ticker)
                calls, puts = get_option_chain(yf_ticker)
                extra_data_store[symbol] = {
                    "recommendations_summary": get_recommendations_summary(yf_ticker),
                    "upgrades_downgrades": get_upgrades_downgrades(yf_ticker),
                    "financials": fin,
                    "balance_sheet": bs,
                    "cashflow": cf,
                    "calls": calls,
                    "puts": puts,
                }

            #dip strategy for stock with best fundamentals
            # ... inside the ENABLE_DIP_STRATEGY block ...
            with profile_step("dip strategy", symbol, enabled=CONFIG["ENABLE_DIP_STRATEGY"] == 1):
                dip_block = {}
                if CONFIG["ENABLE_DIP_STRATEGY"] == 1:
                    dip_pct = calculate_dip_pct(df, days)

                    edates_df = edates if CONFIG["ENABLE_EARNINGS_DATES"] == 1 else None
                    near_earnings = is_near_earnings(edates_df)

                    base_quality_score = calculate_base_quality_score(
                        cash_block if CONFIG["ENABLE_CASH_METRICS"] == 1 else None,
                        altman_block.get("AltmanZ") if CONFIG["ENABLE_ALTMAN_ZSCORE"] == 1 else None,
                        analyst_block if CONFIG["ENABLE_ANALYST_DATA"] == 1 else None,
                        eps_block if CONFIG["ENABLE_EPS_DATA"] == 1 else None,
                    )

                    dip_block["DipPct"] = dip_pct
                    dip_block["NearEarnings"] = near_earnings
                    dip_block["BaseQualityScore"] = base_quality_score
                    # QualityScore and QualityDipBuy are NOT set here — computed after the
                    # full batch runs, once sector-relative valuation is available.

            

            with profile_step("valuation", symbol, enabled=CONFIG["ENABLE_VALUATION"] == 1):
                valuation_block = {}
                if CONFIG["ENABLE_VALUATION"] == 1:
                    valuation_block = get_valuation_metrics(symbol, ticker=yf_ticker)


            # Intraday analysis runs now if the intraday bars are available
            # (intraday_ready() is True, or no background download is in use);
            # otherwise a placeholder is left in the row and the analysis is
            # done in a second pass once the background download finishes.
            intraday_on = CONFIG["ENABLE_INTRADAY"] == 1
            intraday_now = intraday_on and (intraday_ready is None or intraday_ready())
            with profile_step("intraday analysis", symbol, enabled=intraday_now):
                intraday_block = {} 
                if intraday_on:

                    # get_company_info stores sectors with underscores ("Financial_Services"),
                    # but SECTOR_TO_ETF uses spaces -- convert back so multi-word sectors match.
                    sector = company_block.get("Sector")
                    if isinstance(sector, str):
                        sector = sector.replace("_", " ")

                    if intraday_now:
                        if spy_df is None:
                            spy_df = fetch_intraday("SPY")
                        etf = SECTOR_TO_ETF.get(sector)
                        if etf and etf not in sector_cache:
                            sector_cache[etf] = fetch_intraday(etf)

                        intraday_block = analyze_intraday(
                                    symbol,
                                    sector=sector,
                                    spy_df=spy_df,
                                    sector_df=sector_cache.get(etf),
                                    daily_df=df,
                        )
                    else:
                        intraday_block = {_INTRADAY_KEY: sector}   # filled in later, same column position


            if passes_gap_filter:
                aboveSMA = 1 if days[DAYm1_IDX]['SMA50'] < days[DAYm1_IDX]['Close'] else 0
                results2.append({
                    " Ticker ": '"' + symbol + '",',
                    **company_block,
                    **oc_block,
                    "ATR(14) ": f"{daily_atr_val:.2f} ",
                    "Sug. Stop ": f"{structural_stop:.2f} ",
                    "Take Profit ": f"{take_profit_target:.2f} ",
                    # Pre-Market, Pre-Market%, Pre-Mkt Src, Pre-Mkt Time, Pre-Mkt Reason
                    # (placeholders in deferred mode; filled in after the pool finishes)
                    **(pm_cols if CONFIG["SHOW_PREMARKET_PRICE"] == 1 else {}),
                    **({_D1_CLOSE_KEY: d1_close} if premarket_prices is PREMARKET_DEFERRED else {}),
                    "50SMA ": f"{days[DAYm1_IDX]['SMA50']:.2f} ",
                    **sr_block,
                    **vol_block,
                    **g_block,
                    "AboveSMA ": f"{aboveSMA:.0f} ",
                    "52WLow ": f"{week52_low:.2f} ",
                    "52WHigh ": f"{week52_high:.2f} ",
                    "%FromLow ": f"{pct_from_low*100:.1f} ",
                    "%FromHigh ": f"{pct_from_high*100:.1f} ",
                    "RisingVol ": f"{risingVol:.0f} ",
                    "RiseRSI ": f"{rsig:.0f} ",
                    **rsi_block,
                    "Stoch ": f"{stoch_buy_signal} ",
                    "StochK": round(float(df['StochK'].iloc[DAYm1_IDX]), 1),
                    "StochD": round(float(df['StochD'].iloc[DAYm1_IDX]), 1),
                    "GoldenCross": golden_cross,
                    "MACDBuy": macd_buy_signal,
                    "ADX": round(float(df['ADX'].iloc[DAYm1_IDX]), 1),
                    "Trending": trending,
                    "Bollinger": bollinger_bounce,
                    "OBVRising": obv_rising,
                    "VolumeSpike": volume_spike,
                    "Near52WLow": near_low_bounce,
                    "CMFBuy": cmf_buy,
                    "BullDiv": bull_div,
                    "TrendPullback": trend_pullback,
                    "BearDiv": bear_div,
                    "DeathCross": death_cross,
                    "ExhaustionSell": exhaustion_sell,
                    **analyst_block,
                    **finviz_block,
                    **sentiment_block,
                    **eps_block,
                    **earnings_block,
                    **altman_block,
                    **sig,
                    **cash_block,
                    **valuation_block,
                    **intraday_block,
                    **dip_block, 

                })
        except Exception as e:
            if verbose_errors:
                print(f"{symbol}: skipped ({e})")
            continue
    return results2




def evaluate_tickers_parallel(sp500_tickers, data, max_workers=10, 
                              verbose_errors=True, show_progress=True, 
                              finbert_pipeline=None, extra_data_store=None,tv_exchange_map=None,
                              **kwargs):
    results = []

    # --- Pre-market prices (pre_market.py) ---
    # Batched 1-min bars for today's pre-market on SIP (16-min delayed) and IEX
    # (live). This only talks to Alpaca, so it runs in a BACKGROUND thread at
    # the same time as the intraday prefetch and the worker pool (which talk to
    # Yahoo), and its columns are filled into the rows afterwards.
    # Exception: FILTER_PREMARKET_GAP_UP needs the price while building rows,
    # so in that mode it is fetched up front as before.
    premarket_prices = None
    premarket_future = None
    premarket_bg = None
    if CONFIG["SHOW_PREMARKET_PRICE"] == 1:
        if CONFIG["FILTER_PREMARKET_GAP_UP"] == 1:
            premarket_prices = _fetch_premarket(sp500_tickers)
        else:
            premarket_bg = ThreadPoolExecutor(max_workers=1, thread_name_prefix="premarket")
            premarket_future = premarket_bg.submit(_fetch_premarket, sp500_tickers)
            premarket_prices = PREMARKET_DEFERRED

    # --- Intraday 5m bars (intraday_strategy.py) ---
    # Batched yf.download for every ticker + SPY + sector ETFs, run in a
    # BACKGROUND thread while the workers do everything else. The workers never
    # call yf.download themselves, so this is the only yf.download running
    # (yf.download is not safe to run twice at once). A worker that reaches a
    # ticker's intraday step after the download has finished analyses it
    # immediately; earlier tickers are finished in a second pass below.
    intraday_future = None
    intraday_bg = None
    intraday_ready = None
    if CONFIG["ENABLE_INTRADAY"] == 1:
        intraday_bg = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intraday")
        intraday_future = intraday_bg.submit(_prefetch_intraday_bg, list(sp500_tickers))
        intraday_ready = lambda: intraday_future.done() and intraday_future.exception() is None

    def process_one(symbol):
        with profile_step("whole ticker (run_screen)", symbol):
            return evaluate_tickers([symbol], data, verbose_errors=verbose_errors,
                finbert_pipeline=None, extra_data_store=None,tv_exchange_map=None,
                premarket_prices=premarket_prices, intraday_ready=intraday_ready, **kwargs)

    try:
        _run_pool(sp500_tickers, process_one, max_workers, verbose_errors, show_progress, results)

        # Finish intraday analysis for tickers the workers reached before the
        # background download was done.
        if intraday_future is not None:
            _finish_deferred_intraday(results, data, intraday_future, max_workers)
    finally:
        if intraday_bg is not None:
            intraday_bg.shutdown(wait=True)
            clear_prefetched_intraday()
        with profile_step("main: save caches"):
            disk_cache.flush_all()   # save earnings / cash-flow / info / analyst caches for the next run

    # Fill in the pre-market columns now that the background fetch is done.
    if premarket_future is not None:
        try:
            with profile_step("main: wait for pre-market"):   # ~0 if it finished during the pool
                prices = premarket_future.result()
        except Exception as e:
            print(f"Pre-market fetch failed: {e}")
            prices = {}
        finally:
            premarket_bg.shutdown(wait=False)
        for row in results:
           d1_close = row.pop(_D1_CLOSE_KEY, None)
           sym = row.get(" Ticker ", "").strip('",')   # '"AAPL",' -> 'AAPL'
           cols, _ = _premarket_columns(prices.get(sym), d1_close)
           row.update(cols)   # existing keys -> column order is unchanged

    return results


def _prefetch_intraday_bg(tickers) -> None:
    """Background-thread wrapper around prefetch_intraday (timed separately)."""
    with profile_step("intraday prefetch (background)"):
        prefetch_intraday(tickers)


def _finish_deferred_intraday(results, data, intraday_future, max_workers) -> None:
    """Run analyze_intraday for rows that were left with a placeholder, and put
    the intraday columns where the placeholder was (column order unchanged)."""
    with profile_step("main: wait for intraday prefetch"):   # ~0 if it finished during the pool
        try:
            intraday_future.result()
            failed = None
        except Exception as e:
            print(f"Intraday prefetch failed: {e}")
            failed = str(e)

    pending = [row for row in results if _INTRADAY_KEY in row]
    if not pending:
        return
    print(f"Intraday analysis: {len(results) - len(pending)} done during the pool, "
          f"{len(pending)} after the download")

    spy_df = fetch_intraday("SPY") if failed is None else None
    sector_dfs = {}
    if failed is None:
        for etf in {SECTOR_TO_ETF.get(r[_INTRADAY_KEY]) for r in pending} - {None}:
            sector_dfs[etf] = fetch_intraday(etf)

    def analyse(row):
        symbol = row.get(" Ticker ", "").strip('",')
        sector = row.get(_INTRADAY_KEY)
        if failed is not None:
            return {"symbol": symbol, "error": "intraday_prefetch_failed"}
        with profile_step("intraday analysis", symbol):
            try:
                daily_df = data.xs(symbol, level=1, axis=1).dropna()
                return analyze_intraday(symbol, sector=sector, spy_df=spy_df,
                                        sector_df=sector_dfs.get(SECTOR_TO_ETF.get(sector)),
                                        daily_df=daily_df)
            except Exception as e:
                return {"symbol": symbol, "error": f"intraday_failed: {e}"}

    with profile_step("main: deferred intraday analysis"):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            blocks = list(pool.map(analyse, pending))

    for row, block in zip(pending, blocks):
        # Rebuild the row so the intraday columns sit where the placeholder was.
        items = list(row.items())
        row.clear()
        for key, value in items:
            if key == _INTRADAY_KEY:
                row.update(block)
            else:
                row[key] = value


def _fetch_premarket(tickers) -> dict:
    """Batched pre-market prices for all tickers; {} if disabled or failed."""
    prices = prefetch_pre_market(list(tickers), CONFIG)
    if prices is None:
        return {}   # disabled in config: don't fall back to per-ticker calls
    n = sum(1 for d in prices.values() if d and d.get("price") is not None)
    print(f"Pre-market prices: {n}/{len(tickers)} tickers")
    return prices


def _run_pool(sp500_tickers, process_one, max_workers, verbose_errors, show_progress, results):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(process_one, symbol): symbol 
            for symbol in sp500_tickers
        }
        
        futures_iterable = as_completed(future_to_symbol)
        if show_progress:
            futures_iterable = tqdm(futures_iterable, total=len(sp500_tickers), desc="Evaluating Tickers")

        for future in futures_iterable:
            symbol = future_to_symbol[future]
            try:
                rows = future.result()
                if rows:
                    results.extend(rows)
            except Exception as e:
                if verbose_errors:
                    print(f"Error processing {symbol}: {e}")

    return results




def is_market_in_uptrend(benchmark_ticker="SPY", sma_period=200):
    """
    Fetches the benchmark ETF and determines if its latest close 
    is above its 200-day Simple Moving Average.
    """
    try:
        # Fetch 1 year of data (roughly 252 trading days)
        df = yf.download(benchmark_ticker, period="1y", progress=False)
        
        # Handle yfinance multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) < sma_period:
            print(f"Not enough data to calculate {sma_period} SMA for {benchmark_ticker}.")
            return True # Fail open so the screener still runs
            
        # Calculate the 200 SMA
        sma_200 = df['Close'].rolling(window=sma_period).mean()
        
        latest_close = float(df['Close'].iloc[-1])
        latest_sma = float(sma_200.iloc[-1])
        
        is_bullish = latest_close > latest_sma
        print(f"Macro Regime [{benchmark_ticker}]: Close={latest_close:.2f} | 200 SMA={latest_sma:.2f}")
        print(f"Status: {'BULLISH (Risk On)' if is_bullish else 'BEARISH (Risk Off - Halt Entries)'}")
        
        return is_bullish
    except Exception as e:
        print(f"Error fetching market regime: {e}")
        return True # Default to allowing trades if lookup fails
