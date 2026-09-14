
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

# 1. Create the session
yf_session = Session()

# 2. Configure the retry rules
retry = Retry(
    total=2,              # Try up to 2 times
    backoff_factor=1,     # Wait 1s, then 2s
    status_forcelist=[429, 500, 502, 503, 504] # 429 is the rate limit error
)

# 3. Create ONE adapter that handles both the pool size AND the retries
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)

# 4. Mount it to the session
yf_session.mount('https://', adapter)
yf_session.mount('http://', adapter)


def get_premarket_price(ticker: str) -> float | str:
    """Fetches the absolute latest trade price using Alpaca's free IEX feed."""
    
    # The /snapshots endpoint returns the latest trade, quote, and daily bars all at once
    url = "https://data.alpaca.markets/v2/stocks/snapshots"
    
    # We must explicitly request the 'iex' feed for the free tier to work
    params = {
        "symbols": ticker.upper(),
        "feed": "iex"
    }
    
    headers = {
        "APCA-API-KEY-ID": "PKGT4VDNU6I3UJVRNUFYT34PK2",
        "APCA-API-SECRET-KEY": "E9zQAKbXS5ATHJ399iqPGq5GaYqdKQq6DQDjcKrQQQDx"
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            ticker_data = data.get(ticker.upper(), {})
            
            # Navigate the JSON to get the most recent trade price
            latest_trade = ticker_data.get("latestTrade", {})
            price = latest_trade.get("p")
            
            if price:
                return float(price)
                
    except Exception as e:
        print(f"Error fetching Alpaca data: {e}")
        
    return "N/A"

def evaluate_tickers(sp500_tickers, data, verbose_errors=True, finbert_pipeline=None, extra_data_store=None,tv_exchange_map=None):
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

    for symbol in sp500_tickers:
        try:
            df = data.xs(symbol, level=1, axis=1).dropna()
            if len(df) < 200:
                continue



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

            risingVol = (days[DAYm1_IDX]['Volume'] > days[DAYm2_IDX]['Volume']) and (days[DAYm2_IDX]['Volume'] > days[DAYm3_IDX]['Volume'])
            pre_price_str = "N/A"
            latest_price = None

            if CONFIG["SHOW_PREMARKET_PRICE"] == 1:
                try:
                    latest_price = get_premarket_price(symbol)                    
                    pre_price_str = f"{float(latest_price):.2f} "
                    pre_price_pct = (latest_price - days[DAYm1_IDX]['Close'])/days[DAYm1_IDX]['Close']
                except Exception:
                    pass


            passes_gap_filter = (
               CONFIG["FILTER_PREMARKET_GAP_UP"] == 0
               or (latest_price is not None )
            )

            gflags = {i: (1 if days[i]['Close'] > days[i]['Open'] else 0) for i in DAY_IDXS}
            cflags = {i: is_nice_green_candle(days[i]) for i in DAY_IDXS}

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
            c_block = {f"C{DAY_LABELS[i]}": cflags[i] for i in DAY_ORDER}
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
            
            yf_ticker = yf.Ticker(symbol,session=yf_session) if needs_yf_ticker else None

            company_block = {}
            if CONFIG["ENABLE_COMPANY_INFO"] == 1:
                company_info = get_company_info(symbol, yf_ticker)
                company_block["Name"] = company_info["Name"]
                company_block["Sector"] = company_info["Sector"]
                company_block["Industry"] = company_info["Industry"]
                company_block["MarketCap"] = company_info["MarketCap"]

            # --- Analyst ratings / price targets ---
            analyst_block = {}
            if CONFIG["ENABLE_ANALYST_DATA"] == 1:
                
                rec_mean = get_recommendation_mean(yf_ticker)
                targets = get_analyst_price_targets(yf_ticker)
                analyst_block["RecMean"] = rec_mean
                analyst_block["TargetMean"] = targets.get("mean") if isinstance(targets, dict) else None
                analyst_block["TargetHigh"] = targets.get("high") if isinstance(targets, dict) else None
                analyst_block["TargetLow"] = targets.get("low") if isinstance(targets, dict) else None
                
                rec_breakdown = get_yahoo_recommendation_breakdown(symbol, yf_ticker)
                analyst_block.update(rec_breakdown)

                # NEW: derived smart-signal block, reuses same yf_ticker
                smart_signals = extract_smart_analyst_signals(
                    yf_ticker, current_price=days[DAYm1_IDX]['Close']
                )
                analyst_block.update(smart_signals)

            # --- TradingView TA summary rating ---
            tv_block = {}
            if CONFIG["ENABLE_TRADINGVIEW"] == 1:
                tv_rating, tv_votes = get_tradingview_rating_cached(symbol, tv_exchange_map or {})
                tv_block["TVRating"] = tv_rating
                tv_block["TVBuy"] = tv_votes.get("BUY") if tv_votes else None
                tv_block["TVSell"] = tv_votes.get("SELL") if tv_votes else None
                tv_block["TVNeutral"] = tv_votes.get("NEUTRAL") if tv_votes else None

            # --- Finviz target price / short float ---
            finviz_block = {}
            if CONFIG["ENABLE_FINVIZ"] == 1:
                fz_target, fz_short_float = get_finviz_fundamentals(symbol)
                finviz_block["FinvizTarget"] = fz_target
                finviz_block["FinvizShortFloat"] = fz_short_float

            # --- FinBERT news sentiment ---
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

            # --- Short interest ---
            short_block = {}
            if CONFIG["ENABLE_SHORT_INTEREST"] == 1:
                short_pct, short_ratio = get_short_interest(yf_ticker)
                short_block["ShortPctFloat"] = short_pct
                short_block["ShortRatio"] = short_ratio

            # --- EPS revisions / trend / estimate (current-quarter row, if present) ---
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
                try:
                    est_full = get_earnings_estimate(yf_ticker)
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
            earnings_block = {}
            if CONFIG["ENABLE_EARNINGS_DATES"] == 1:
                try:
                    edates = get_earnings_dates(yf_ticker)
                    if edates is not None and not edates.empty:
                        earnings_block["NextEarningsDate"] = str(edates.index[0].date())
                except Exception:
                    pass

            # --- Altman Z-score (Financial Modeling Prep) ---
            altman_block = {}
            if CONFIG["ENABLE_ALTMAN_ZSCORE"] == 1:
                altman_block["AltmanZ"] = get_altman_zscore_exact(symbol,yf_ticker)

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
            dip_block = {}
            if CONFIG["ENABLE_DIP_STRATEGY"] == 1:
                dip_pct = calculate_dip_pct(df, days)

                edates_df = None
                if CONFIG["ENABLE_EARNINGS_DATES"] == 1:
                    edates_df = get_earnings_dates(yf_ticker)
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


            valuation_block = {}
            if CONFIG["ENABLE_VALUATION"] == 1:
                valuation_block = get_valuation_metrics(symbol, ticker=yf_ticker)

            if passes_gap_filter:
                aboveSMA = 1 if days[DAYm1_IDX]['SMA50'] < days[DAYm1_IDX]['Close'] else 0
                results2.append({
                    " Ticker ": symbol,
                    **company_block,
                    **oc_block,
                    "ATR(14) ": f"{daily_atr_val:.2f} ",
                    "Sug. Stop ": f"{structural_stop:.2f} ",
                    "Take Profit ": f"{take_profit_target:.2f} ",
                    **({"Pre-Market": pre_price_str} if CONFIG["SHOW_PREMARKET_PRICE"] == 1 else {}),
                    **({"Pre-Market%": f"{pre_price_pct*100:.1f} ",} if CONFIG["SHOW_PREMARKET_PRICE"] == 1 else {}),
                    "50SMA ": f"{days[DAYm1_IDX]['SMA50']:.2f} ",
                    **sr_block,
                    **vol_block,
                    **g_block,
                    **c_block,
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
                    **tv_block,
                    **finviz_block,
                    **sentiment_block,
                    **short_block,
                    **eps_block,
                    **earnings_block,
                    **altman_block,
                    **sig,
                    **cash_block,
                    **valuation_block,
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

    def process_one(symbol):
        return evaluate_tickers([symbol], data, verbose_errors=False, 
            finbert_pipeline=None, extra_data_store=None,tv_exchange_map=None, **kwargs)

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
