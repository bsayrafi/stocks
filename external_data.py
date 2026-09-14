from constants import CONFIG
import yfinance as yf
import logging
from tradingview_ta import TA_Handler, Interval
# --- yfinance: analyst ratings / targets ---

def get_recommendation_mean(ticker):
    """Average analyst recommendation score (lower = more bullish), or None."""
    try:
        return ticker.info.get('recommendationMean')
    except Exception:
        return None


def get_recommendations_summary(ticker):
    """Analyst recommendations summary DataFrame, or None."""
    try:
        return ticker.recommendations_summary
    except Exception:
        return None


def get_upgrades_downgrades(ticker):
    """Upgrades/downgrades history DataFrame, or None."""
    try:
        return ticker.upgrades_downgrades
    except Exception:
        return None


def get_analyst_price_targets(ticker):
    """Analyst price target dict (low/high/mean/median/current), or None."""
    try:
        return ticker.analyst_price_targets
    except Exception:
        return None


# --- yfinance: earnings / estimates ---

def get_eps_revisions(ticker):
    try:
        return ticker.get_eps_revisions()
    except Exception:
        return None


def get_eps_trend(ticker):
    try:
        return ticker.get_eps_trend()
    except Exception:
        return None


def get_earnings_estimate(ticker):
    try:
        return ticker.get_earnings_estimate()
    except Exception:
        return None

def get_PE(ticker):
    info = ticker.info
    try:
        return info.get("trailingPE")
    except Exception:
        return None

def get_FPE(ticker):
    info = ticker.info
    try:
        return info.get("forwardPE")
    except Exception:
        return None

def get_earnings_dates(ticker, limit=CONFIG["EPS_DATES_LIMIT"]):
    try:
        return ticker.get_earnings_dates(limit=limit)
    except Exception:
        return None


# --- yfinance: short interest / options ---

def get_short_interest(ticker):
    """(short_pct_of_float, short_ratio_days_to_cover), or (None, None)."""
    try:
        info = ticker.info
        return info.get("shortPercentOfFloat"), info.get("shortRatio")
    except Exception:
        return None, None


def get_option_chain(ticker, expiry_index=CONFIG["OPTIONS_EXPIRY_INDEX"]):
    """(calls_df, puts_df) for the given expiration index, or (None, None)."""
    try:
        exp_dates = ticker.options
        if not exp_dates:
            return None, None
        opt_chain = ticker.option_chain(exp_dates[expiry_index])
        return opt_chain.calls, opt_chain.puts
    except Exception:
        return None, None


# --- yfinance: financial statements ---

def get_financial_statements(ticker):
    """(financials, balance_sheet, cashflow) DataFrames, or (None, None, None)."""
    try:
        return ticker.financials, ticker.balance_sheet, ticker.cashflow
    except Exception:
        return None, None, None


# --- yfinance news + FinBERT sentiment ---

def build_finbert_pipeline(model_name=CONFIG["FINBERT_MODEL_NAME"]):
    """
    Load FinBERT once and return a sentiment-analysis pipeline.
    Call this ONCE per run and reuse it — reloading per ticker is very slow.
    """
    try:
        from transformers import BertTokenizer, BertForSequenceClassification, pipeline
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception:
        return None


def get_news_sentiment(ticker, headline_limit=CONFIG["NEWS_HEADLINE_LIMIT"], nlp_pipeline=None):
    """List of (headline, sentiment) tuples for the latest news, or []."""
    if nlp_pipeline is None:
        return []
    try:
        news = ticker.news
        results = []
        for item in news[:headline_limit]:
            content = item.get('content', {})
            headline = content.get('title')
            if not headline:
                continue
            sentiment = nlp_pipeline(headline)
            results.append((headline, sentiment))
        return results
    except Exception:
        return []

YF_TO_TV_EXCHANGE = {
    "NMS": "NASDAQ",   # NASDAQ Global Select
    "NGM": "NASDAQ",   # NASDAQ Global Market
    "NCM": "NASDAQ",   # NASDAQ Capital Market
    "NYQ": "NYSE",
    "NYS": "NYSE",
    "ASE": "AMEX",
    "PCX": "AMEX",     # NYSE Arca often maps cleanly to AMEX on TV
    "BATS": "BATS",
}


def get_tradingview_exchange(yf_ticker):
    """Derive TradingView's expected exchange string from yfinance's info."""
    try:
        yf_exch = yf_ticker.info.get("exchange")
        return YF_TO_TV_EXCHANGE.get(yf_exch)  # None if unmapped
    except Exception:
        return None

        

def build_tv_exchange_map(sp500_tickers):
    tv_exchange_map = {}
    for symbol in sp500_tickers:
        try:
            t = yf.Ticker(symbol)
            tv_exch = get_tradingview_exchange(t)
            tv_exchange_map[symbol] = tv_exch  # may be None if unmapped
        except Exception:
            tv_exchange_map[symbol] = None
    return tv_exchange_map



# --- TradingView TA ---

import json, os, time, random
from datetime import date

TV_CACHE_FILE = "tv_ratings_cache.json"

def load_tv_cache():
    if os.path.exists(TV_CACHE_FILE):
        with open(TV_CACHE_FILE) as f:
            cache = json.load(f)
        if cache.get("_date") != str(date.today()):
            return {"_date": str(date.today())}
        return cache
    return {"_date": str(date.today())}

def save_tv_cache(cache):
    with open(TV_CACHE_FILE, "w") as f:
        json.dump(cache, f)

tv_cache = load_tv_cache()

def get_tradingview_rating(symbol, tv_exchange_map, screener="america", interval=None,
                            max_retries=3, base_delay=3.0):
    """(recommendation, vote_breakdown_dict) from TradingView TA, or (None, None).
    Tries the mapped exchange first, falls back to NASDAQ/NYSE/AMEX,
    with exponential backoff on 429s."""
    from tradingview_ta import TA_Handler, Interval
    if interval is None:
        interval = Interval.INTERVAL_1_DAY

    known_exch = (tv_exchange_map or {}).get(symbol)
    candidates = [known_exch] if known_exch else []
    candidates += [e for e in ["NASDAQ", "NYSE", "AMEX"] if e != known_exch]

    for exch in candidates:
        for attempt in range(max_retries):
            try:
                handler = TA_Handler(symbol=symbol, screener=screener, exchange=exch, interval=interval)
                analysis = handler.get_analysis()
                return analysis.summary["RECOMMENDATION"], analysis.summary
            except Exception as e:
                if "429" in str(e):
                    time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
                else:
                    break
    return None, None


def get_tradingview_rating_cached(symbol, tv_exchange_map):
    """Persists to disk, refreshed once per calendar day. Repeated
    evaluate_tickers() calls within the same day hit the cache, not the API."""
    if symbol in tv_cache:
        rec, summary = tv_cache[symbol]
        return rec, summary
    rec, summary = get_tradingview_rating(symbol, tv_exchange_map)
    tv_cache[symbol] = [rec, summary]
    save_tv_cache(tv_cache)
    time.sleep(2)
    return rec, summary



# --- Finviz ---

def get_finviz_ratings(symbol):
    """Finviz analyst ratings log DataFrame, or None."""
    try:
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol)
        return stock.ticker_outer_ratings()
    except Exception:
        return None


def get_finviz_fundamentals(symbol):
    """(target_price, short_float) from finviz fundamentals, or (None, None)."""
    try:
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol)
        fundament_dict = stock.ticker_fundament()
        return fundament_dict.get("Target Price"), fundament_dict.get("Short Float")
    except Exception:
        return None, None


# --- Financial Modeling Prep (fundamentalanalysis) ---

def get_altman_zscore(symbol, api_key, period="annual"):
    """Most recent Altman Z-Score via FMP, or None."""
    try:
        import fundamentalanalysis as fa
        key_metrics = fa.key_metrics(symbol, api_key, period=period)
        return float(key_metrics.loc['altmanZScore'].iloc[0])
    except Exception:
        return None





def get_altman_zscore_exact(symbol: str, ticker=None):
    #ticker = yf.Ticker(symbol)

    q_bs = ticker.quarterly_balance_sheet.reindex(
        columns=sorted(ticker.quarterly_balance_sheet.columns, reverse=True)
    )
    q_inc = ticker.quarterly_financials.reindex(
        columns=sorted(ticker.quarterly_financials.columns, reverse=True)
    )

    mrq_bs = q_bs.iloc[:, 0]
    ttm_inc = q_inc.iloc[:, :4].sum(axis=1)

    if "Total Assets" not in mrq_bs.index:
        return None
    total_assets = mrq_bs.loc["Total Assets"]

    if "Working Capital" in mrq_bs.index:
        working_capital = mrq_bs.loc["Working Capital"]
    elif "Current Assets" in mrq_bs.index and "Current Liabilities" in mrq_bs.index:
        working_capital = mrq_bs.loc["Current Assets"] - mrq_bs.loc["Current Liabilities"]
    else:
        return None

    retained_earnings = mrq_bs.get("Retained Earnings", 0)
    ebit = ttm_inc.get("EBIT", ttm_inc.get("Operating Income"))
    if ebit is None:
        return None

    if "Total Revenue" not in ttm_inc.index:
        return None
    total_revenue = ttm_inc.loc["Total Revenue"]

    market_cap = ticker.info.get("marketCap", ticker.fast_info.get("market_cap", 0))

    if "Stockholders Equity" in mrq_bs.index:
        equity = mrq_bs.loc["Stockholders Equity"]
    elif "Total Stockholder Equity" in mrq_bs.index:
        equity = mrq_bs.loc["Total Stockholder Equity"]
    else:
        return None

    total_liabilities = total_assets - equity
    if total_liabilities <= 0:
        return None  # avoid nonsensical/divide-by-zero ratios

    x1 = (working_capital / total_assets) * 1.2
    x2 = (retained_earnings / total_assets) * 1.4
    x3 = (ebit / total_assets) * 3.3
    x4 = (market_cap / total_liabilities) * 0.6
    x5 = (total_revenue / total_assets) * 1.0

    z_score = x1 + x2 + x3 + x4 + x5
    return round(z_score, 2)


def extract_smart_analyst_signals(yf_ticker, current_price):
    """
    Transforms raw Wall Street analyst data into quantitative signals.
    Returns N/A-safe defaults if data is missing so it never breaks row alignment.
    """
    analyst_block = {
        "TargetUpsidePct": None,
        "AnalystBias": "NEUTRAL",
        "EPSRevMomentum": 0,
    }

    try:
        info = yf_ticker.info

        # 1. Price Target Spread
        target_mean = info.get("targetMeanPrice")
        if target_mean and current_price:
            upside = ((target_mean - current_price) / current_price) * 100
            analyst_block["TargetUpsidePct"] = round(upside, 1)

        # 2. Rating Recommendation Mean (1.0 = Strong Buy, 3.0 = Hold, 5.0 = Sell)
        rec_mean = info.get("recommendationMean")
        if rec_mean:
            if rec_mean <= 1.8:
                analyst_block["AnalystBias"] = "BULLISH"
            elif rec_mean >= 2.8:
                analyst_block["AnalystBias"] = "BEARISH/HOLD"

        # 3. EPS Revision Direction (7-day momentum)
        eps_rev = yf_ticker.eps_revisions
        if eps_rev is not None and not eps_rev.empty and '0q' in eps_rev.index:
            up_7d = eps_rev.loc['0q'].get('upLast7days', 0) or 0
            down_7d = eps_rev.loc['0q'].get('downLast7days', 0) or 0
            if up_7d > down_7d:
                analyst_block["EPSRevMomentum"] = 1
            elif down_7d > up_7d:
                analyst_block["EPSRevMomentum"] = -1

    except Exception:
        pass

    return analyst_block

def get_cash_metrics(symbol, ticker=None):
    """
    Cash flow and balance-sheet liquidity metrics for a symbol.
    Returns a dict; any value defaults to None if the underlying data or
    line-item label is missing for this ticker.
    """
    import yfinance as yf
    if ticker is None:
        ticker = yf.Ticker(symbol)

    result = {
        "LatestFCF": None, "FCFYield": None, "FCFMargin": None,
        "OperatingCashFlow": None, "CapEx": None,
        "MarketCap": None, "TotalCash": None, "TotalDebt": None,
        "NetCashPosition": None, "CashToDebt": None,
        "CurrentRatio": None, "QuickRatio": None,
        "pegRatio": None,
    }

    # --- Cash flow statement ---
    try:
        cf = ticker.cashflow
        if cf is not None and not cf.empty:
            op_cash = cf.loc["Operating Cash Flow"] if "Operating Cash Flow" in cf.index else None
            capex_label = "Capital Expenditure" if "Capital Expenditure" in cf.index else (
                "Capital Expenditures" if "Capital Expenditures" in cf.index else None
            )
            capex = cf.loc[capex_label] if capex_label else None

            if op_cash is not None and not op_cash.empty:
                result["OperatingCashFlow"] = float(op_cash.iloc[0])
            if capex is not None and not capex.empty:
                result["CapEx"] = float(capex.iloc[0])

            if "Free Cash Flow" in cf.index:
                fcf_series = cf.loc["Free Cash Flow"]
            elif op_cash is not None and capex is not None:
                fcf_series = op_cash + capex  # capex is typically already negative
            else:
                fcf_series = None

            if fcf_series is not None and not fcf_series.empty:
                result["LatestFCF"] = float(fcf_series.iloc[0])
    except Exception:
        pass

    # --- .info: market cap, cash, debt, liquidity ratios ---
    try:
        info = ticker.info

        market_cap = info.get("marketCap")
        if market_cap:
            result["MarketCap"] = float(market_cap)
            if result["LatestFCF"] is not None:
                result["FCFYield"] = round((result["LatestFCF"] / market_cap) * 100, 2)

        total_revenue = info.get("totalRevenue")
        if total_revenue and result["LatestFCF"] is not None:
            result["FCFMargin"] = round((result["LatestFCF"] / total_revenue) * 100, 2)

        total_cash = info.get("totalCash")
        if total_cash:
            result["TotalCash"] = float(total_cash)

        total_debt = info.get("totalDebt")
        if total_debt:
            result["TotalDebt"] = float(total_debt)

        if result["TotalCash"] is not None and result["TotalDebt"] is not None:
            result["NetCashPosition"] = result["TotalCash"] - result["TotalDebt"]
            if result["TotalDebt"] != 0:
                result["CashToDebt"] = round(result["TotalCash"] / result["TotalDebt"], 2)

        current_ratio = info.get("currentRatio")
        if current_ratio:
            result["CurrentRatio"] = round(float(current_ratio), 2)

        quick_ratio = info.get("quickRatio")
        if quick_ratio:
            result["QuickRatio"] = round(float(quick_ratio), 2)

        peg_ratio =  info.get("pegRatio")
        if peg_ratio:
            result["pegRatio"] = round(float(peg_ratio), 2)

    except Exception:
        pass

    return result

def get_company_info(symbol, ticker=None):
    """
    Basic company identification: name, sector, industry, market cap.
    Returns a dict; any value defaults to None if unavailable for this ticker.
    """
    import yfinance as yf
    if ticker is None:
        ticker = yf.Ticker(symbol)

    result = {"Name": None, "Sector": None, "Industry": None, "MarketCap": None}
    try:
        info = ticker.info
        temp = info.get("shortName") or info.get("longName")
        result["Name"] = temp.replace(" ", "_") if isinstance(temp, str) else temp
        result["Sector"] = info.get("sector").replace(" ", "_") if isinstance(info.get("sector"), str) else info.get("sector")
        result["Industry"] = info.get("industry").replace(" ", "_") if isinstance(info.get("industry"), str) else info.get("industry")

        market_cap = info.get("marketCap")
        if market_cap:
            result["MarketCap"] = float(market_cap)
    except Exception:
        pass

    return result

def get_yahoo_recommendation_breakdown(symbol, ticker=None):
    """
    Analyst recommendation counts from Yahoo Finance for the current month
    (strongBuy, buy, hold, sell, strongSell), plus simplified Buy/Neutral/Sell
    rollups and the total number of analysts.
    Returns a dict; values default to None if unavailable for this ticker.
    """
    import yfinance as yf
    if ticker is None:
        ticker = yf.Ticker(symbol)

    result = {
        "StrongBuy": None, "Buy": None, "Hold": None, "Sell": None, "StrongSell": None,
        "TotalAnalysts": None, "BuyCount": None, "NeutralCount": None, "SellCount": None,
    }

    try:
        recs = ticker.recommendations
        if recs is not None and not recs.empty:
            # "0m" = current month; fall back to the first row if that label isn't present
            row = recs[recs["period"] == "0m"]
            row = row.iloc[0] if not row.empty else recs.iloc[0]

            strong_buy = int(row.get("strongBuy", 0) or 0)
            buy = int(row.get("buy", 0) or 0)
            hold = int(row.get("hold", 0) or 0)
            sell = int(row.get("sell", 0) or 0)
            strong_sell = int(row.get("strongSell", 0) or 0)

            result["StrongBuy"] = strong_buy
            result["Buy"] = buy
            result["Hold"] = hold
            result["Sell"] = sell
            result["StrongSell"] = strong_sell
            result["TotalAnalysts"] = strong_buy + buy + hold + sell + strong_sell

            result["BuyCount"] = strong_buy + buy
            result["NeutralCount"] = hold
            result["SellCount"] = sell + strong_sell
    except Exception:
        pass
    
    try:
        result["RecommendationKey"] = ticker.info.get("recommendationKey")
    except Exception:
        pass

    return result


def get_valuation_metrics(symbol, ticker=None):
    """
    Core valuation ratios: trailing/forward P/E, PEG, EV/EBITDA, Price/FCF.
    Returns a dict; values default to None if unavailable for this ticker.
    Uses ticker.info for most fields (fast, single call), and derives
    Price/FCF from market cap + the cash flow statement when possible.
    """
    import yfinance as yf
    if ticker is None:
        ticker = yf.Ticker(symbol)

    result = {
        "TrailingPE": None, "ForwardPE": None, "PEG": None,
        "EVToEBITDA": None, "PriceToFCF": None,
    }

    try:
        info = ticker.info

        result["TrailingPE"] = info.get("trailingPE")
        result["ForwardPE"] = info.get("forwardPE")

        # yfinance sometimes exposes "trailingPegRatio" instead of "pegRatio"
        peg = info.get("pegRatio")
        if peg is None:
            peg = info.get("trailingPegRatio")
        result["PEG"] = peg

        result["EVToEBITDA"] = info.get("enterpriseToEbitda")

        market_cap = info.get("marketCap")
        if market_cap:
            try:
                cf = ticker.cashflow
                if cf is not None and not cf.empty:
                    if "Free Cash Flow" in cf.index:
                        fcf = cf.loc["Free Cash Flow"].iloc[0]
                    elif "Operating Cash Flow" in cf.index:
                        op_cash = cf.loc["Operating Cash Flow"].iloc[0]
                        capex_label = "Capital Expenditure" if "Capital Expenditure" in cf.index else (
                            "Capital Expenditures" if "Capital Expenditures" in cf.index else None
                        )
                        capex = cf.loc[capex_label].iloc[0] if capex_label else 0
                        fcf = op_cash + capex
                    else:
                        fcf = None

                    if fcf and fcf > 0:
                        result["PriceToFCF"] = round(market_cap / fcf, 2)
            except Exception:
                pass
    except Exception:
        pass

    for key in ("TrailingPE", "ForwardPE", "PEG", "EVToEBITDA"):
        if result[key] is not None:
            try:
                result[key] = round(float(result[key]), 2)
            except Exception:
                result[key] = None

    return result

