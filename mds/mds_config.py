"""
mds_config.py
=============
Single mutable CONFIG for the Multi-Day Swing (MDS) strategy. Every mds_* module
imports THIS dict (same pattern as constants.CONFIG) so a flag flipped in a
runner propagates everywhere.

Pillars -> rules (see mds_strategy.py for the implementation of each):
  1. No prediction: every gate is a *state* (trend is up, money is flowing in),
     never a forecast.
  2. Institutional money: anchored VWAP from the channel's swing low, up/down
     volume ratio, CMF, OBV slope, institutional ownership (info only).
  3. VWAP / POC / volume profile: composite 15m profile (POC/VAH/VAL) for
     location + stops/targets; session VWAP for the 15m trigger.
  4. Fundamentals: analyst Buy/Strong Buy, quality checks.
  5. Beta <= 2.   6. 0 < PEG < 4.
  7. Sector rotation: RRG-style RS-Ratio / RS-Momentum of SPDR sector ETFs vs SPY.
  8. SL / TP from support / resistance + volume-profile levels.
  9. Linear regression channel: slope > 0 and R^2 >= 0.5.
 10. Entry on the 15-minute timeframe.
"""

MDS_CONFIG = {
    # ------------------------------------------------------------- universe
    # Finviz prefilter (cheap, server-side). Exact thresholds are re-checked in
    # Python on yfinance data, because Finviz and Yahoo values differ a bit.
    "FINVIZ_FILTERS": {
        "Country": "USA",
        "Analyst Recom.": "Buy or better",
        "Beta": "Under 2",
        "Average Volume": "Over 500K",
        "Price": "Over $10",
        "P/E": "Profitable (>0)",
        "200-Day Simple Moving Average": "Price above SMA200",
    },
    "MARKET_CAP_OPTIONS": {
        0: "-Small (under $2bln)",
        1: "Mid ($2bln to $10bln)",
        2: "+Mid (over $2bln)",
        3: "+Micro (over $50mln)",
    },

    # ------------------------------------------------------------- data
    "MARKET_TZ": "America/New_York",
    "SESSION_OPEN": "09:30",
    "SESSION_CLOSE": "16:00",
    "DAILY_HISTORY_DAYS": 420,        # scanner: enough for 200-day warm-up + RRG
    "INTRADAY_HISTORY_DAYS": 45,      # scanner: 15m bars (yfinance keeps ~60d)
    "SCANNER_SOURCE": "yfinance",     # "yfinance" (near real-time) or "alpaca" (SIP, 15-min delayed on free plan)
    "ALPACA_SIP_BUFFER_MIN": 16,      # free plan: SIP bars must be >= 15 min old
    "CACHE_DIR": "cache/mds",

    # ------------------------------------------------------------- market regime
    "BENCHMARK": "SPY",
    "REQUIRE_MARKET_UPTREND": True,   # SPY close > SMA(200) -- no new longs in a bear market
    "MARKET_SMA": 200,

    # ------------------------------------------------------------- fundamentals (pillars 4-6)
    "ALLOWED_RECOMMENDATIONS": ("buy", "strong_buy"),
    "MIN_ANALYSTS": 5,                # thin coverage is a red flag
    "MAX_BETA": 2.0,
    "REQUIRE_BETA": True,             # missing beta -> fail
    "MAX_PEG": 4.0,
    "MIN_PEG": 0.0,                   # PEG <= 0 = negative growth -> fail
    "REQUIRE_PEG": True,              # missing PEG (and not computable) -> fail
    "MIN_QUALITY_CHECKS": 3,          # of 5: margin>0, revenue growth>0, FCF>0, EPS revisions up, D/E ok
    "MAX_DEBT_TO_EQUITY": 200.0,      # yfinance reports D/E in %, 200 = 2.0x
    "EARNINGS_BLACKOUT_DAYS": 15,     # skip if earnings fall inside the max hold window (calendar days ~ trading 10-11)

    # ------------------------------------------------------------- sector rotation (pillar 7)
    "SECTOR_ETFS": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC"],
    "RRG_RATIO_WINDOW": 50,           # RS-Ratio = 100 * RS / SMA(RS, window)
    "RRG_MOMENTUM_WINDOW": 10,        # RS-Momentum = 100 * RS-Ratio / RS-Ratio.shift(window)
    "RRG_SMOOTH": 5,                  # EMA smoothing on RS before the ratio (reduces daily noise)
    # Backtest 2019-2026 (daily, ~370 names): Improving +0.19R/trade, Leading -0.02R,
    # Weakening -0.04R -- stable in-sample and out-of-sample. So only Improving by default.
    "ALLOWED_QUADRANTS": ("Improving",),
    "REQUIRE_SECTOR_ROTATION": True,
    "UNKNOWN_SECTOR_PASSES": False,   # backtest: delisted names with no sector label

    # ------------------------------------------------------------- trend (pillars 1, 9)
    "LRC_LENGTH": 50,                 # daily bars in the regression window
    "LRC_DEV": 2.0,                   # channel half-width in residual std devs
    "LRC_MIN_R2": 0.0,                # backtest: R^2 >= 0.5 cut trades by 40% without raising R/trade -> off (slope > 0 kept)
    "LRC_MIN_SLOPE_PCT_DAY": 0.0,     # slope > 0 (expressed as % of price per day)
    "LRC_MAX_POSITION_PCT": 75,       # don't buy in the top quarter of the channel (0 = lower band, 100 = upper)
    "REQUIRE_EMA_STACK": False,       # EMA_FAST > EMA_SLOW and close > EMA_SLOW (backtest: no added value)
    "EMA_FAST": 21,
    "EMA_SLOW": 50,
    "REQUIRE_ADX": False,             # ADX >= ADX_MIN and +DI > -DI (backtest: high ADX was worse, not better)
    "ADX_PERIOD": 14,
    "ADX_MIN": 20,
    "ATR_PERIOD": 14,
    "RS_LOOKBACK": 63,                # stock vs SPY relative strength (~3 months), informational + ranking

    # ------------------------------------------------------------- institutional flow (pillar 2)
    "UDV_WINDOW": 50,                 # up-volume / down-volume ratio window
    "UDV_MIN": 1.0,
    "CMF_PERIOD": 20,
    "CMF_MIN": 0.0,
    "OBV_SLOPE_WINDOW": 20,
    "REQUIRE_ABOVE_AVWAP": True,      # price above VWAP anchored at the channel's lowest low
    "MIN_FLOW_CHECKS": 2,             # of 3: UDV, CMF, OBV slope

    # ------------------------------------------------------------- volume profile / levels (pillars 3, 8)
    "VP_SESSIONS": 20,                # composite profile over the last N sessions of 15m bars
    "VP_BINS": 50,
    "SR_FRACTAL_WINDOW": 3,           # daily fractal: bar is a swing if it's the extreme of 2n+1 bars
    "SR_CLUSTER_PCT": 0.015,
    "SR_LOOKBACK_DAYS": 120,
    "SR_NUM_LEVELS": 4,
    "STOP_ATR_CUSHION": 0.25,         # stop = support - cushion * ATR
    "MIN_STOP_ATR": 0.75,             # structure closer than this -> use the next level down
    "MAX_STOP_ATR": 3.0,              # structure further than this -> no trade (risk too wide)
    "MIN_RR": 2.0,                    # reward to TP1 / risk, checked at the actual 15m entry price
    "SETUP_MAX_PULLBACK_ATR": 1.0,    # if R:R at the close < MIN_RR, keep the setup only when the price giving
                                      # MIN_RR ("max_entry") is within this many ATR below the close
    "MIN_ENTRY_RISK_ATR": 0.5,        # entry must still be >= this far above the stop (no hair-trigger stops)
    "TP2_MIN_GAP_ATR": 1.0,           # TP2 = first resistance at least this far above TP1
    "MIN_TARGET_ATR": 1.0,            # ignore resistance closer than this (noise)
    "VP_LEVELS_AS_TARGETS": False,    # VAH/POC above price as targets? (POC below-value-area case always counts)

    # ------------------------------------------------------------- 15m entry (pillar 10)
    "SETUP_VALID_SESSIONS": 3,        # a daily setup can trigger during the next N sessions
    "ENTRY_START": "09:45",           # skip the opening 15m bar
    "ENTRY_END": "15:30",             # last bar that may trigger
    "ENTRY_EMA": 9,
    "ENTRY_MIN_RVOL": 1.0,            # 15m bar volume vs same time-of-day average
    "RVOL_LOOKBACK_SESSIONS": 10,
    "ENTRY_MAX_VWAP_SIGMA": 1.0,      # VWAP reclaim: don't chase above session VWAP + 1 sigma
    "LEVEL_TOUCH_ATR": 0.2,           # level bounce: bar low within 0.2 daily ATR of a daily support
    "BOUNCE_MAX_BELOW_VWAP_SIGMA": 1.0,  # level bounce: close no more than 1 sigma below session VWAP
    "MAX_GAP_ATR": 1.0,               # skip a session that gaps up > 1 ATR above the setup close

    # ------------------------------------------------------------- trade management
    "TP1_FRACTION": 0.5,              # sell this fraction at TP1 ...
    "MOVE_STOP_TO_BE_AT_TP1": True,   # ... and move the stop to breakeven
    "MAX_HOLD_SESSIONS": 15,
    "EXIT_ON_CLOSE_BELOW_LRC_LOWER": True,

    # ------------------------------------------------------------- portfolio (backtest)
    "STARTING_CAPITAL": 100_000,
    "RISK_PCT_PER_TRADE": 0.01,
    "MAX_POSITIONS": 8,
    "MAX_POSITION_PCT": 0.20,         # cap any single position at 20% of equity
    "MAX_PER_SECTOR": 3,
    "COST_BPS_PER_SIDE": 5,

    # ------------------------------------------------------------- output
    "REPORT_DIR": "reports",
    "LIVE_DIR": "live",
    "NTFY_ENABLED": True,
    "MAX_WORKERS": 8,
    "CHART_DAYS": 90,
}
