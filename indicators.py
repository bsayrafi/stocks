
import pandas as pd
import numpy as np
from constants import CONFIG, DAYm1_IDX, DAYm2_IDX, DAYm3_IDX

   
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stoch_rsi(close, rsi_period=14, stoch_period=CONFIG["STOCH_PERIOD"], smooth_k=CONFIG["SMOOTH_K"], smooth_d=CONFIG["SMOOTH_D"]):
    rsi = calculate_rsi(close, period=rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_k = stoch_rsi.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(smooth_d).mean()
    return stoch_k, stoch_d


def is_stoch_rsi_buy_signal(df, oversold_level=CONFIG["OVERSOLD_LEVEL"], require_oversold=True):
    k_prev, k_curr = df['StochK'].iloc[DAYm2_IDX], df['StochK'].iloc[DAYm1_IDX]
    d_prev, d_curr = df['StochD'].iloc[DAYm2_IDX], df['StochD'].iloc[DAYm1_IDX]
    crossed_up = (k_prev <= d_prev) and (k_curr > d_curr)
    if require_oversold:
        in_oversold = (k_prev < oversold_level) or (d_prev < oversold_level)
        signal = crossed_up and in_oversold
    else:
        signal = crossed_up
    return 1 if signal else 0


def is_nice_green_candle(day, wick_threshold=CONFIG["WICK_THRESHOLD"], body_threshold=CONFIG["BODY_THRESHOLD"]):
    o, h, l, c = day['Open'], day['High'], day['Low'], day['Close']
    is_bullish = c > o
    lower_wick_pct = (o - l) / o if o != 0 else float('inf')
    upper_wick_pct = (h - c) / c if c != 0 else float('inf')
    body_pct = (c - o) / o if o != 0 else 0
    is_nice = (is_bullish and lower_wick_pct <= wick_threshold and upper_wick_pct <= wick_threshold and body_pct >= body_threshold)
    return 1 if is_nice else 0


def calculate_sma(close, fast_period=CONFIG["SMA_FAST"], slow_period=CONFIG["SMA_SLOW"]):
    sma_fast = close.rolling(fast_period).mean()
    sma_slow = close.rolling(slow_period).mean()
    return sma_fast, sma_slow


def is_golden_cross(df):
    fast_prev, fast_curr = df['SMA9'].iloc[DAYm2_IDX], df['SMA9'].iloc[DAYm1_IDX]
    slow_prev, slow_curr = df['SMA50'].iloc[DAYm2_IDX], df['SMA50'].iloc[DAYm1_IDX]
    crossed_up = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
    return 1 if crossed_up else 0


def calculate_macd(close, fast_period=CONFIG["MACD_FAST"], slow_period=CONFIG["MACD_SLOW"], signal_period=CONFIG["MACD_SIGNAL"]):
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def is_macd_buy_signal(df):
    macd_prev, macd_curr = df['MACD'].iloc[DAYm2_IDX], df['MACD'].iloc[DAYm1_IDX]
    signal_prev, signal_curr = df['MACDSignal'].iloc[DAYm2_IDX], df['MACDSignal'].iloc[DAYm1_IDX]
    crossed_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
    return 1 if crossed_up else 0


def calculate_adx(df, period=CONFIG["ADX_PERIOD"]):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx


def is_trending(df, threshold=CONFIG["ADX_TREND_THRESHOLD"]):
    adx_curr = df['ADX'].iloc[DAYm1_IDX]
    return 1 if adx_curr > threshold else 0


def calculate_bollinger_bands(close, period=CONFIG["BB_PERIOD"], num_std=CONFIG["BB_NUM_STD"]):
    mid_band = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper_band = mid_band + (num_std * std)
    lower_band = mid_band - (num_std * std)
    return upper_band, mid_band, lower_band


def is_bollinger_bounce(df):
    close_curr = df['Close'].iloc[DAYm1_IDX]
    lower_curr = df['BBLower'].iloc[DAYm1_IDX]
    low_curr = df['Low'].iloc[DAYm1_IDX]
    touched_lower = low_curr <= lower_curr
    closed_above_lower = close_curr > lower_curr
    return 1 if (touched_lower and closed_above_lower) else 0


def calculate_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


def is_obv_rising(df, lookback=CONFIG["OBV_LOOKBACK"]):
    recent_obv = df['OBV'].iloc[-lookback:]
    return 1 if recent_obv.is_monotonic_increasing else 0


def is_volume_spike(df, multiplier=CONFIG["VOLUME_SPIKE_MULTIPLIER"], avg_period=CONFIG["VOLUME_AVG_PERIOD"]):
    avg_volume = df['Volume'].iloc[-avg_period:].mean()
    vol_curr = df['Volume'].iloc[DAYm1_IDX]
    return 1 if vol_curr >= (multiplier * avg_volume) else 0


def is_near_52week_low_bounce(d3_close, week52_low, threshold=CONFIG["NEAR_LOW_BOUNCE_THRESHOLD"]):
    pct_above_low = (d3_close - week52_low) / week52_low
    return 1 if pct_above_low <= threshold else 0


def calculate_cmf(df, period=20):
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
    mfv = mfm * df['Volume']
    cmf = mfv.rolling(period).sum() / df['Volume'].rolling(period).sum()
    return cmf

def is_cmf_buy_signal(df, period=20, threshold=0.05):
    cmf = calculate_cmf(df, period)
    # crossed from negative/neutral into meaningful positive territory
    return int(cmf.iloc[-2] < threshold and cmf.iloc[-1] >= threshold)


def is_bullish_divergence(df, lookback=20):
    window = df.iloc[-lookback:]
    price_low_idx = window['Close'].idxmin()
    rsi_at_price_low = window.loc[price_low_idx, 'RSI']
    # find the most recent local low after that point
    later = window.loc[price_low_idx:]
    if len(later) < 3:
        return 0
    recent_low_idx = later['Close'].iloc[1:].idxmin()
    recent_close = later.loc[recent_low_idx, 'Close']
    recent_rsi = later.loc[recent_low_idx, 'RSI']
    price_low = window.loc[price_low_idx, 'Close']
    return int(recent_close <= price_low and recent_rsi > rsi_at_price_low)

def is_trend_pullback_buy(df):
    last = df.iloc[-1]
    adx_rising = df['ADX'].iloc[-1] > df['ADX'].iloc[-5]
    near_sma = abs(last['Close'] - last['SMA50']) / last['SMA50'] < 0.02
    uptrend = last['SMA50'] > df['SMA50'].iloc[-20]
    return int(adx_rising and near_sma and uptrend and last['ADX'] > 20)

def is_bearish_divergence(df, lookback=20):
    window = df.iloc[-lookback:]
    price_high_idx = window['Close'].idxmax()
    rsi_at_price_high = window.loc[price_high_idx, 'RSI']
    later = window.loc[price_high_idx:]
    if len(later) < 3:
        return 0
    recent_high_idx = later['Close'].iloc[1:].idxmax()
    recent_close = later.loc[recent_high_idx, 'Close']
    recent_rsi = later.loc[recent_high_idx, 'RSI']
    price_high = window.loc[price_high_idx, 'Close']
    return int(recent_close >= price_high and recent_rsi < rsi_at_price_high)

def is_death_cross(df):
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    return int(sma50.iloc[-2] > sma200.iloc[-2] and sma50.iloc[-1] <= sma200.iloc[-1])

def is_exhaustion_sell(df):
    last = df.iloc[-1]
    overbought = last['RSI'] > 70 or df['StochK'].iloc[-1] > 80
    red_candle = last['Close'] < last['Open']
    closed_near_low = (last['Close'] - last['Low']) / (last['High'] - last['Low'] + 1e-9) < 0.25
    return int(overbought and red_candle and closed_near_low)

def calculate_support_resistance(df, fractal_window=CONFIG["SR_FRACTAL_WINDOW"],
                                  cluster_pct=CONFIG["SR_CLUSTER_PCT"],
                                  lookback_days=CONFIG["SR_LOOKBACK_DAYS"],
                                  num_levels=CONFIG["SR_NUM_LEVELS"]):
    """
    Standard fractal-based support/resistance for short-term trading.

    A bar is a resistance point if its High is the highest in the window
    around it; a support point if its Low is the lowest. Nearby points are
    merged into single levels (average price), each with a touch count.

    Returns:
        {
            "resistance": [price, price, price],  # nearest 3 above current price
            "support":    [price, price, price],  # nearest 3 below current price
        }
    """
    recent = df.iloc[-lookback_days:] if len(df) > lookback_days else df
    highs, lows, closes = recent['High'], recent['Low'], recent['Close']
    current_price = closes.iloc[-1]

    swing_highs, swing_lows = [], []
    n = fractal_window

    for i in range(n, len(recent) - n):
        if highs.iloc[i] == highs.iloc[i - n:i + n + 1].max():
            swing_highs.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i - n:i + n + 1].min():
            swing_lows.append(lows.iloc[i])

    def cluster(levels, pct):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            avg = sum(clusters[-1]) / len(clusters[-1])
            if abs(lvl - avg) / avg <= pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        return [round(sum(c) / len(c), 2) for c in clusters]

    resistance_levels = sorted(l for l in cluster(swing_highs, cluster_pct) if l > current_price)
    support_levels = sorted((l for l in cluster(swing_lows, cluster_pct) if l < current_price), reverse=True)

    return {
        "resistance": resistance_levels[:num_levels],
        "support": support_levels[:num_levels],
    }





def compute_signals(df, days, DAYm1_IDX, week52_low):
    """Call this once per symbol, after df has SMA9/SMA50/RSI/StochK/StochD/
    MACD/MACDSignal/ADX/BBLower/OBV columns populated."""
    return {
        # --- buy signals ---
        "StochBuy":       is_stoch_rsi_buy_signal(df),
        "GoldenCross":     is_golden_cross(df),
        "MACDBuy":         is_macd_buy_signal(df),
        "Trending":        is_trending(df),
        "Bollinger":       is_bollinger_bounce(df),
        "OBVRising":       is_obv_rising(df),
        "VolumeSpike":     is_volume_spike(df),
        "Near52WLow":      is_near_52week_low_bounce(days[DAYm1_IDX]['Close'], week52_low),
        "NiceGreenCandle": is_nice_green_candle(days[DAYm1_IDX]),
        "CMFBuy":          is_cmf_buy_signal(df),
        "BullDiv":         is_bullish_divergence(df),
        "TrendPullback":   is_trend_pullback_buy(df),
        # --- sell signals ---
        "BearDiv":         is_bearish_divergence(df),
        "DeathCross":      is_death_cross(df),
        "ExhaustionSell":  is_exhaustion_sell(df),
    }


def buy_score(sig):
    """Weighted score out of 10. Trend/volume-confirmation signals count
    more than single-indicator momentum blips."""
    weights = {
        "GoldenCross": 2, "TrendPullback": 2, "CMFBuy": 2,
        "MACDBuy": 1, "StochBuy": 1, "BullDiv": 1,
        "OBVRising": 1, "Bollinger": 1, "VolumeSpike": 1,
        "Near52WLow": 1, "NiceGreenCandle": 0.5, "Trending": 0.5,
    }
    return round(sum(weights[k] * sig[k] for k in weights), 1)


def sell_score(sig):
    """Weighted score out of 10. Any one of these firing alone is already
    worth attention -- these are lower-frequency, higher-signal events."""
    weights = {"DeathCross": 4, "ExhaustionSell": 4, "BearDiv": 2}
    return round(sum(weights[k] * sig[k] for k in weights), 1)


def is_high_quality_buy(sig):
    """Confluence filter: require agreement across trend + momentum + volume,
    not just a raw score, since several signals are correlated (e.g. GoldenCross
    and MACDBuy both just say 'trend turned up')."""
    trend_ok = sig["GoldenCross"] or sig["TrendPullback"] or sig["Trending"]
    momentum_ok = sig["MACDBuy"] or sig["StochBuy"] or sig["BullDiv"]
    volume_ok = sig["OBVRising"] or sig["CMFBuy"] or sig["VolumeSpike"]
    no_active_sell = not (sig["DeathCross"] or sig["ExhaustionSell"])
    return int(trend_ok and momentum_ok and volume_ok and no_active_sell)
