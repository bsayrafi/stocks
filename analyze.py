import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

INPUT_FILE = "/content/drive/MyDrive/stock_screener/market_data.pkl"

# --- Constants: Day indices ---
DAY3_IDX = -1  # most recent completed session
DAY2_IDX = DAY3_IDX - 1
DAY1_IDX = DAY3_IDX - 2
LOOKBACK_IDX = DAY3_IDX

# get premarket
PREMARKET = 0

# --- Constants: SMA / Golden Cross ---
SMA_FAST = 9
SMA_SLOW = 50

# --- Constants: MACD ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- Constants: ADX ---
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25  # above this = trend strong enough to trust signals

# --- Constants: Bollinger Bands ---
BB_PERIOD = 20
BB_NUM_STD = 2

# --- Constants: OBV / Volume ---
OBV_LOOKBACK = 5           # days to check OBV rising trend
VOLUME_SPIKE_MULTIPLIER = 1.5  # today's volume vs its own recent average
VOLUME_AVG_PERIOD = 20

# --- Constants: 52-week high/low ---
PCT_OFF_HIGH_THRESHOLD = 0.20  # D3 close must be this % below 52w high
NEAR_LOW_BOUNCE_THRESHOLD = 0.05  # D3 close within this % of 52w low = "near low"

WICK_THRESHOLD = 0.01  # max wick size as a % of price
BODY_THRESHOLD = 0.005  # min body size as a % of open price

# --- Constants: RSI / Stochastic RSI ---
STOCH_PERIOD = 14
SMOOTH_K = 3
SMOOTH_D = 3
OVERSOLD_LEVEL = 20


def calculate_stoch_rsi(close, rsi_period=14, stoch_period=STOCH_PERIOD,
                         smooth_k=SMOOTH_K, smooth_d=SMOOTH_D):
    rsi = calculate_rsi(close, period=rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_k = stoch_rsi.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(smooth_d).mean()
    return stoch_k, stoch_d


def is_stoch_rsi_buy_signal(df, oversold_level=OVERSOLD_LEVEL, require_oversold=True):
    """
    Returns 1 if %K crossed above %D on DAY3_IDX (vs DAY2_IDX), 0 otherwise.
    If require_oversold=True, the cross must also occur below oversold_level.
    """
    k_prev, k_curr = df['StochK'].iloc[DAY2_IDX], df['StochK'].iloc[DAY3_IDX]
    d_prev, d_curr = df['StochD'].iloc[DAY2_IDX], df['StochD'].iloc[DAY3_IDX]

    crossed_up = (k_prev <= d_prev) and (k_curr > d_curr)

    if require_oversold:
        in_oversold = (k_prev < oversold_level) or (d_prev < oversold_level)
        signal = crossed_up and in_oversold
    else:
        signal = crossed_up

    return 1 if signal else 0


def is_nice_green_candle(day, wick_threshold=WICK_THRESHOLD, body_threshold=BODY_THRESHOLD):
    """
    Returns 1 if the candle is a 'nice green candle':
      - Close > Open (bullish body)
      - Low is not far below Open (small lower wick)
      - High is not far above Close (small upper wick)
    Returns 0 otherwise.
    """
    o, h, l, c = day['Open'], day['High'], day['Low'], day['Close']

    is_bullish = c > o
    lower_wick_pct = (o - l) / o if o != 0 else float('inf')
    upper_wick_pct = (h - c) / c if c != 0 else float('inf')
    body_pct = (c - o) / o if o != 0 else 0

    is_nice = (
        is_bullish
        and lower_wick_pct <= wick_threshold
        and upper_wick_pct <= wick_threshold
        and body_pct >= body_threshold
    )

    return 1 if is_nice else 0


def calculate_rsi(close, period=14):
    """
    Calculate the 14-period RSI for a Close price series.
    Returns an RSI series aligned with the input.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(close, fast_period=SMA_FAST, slow_period=SMA_SLOW):
    """Calculate fast and slow SMAs. Returns (sma_fast, sma_slow) Series."""
    sma_fast = close.rolling(fast_period).mean()
    sma_slow = close.rolling(slow_period).mean()
    return sma_fast, sma_slow


def is_golden_cross(df):
    """Returns 1 if SMA_FAST crossed above SMA_SLOW on DAY3_IDX (vs DAY2_IDX)."""
    fast_prev, fast_curr = df['SMA9'].iloc[DAY2_IDX], df['SMA9'].iloc[DAY3_IDX]
    slow_prev, slow_curr = df['SMA50'].iloc[DAY2_IDX], df['SMA50'].iloc[DAY3_IDX]

    crossed_up = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
    return 1 if crossed_up else 0


def calculate_macd(close, fast_period=MACD_FAST, slow_period=MACD_SLOW, signal_period=MACD_SIGNAL):
    """Calculate MACD line, signal line, and histogram. Returns (macd_line, signal_line, histogram) Series."""
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def is_macd_buy_signal(df):
    """Returns 1 if MACD line crossed above signal line on DAY3_IDX (vs DAY2_IDX)."""
    macd_prev, macd_curr = df['MACD'].iloc[DAY2_IDX], df['MACD'].iloc[DAY3_IDX]
    signal_prev, signal_curr = df['MACDSignal'].iloc[DAY2_IDX], df['MACDSignal'].iloc[DAY3_IDX]

    crossed_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
    return 1 if crossed_up else 0


def calculate_adx(df, period=ADX_PERIOD):
    """Calculate ADX (trend strength) for a DataFrame with High/Low/Close columns. Returns a Series."""
    high, low, close = df['High'], df['Low'], df['Close']

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

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


def is_trending(df, threshold=ADX_TREND_THRESHOLD):
    """Returns 1 if ADX on DAY3_IDX is above threshold (trend strong enough to trust)."""
    adx_curr = df['ADX'].iloc[DAY3_IDX]
    return 1 if adx_curr > threshold else 0


def calculate_bollinger_bands(close, period=BB_PERIOD, num_std=BB_NUM_STD):
    """Calculate Bollinger Bands. Returns (upper_band, mid_band, lower_band) Series."""
    mid_band = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper_band = mid_band + (num_std * std)
    lower_band = mid_band - (num_std * std)
    return upper_band, mid_band, lower_band


def is_bollinger_bounce(df):
    """Returns 1 if D3 close bounced off/near the lower Bollinger Band."""
    close_curr = df['Close'].iloc[DAY3_IDX]
    lower_curr = df['BBLower'].iloc[DAY3_IDX]
    low_curr = df['Low'].iloc[DAY3_IDX]

    touched_lower = low_curr <= lower_curr
    closed_above_lower = close_curr > lower_curr

    return 1 if (touched_lower and closed_above_lower) else 0


def calculate_obv(close, volume):
    """Calculate On-Balance Volume. Returns a Series."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


def is_obv_rising(df, lookback=OBV_LOOKBACK):
    """Returns 1 if OBV has been rising over the last `lookback` days."""
    recent_obv = df['OBV'].iloc[-lookback:]
    return 1 if recent_obv.is_monotonic_increasing else 0


def is_volume_spike(df, multiplier=VOLUME_SPIKE_MULTIPLIER, avg_period=VOLUME_AVG_PERIOD):
    """Returns 1 if D3 volume is at least `multiplier`x the recent average volume."""
    avg_volume = df['Volume'].iloc[-avg_period:].mean()
    vol_curr = df['Volume'].iloc[DAY3_IDX]
    return 1 if vol_curr >= (multiplier * avg_volume) else 0


def is_near_52week_low_bounce(d3_close, week52_low, threshold=NEAR_LOW_BOUNCE_THRESHOLD):
    """Returns 1 if D3 close is within `threshold` % of the 52-week low."""
    pct_above_low = (d3_close - week52_low) / week52_low
    return 1 if pct_above_low <= threshold else 0


def evaluate_tickers(sp500_tickers, data, verbose_errors=True):
    """
    Evaluate each ticker across the last 3 completed sessions.
    Computes SMA9, SMA50, RSI, and checks for strictly declining volume.
    Returns a list of dicts, one per qualifying ticker.
    """
    results2 = []

    for symbol in sp500_tickers:
        try:
            df = data.xs(symbol, level=1, axis=1).dropna()
            if len(df) < 200:
                continue

            week52_high = df['Close'].max()
            week52_low = df['Close'].min()

            # --- Compute all indicator columns FIRST, before slicing d1/d2/d3 ---
            df['SMA9'], df['SMA50'] = calculate_sma(df['Close'])
            df['RSI'] = calculate_rsi(df['Close'])
            df['StochK'], df['StochD'] = calculate_stoch_rsi(df['Close'])
            df['MACD'], df['MACDSignal'], df['MACDHist'] = calculate_macd(df['Close'])
            df['ADX'] = calculate_adx(df)
            df['BBUpper'], df['BBMid'], df['BBLower'] = calculate_bollinger_bands(df['Close'])
            df['OBV'] = calculate_obv(df['Close'], df['Volume'])

            # --- THEN take the row snapshots, once everything exists ---
            d1, d2, d3 = df.iloc[LOOKBACK_IDX-2], df.iloc[LOOKBACK_IDX-1], df.iloc[LOOKBACK_IDX]

            # Condition 3: Strictly rising daily volume across the 3 days
            risingVol = (d3['Volume'] > d2['Volume']) and (d2['Volume'] > d1['Volume'])
            pre_price_str = "N/A"
            latest_price = None

            # Fetch current pre-market price using 1-minute extended hours data
            if PREMARKET == 1:
                try:
                    pre_df = yf.download(symbol, period="1d", interval="1m", prepost=True, progress=False, auto_adjust=False)
                    if not pre_df.empty:
                        latest_price = pre_df['Close'].iloc[-1]
                        if isinstance(latest_price, pd.Series):
                            latest_price = latest_price.iloc[0]
                        pre_price_str = f"{float(latest_price):.2f} "
                except Exception:
                    pass

            d1g = 0
            d2g = 0
            d3g = 0
            rsig = 0
            smadist = 0
            if d3['Close'] > d3['Open']:
                d3g = 1
            if d2['Close'] > d2['Open']:
                d2g = 1
            if d1['Close'] > d1['Open']:
                d1g = 1
            if d1['RSI'] < d2['RSI'] and d2['RSI'] < d3['RSI']:
                rsig = 1
            if d3['Close'] < d3['SMA50']:
                smadist = (d3['SMA50'] - d3['Close']) / d3['SMA50']

            # --- 20% off 52-week high filter ---
            pct_from_high = (week52_high - d3['Close']) / week52_high
            pct_from_low = (d3['Close'] - week52_low) / week52_low

            c1 = is_nice_green_candle(d1)
            c2 = is_nice_green_candle(d2)
            c3 = is_nice_green_candle(d3)

            # --- Signals ---
            stoch_buy_signal = is_stoch_rsi_buy_signal(df)
            golden_cross = is_golden_cross(df)
            macd_buy_signal = is_macd_buy_signal(df)
            trending = is_trending(df)
            bollinger_bounce = is_bollinger_bounce(df)
            obv_rising = is_obv_rising(df)
            volume_spike = is_volume_spike(df)
            near_low_bounce = is_near_52week_low_bounce(d3['Close'], week52_low)

            if PREMARKET == 0 or (PREMARKET == 1 and latest_price is not None and latest_price > d3['Close']):
                aboveSMA = 0
                if d3['SMA50'] < d3['Close']:
                    aboveSMA = 1
                results2.append({
                    "Ticker ": symbol,
                    "D1Open ": f"{d1['Open']:.2f} ",
                    "D1Close ": f"{d1['Close']:.2f} ",
                    "D2Open ": f"{d2['Open']:.2f} ",
                    "D2Close ": f"{d2['Close']:.2f} ",
                    "D3Open ": f"{d3['Open']:.2f} ",
                    "D3Close ": f"{d3['Close']:.2f} ",
                    **({"Pre-Market": pre_price_str} if PREMARKET == 1 else {}),
                    "50SMA ": f"{d3['SMA50']:.2f} ",
                    "G1 ": f"{d1g:.0f} ",
                    "G2 ": f"{d2g:.0f} ",
                    "G3 ": f"{d3g:.0f} ",
                    "C1": c1,
                    "C2": c2,
                    "C3": c3,
                    "AboveSMA ": f"{aboveSMA:.0f} ",
                    "52WLow ": f"{week52_low:.2f} ",
                    "52WHigh ": f"{week52_high:.2f} ",
                    "RisingVol ": f"{risingVol:.0f} ",
                    "RiseRSI ": f"{rsig:.0f} ",
                    "RSI1 ": f"{round(float(d1['RSI']), 1)} ",
                    "RSI2 ": f"{round(float(d2['RSI']), 1)} ",
                    "RSI3 ": f"{round(float(d3['RSI']), 1)} ",
                    "Stoch ": f"{stoch_buy_signal} ",
                    "StochK": round(float(df['StochK'].iloc[DAY3_IDX]), 1),
                    "StochD": round(float(df['StochD'].iloc[DAY3_IDX]), 1),
                    "GoldenCross": golden_cross,
                    "MACDBuy": macd_buy_signal,
                    "ADX": round(float(df['ADX'].iloc[DAY3_IDX]), 1),
                    "Trending": trending,
                    "Bollinger": bollinger_bounce,
                    "OBVRising": obv_rising,
                    "VolumeSpike": volume_spike,
                    "Near52WLow": near_low_bounce
                })

        except Exception as e:
            if verbose_errors:
                print(f"{symbol}: skipped ({e})")
            continue

    return results2


def main():
    with open(INPUT_FILE, "rb") as f:
        cached = pickle.load(f)

    sp500_tickers = cached["tickers"]
    data = cached["data"]

    results2 = evaluate_tickers(sp500_tickers, data)

    results_df2 = pd.DataFrame(results2)
    if not results_df2.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(results_df2.to_string(index=False))
    else:
        print("Error!")


if __name__ == "__main__":
    main()
