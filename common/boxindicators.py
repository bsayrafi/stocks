"""
indicators.py
-------------
Standalone technical indicator calculations used to detect low-volume
and choppy/range-bound conditions before applying a breakout system
(e.g. Darvas Box Theory).

All functions take a pandas.DataFrame with at least these columns:
    'Open', 'High', 'Low', 'Close', 'Volume'
(case-insensitive column names are normalized on import).

No external dependencies beyond pandas / numpy.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard capitalized OHLCV column names."""
    rename_map = {c: c.strip().capitalize() for c in df.columns}
    df = df.rename(columns=rename_map)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    return df


def true_range(df: pd.DataFrame) -> pd.Series:
    """Classic True Range (Wilder)."""
    df = _normalize_columns(df)
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.name = "TR"
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (RMA)."""
    tr = true_range(df)
    atr_series = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    atr_series.name = f"ATR_{period}"
    return atr_series


def atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR expressed as a percentage of closing price. Useful for
    normalizing volatility across stocks of different price levels."""
    df = _normalize_columns(df)
    a = atr(df, period)
    pct = (a / df["Close"]) * 100
    pct.name = f"ATR_PCT_{period}"
    return pct


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (Wilder).

    Returns a DataFrame with columns: '+DI', '-DI', 'ADX'.
    ADX below ~20-25 generally signals a weak/non-trending (choppy) market.
    """
    df = _normalize_columns(df)
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_smooth = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smooth)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx_series = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    out = pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx_series})
    return out


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Choppiness Index (CHOP), bounded 0-100.
    > ~61  -> choppy / ranging market
    < ~38  -> trending market
    """
    df = _normalize_columns(df)
    tr = true_range(df)
    tr_sum = tr.rolling(period).sum()

    high_max = df["High"].rolling(period).max()
    low_min = df["Low"].rolling(period).min()
    rng = (high_max - low_min).replace(0, np.nan)

    chop = 100 * np.log10(tr_sum / rng) / np.log10(period)
    chop.name = f"CHOP_{period}"
    return chop


def bollinger_band_width(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands and normalized band width.

    Returns DataFrame with 'BB_MID', 'BB_UPPER', 'BB_LOWER', 'BB_WIDTH'
    where BB_WIDTH = (upper - lower) / mid  (a volatility-contraction measure).
    """
    df = _normalize_columns(df)
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid

    return pd.DataFrame({
        "BB_MID": mid,
        "BB_UPPER": upper,
        "BB_LOWER": lower,
        "BB_WIDTH": width,
    })


def relative_volume(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Today's volume divided by the rolling average volume over `period`
    (average excludes the current bar to avoid self-inflation).
    """
    df = _normalize_columns(df)
    avg_vol = df["Volume"].shift(1).rolling(period).mean()
    rel_vol = df["Volume"] / avg_vol
    rel_vol.name = f"REL_VOL_{period}"
    return rel_vol


def dollar_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling average dollar volume (Close * Volume) over `period` days."""
    df = _normalize_columns(df)
    dv = df["Close"] * df["Volume"]
    avg_dv = dv.rolling(period).mean()
    avg_dv.name = f"AVG_DOLLAR_VOL_{period}"
    return avg_dv
