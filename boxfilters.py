"""
filters.py
----------
Pre-breakout filtering pipeline: liquidity, trend context, volatility
contraction, and box structure detection.

The goal of every function here is to answer one question:
"Is this instrument's recent action clean enough that a box breakout
signal can be trusted?" -- as opposed to being low-volume noise or an
untrendy chop.

Each filter returns a pandas.Series (or DataFrame) aligned to the
input index so results can be inspected bar-by-bar, not just as a
single pass/fail for "today."
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from boxindicators import (
    _normalize_columns,
    atr,
    atr_percent,
    adx,
    choppiness_index,
    bollinger_band_width,
    relative_volume,
    dollar_volume,
)


# ---------------------------------------------------------------------
# 1. Liquidity filters
# ---------------------------------------------------------------------

def liquidity_filter(
    df: pd.DataFrame,
    min_price: float = 5.0,
    min_avg_volume: float = 500_000,
    min_avg_dollar_volume: float = 10_000_000,
    lookback: int = 20,
) -> pd.Series:
    """
    Boolean series: True where the instrument is liquid enough to trust
    its price action (avoids thin/penny-stock noise).
    """
    df = _normalize_columns(df)
    avg_vol = df["Volume"].rolling(lookback).mean()
    avg_dv = dollar_volume(df, lookback)

    passes = (
        (df["Close"] >= min_price)
        & (avg_vol >= min_avg_volume)
        & (avg_dv >= min_avg_dollar_volume)
    )
    passes.name = "LIQUIDITY_OK"
    return passes


# ---------------------------------------------------------------------
# 2. Trend context filters
# ---------------------------------------------------------------------

def trend_filter(
    df: pd.DataFrame,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    ma_period: int = 50,
    require_rising_ma: bool = True,
    ma_slope_lookback: int = 5,
) -> pd.Series:
    """
    Boolean series: True where there is a real underlying trend
    (as opposed to a directionless, choppy tape).

    Requires ADX above `adx_threshold`, and optionally a rising
    moving average (price structurally trending up, not just volatile).
    """
    df = _normalize_columns(df)
    adx_df = adx(df, adx_period)
    ma = df["Close"].rolling(ma_period).mean()

    trending = adx_df["ADX"] >= adx_threshold

    if require_rising_ma:
        ma_slope = ma.diff(ma_slope_lookback)
        rising = ma_slope > 0
        trending = trending & rising

    trending.name = "TREND_OK"
    return trending


def choppiness_filter(df: pd.DataFrame, period: int = 14, max_chop: float = 61.0) -> pd.Series:
    """
    Boolean series: True where the Choppiness Index is BELOW `max_chop`,
    i.e. the market is NOT in a choppy/ranging state right now.

    Named NOT_CHOPPY (rather than "CHOP_OK") deliberately -- True means
    "chop is absent," so there's no inversion to remember when reading
    the column. Use this as a complement (or alternative) to
    trend_filter's ADX check.
    """
    chop = choppiness_index(df, period)
    ok = chop < max_chop
    ok.name = "NOT_CHOPPY"
    return ok


# ---------------------------------------------------------------------
# 3. Volatility contraction (consolidation quality)
# ---------------------------------------------------------------------

def volatility_contraction_filter(
    df: pd.DataFrame,
    bb_period: int = 20,
    lookback: int = 100,
    percentile_threshold: float = 30.0,
) -> pd.Series:
    """
    Boolean series: True where the current Bollinger Band width sits in
    the bottom `percentile_threshold` percentile of its own trailing
    `lookback` history -- i.e. the stock is currently unusually tight
    (a real consolidation), not just chopping around with normal noise.
    """
    bb = bollinger_band_width(df, bb_period)
    width = bb["BB_WIDTH"]

    def _pct_rank(window: pd.Series) -> float:
        current = window.iloc[-1]
        return (window <= current).mean() * 100

    pct_rank = width.rolling(lookback).apply(_pct_rank, raw=False)
    ok = pct_rank <= percentile_threshold
    ok.name = "VOL_CONTRACTION_OK"
    return ok


# ---------------------------------------------------------------------
# 4. Box structure detection
# ---------------------------------------------------------------------

def detect_box(
    df: pd.DataFrame,
    window: int = 20,
    touch_tolerance_pct: float = 2.0,
    min_touches: int = 2,
    min_box_height_atr: float = 1.5,
    max_box_height_atr: float = 6.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    For each bar, look back `window` days and identify a candidate box
    (top = rolling high, bottom = rolling low over that window), then
    check whether the box is a *real* consolidation:

      - price touched near the top at least `min_touches` times
      - price touched near the bottom at least `min_touches` times
      - box height is a reasonable multiple of ATR (not too tight to be
        noise, not too wide to be a real trading range)

    Returns a DataFrame with columns:
        'BOX_TOP', 'BOX_BOTTOM', 'BOX_HEIGHT_ATR',
        'TOP_TOUCHES', 'BOTTOM_TOUCHES', 'BOX_VALID'
    """
    df = _normalize_columns(df)
    a = atr(df, atr_period)

    box_top = df["High"].rolling(window).max()
    box_bottom = df["Low"].rolling(window).min()
    box_height = box_top - box_bottom
    box_height_atr = box_height / a

    top_touches = pd.Series(index=df.index, dtype=float)
    bottom_touches = pd.Series(index=df.index, dtype=float)

    highs = df["High"].values
    lows = df["Low"].values
    tops = box_top.values
    bottoms = box_bottom.values

    n = len(df)
    for i in range(n):
        if i < window - 1 or np.isnan(tops[i]) or np.isnan(bottoms[i]):
            top_touches.iloc[i] = np.nan
            bottom_touches.iloc[i] = np.nan
            continue
        start = i - window + 1
        seg_high = highs[start : i + 1]
        seg_low = lows[start : i + 1]

        top_band = tops[i] * (touch_tolerance_pct / 100.0)
        bottom_band = bottoms[i] * (touch_tolerance_pct / 100.0)

        top_touches.iloc[i] = np.sum(seg_high >= (tops[i] - top_band))
        bottom_touches.iloc[i] = np.sum(seg_low <= (bottoms[i] + bottom_band))

    box_valid = (
        (top_touches >= min_touches)
        & (bottom_touches >= min_touches)
        & (box_height_atr >= min_box_height_atr)
        & (box_height_atr <= max_box_height_atr)
    )

    out = pd.DataFrame({
        "BOX_TOP": box_top,
        "BOX_BOTTOM": box_bottom,
        "BOX_HEIGHT_ATR": box_height_atr,
        "TOP_TOUCHES": top_touches,
        "BOTTOM_TOUCHES": bottom_touches,
        "BOX_VALID": box_valid,
    })
    return out


# ---------------------------------------------------------------------
# 5. Breakout confirmation
# ---------------------------------------------------------------------

def breakout_confirmation(
    df: pd.DataFrame,
    box_df: pd.DataFrame,
    breakout_pct: float = 0.0,
    rel_vol_threshold: float = 1.5,
    rel_vol_period: int = 50,
) -> pd.Series:
    """
    Boolean series: True on bars where price closes above the prior
    bar's box top by at least `breakout_pct`, AND volume is at least
    `rel_vol_threshold` times the trailing average (relative volume).

    `box_df` should be the output of detect_box(), already aligned to df.
    Uses the *previous* bar's box (shift(1)) so the breakout is measured
    against a box that existed before today's bar, not one that includes it.
    """
    df = _normalize_columns(df)
    prior_box_top = box_df["BOX_TOP"].shift(1)
    prior_box_valid = box_df["BOX_VALID"].shift(1)

    price_break = df["Close"] > prior_box_top * (1 + breakout_pct / 100.0)
    rel_vol = relative_volume(df, rel_vol_period)
    volume_confirmed = rel_vol >= rel_vol_threshold

    breakout = price_break & volume_confirmed & prior_box_valid.fillna(False)
    breakout.name = "BREAKOUT_CONFIRMED"
    return breakout


# ---------------------------------------------------------------------
# 6. Combined pipeline
# ---------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    *,
    min_price: float = 5.0,
    min_avg_volume: float = 500_000,
    min_avg_dollar_volume: float = 10_000_000,
    liquidity_lookback: int = 20,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    ma_period: int = 50,
    max_chop: float = 61.0,
    bb_period: int = 20,
    vol_contraction_lookback: int = 100,
    vol_contraction_percentile: float = 30.0,
    box_window: int = 20,
    touch_tolerance_pct: float = 2.0,
    min_touches: int = 2,
    min_box_height_atr: float = 1.5,
    max_box_height_atr: float = 6.0,
    atr_period: int = 14,
    breakout_pct: float = 0.0,
    rel_vol_threshold: float = 1.5,
    rel_vol_period: int = 50,
) -> pd.DataFrame:
    """
    Run the full filter pipeline and return one combined DataFrame,
    bar-by-bar, with every intermediate flag plus the final
    'SIGNAL' column (True only when every upstream condition holds).

    Order of evaluation mirrors the write-up:
      liquidity -> trend/choppiness -> volatility contraction ->
      box validity -> breakout confirmation
    """
    df = _normalize_columns(df)

    liq_ok = liquidity_filter(df, min_price, min_avg_volume, min_avg_dollar_volume, liquidity_lookback)
    trend_ok = trend_filter(df, adx_period, adx_threshold, ma_period)
    not_choppy = choppiness_filter(df, adx_period, max_chop)
    chop_value = choppiness_index(df, adx_period)
    vol_contraction_ok = volatility_contraction_filter(
        df, bb_period, vol_contraction_lookback, vol_contraction_percentile
    )
    box_df = detect_box(
        df, box_window, touch_tolerance_pct, min_touches,
        min_box_height_atr, max_box_height_atr, atr_period,
    )
    breakout_ok = breakout_confirmation(
        df, box_df, breakout_pct, rel_vol_threshold, rel_vol_period
    )

    # Same trailing-average-volume calc used inside relative_volume(), broken
    # out here so callers can see the actual share count a breakout would
    # need, not just the pass/fail boolean.
    avg_volume = df["Volume"].shift(1).rolling(rel_vol_period).mean()
    required_breakout_volume = avg_volume * rel_vol_threshold

    # Distance from today's close to the CURRENT box top/bottom, as a
    # percentage move -- how far price still has to travel to break out
    # (positive) or break down (negative distance to the bottom).
    pct_to_box_top = (box_df["BOX_TOP"] - df["Close"]) / df["Close"] * 100
    pct_to_box_bottom = (df["Close"] - box_df["BOX_BOTTOM"]) / df["Close"] * 100

    result = pd.DataFrame({
        "Close": df["Close"],
        "Volume": df["Volume"],
        "AVG_VOLUME_50D": avg_volume,
        "REQUIRED_BREAKOUT_VOLUME": required_breakout_volume,
        "LIQUIDITY_OK": liq_ok,
        "TREND_OK": trend_ok,
        "NOT_CHOPPY": not_choppy,
        "CHOP_VALUE": chop_value,
        "VOL_CONTRACTION_OK": vol_contraction_ok,
        "BOX_TOP": box_df["BOX_TOP"],
        "BOX_BOTTOM": box_df["BOX_BOTTOM"],
        "PCT_TO_BOX_TOP": pct_to_box_top,
        "PCT_ABOVE_BOX_BOTTOM": pct_to_box_bottom,
        "BOX_HEIGHT_ATR": box_df["BOX_HEIGHT_ATR"],
        "TOP_TOUCHES": box_df["TOP_TOUCHES"],
        "BOTTOM_TOUCHES": box_df["BOTTOM_TOUCHES"],
        "BOX_VALID": box_df["BOX_VALID"],
        "BREAKOUT_CONFIRMED": breakout_ok,
    })

    result["SIGNAL"] = (
        result["LIQUIDITY_OK"]
        & result["TREND_OK"]
        & result["NOT_CHOPPY"]
        & result["BOX_VALID"]
        & result["BREAKOUT_CONFIRMED"]
    )
    return result
