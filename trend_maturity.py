"""
trend_maturity.py
=====================
Answers "has this trend just started, or has it been running a while" --
INFORMATIONAL ONLY. This is deliberately NOT wired in as an entry filter.

Why not a filter: I tested it the same way the ADX/RSI entry filters were
validated (see trend_tsm_backtest.py's SIM_CONFIG comment) -- bucketing all 529
historical trades with usable data by these three maturity measures and
checking mean return per bucket. Unlike ADX/RSI, there's no clean relationship:
correlation with trade return is ~0 for two of the three measures (-0.03, -0.01)
and only weakly POSITIVE for the third (+0.11, horizon_skew) -- if anything,
entries near a 52-week high (the "most mature" bucket) drove the majority of
this strategy's total historical profit. Likely reason: this is a momentum
strategy built on a 250-day lookback, so it structurally tends to confirm
entries into moves that have already run for a while, and in liquid large caps
"already strong" often keeps being strong -- filtering that out would likely
cut good trades along with any genuinely stale ones, the same trap the
stop-loss test fell into. So: shown for awareness, not used to gate entries.

Three measures, all causal (no lookahead -- computed from bars up to and
including the row's own date, same convention as build_daily_trend_context):
  - days_above_ema200: consecutive days close has stayed above its 200-day EMA.
    0 means price is currently below its 200d average even though the trend
    gate fired -- can happen since the gate is vol-adjusted multi-horizon
    momentum, not a literal moving-average test.
  - pct_in_252d_range: where today's close sits within the past year's
    high-low range. 0.0 = at the 252-day low, 1.0 = at the 252-day high.
  - horizon_skew: long-horizon (250d) vol-adjusted momentum minus short-horizon
    (20d). Positive & large = most of the move already happened over the past
    year (long horizon dominates); near/below zero = the move is recent and
    the long lookback hasn't caught up yet.
"""

import numpy as np
import pandas as pd


def compute_maturity(df):
    """df: a symbol's daily bars with at least open/high/low/close and an
    'ema200' column (as produced by trend_data_pipeline.add_indicators).
    Returns df with three new columns added; does not mutate the input."""
    df = df.copy()

    above200 = df["close"] > df["ema200"]
    grp = (above200 != above200.shift()).cumsum()
    streak = above200.groupby(grp).cumcount() + 1
    df["days_above_ema200"] = np.where(above200, streak, 0)

    roll_low = df["close"].rolling(252, min_periods=60).min()
    roll_high = df["close"].rolling(252, min_periods=60).max()
    df["pct_in_252d_range"] = (df["close"] - roll_low) / (roll_high - roll_low).replace(0, np.nan)

    ret20 = df["close"].pct_change(20)
    vol20 = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
    ret250 = df["close"].pct_change(250)
    vol250 = df["close"].pct_change().rolling(250).std() * np.sqrt(252)
    df["horizon_skew"] = (ret250 / vol250.replace(0, np.nan)) - (ret20 / vol20.replace(0, np.nan))

    return df


def describe_stage(days_above_ema200, pct_in_252d_range):
    """One human-readable label for the digest. Purely descriptive -- see
    module docstring for why this isn't used to accept/reject a trade."""
    if pd.isna(days_above_ema200) or days_above_ema200 == 0:
        age_label = "below its 200d avg"
    elif days_above_ema200 <= 5:
        age_label = "just started"
    elif days_above_ema200 <= 20:
        age_label = "early"
    elif days_above_ema200 <= 60:
        age_label = "established"
    else:
        age_label = f"running {int(days_above_ema200)}d"

    range_note = ""
    if pd.notna(pct_in_252d_range):
        if pct_in_252d_range >= 0.95:
            range_note = ", near 52w high"
        elif pct_in_252d_range <= 0.4:
            range_note = ", well off highs"

    return f"{age_label}{range_note}"
