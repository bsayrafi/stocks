"""
macro_calendar.py
====================
Lightweight lookup for scheduled macro-economic events (FOMC decisions,
and anything else you add) so the pipeline and live script can flag or
filter FVGs that formed right around a known catalyst — those move for
reasons the model was never trained to recognize, and behave differently
from an ordinary technical setup (see fvg_live_signal.py's --exclude-macro).

Add more event types by appending rows to macro_calendar.csv in the same
format (event, date, time_et). FOMC dates are pre-set for 2026 from the
Fed's published schedule. CPI/PPI/NFP dates aren't included yet — add
them from bls.gov's release schedule if you want those covered too.
"""

import pandas as pd

DEFAULT_CALENDAR_PATH = "macro_calendar.csv"


def load_calendar(path=DEFAULT_CALENDAR_PATH):
    df = pd.read_csv(path)
    # combine date + time_et into a single US/Eastern timestamp
    naive = pd.to_datetime(df["date"] + " " + df["time_et"])
    df["event_time"] = naive.dt.tz_localize("America/New_York")
    return df[["event", "event_time"]]


def nearest_event(ts, calendar, window_before_min=30, window_after_min=180):
    """Returns (event_name, minutes_from_event) for the closest calendar
    event within the window around ts, or (None, None) if none qualify.
    ts must be tz-aware; window_before/after are relative to event_time
    (e.g. default flags anything from 30 min before an FOMC statement to
    3 hours after it — covers the statement, press conference, and the
    initial digestion period)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = ts.tz_convert("America/New_York")

    deltas = (ts_et - calendar["event_time"]).dt.total_seconds() / 60.0  # minutes after event
    in_window = (deltas >= -window_before_min) & (deltas <= window_after_min)
    if not in_window.any():
        return None, None
    idx = deltas[in_window].abs().idxmin()
    return calendar.loc[idx, "event"], round(deltas[idx], 1)
