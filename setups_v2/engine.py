"""
The v2 setup engine: context -> setup -> confirmations -> risk -> verdict (evaluate_setups).
Pure pandas / numpy; the live report and backtest_v2.py run exactly this code.
Split out of enrich_html_v2.py, which re-exports every name (enrich_html_v2.X still works).
"""

import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------------- signal

# ================================================================ SETUP ENGINE
# Replaces the old "hard requirements + soft score" gate.
#
#   Layer 1  CONTEXT     what is the daily trend doing?   (daily_context)
#   Layer 2  SETUP       where is price relative to structure? five detectors:
#                        FRESH_BREAKOUT, SUPPORT_BOUNCE, MA_BOUNCE, PULLBACK, MA_RECLAIM
#   Layer 3  CONFIRM     setup-specific evidence (volume, stochastic, candle, structure, 1h timing)
#   Layer 4  RISK        setup-specific stop, nearest real resistance as target, R:R >= SETUP_MIN_R
#   Layer 5  VERDICT     BUY (all required + enough confirmations + R:R) / WATCH / WAIT
#
# All setup logic runs on DAILY bars (built from the hourly download); 4h and 1h
# only supply candle patterns and entry timing.
#
# STRATEGY (updated after the 2-year backtest, see backtest_v2.py):
#   * Traded setup: pullback in an uptrend ONLY. It carried the backtest (323 of 414 trades, +0.24R vs matched random days,
#     t 2.1). MA bounce and support bounce (too few trades / no measurable edge), MA reclaim (mixed) and fresh breakouts
#     (-0.5R vs random days) are still detected and shown for information but are NOT traded.
#   * Entry: the usual BUY definition (structure + turn + confirmations + R:R >= 1.5 to the nearest real resistance).
#   * Exit: the setup's structural stop; NO fixed profit target. Once price trades +2R above the entry,
#     trail the stop under the daily EMA20 (raise only); leave after 90 trading days at the latest.
#     In the backtest this trailing exit beat the fixed stop/target exit; the fixed target was the weaker exit.
#   * The old A/B grade is no longer shown: grade A was not better than grade B once breakouts were removed.

SETUP_LABELS = {
    "FRESH_BREAKOUT": "Fresh breakout",
    "SUPPORT_BOUNCE": "Support bounce",
    "MA_BOUNCE": "MA bounce",
    "PULLBACK": "Pullback in uptrend",
    "MA_RECLAIM": "MA reclaim (early trend)",
}

# What the 2-year backtest said about each traded setup: BUY minus matched random nearby days (EMA20-trail exit,
# BUY days, signal_close entry). Shown on the card so the level of evidence is never hidden.
SETUP_EVIDENCE = {
    "PULLBACK": "Backtest (2y, 323 trades): +0.37R per trade, +0.24R vs matched random days (t 2.1) - the only setup with a measurable edge.",
    "MA_BOUNCE": "Not traded. Backtest: 36 trades, +0.29R per trade, +0.35R vs matched random days (t 0.9) - too few trades to conclude.",
    "SUPPORT_BOUNCE": "Not traded. Backtest: 55 trades, +0.28R per trade but only +0.09R vs matched random days (t 0.3) - no measurable edge.",
    "MA_RECLAIM": "Not traded. Backtest: 70 trades, -0.33R vs control (t -1.6) on BUY days but +0.04R on pre-turn days - no measurable edge.",
    "FRESH_BREAKOUT": "Not traded. Backtest: 677 trades, -0.52R vs control (t -5.4) - breakouts as defined here lose to random days.",
}

NOT_TRADED_WHY = {
    "FRESH_BREAKOUT": "the backtest showed it trailing random days by about 0.5R",
    "MA_RECLAIM": "the backtest found no measurable edge",
    "MA_BOUNCE": "too few backtest trades to show an edge",
    "SUPPORT_BOUNCE": "the backtest found no measurable edge",
}


# ---------------------------------------------------------------- helpers

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _pct_slope(s: pd.Series, n: int = 5):
    s = s.dropna()
    if len(s) <= n or s.iloc[-1 - n] == 0:
        return None
    return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def _pivots(d: pd.DataFrame, pivot: int):
    """Swing lows / highs as [(position, price)]: the lowest low / highest high of
    `pivot` bars on each side. Equal neighbours are collapsed into one pivot."""
    lows = d["Low"].to_numpy(dtype=float)
    highs = d["High"].to_numpy(dtype=float)
    n = len(d)
    pl = [(i, lows[i]) for i in range(pivot, n - pivot) if lows[i] == lows[i - pivot:i + pivot + 1].min()]
    ph = [(i, highs[i]) for i in range(pivot, n - pivot) if highs[i] == highs[i - pivot:i + pivot + 1].max()]

    def dedupe(pts):
        out = []
        for i, p in pts:
            if out and i - out[-1][0] <= pivot:
                continue
            out.append((i, p))
        return out
    return dedupe(pl), dedupe(ph)


def swing_levels_all(d: pd.DataFrame, atr: float, pivot: int, merge_atr: float) -> list:
    """Every swing high AND swing low clustered into levels (old resistance can
    become support and vice versa). Each level: price, touches, kinds ('H'/'L'), last_date."""
    if d is None or len(d) < 2 * pivot + 1 or not atr:
        return []
    pl, ph = _pivots(d, pivot)
    pts = [(p, d.index[i], "L") for i, p in pl] + [(p, d.index[i], "H") for i, p in ph]
    pts.sort(key=lambda x: x[0])
    tol = merge_atr * atr
    groups = []
    for pt in pts:
        if groups and pt[0] - groups[-1][-1][0] <= tol:
            groups[-1].append(pt)
        else:
            groups.append([pt])
    return [{"price": round(float(np.mean([g[0] for g in grp])), 2),
             "touches": len(grp),
             "kinds": sorted({g[2] for g in grp}),
             "last_date": max(g[1] for g in grp).date().isoformat()} for grp in groups]


def intraday_rvol(hourly: pd.DataFrame, tz: str, lookback: int = 20):
    """Today's cumulative volume so far vs the average cumulative volume at the
    SAME number of hourly bars on the previous days (so a half-finished session
    is compared fairly). None when there is not enough history."""
    h = hourly.dropna(subset=["Volume"])
    if h.empty:
        return None
    idx = h.index.tz_convert(tz) if h.index.tz is not None else h.index
    dates = pd.Series(idx.date, index=h.index)
    by_day = {k: g["Volume"].to_numpy(dtype=float) for k, g in h.groupby(dates)}
    days = sorted(by_day)
    k = len(by_day[days[-1]])
    prev = [by_day[x] for x in days[:-1] if len(by_day[x]) >= k][-lookback:]
    if len(prev) < 5:
        return None
    avg = float(np.mean([p[:k].sum() for p in prev]))
    return float(by_day[days[-1]].sum() / avg) if avg > 0 else None


# ---------------------------------------------------------------- layer 1: context

def daily_context(d: pd.DataFrame, atr: float, cfg: dict) -> dict:
    close = d["Close"]
    ema9, ema20, ema50 = _ema(close, 9), _ema(close, 20), _ema(close, 50)
    sma50 = close.rolling(50).mean()
    c = float(close.iloc[-1])

    # Trend structure on MAJOR swings (STRUCT_PIVOT bars each side). Minor 2-bar pivots are noise: every
    # pullback undercuts some minor low, and they go stale. The uptrend is intact while
    #   - the last major swing low is higher than the one before AND nothing has undercut it since, and
    #   - the last major swing high is higher than the one before, or a newer high has already exceeded it.
    sw = d.iloc[-cfg["STRUCT_DAYS"]:]
    pl, ph = _pivots(sw, cfg["STRUCT_PIVOT"])
    lows_a, highs_a = sw["Low"].to_numpy(dtype=float), sw["High"].to_numpy(dtype=float)
    tol = cfg["PULLBACK_LOW_TOL_ATR"] * atr
    higher_lows, low_since = False, None
    if len(pl) >= 2:
        i, p_last = pl[-1]
        low_since = float(lows_a[i + 1:].min()) if i + 1 < len(lows_a) else float(p_last)
        higher_lows = bool(p_last > pl[-2][1] and low_since >= p_last - tol)
    higher_highs, high_since = False, None
    if len(ph) >= 2:
        j, h_last = ph[-1]
        high_since = float(highs_a[j + 1:].max()) if j + 1 < len(highs_a) else float(h_last)
        higher_highs = bool(max(h_last, high_since) > ph[-2][1])

    def run_above(ma: pd.Series) -> int:
        cl, m, k = close.to_numpy(dtype=float), ma.to_numpy(dtype=float), 0
        for i in range(len(cl) - 1, -1, -1):
            if np.isnan(m[i]) or cl[i] <= m[i]:
                break
            k += 1
        return k

    days_above_ema50, days_above_sma50 = run_above(ema50), run_above(sma50)
    below_before = 0
    if 0 < days_above_ema50 < len(close):
        seg = slice(-(days_above_ema50 + 15), -days_above_ema50)
        below_before = int((close.iloc[seg] < ema50.iloc[seg]).sum())

    e50 = float(ema50.iloc[-1])
    slope50 = _pct_slope(ema50, 5)
    above = c > e50
    if above and 1 <= days_above_ema50 <= cfg["RECLAIM_MAX_DAYS"] and below_before >= cfg["RECLAIM_MIN_BELOW"]:
        state = "EARLY_UPTREND"      # just reclaimed EMA50 from below - not yet an established trend
    elif above and slope50 is not None and slope50 > 0 and higher_lows and higher_highs:
        state = "UPTREND"            # rising EMA50 AND higher highs AND higher lows
    elif above:
        state = "UPTREND_WEAK"       # above EMA50 but EMA50 flat/falling, or the swings are not both higher yet
    elif (not above) and higher_lows and atr and (c - e50) / atr >= -cfg["CORRECTION_MAX_BELOW_ATR"]:
        state = "CORRECTION"         # dipped just under EMA50 but the swing structure is intact
    elif slope50 is not None and slope50 < 0:
        state = "DOWNTREND"
    else:
        state = "RANGE"

    return {
        "state": state, "close": round(c, 2), "atr": round(float(atr), 2),
        "ema9": round(float(ema9.iloc[-1]), 2), "ema20": round(float(ema20.iloc[-1]), 2),
        "ema50": round(e50, 2),
        "sma50": None if np.isnan(sma50.iloc[-1]) else round(float(sma50.iloc[-1]), 2),
        "ema50_slope_pct": None if slope50 is None else round(slope50, 2),
        "ext_ema50_atr": round((c - e50) / atr, 2) if atr else 0.0,
        "higher_lows": bool(higher_lows), "higher_highs": bool(higher_highs),
        "low_since_swing": None if low_since is None else round(low_since, 2),
        "high_since_swing": None if high_since is None else round(high_since, 2),
        "last_swing_lows": [round(float(p), 2) for _, p in pl[-2:]],
        "last_swing_highs": [round(float(p), 2) for _, p in ph[-2:]],
        "days_above_ema50": days_above_ema50, "days_above_sma50": days_above_sma50,
        "closes_below_before_reclaim": below_before,
        "_ema9": ema9, "_ema20": ema20, "_ema50": ema50, "_sma50": sma50,   # series, stripped before reporting
    }


# ---------------------------------------------------------------- layer 2/3: setup detectors
# Each detector returns None (setup not present) or
#   {"type", "level", "level_desc", "stop_ref", "required": [(label, ok)],
#    "confirms": [(label, ok)], "notes": [str]}

def _candle_hits(env: dict):
    hits = []
    for tf in ("1D", "4h", "1h"):
        hits += [f"{p} ({tf})" for p in env["res"][tf]["patterns_detected"]]
    return bool(hits), hits


def _rejection(env: dict, test_pos: int) -> bool:
    """Price refused to stay down: the close is back above the test bar's close, or
    (when the test bar is today) it closes in the upper half of its range."""
    d = env["d"]
    if test_pos < len(d) - 1:
        return bool(env["c"] > float(d["Close"].iloc[test_pos]))
    row = d.iloc[-1]
    rng = float(row["High"] - row["Low"])
    return bool(rng > 0 and (env["c"] - float(row["Low"])) / rng >= 0.5)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    up, dn = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def _dip_volume_contracting(env: dict, hi_pos: int):
    """Is selling volume FADING toward the end of the dip? Daily volume of each dip day is
    expressed as a multiple of the prior 20-day average (so quiet and busy stocks compare fairly).
    The last 2-3 dip days must average lower than the earlier dip days AND at or below
    DIP_TAIL_MAX_RVOL (default 1.0x normal). Today's still-forming bar IS included, but as its
    time-of-day-adjusted pace (today's volume so far vs the usual volume by this hour), never as
    its raw partial total, which would always look like a dry-up."""
    r = env["vol_ratio"]
    live = None
    if env["in_progress"]:
        dip = r[hi_pos + 1:len(r) - 1]
        live = env["vol"]["rvol_live"]
        dip = dip[~np.isnan(dip)]
        if live is not None:
            dip = np.append(dip, float(live))
    else:
        dip = r[hi_pos + 1:]
        dip = dip[~np.isnan(dip)]
    if len(dip) < 3:
        return False, f"only {len(dip)} dip day(s) so far - too short to judge"
    t = env["dip_tail_days"] or (2 if len(dip) <= 4 else 3)
    if len(dip) < t + 1:
        return False, f"only {len(dip)} dip day(s) so far - too short for a {t}-day tail"
    head, tail = dip[:-t], dip[-t:]
    hm, tm = float(head.mean()), float(tail.mean())
    today = f", incl. today at {float(live):.2f}x pace" if live is not None else ""
    return bool(tm < hm and tm <= env["dip_tail_max"]), \
        f"last {t} dip days {tm:.2f}x vs earlier dip days {hm:.2f}x of normal volume{today}"


def _stoch_reset_turn(env: dict) -> bool:
    """Daily stochastic dipped into the lower zone in the last 5 bars and is turning up."""
    return bool(env["reset_min"] <= env["reset_level"] and env["k_rising"])


def _reset_label(env: dict) -> str:
    return f"Daily {env['reset_name']} reset (<= {env['reset_level']:g} in last {env['reset_bars']} bars)"


def detect_breakout(env: dict, cfg: dict):
    d, a, c, ctx = env["d"], env["atr"], env["c"], env["ctx"]
    if ctx["state"] not in ("UPTREND", "EARLY_UPTREND", "UPTREND_WEAK"):
        return None
    F = cfg["BREAKOUT_FRESH_BARS"]
    closes = d["Close"].to_numpy(dtype=float)
    if len(closes) < F + cfg["BREAKOUT_BASE_BARS"] + 2:
        return None
    base = d.iloc[-(F + cfg["BREAKOUT_BASE_BARS"]):-F]
    cands = [(float(base["High"].max()), 1, f"{cfg['BREAKOUT_BASE_BARS']}-day high")]
    cands += [(L["price"], L["touches"], f"swing level, {L['touches']} touches") for L in env["levels"]
              if L["touches"] >= cfg["BREAKOUT_MIN_TOUCHES"]]

    hit = None
    for p, touches, src in cands:
        if c <= p * 1.001:
            continue                                    # not above the level
        if closes[-1 - F] > p:
            continue                                    # was already above it before the fresh window
        if (c - p) / a > cfg["BREAKOUT_MAX_EXT_ATR"]:
            continue                                    # too far above the level - chasing
        if hit is None or (p, touches) > (hit[0], hit[1]):
            hit = (p, touches, src)                     # the highest level cleared
    if hit is None:
        return None
    p, touches, src = hit
    bars_since = next((k for k in range(F, 0, -1) if closes[-k] > p), 1)

    vol = env["vol"]
    vmax = max([v for v in [vol["rvol_live"]] + vol["done_ratios"][-F:] if v is not None] or [0.0])
    row = d.iloc[-1]
    rng = float(row["High"] - row["Low"])
    close_pos = (c - float(row["Low"])) / rng if rng > 0 else 0.5
    pre = d.iloc[-(F + 10):-F]
    base_span_atr = float(pre["High"].max() - pre["Low"].min()) / a

    return {
        "type": "FRESH_BREAKOUT", "level": round(p, 2),
        "level_desc": f"broke {p:.2f} ({src}) {bars_since} bar(s) ago, now {(c - p) / a:.1f} ATR above it",
        "stop_ref": p - cfg["BREAKOUT_STOP_ATR"] * a,
        "required": [(f"Volume >= {cfg['BREAKOUT_RVOL']:g}x average (best of live/last {F} bars: {vmax:.1f}x)",
                      vmax >= cfg["BREAKOUT_RVOL"])],
        "confirms": [
            ("Close in the top 40% of the bar's range", close_pos >= 0.6),
            (f"Tight base before the break (<= {cfg['BREAKOUT_BASE_MAX_ATR']:g} ATR, was {base_span_atr:.1f})",
             base_span_atr <= cfg["BREAKOUT_BASE_MAX_ATR"]),
            ("Daily %K above 50 and not rolling over (%K >= %D or %K < 80)",
             env["k"] > 50 and (env["k_above_d"] or env["k"] < 80)),
            ("Higher lows in place", ctx["higher_lows"]),
        ],
        "notes": [],
    }


def detect_support_bounce(env: dict, cfg: dict):
    d, a, c, ctx = env["d"], env["atr"], env["c"], env["ctx"]
    if ctx["state"] == "DOWNTREND":
        return None
    F = 3
    lows, closes = d["Low"].to_numpy(dtype=float), d["Close"].to_numpy(dtype=float)
    recent_low = float(lows[-F:].min())
    test_pos = len(d) - F + int(np.argmin(lows[-F:]))
    best = None
    for L in env["levels"]:
        p = L["price"]
        strength = L["touches"] + (1 if len(L["kinds"]) > 1 else 0)
        if strength < cfg["SUPPORT_MIN_STRENGTH"] or p > c:
            continue
        tested = recent_low <= p + cfg["SUPPORT_TEST_ATR"] * a
        pierce_ok = recent_low >= p - cfg["SUPPORT_MAX_PIERCE_ATR"] * a
        held = float(closes[-F:].min()) >= p - 0.1 * a
        away = (c - p) / a
        if tested and pierce_ok and held and away <= cfg["SUPPORT_MAX_AWAY_ATR"]:
            if best is None or p > best["price"]:
                best = {**L, "strength": strength}
    if best is None:
        return None
    p = best["price"]
    held_bars = int((closes[test_pos:] >= p - 0.1 * a).sum())
    candle_ok, candle_names = _candle_hits(env)
    v = env["vol"]["rvol_live"]
    return {
        "type": "SUPPORT_BOUNCE", "level": p,
        "level_desc": (f"support {p:.2f} ({best['touches']} touches, "
                       f"{'flipped resistance' if len(best['kinds']) > 1 else 'swing low'}), "
                       f"low {recent_low:.2f} tested it, {(c - p) / a:.1f} ATR above now"),
        "stop_ref": min(p - 0.5 * a, recent_low - 0.1 * a),
        "required": [
            (f"Support held - no daily close below it ({held_bars} close(s) since the test)", True),
            ("Rejection - closed back above the test bar / upper half of range", _rejection(env, test_pos)),
        ],
        "confirms": [
            ("Bullish candle" + (f" ({', '.join(candle_names)})" if candle_ok else ""), candle_ok),
            ("Volume on the bounce >= average", v is not None and v >= 1.0),
            (_reset_label(env) + " and turning up", _stoch_reset_turn(env)),
            ("Trend context intact (UPTREND / EARLY_UPTREND, higher lows)",
             ctx["state"] in ("UPTREND", "EARLY_UPTREND") and ctx["higher_lows"]),
        ],
        "notes": [],
    }


def detect_ma_bounce(env: dict, cfg: dict):
    d, a, c, ctx = env["d"], env["atr"], env["c"], env["ctx"]
    if ctx["state"] != "UPTREND":
        return None
    F = 3
    lows, closes = d["Low"].to_numpy(dtype=float), d["Close"].to_numpy(dtype=float)
    recent_low = float(lows[-F:].min())
    test_pos = len(d) - F + int(np.argmin(lows[-F:]))
    for name, ma in (("EMA20", ctx["_ema20"]), ("EMA50", ctx["_ema50"]), ("SMA50", ctx["_sma50"])):
        m = ma.iloc[-1]
        if np.isnan(m):
            continue
        m = float(m)
        if name == "EMA20" and not ctx["ema20"] > ctx["ema50"]:
            continue                                            # need the EMA20 > EMA50 stack
        slope = _pct_slope(ma, 5)
        rising = slope is not None and slope > 0
        tested = recent_low <= m + cfg["MA_TEST_ATR"] * a
        pierce_ok = recent_low >= m - cfg["MA_MAX_PIERCE_ATR"] * a
        held = float(closes[-F:].min()) >= m - 0.1 * a
        above = 0 < (c - m) / a <= cfg["MA_MAX_AWAY_ATR"]
        if not (tested and pierce_ok and held and above):
            continue
        candle_ok, candle_names = _candle_hits(env)
        L = min(12, len(d))                      # the dip starts at the highest high of the last 12 daily bars
        hi_pos = len(d) - L + int(np.argmax(d["High"].to_numpy(dtype=float)[-L:]))
        dry, dry_detail = _dip_volume_contracting(env, hi_pos)
        return {
            "type": "MA_BOUNCE", "level": round(m, 2),
            "level_desc": f"{name} {m:.2f} (rising) tested by low {recent_low:.2f}, {(c - m) / a:.1f} ATR above now",
            "stop_ref": min(recent_low - 0.25 * a, m - 0.75 * a),
            "required": [
                (f"{name} rising", rising),
                (f"{name} held - no daily close below it", True),
                ("Rejection - closed back above the test bar / upper half of range", _rejection(env, test_pos)),
            ],
            "confirms": [
                ("Bullish candle" + (f" ({', '.join(candle_names)})" if candle_ok else ""), candle_ok),
                ("Volume fading into the end of the dip (" + dry_detail + ")", dry),
                (_reset_label(env) + " and turning up", _stoch_reset_turn(env)),
                ("Higher lows in place", ctx["higher_lows"]),
            ],
            "notes": [],
        }
    return None


def _launch_low(d: pd.DataFrame, hi_pos: int, bars: int, min_leg: float, max_back: int = 40):
    """The last MAJOR swing low before the impulse high: the higher low that launched the up-leg.
    It is the lowest low of `bars` daily bars on each side, looking only at bars up to the high,
    so a pullback that undercuts it cannot hide it. Must sit at least `min_leg` below the high
    (ignores small wiggles right under the top). Returns (position, price) or None."""
    lows = d["Low"].to_numpy(dtype=float)
    top = float(d["High"].to_numpy(dtype=float)[hi_pos])
    for i in range(hi_pos - 2, max(bars, hi_pos - max_back) - 1, -1):
        lo, hi = max(0, i - bars), min(i + bars, hi_pos) + 1
        if lows[i] == lows[lo:hi].min() and top - lows[i] >= min_leg:
            return i, float(lows[i])
    return None


def detect_pullback(env: dict, cfg: dict):
    d, a, c, ctx = env["d"], env["atr"], env["c"], env["ctx"]
    if ctx["state"] != "UPTREND":
        return None
    n = len(d)
    L = min(cfg["PULLBACK_LOOKBACK"], n)
    highs = d["High"].to_numpy(dtype=float)
    hi_pos = n - L + int(np.argmax(highs[-L:]))
    if n - 1 - hi_pos < 3:
        return None                                     # the high is too recent to be a pullback
    H = float(highs[hi_pos])
    leg_low = float(d["Low"].iloc[max(0, hi_pos - cfg["PULLBACK_LEG_BARS"]):hi_pos + 1].min())
    leg = H - leg_low
    if leg < cfg["PULLBACK_MIN_LEG_ATR"] * a:
        return None
    pb = d.iloc[hi_pos + 1:]
    pb_low = float(pb["Low"].min())
    depth = (H - pb_low) / leg
    if H - pb_low < cfg["PULLBACK_MIN_DROP_ATR"] * a:
        return None                                     # too shallow to be a real pullback
    if not (cfg["PULLBACK_MIN_DEPTH"] <= depth <= cfg["PULLBACK_MAX_DEPTH"]):
        return None
    if H - c < 0.5 * a:
        return None                                     # already back near the high - not a pullback anymore
    zone = [("EMA20", ctx["ema20"]), ("EMA50", ctx["ema50"])]
    if ctx["sma50"] is not None:
        zone.append(("SMA50", ctx["sma50"]))
    zone += [(f"swing level {lv['price']:.2f}", lv["price"]) for lv in env["levels"]]
    near = [nm for nm, v in zone if abs(pb_low - v) <= env["value_zone_atr"] * a]
    dry, dry_detail = _dip_volume_contracting(env, hi_pos)
    launch = _launch_low(d, hi_pos, cfg["PULLBACK_STRUCT_BARS"], cfg["PULLBACK_MIN_DROP_ATR"] * a)
    if launch is not None:
        ref, where = launch[1], f"launch low {launch[1]:.2f} on {d.index[launch[0]].date()}"
    else:                                   # no clear swing low before the high: fall back to the start of the leg
        ref, where = leg_low, f"start of the leg {leg_low:.2f}"
    struct_ok = bool(pb_low >= ref - cfg["PULLBACK_LOW_TOL_ATR"] * a)
    struct_desc = f"{where}; pullback low {pb_low:.2f}, {'holds by' if pb_low >= ref else 'undercuts by'} {abs(pb_low - ref):.2f}"
    candle_ok, candle_names = _candle_hits(env)
    return {
        "type": "PULLBACK", "level": round(pb_low, 2),
        "level_desc": (f"pulled back {depth * 100:.0f}% of the {leg_low:.2f}->{H:.2f} leg "
                       f"(low {pb_low:.2f}, {n - 1 - hi_pos} bars since the high)"),
        "stop_ref": pb_low - 0.25 * a,
        "required": [
            ("Structure intact - pullback low holds above the last major swing low (" + struct_desc + ")", struct_ok),
        ],
        "confirms": [
            ("Volume fading into the end of the dip (" + dry_detail + ")", dry),
            (_reset_label(env), env["reset_min"] <= env["reset_level"]),
            ("Pullback low sat at value" + (f" ({', '.join(near)})" if near else " (EMA20/EMA50/SMA50/swing level)"),
             bool(near)),
            ("Bullish candle" + (f" ({', '.join(candle_names)})" if candle_ok else ""), candle_ok),
        ],
        "notes": [],
    }


def detect_reclaim(env: dict, cfg: dict):
    d, a, c, ctx = env["d"], env["atr"], env["c"], env["ctx"]
    if ctx["state"] != "EARLY_UPTREND":
        return None
    k = ctx["days_above_ema50"]
    vratios = env["vol"]["done_ratios"] + ([env["vol"]["rvol_live"]] if env["vol"]["rvol_live"] is not None else [])
    since = [v for v in vratios[-k:] if v is not None]
    lows = d["Low"].to_numpy(dtype=float)
    low_since = float(lows[-k:].min())
    low_before = float(lows[-(k + 15):-k].min()) if len(lows) > k + 15 else float(lows[:-k].min())
    sma_ok = ctx["sma50"] is not None and c > ctx["sma50"]
    return {
        "type": "MA_RECLAIM", "level": ctx["ema50"],
        "level_desc": (f"closed above EMA50 {ctx['ema50']:.2f} for {k} day(s) after "
                       f"{ctx['closes_below_before_reclaim']} closes below it in the prior 15"
                       + (f"; SMA50 {ctx['sma50']:.2f} also reclaimed" if sma_ok else "")),
        "stop_ref": min(ctx["ema50"] - 0.75 * a, low_since - 0.25 * a),
        "required": [
            (f"Held {cfg['RECLAIM_MIN_HOLD_DAYS']}+ closes above EMA50 (has {k})", k >= cfg["RECLAIM_MIN_HOLD_DAYS"]),
            (f"Not extended (<= {cfg['RECLAIM_MAX_EXT_ATR']:g} ATR above EMA50, is {ctx['ext_ema50_atr']:.1f})",
             ctx["ext_ema50_atr"] <= cfg["RECLAIM_MAX_EXT_ATR"]),
        ],
        "confirms": [
            ("Higher low since the reclaim (low above the prior 15-bar low)", low_since > low_before),
            ("EMA9 above EMA20 (short-term turn)", ctx["ema9"] > ctx["ema20"]),
            ("Volume on the reclaim >= 1.2x average", bool(since) and max(since) >= 1.2),
            ("Daily %K above %D and not overbought (< 80)", bool(env["k_above_d"] and env["k"] < 80)),
            ("SMA50 reclaimed as well", sma_ok),
        ],
        "notes": [],
    }


# ---------------------------------------------------------------- layer 4: risk

def trade_plan(setup: dict, price: float, atr: float, target_levels: list, cfg: dict, majors=()) -> dict:
    """Stop = the setup's structural stop, but never closer than MIN_RISK_ATR (a tiny
    stop would fake a great R:R).
    Target = the nearest REAL resistance above price. Real means a swing level with
    >= OVERHEAD_MIN_TOUCHES touches, or a 20-bar / full-window / 52-week high. Single-touch minor
    pivots are never targets (they are only listed as 'overhead'), so R:R does not jump when price
    crosses a minor pivot. A real level inside MIN_TARGET_ATR blocks the trade (tiny R:R on purpose);
    with clear air above, the target is TARGET_FALLBACK_R x risk."""
    stop = min(setup["stop_ref"], price - cfg["MIN_RISK_ATR"] * atr)
    risk = price - stop
    room = cfg["MIN_TARGET_ATR"] * atr
    need = cfg["OVERHEAD_MIN_TOUCHES"]
    real = [(lv["price"], lv["touches"], f"swing level, {lv['touches']} touch(es)")
            for lv in target_levels if lv["price"] > price and lv["touches"] >= need]
    real += [(p, need, tag) for tag, p in majors if p > price]
    real.sort()
    merged = []                       # levels within 0.1 ATR are one resistance (swing level + period high + 52w high...)
    for p, n, tag in real:
        if merged and p - merged[-1][0] <= 0.1 * atr:
            prev = merged[-1]
            parts = prev[2].split(" + ")
            merged[-1] = (prev[0], max(prev[1], n), prev[2] if tag in parts else prev[2] + " + " + tag)
        else:
            merged.append((p, n, tag))
    minor = [lv["price"] for lv in target_levels
             if price < lv["price"] < price + room and lv["touches"] < need]
    blocking = [x for x in merged if x[0] < price + room]
    if merged:
        p, _, tag = merged[0]
        target = p
        src = (f"T1 - overhead resistance ({tag}) inside {cfg['MIN_TARGET_ATR']:g} ATR" if blocking
               else f"T1 - nearest real resistance ({tag})")
    else:
        target, src = price + cfg["TARGET_FALLBACK_R"] * risk, f"{cfg['TARGET_FALLBACK_R']:g}R - no resistance above"
    rr = (target - price) / risk if risk > 0 else 0.0
    # entry price at which the SAME stop and target would give exactly SETUP_MIN_RR
    min_rr = cfg["SETUP_MIN_RR"]
    buy_below = (target + min_rr * stop) / (1 + min_rr)
    # when R:R is already fine, the same formula is the HIGHEST entry that still gives min R:R
    return {"buy_below": round(buy_below, 2) if rr < min_rr and stop < buy_below < price else None,
            "buy_up_to": round(buy_below, 2) if rr >= min_rr and price <= buy_below < target else None,
            "breakout_above": round(blocking[0][0], 2) if blocking else None,
            "stop": round(stop, 2), "target": round(target, 2), "rr": round(rr, 2),
            "target_src": src, "risk": round(risk, 2),
            "overhead": sorted(round(x, 2) for x in ([b[0] for b in blocking] + minor))}


# ---------------------------------------------------------------- layer 5: verdict

def evaluate_setups(daily: pd.DataFrame, daily_atr: float, results: dict, stoch_d: pd.DataFrame,
                    hourly: pd.DataFrame, h1_macd_ok: bool, above_vwap, in_progress: bool, cfg: dict,
                    high_52w=None, vwap=None, shared: dict = None) -> dict:
    """`shared` (optional dict) lets several calls for the SAME day with different item settings (the backtest's
    variants) reuse the parts that do not depend on those settings."""
    sh = shared if shared is not None else {}
    if "d" not in sh:
        d0 = daily.dropna(subset=["Close"])
        a0 = float(daily_atr)
        sh["d"], sh["a"] = d0, a0
        sh["ctx"] = daily_context(d0, a0, cfg)
        sh["ratio"] = (d0["Volume"] / d0["Volume"].rolling(20).mean().shift(1))
        sh["levels"] = swing_levels_all(d0.iloc[-cfg.get("SR_DAYS", 60):], a0, cfg.get("SR_PIVOT", 2),
                                        cfg.get("SR_MERGE_ATR", 0.5))
        sh["target_levels"] = swing_levels_all(d0.iloc[-cfg["TARGET_DAYS"]:], a0, cfg.get("SR_PIVOT", 2),
                                               cfg.get("SR_MERGE_ATR", 0.5))
        sh["k"] = stoch_d["%K"].dropna()
        sh["rsi"] = _rsi(d0["Close"], 14)
    d, a, ctx, ratio = sh["d"], sh["a"], sh["ctx"], sh["ratio"]
    c = float(d["Close"].iloc[-1])

    # volume: ratio of each daily bar to the PRIOR 20-bar average; today's bar (still
    # forming) is compared on a same-time-of-day basis instead of its partial total
    done = ratio.iloc[:-1] if in_progress else ratio
    rvol_live = intraday_rvol(hourly, cfg.get("MARKET_TZ", "America/New_York")) if in_progress \
        else (None if np.isnan(ratio.iloc[-1]) else float(ratio.iloc[-1]))
    vol = {"rvol_live": None if rvol_live is None else round(rvol_live, 2),
           "done_ratios": [round(float(x), 2) for x in done.dropna().tail(5)]}

    k = sh["k"]
    env = {
        "d": d, "atr": a, "c": c, "ctx": ctx, "res": results, "vol": vol,
        "vol_ratio": ratio.to_numpy(dtype=float), "in_progress": bool(in_progress),
        "dip_tail_max": cfg["DIP_TAIL_MAX_RVOL"], "dip_tail_days": cfg["DIP_TAIL_DAYS"],
        "value_zone_atr": cfg["VALUE_ZONE_ATR"],
        "levels": sh["levels"],
        "k": float(k.iloc[-1]), "k_rising": bool(k.iloc[-1] > k.iloc[-2]),
        "k_min5": float(k.iloc[-5:].min()),
        "reset_min": float((sh["rsi"] if cfg["RESET_INDICATOR"] == "rsi" else k).iloc[-cfg["RESET_BARS"]:].min()),
        "reset_level": cfg["RESET_LEVEL"], "reset_bars": cfg["RESET_BARS"],
        "reset_name": "RSI" if cfg["RESET_INDICATOR"] == "rsi" else "stochastic",
        "k_above_d": bool(results["1D"]["k_above_d"]),
        "timing_ok": bool(h1_macd_ok and above_vwap is not False),
    }
    target_levels = sh["target_levels"]

    # major highs = real resistance even without swing touches (today's own bar is excluded)
    def major_highs(skip_bars: int) -> list:
        """skip_bars = how many of the most recent bars to leave out: today's own bar always,
        and for a fresh breakout its own run-up bars (they are the move, not resistance)."""
        prior, out = d.iloc[:-skip_bars], []
        if len(prior) >= 5:
            out.append(("20-bar high", float(prior["High"].iloc[-20:].max())))
            out.append((f"{min(len(prior), cfg['TARGET_DAYS'])}-bar high",
                        float(prior["High"].iloc[-cfg["TARGET_DAYS"]:].max())))
        if high_52w and float(high_52w) > float(d["High"].iloc[-1]) + 1e-6:
            out.append(("52-week high", float(high_52w)))
        return out

    # TURN GATE (required for every BUY): the reversal must show on the daily AND the hourly chart.
    #   daily: a GREEN candle (close above the day's open) that closes in the upper half of its range;
    #          a gap-up that fades into a red candle is selling, not a turn, even if it is above yesterday's close
    #   1h   : above the session VWAP and (1h MACD bullish or 1h %K above %D)
    prev_c = float(d["Close"].iloc[-2]) if len(d) > 1 else c
    row = d.iloc[-1]
    rng = float(row["High"] - row["Low"])
    close_pos = (c - float(row["Low"])) / rng if rng > 0 else 0.5
    day_open = float(row["Open"])
    daily_turn = bool(c > day_open and close_pos >= 0.5)
    daily_level = max(day_open, float(row["Low"]) + 0.5 * rng)      # price that must hold into the close
    h1r = results["1h"]
    h1_turn = bool(above_vwap is not False and (h1_macd_ok or h1r["k_above_d"]))
    # price that would flip the turn checks (a guide only: MACD / %K must follow)
    turn_above = None
    if not (daily_turn and h1_turn):
        turn_above = max([x for x in (daily_level if not daily_turn else None,
                                      float(vwap) if (vwap is not None and above_vwap is False) else None)
                          if x is not None] or [None]) if (not daily_turn or above_vwap is False) else None
    turn = [
        (f"Daily turn - green candle closing in the upper half ({'green' if c > day_open else 'red'} candle, open {day_open:.2f} -> {c:.2f}; "
         f"{(c / prev_c - 1) * 100:+.1f}% vs prior close {prev_c:.2f}; closes at {close_pos * 100:.0f}% of today's range)",
         daily_turn),
        (f"1h turn - above session VWAP and momentum up (VWAP {'ok' if above_vwap is not False else 'below'}, "
         f"MACD {'bullish' if h1_macd_ok else 'not bullish'}, %K {'>' if h1r['k_above_d'] else '<'} %D)", h1_turn),
    ]

    setups, not_traded = [], []
    for fn in (detect_breakout, detect_support_bounce, detect_ma_bounce, detect_pullback, detect_reclaim):
        s = fn(env, cfg)
        if s is None:
            continue
        if s["type"] not in cfg["TRADEABLE_SETUPS"]:
            not_traded.append(s["type"])                        # shown for information only
            continue
        s.update(trade_plan(s, c, a, target_levels, cfg,
                            major_highs(cfg["BREAKOUT_FRESH_BARS"] if s["type"] == "FRESH_BREAKOUT" else 1)))
        s["n_confirms"] = sum(ok for _, ok in s["confirms"])
        s["min_confirms"] = cfg["MIN_CONFIRMS"][s["type"]]
        fails = [lbl for lbl, ok in s["required"] if not ok]
        s["turn"] = turn
        if not (daily_turn and h1_turn):
            missing = [nm for nm, ok in (("daily", daily_turn), ("1h", h1_turn)) if not ok]
            fails.append(f"no turn yet on the {' and '.join(missing)} chart" + ("s" if len(missing) > 1 else ""))
        if s["n_confirms"] < s["min_confirms"]:
            fails.append(f"confirmations {s['n_confirms']}/{len(s['confirms'])} (need {s['min_confirms']})")
        if s["rr"] < cfg["SETUP_MIN_RR"]:
            fails.append(f"R:R {s['rr']:.2f} below {cfg['SETUP_MIN_RR']:g}")
        if s["type"] != "FRESH_BREAKOUT" and ctx["ext_ema50_atr"] > cfg["MAX_EXT_ATR"]:
            fails.append(f"extended {ctx['ext_ema50_atr']:.1f} ATR above EMA50")
        s["fails"], s["buy"] = fails, not fails
        # would this be a BUY if the confirmation COUNT did not matter? (used by the backtest's confirmation sweep)
        s["cand"] = not [f for f in fails if not f.startswith("confirmations ")]
        s["near"] = bool(all(ok for _, ok in s["required"])
                         and s["n_confirms"] >= s["min_confirms"] - cfg["WATCH_CONFIRM_GAP"])
        setups.append(s)

    setups.sort(key=lambda s: (s["buy"], s["near"], s["n_confirms"], s["rr"]), reverse=True)
    best = setups[0] if setups else None
    cands = [x for x in setups if x["cand"]]
    conf_candidate = None
    if cands:
        cb = max(cands, key=lambda x: (x["n_confirms"], x["rr"]))
        conf_candidate = {"type": cb["type"], "n_confirms": cb["n_confirms"], "of": len(cb["confirms"]),
                          "stop": cb["stop"], "target": cb["target"], "rr": cb["rr"], "buy_up_to": cb["buy_up_to"]}

    # levels worth an alert when there is no trade yet
    watch = [(nm if v < c else f"{nm} (reclaim)", v)
             for nm, v in (("EMA20", ctx["ema20"]), ("EMA50", ctx["ema50"]), ("SMA50", ctx["sma50"]))
             if v is not None and (v < c or ctx["state"] in ("CORRECTION", "RANGE"))]
    sup = [lv for lv in target_levels if lv["price"] < c and lv["touches"] + (len(lv["kinds"]) > 1) >= 2]
    if sup:
        watch.append(("nearest support", max(sup, key=lambda lv: lv["price"])["price"]))
    watch.sort(key=lambda x: -x[1])

    watch_kind = None
    if best is not None and best["buy"]:
        verdict = "BUY"
    elif best is not None and best["near"]:
        verdict = "WATCH"
        # price: R:R is the problem, wait for a cheaper entry; trigger: wait for the turn / confirmation
        watch_kind = "price" if any(f.startswith("R:R") for f in best["fails"]) else "trigger"
    else:
        verdict = "WAIT"

    if best is not None:
        wait_reason = "" if best["buy"] else (
            "" if verdict == "WATCH" else f"weak {SETUP_LABELS[best['type']].lower()}: ") + "; ".join(best["fails"])
    elif ctx["state"] == "CORRECTION":
        wait_reason = (f"under EMA50 by {abs(ctx['ext_ema50_atr']):.1f} ATR with a higher low intact - "
                       "only pullbacks in a confirmed uptrend are traded, so wait for an EMA50 reclaim and a rebuilt uptrend")
    elif ctx["state"] == "DOWNTREND":
        wait_reason = "downtrend (below a falling EMA50) - no long setup"
    elif ctx["state"] == "RANGE":
        wait_reason = "no trend (below a flat EMA50) - only pullbacks in a confirmed uptrend are traded"
    elif ctx["state"] == "UPTREND_WEAK":
        gaps = [nm for nm, ok in (("higher highs", ctx["higher_highs"]), ("higher lows", ctx["higher_lows"]),
                                  ("a rising EMA50", (ctx["ema50_slope_pct"] or 0) > 0)) if not ok]
        wait_reason = ("above the EMA50 but not a clean uptrend yet (missing: " + ", ".join(gaps) +
                       ") - the pullback setup needs higher highs, higher lows and a rising EMA50")
    elif ctx["ext_ema50_atr"] > 2.0:
        wait_reason = (f"extended {ctx['ext_ema50_atr']:.1f} ATR above EMA50 - "
                       "wait for a pullback to value")
    else:
        wait_reason = "trend intact, but no pullback setup has formed (a dip of 20-65% of the last leg that holds above the last major swing low)"

    # concrete conditions that would turn a WATCH into a BUY (all must hold together)
    buy_when = []
    if best is not None and not best["buy"]:
        def short(t):
            return re.sub(r"\s*\(.*?\)", "", t).strip()
        min_rr = cfg["SETUP_MIN_RR"]
        for lbl, ok in best["required"]:
            if not ok:
                buy_when.append(f"Setup: {short(lbl)}")
        turn_items = []
        if not daily_turn:
            turn_items.append(f"daily: a green candle - trade above {daily_level:.2f} (today's open / mid-range) and hold it into the close")
        if not h1_turn:
            bits = []
            if above_vwap is False:
                bits.append("reclaim the session VWAP" + (f" ({float(vwap):.2f})" if vwap else ""))
            if not (h1_macd_ok or h1r["k_above_d"]):
                bits.append("1h MACD bullish or 1h %K above %D")
            turn_items.append("1h: " + " and ".join(bits))
        if best["rr"] < min_rr:
            # the entry price is the problem: it has to come first, and the turn must then re-form there
            opts = []
            if best.get("buy_below"):
                opts.append(f"pull back to {best['buy_below']:.2f} or lower (R:R {min_rr:g}+ with the same stop and target)")
            buy_when.append("Price: " + " OR ".join(opts) if opts
                            else f"Price: R:R is {best['rr']:.2f} (resistance at {best['target']:.2f} is too close), it needs {min_rr:g}+")
            if turn_items:
                buy_when.append("Then the turn has to show at that price - " + "; ".join(turn_items))
        elif turn_items:
            buy_when.append("Turn - " + "; ".join(turn_items))
            if turn_above and best.get("buy_up_to") and turn_above > best["buy_up_to"] and turn_above > best["stop"]:
                rr_at = (best["target"] - turn_above) / (turn_above - best["stop"])
                buy_when.append(f"Note: {turn_above:.2f} is above the buy-up-to price {best['buy_up_to']:.2f} "
                                f"(R:R there would be {rr_at:.2f}), so it cannot qualify at that level today - "
                                "it needs a fresh setup on a later day")
        need = best["min_confirms"] - best["n_confirms"]
        if need > 0:
            miss = [short(l) for l, ok in best["confirms"] if not ok]
            buy_when.append(f"Confirmations: any {need} of: " + (", or ".join(miss) if need == 1 else "; ".join(miss)))
        if best["type"] != "FRESH_BREAKOUT" and ctx["ext_ema50_atr"] > cfg["MAX_EXT_ATR"]:
            buy_when.append(f"Extension: pull back closer to the EMA50 (now {ctx['ext_ema50_atr']:.1f} ATR above, "
                            f"max {cfg['MAX_EXT_ATR']:g})")

    # how far from a BUY: fixable-soon items are cheap, a pullback or a broken structure is not
    #   turn (daily / 1h) = 1 each; missing confirmation = 1 each; price (R:R) = 2 + ATRs to the entry zone (max 3), or 3
    #   with no entry zone; extended = 2; failed structural requirement = 3
    distance = None
    if best is not None:
        distance = 0.0
        if not best["buy"]:
            distance += sum(1 for ok in (daily_turn, h1_turn) if not ok)
            distance += sum(3 for _, ok in best["required"] if not ok)
            distance += max(0, best["min_confirms"] - best["n_confirms"])
            if best["rr"] < cfg["SETUP_MIN_RR"]:
                distance += (2 + min(3.0, (c - best["buy_below"]) / a)) if best.get("buy_below") else 3
            if best["type"] != "FRESH_BREAKOUT" and ctx["ext_ema50_atr"] > cfg["MAX_EXT_ATR"]:
                distance += 2

    if best is None and not_traded:
        wait_reason += "; " + "; ".join(
            f"{SETUP_LABELS[t]} present but not traded ({NOT_TRADED_WHY.get(t, 'not a traded setup')})"
            for t in not_traded)

    grade = "-"
    if best is not None:
        grade = ("A" if best["n_confirms"] >= 4 and best["rr"] >= 2 else "B") if best["buy"] else (
            "C" if verdict == "WATCH" else "D")

    # "Turning up near the EMA50" (watch only, never a BUY, not backtested): no traded setup, the close is at or below
    # the EMA50 by at most TURN_WATCH_MAX_ATR, and both turn checks pass (the same daily + 1h turn a BUY needs).
    ext50 = ctx["ext_ema50_atr"]
    turning_up = bool(best is None and cfg.get("TURN_WATCH", True) and ext50 is not None
                      and -cfg.get("TURN_WATCH_MAX_ATR", 1.5) <= ext50 <= 0 and daily_turn and h1_turn)
    if turning_up:
        wait_reason += (f"; TURNING UP near the EMA50 ({abs(ext50):.2f} ATR below it, daily and 1h turn both pass) - "
                        "watch for an EMA50 reclaim; not a traded setup")

    public_ctx = {kk: vv for kk, vv in ctx.items() if not kk.startswith("_")}
    return {
        "signal": verdict, "grade": grade,
        "turning_up": turning_up,
        "day_chg_pct": round((c / prev_c - 1) * 100, 2) if prev_c else None,
        "setup": best["type"] if best else None,
        "setup_label": SETUP_LABELS[best["type"]] if best else None,
        "context": public_ctx, "vol": vol,
        "level": best["level"] if best else None,
        "level_desc": best["level_desc"] if best else "",
        "required": best["required"] if best else [],
        "turn": best["turn"] if best else turn,
        "confirms": best["confirms"] if best else [],
        "n_confirms": best["n_confirms"] if best else 0,
        "min_confirms": best["min_confirms"] if best else 0,
        "fails": best["fails"] if best else [],
        "stop": best["stop"] if best else None,
        "target": best["target"] if best else None,
        "rr": best["rr"] if best else None,
        "target_src": best["target_src"] if best else "",
        "overhead": best["overhead"] if best else [],
        "buy_below": best["buy_below"] if best else None,
        "buy_up_to": best["buy_up_to"] if best else None,
        "breakout_above": best["breakout_above"] if best else None,
        "watch_kind": watch_kind,
        "not_traded": [SETUP_LABELS[t] for t in not_traded],
        "conf_candidate": conf_candidate,
        "risk": best["risk"] if best else None,
        "arm_level": round(c + cfg["TRAIL_ARM_R"] * best["risk"], 2) if best else None,
        "trail_now": ctx["ema20"] if cfg["TRAIL_MA"] == "EMA20" else ctx["ema50"],
        "pre_turn": bool(best is not None and not best["buy"] and len(best["fails"]) == 1
                         and best["fails"][0].startswith("no turn yet")),      # a BUY in every respect except the turn gate
        "buy_when": buy_when,
        "distance": None if distance is None else round(distance, 1),
        "turn_above": round(turn_above, 2) if turn_above else None,
        "wait_reason": wait_reason,
        "watch_levels": [(nm, round(v, 2)) for nm, v in watch],
        "all_setups": [{"type": s["type"], "label": SETUP_LABELS[s["type"]], "buy": s["buy"],
                        "n_confirms": s["n_confirms"], "rr": s["rr"], "fails": s["fails"],
                        "near": s["near"]} for s in setups],
    }
