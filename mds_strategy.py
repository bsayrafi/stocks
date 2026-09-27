"""
mds_strategy.py
===============
Multi-Day Swing (MDS) strategy -- the rules, shared 1:1 by the live scanner
(run_mds_scanner.py) and the backtest (mds_backtest.py), so what you backtest
is exactly what you trade.

Layers (every one is a STATE check, nothing is forecast):

  0. Market regime     SPY close > SMA200
  1. Sector rotation   RRG-style RS-Ratio / RS-Momentum of the stock's SPDR
                       sector ETF vs SPY -> quadrant must be Leading/Improving
  2. Fundamentals      analyst Buy/Strong Buy, beta <= 2, 0 < PEG < 4,
                       >= 3/5 quality checks, no earnings inside the hold window
  3. Trend             50-day linear regression channel: slope > 0, R^2 >= 0.5,
                       not in the top quarter of the channel; EMA21 > EMA50,
                       ADX >= 20 with +DI > -DI
  4. Institutional     price above the VWAP anchored at the channel's lowest low
     flow              (the buyers since the low are in profit and defending)
                       + >= 2/3 of: up/down volume ratio > 1, CMF > 0, OBV rising
  5. Levels            composite 15m volume profile (POC/VAH/VAL) + daily fractal
                       support/resistance -> structural stop and targets, R:R gate
  6. 15m entry         VWAP hold/reclaim on a green 15m bar above EMA9 with
                       time-of-day relative volume >= 1, not stretched above VWAP+1sd,
                       R:R >= MIN_RR at the actual price

Reused from the existing codebase:
  boxindicators.atr / adx, indicators.calculate_support_resistance,
  intraday_strategy.compute_volume_profile / SECTOR_TO_ETF, cached_ticker.CachedTicker.
The rolling regression channel here is a vectorized form of
enrich_html.linear_regression_channel (identical last-bar values -- checked in
the self-test at the bottom).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from mds_config import MDS_CONFIG as CFG
import boxindicators as bxi
from indicators import calculate_support_resistance
from intraday_strategy import compute_volume_profile, SECTOR_TO_ETF as _BASE_SECTOR_MAP

ET = CFG["MARKET_TZ"]

# Finviz / Yahoo / GICS sector names -> SPDR ETF
SECTOR_TO_ETF = dict(_BASE_SECTOR_MAP)
SECTOR_TO_ETF.update({"Financial": "XLF", "Consumer Goods": "XLP", "Services": "XLY"})

ETF_TO_SECTOR = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLE": "Energy", "XLI": "Industrials", "XLB": "Materials",
    "XLRE": "Real Estate", "XLU": "Utilities", "XLC": "Communication Services",
}


def sector_etf(sector: str | None) -> str | None:
    if not sector or not isinstance(sector, str):
        return None
    return SECTOR_TO_ETF.get(sector.strip())


# =========================================================================== 0. market regime

def market_regime(spy: pd.DataFrame, cfg=CFG) -> pd.Series:
    """True on days SPY closed above its SMA(MARKET_SMA)."""
    sma = spy["Close"].rolling(cfg["MARKET_SMA"]).mean()
    return (spy["Close"] > sma).rename("market_ok")


# =========================================================================== 1. sector rotation

def sector_rotation(etf_closes: dict[str, pd.Series], spy_close: pd.Series, cfg=CFG) -> dict[str, pd.DataFrame]:
    """RRG-style relative rotation, per ETF, per day.

    RS           = ETF / SPY (EMA-smoothed)
    RS-Ratio     = 100 * RS / SMA(RS, RRG_RATIO_WINDOW)      > 100: outperforming its own trend
    RS-Momentum  = 100 * RS-Ratio / RS-Ratio[t - RRG_MOMENTUM_WINDOW]   > 100: improving
    Quadrants    Leading (>100, >100), Weakening (>100, <100),
                 Lagging (<100, <100), Improving (<100, >100)
    Institutions rotate money INTO Improving -> Leading sectors; that's where we fish.
    """
    out = {}
    for etf, close in etf_closes.items():
        a, b = close.align(spy_close, join="inner")
        rs = (a / b).ewm(span=cfg["RRG_SMOOTH"], adjust=False).mean()
        ratio = 100 * rs / rs.rolling(cfg["RRG_RATIO_WINDOW"]).mean()
        mom = 100 * ratio / ratio.shift(cfg["RRG_MOMENTUM_WINDOW"])
        quad = np.select(
            [(ratio >= 100) & (mom >= 100), (ratio >= 100) & (mom < 100),
             (ratio < 100) & (mom < 100), (ratio < 100) & (mom >= 100)],
            ["Leading", "Weakening", "Lagging", "Improving"], default="n/a")
        df = pd.DataFrame({"rs_ratio": ratio, "rs_momentum": mom, "quadrant": quad}, index=ratio.index)
        df.loc[ratio.isna() | mom.isna(), "quadrant"] = "n/a"
        out[etf] = df
    return out


def sector_table(rot: dict[str, pd.DataFrame], date=None) -> pd.DataFrame:
    """Snapshot of every sector on `date` (default: last day), best first."""
    rows = []
    for etf, df in rot.items():
        d = df.loc[:date].iloc[-1] if date is not None else df.iloc[-1]
        rows.append({"etf": etf, "sector": ETF_TO_SECTOR.get(etf, etf), "rs_ratio": round(d.rs_ratio, 2),
                     "rs_momentum": round(d.rs_momentum, 2), "quadrant": d.quadrant})
    t = pd.DataFrame(rows)
    order = {"Leading": 0, "Improving": 1, "Weakening": 2, "Lagging": 3, "n/a": 4}
    t["_o"] = t["quadrant"].map(order)
    return t.sort_values(["_o", "rs_ratio"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def sector_ok(rot: dict[str, pd.DataFrame], etf: str | None, date, cfg=CFG) -> tuple[bool, str]:
    if not cfg["REQUIRE_SECTOR_ROTATION"]:
        return True, "off"
    if etf is None or etf not in rot:
        return bool(cfg["UNKNOWN_SECTOR_PASSES"]), "unknown"
    df = rot[etf].loc[:date]
    if df.empty:
        return False, "n/a"
    q = df["quadrant"].iloc[-1]
    return q in cfg["ALLOWED_QUADRANTS"], q


# =========================================================================== 2. fundamentals

def fetch_fundamental_snapshot(symbol: str) -> dict:
    """yfinance .info + current-quarter EPS trend, cached per trading day via CachedTicker."""
    import constants
    constants.CONFIG.setdefault("CACHE_INFO_DAILY", 1)
    constants.CONFIG.setdefault("CACHE_ANALYST_DAILY", 1)
    constants.CONFIG.setdefault("EARNINGS_FRESH_WINDOW_DAYS", 2)
    from cached_ticker import CachedTicker

    snap = {"symbol": symbol, "info": {}, "eps_current": None, "eps_30d_ago": None}
    try:
        tkr = CachedTicker(symbol)
        snap["info"] = tkr.get_info() or {}
        try:
            trend = tkr.get_eps_trend()
            if trend is not None and not trend.empty and "0q" in trend.index:
                snap["eps_current"] = _num(trend.loc["0q"].get("current"))
                snap["eps_30d_ago"] = _num(trend.loc["0q"].get("30daysAgo"))
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        snap["error"] = str(e)
    return snap


def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def peg_ratio(info: dict) -> tuple[float | None, str]:
    for key in ("trailingPegRatio", "pegRatio"):
        v = _num(info.get(key))
        if v is not None:
            return v, key
    fpe, g = _num(info.get("forwardPE")), _num(info.get("earningsGrowth"))
    if fpe is not None and g is not None and g > 0:
        return fpe / (g * 100), "fwdPE/earningsGrowth"
    return None, "missing"


def next_earnings(info: dict, now: pd.Timestamp | None = None) -> pd.Timestamp | None:
    now = now or pd.Timestamp.now(tz="UTC")
    stamps = []
    for key in ("earningsTimestampStart", "earningsTimestamp", "earningsTimestampEnd"):
        v = _num(info.get(key))
        if v:
            ts = pd.Timestamp(int(v), unit="s", tz="UTC")
            if ts >= now - pd.Timedelta(hours=12):
                stamps.append(ts)
    return min(stamps) if stamps else None


def fundamentals_gate(snap: dict, cfg=CFG) -> dict:
    """Returns {'pass': bool, 'fails': [...], plus the values used}."""
    info = snap.get("info") or {}
    fails = []

    rec = str(info.get("recommendationKey") or "none").lower()
    n_an = _num(info.get("numberOfAnalystOpinions")) or 0
    if rec not in cfg["ALLOWED_RECOMMENDATIONS"]:
        fails.append(f"rating={rec}")
    if n_an < cfg["MIN_ANALYSTS"]:
        fails.append(f"analysts={int(n_an)}")

    beta = _num(info.get("beta"))
    if beta is None:
        if cfg["REQUIRE_BETA"]:
            fails.append("beta=missing")
    elif beta > cfg["MAX_BETA"]:
        fails.append(f"beta={beta:.2f}")

    peg, peg_src = peg_ratio(info)
    if peg is None:
        if cfg["REQUIRE_PEG"]:
            fails.append("peg=missing")
    elif not (cfg["MIN_PEG"] < peg < cfg["MAX_PEG"]):
        fails.append(f"peg={peg:.2f}")

    # quality: 5 checks, need MIN_QUALITY_CHECKS
    margin = _num(info.get("profitMargins"))
    revg = _num(info.get("revenueGrowth"))
    fcf = _num(info.get("freeCashflow"))
    de = _num(info.get("debtToEquity"))
    eps_now, eps_ago = snap.get("eps_current"), snap.get("eps_30d_ago")
    checks = {
        "margin>0": margin is not None and margin > 0,
        "revenue_growth>0": revg is not None and revg > 0,
        "fcf>0": fcf is not None and fcf > 0,
        "eps_revisions_up": eps_now is not None and eps_ago is not None and eps_now >= eps_ago,
        "debt_ok": de is not None and de <= cfg["MAX_DEBT_TO_EQUITY"],
    }
    n_quality = int(sum(checks.values()))
    if n_quality < cfg["MIN_QUALITY_CHECKS"]:
        fails.append(f"quality={n_quality}/5")

    ne = next_earnings(info)
    days_to_earn = None
    if ne is not None:
        days_to_earn = (ne - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400
        if days_to_earn <= cfg["EARNINGS_BLACKOUT_DAYS"]:
            fails.append(f"earnings_in={days_to_earn:.0f}d")

    return {
        "pass": not fails, "fails": fails,
        "name": info.get("shortName") or info.get("longName") or snap.get("symbol"),
        "sector": info.get("sector"), "industry": info.get("industry"),
        "recommendation": rec, "rec_mean": _num(info.get("recommendationMean")), "analysts": int(n_an),
        "beta": beta, "peg": peg, "peg_src": peg_src,
        "fwd_pe": _num(info.get("forwardPE")), "market_cap": _num(info.get("marketCap")),
        "target_mean": _num(info.get("targetMeanPrice")),
        "inst_own": _num(info.get("heldPercentInstitutions")),
        "quality": checks, "quality_n": n_quality,
        "next_earnings": ne.tz_convert(ET).strftime("%Y-%m-%d") if ne is not None else None,
        "days_to_earnings": round(days_to_earn, 1) if days_to_earn is not None else None,
    }


# =========================================================================== 3-4. daily features (vectorized, causal)

def rolling_lrc(close: pd.Series, n: int, dev: float) -> pd.DataFrame:
    """Rolling least-squares channel over the last n bars, evaluated at each bar.
    Same maths as enrich_html.linear_regression_channel (population std of
    residuals), just computed for every day at once."""
    ramp = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    mean_y = close.rolling(n).mean()
    std_y = close.rolling(n).std(ddof=0)
    corr = close.rolling(n).corr(ramp)
    std_x = np.sqrt((n * n - 1) / 12.0)
    slope = corr * std_y / std_x
    r2 = corr ** 2
    resid_sd = std_y * np.sqrt((1 - r2).clip(lower=0))
    mid = mean_y + slope * (n - 1) / 2.0
    upper, lower = mid + dev * resid_sd, mid - dev * resid_sd
    width = (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "lrc_slope": slope,
        "lrc_slope_pct_day": slope / mid * 100,
        "lrc_r2": r2,
        "lrc_mid": mid, "lrc_upper": upper, "lrc_lower": lower,
        "lrc_pos_pct": (close - lower) / width * 100,
    })


def anchored_vwap_at_window_low(df: pd.DataFrame, n: int) -> pd.Series:
    """For each day D: VWAP anchored at the lowest low of the last n bars up to D.
    = average price paid by everyone who bought since the channel's low."""
    low = df["Low"].to_numpy(float)
    tp = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    out = np.full(len(df), np.nan)
    if len(df) < n:
        return pd.Series(out, index=df.index, name="avwap")
    cpv = np.concatenate([[0.0], np.cumsum(tp * vol)])
    cv = np.concatenate([[0.0], np.cumsum(vol)])
    arg = np.argmin(sliding_window_view(low, n), axis=1)            # position of the low inside each window
    ends = np.arange(n - 1, len(df))
    anchors = ends - (n - 1) + arg
    num = cpv[ends + 1] - cpv[anchors]
    den = cv[ends + 1] - cv[anchors]
    out[ends] = np.divide(num, den, out=np.full(len(ends), np.nan), where=den > 0)
    return pd.Series(out, index=df.index, name="avwap")


def daily_features(df: pd.DataFrame, spy_close: pd.Series | None = None, cfg=CFG) -> pd.DataFrame:
    """All daily indicators + gates. Row D uses only data up to and including D."""
    f = pd.DataFrame(index=df.index)
    c, v = df["Close"], df["Volume"]
    f["close"] = c
    f["ema_fast"] = c.ewm(span=cfg["EMA_FAST"], adjust=False).mean()
    f["ema_slow"] = c.ewm(span=cfg["EMA_SLOW"], adjust=False).mean()
    f["atr"] = bxi.atr(df, cfg["ATR_PERIOD"])
    a = bxi.adx(df, cfg["ADX_PERIOD"])
    f["adx"], f["pdi"], f["mdi"] = a["ADX"], a["+DI"], a["-DI"]

    f = f.join(rolling_lrc(c, cfg["LRC_LENGTH"], cfg["LRC_DEV"]))
    f["avwap"] = anchored_vwap_at_window_low(df, cfg["LRC_LENGTH"])

    # --- institutional flow
    chg = c.diff()
    upv = v.where(chg > 0, 0.0).rolling(cfg["UDV_WINDOW"]).sum()
    dnv = v.where(chg < 0, 0.0).rolling(cfg["UDV_WINDOW"]).sum()
    f["udv_ratio"] = upv / dnv.replace(0, np.nan)
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((c - df["Low"]) - (df["High"] - c)) / rng
    f["cmf"] = (mfm.fillna(0) * v).rolling(cfg["CMF_PERIOD"]).sum() / v.rolling(cfg["CMF_PERIOD"]).sum()
    obv = (np.sign(chg).fillna(0) * v).cumsum()
    ramp = pd.Series(np.arange(len(c), dtype=float), index=c.index)
    f["obv_trend"] = obv.rolling(cfg["OBV_SLOPE_WINDOW"]).corr(ramp)        # -1..+1, sign = OBV slope
    f["vol_avg50"] = v.rolling(50).mean()

    if spy_close is not None:
        s = spy_close.reindex(c.index).ffill()
        lb = cfg["RS_LOOKBACK"]
        f["rs_spy"] = (c / c.shift(lb)) / (s / s.shift(lb)) - 1

    # multi-horizon, volatility-adjusted time-series momentum (same as trend_data_pipeline / TSM project)
    dret = c.pct_change()
    moms = [c.pct_change(n) / (dret.rolling(n).std() * np.sqrt(252)).replace(0, np.nan) for n in cfg.get("TSM_HORIZONS", (20, 60, 120, 250))]
    f["tsm_mom"] = pd.concat(moms, axis=1).mean(axis=1)

    # --- gates
    on = pd.Series(True, index=c.index)
    ema_ok = ((f["ema_fast"] > f["ema_slow"]) & (c > f["ema_slow"])) if cfg.get("REQUIRE_EMA_STACK", True) else on
    adx_ok = ((f["adx"] >= cfg["ADX_MIN"]) & (f["pdi"] > f["mdi"])) if cfg.get("REQUIRE_ADX", True) else on
    f["trend_ok"] = (
        (f["lrc_slope_pct_day"] > cfg["LRC_MIN_SLOPE_PCT_DAY"]) & (f["lrc_r2"] >= cfg["LRC_MIN_R2"])
        & ema_ok & adx_ok
        & ((f["tsm_mom"] > 0) if cfg.get("REQUIRE_TSM_MOMENTUM", False) else on)
    )
    f["position_ok"] = f["lrc_pos_pct"] <= cfg["LRC_MAX_POSITION_PCT"]
    f["flow_n"] = ((f["udv_ratio"] > cfg["UDV_MIN"]).astype(int) + (f["cmf"] > cfg["CMF_MIN"]).astype(int)
                   + (f["obv_trend"] > 0).astype(int))
    above = (c > f["avwap"]) if cfg["REQUIRE_ABOVE_AVWAP"] else pd.Series(True, index=c.index)
    f["flow_ok"] = above & (f["flow_n"] >= cfg["MIN_FLOW_CHECKS"])
    f["tech_ok"] = f["trend_ok"] & f["position_ok"] & f["flow_ok"]
    return f


def tech_fail_reasons(row: pd.Series, cfg=CFG) -> list[str]:
    r = []
    if not row.get("lrc_slope_pct_day", 0) > cfg["LRC_MIN_SLOPE_PCT_DAY"]:
        r.append(f"lrc_slope={row.get('lrc_slope_pct_day', np.nan):.2f}%/d")
    if not row.get("lrc_r2", 0) >= cfg["LRC_MIN_R2"]:
        r.append(f"r2={row.get('lrc_r2', np.nan):.2f}")
    if cfg.get("REQUIRE_EMA_STACK", True) and not (row["ema_fast"] > row["ema_slow"] and row["close"] > row["ema_slow"]):
        r.append("ema_stack")
    if cfg.get("REQUIRE_ADX", True) and not (row["adx"] >= cfg["ADX_MIN"] and row["pdi"] > row["mdi"]):
        r.append(f"adx={row['adx']:.0f}")
    if not row.get("position_ok", False):
        r.append(f"chan_pos={row.get('lrc_pos_pct', np.nan):.0f}%")
    if cfg["REQUIRE_ABOVE_AVWAP"] and not row["close"] > row["avwap"]:
        r.append("below_avwap")
    if row.get("flow_n", 0) < cfg["MIN_FLOW_CHECKS"]:
        r.append(f"flow={int(row.get('flow_n', 0))}/3")
    return r


# =========================================================================== 5. volume profile + levels

def composite_profile(intra: pd.DataFrame | None, upto_date, sessions: int, bins: int, by_session: dict | None = None):
    """Volume profile over the last `sessions` sessions of 15m bars up to and including upto_date."""
    if by_session:
        upto = pd.Timestamp(upto_date).date()
        keys = [k for k in by_session if k <= upto][-sessions:]
        if not keys:
            return None
        return compute_volume_profile(pd.concat([by_session[k] for k in keys]), bins=bins)
    if intra is None or intra.empty:
        return None
    dates = intra.index.date
    upto = pd.Timestamp(upto_date).date()
    uniq = np.unique(dates[dates <= upto])
    if len(uniq) == 0:
        return None
    first = uniq[-sessions] if len(uniq) >= sessions else uniq[0]
    window = intra[(dates >= first) & (dates <= upto)]
    return compute_volume_profile(window, bins=bins)


@dataclass
class Setup:
    symbol: str
    date: str
    close: float
    atr: float
    stop: float
    stop_src: str
    tp1: float
    tp1_src: str
    tp2: float
    tp2_src: str
    rr: float                           # R:R to TP1 at the setup close (info)
    max_entry: float = 0.0              # highest 15m entry that still gives MIN_RR
    poc: float | None = None
    vah: float | None = None
    val: float | None = None
    avwap: float | None = None
    lrc_r2: float | None = None
    lrc_slope_pct_day: float | None = None
    lrc_lower: float | None = None
    lrc_upper: float | None = None
    lrc_pos_pct: float | None = None
    udv_ratio: float | None = None
    cmf: float | None = None
    obv_trend: float | None = None
    flow_n: int | None = None
    adx: float | None = None
    rs_spy: float | None = None
    sector: str | None = None
    sector_etf: str | None = None
    quadrant: str | None = None
    score: float = 0.0
    supports: list = field(default_factory=list)
    resistances: list = field(default_factory=list)
    fundamentals: dict | None = None

    def to_dict(self):
        return asdict(self)


def build_levels(daily_upto: pd.DataFrame, row: pd.Series, vp, cfg=CFG) -> dict | None:
    """Structural stop + targets around the reference price (last close).

    Stop    : the highest support that is at least MIN_STOP_ATR below price, minus
              STOP_ATR_CUSHION * ATR. Supports = daily fractal swing lows, VAL, POC,
              anchored VWAP. Too far (> MAX_STOP_ATR) -> None (no trade).
    Targets : resistances at least MIN_TARGET_ATR above price = daily fractal swing
              highs, the 252-day high and the channel's upper band. Volume-profile
              levels are targets only when VP_LEVELS_AS_TARGETS (in a trend the
              20-session VAH is just the top of the latest balance, which the trend
              is expected to migrate through) -- except the POC when price sits
              below the value area, where it acts as a magnet/ceiling.
              TP1 = nearest, TP2 = next one >= TP2_MIN_GAP_ATR beyond TP1 (or TP1 + 1R).
    """
    px, atr = float(row["close"]), float(row["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    sr = calculate_support_resistance(daily_upto, fractal_window=cfg["SR_FRACTAL_WINDOW"],
                                      cluster_pct=cfg["SR_CLUSTER_PCT"],
                                      lookback_days=cfg["SR_LOOKBACK_DAYS"],
                                      num_levels=cfg["SR_NUM_LEVELS"])
    sup = [(lv, "swing_low") for lv in sr["support"]]
    res = [(lv, "swing_high") for lv in sr["resistance"]]
    if vp is not None:
        for lv, name in ((vp.val, "VAL"), (vp.poc, "POC"), (vp.vah, "VAH")):
            if lv < px:
                sup.append((float(lv), name))
            elif cfg["VP_LEVELS_AS_TARGETS"] or (name == "POC" and px < vp.val):
                res.append((float(lv), name))
    if np.isfinite(row.get("avwap", np.nan)) and row["avwap"] < px:
        sup.append((float(row["avwap"]), "AVWAP"))
    hi252 = float(daily_upto["High"].iloc[-252:].max())
    if hi252 > px:
        res.append((hi252, "52w_high"))
    if np.isfinite(row.get("lrc_upper", np.nan)) and row["lrc_upper"] > px:
        res.append((float(row["lrc_upper"]), "LRC_upper"))

    # --- stop
    sup = sorted({(round(l, 2), s) for l, s in sup if l < px}, key=lambda x: -x[0])
    stop, stop_src = None, None
    for lv, src in sup:
        cand = lv - cfg["STOP_ATR_CUSHION"] * atr
        dist = px - cand
        if dist < cfg["MIN_STOP_ATR"] * atr:
            continue
        if dist > cfg["MAX_STOP_ATR"] * atr:
            break
        stop, stop_src = cand, src
        break
    if stop is None:
        return None

    # --- targets
    res = sorted({(round(l, 2), s) for l, s in res if l >= px + cfg["MIN_TARGET_ATR"] * atr}, key=lambda x: x[0])
    risk = px - stop
    if not res:
        return None
    tp1, tp1_src = res[0]
    beyond = [(l, s) for l, s in res if l >= tp1 + cfg["TP2_MIN_GAP_ATR"] * atr]
    if beyond:
        tp2, tp2_src = beyond[0]
    else:
        tp2, tp2_src = tp1 + risk, "TP1+1R"
    rr = cfg["MIN_RR"]
    return {
        "stop": round(stop, 2), "stop_src": stop_src, "tp1": tp1, "tp1_src": tp1_src,
        "tp2": round(tp2, 2), "tp2_src": tp2_src, "rr": round((tp1 - px) / risk, 2),
        # highest entry that still gives R:R >= MIN_RR to TP1
        "max_entry": round((tp1 + rr * stop) / (1 + rr), 2),
        "supports": [l for l, _ in sup[:5]], "resistances": [l for l, _ in res[:5]],
    }


QUAD_SCORE = {"Leading": 1.0, "Improving": 0.7, "Weakening": 0.3, "Lagging": 0.0}


def rank_score(s: Setup) -> float:
    """0..100 ranking when more setups fire than slots: trend quality, flow,
    sector strength, relative strength, R:R. Ranking only -- not a gate."""
    rr = min((s.rr or 0) / 4.0, 1.0)
    rs = float(np.clip(((s.rs_spy or 0) + 0.2) / 0.4, 0, 1))
    return round(100 * (0.30 * (s.lrc_r2 or 0) + 0.20 * (s.flow_n or 0) / 3 + 0.20 * QUAD_SCORE.get(s.quadrant, 0.5)
                        + 0.15 * rs + 0.15 * rr), 1)


def evaluate_setup(symbol: str, daily: pd.DataFrame, feats: pd.DataFrame, date, intra: pd.DataFrame | None,
                   sector: str | None, rot: dict | None, cfg=CFG,
                   by_session: dict | None = None) -> tuple[Setup | None, list[str]]:
    """Daily setup on `date` (a completed daily bar). Returns (Setup | None, fail reasons)."""
    row = feats.loc[date]
    if not bool(row["tech_ok"]):
        return None, tech_fail_reasons(row, cfg)
    etf = sector_etf(sector)
    quad = "off"
    if rot is not None:
        ok, quad = sector_ok(rot, etf, date, cfg)
        if not ok:
            return None, [f"sector={etf}:{quad}"]
    vp = composite_profile(intra, date, cfg["VP_SESSIONS"], cfg["VP_BINS"], by_session)
    daily_upto = daily.loc[:date]
    lv = build_levels(daily_upto, row, vp, cfg)
    if lv is None:
        return None, ["no_structural_stop_or_target"]
    # R:R gate. The trade is taken on a 15m pullback, not at today's close, so the
    # setup qualifies when the price that still yields MIN_RR (max_entry) is
    # within a normal pullback of the close, and still leaves a real stop distance.
    atr = float(row["atr"])
    if lv["rr"] < cfg["MIN_RR"]:
        if lv["max_entry"] < float(row["close"]) - cfg["SETUP_MAX_PULLBACK_ATR"] * atr:
            return None, [f"rr={lv['rr']:.2f}, needs pullback to {lv['max_entry']:.2f}"]
    if lv["max_entry"] - lv["stop"] < cfg["MIN_ENTRY_RISK_ATR"] * atr:
        return None, ["room_too_tight"]
    s = Setup(
        symbol=symbol, date=str(pd.Timestamp(date).date()), close=round(float(row["close"]), 2),
        atr=round(float(row["atr"]), 2), stop=lv["stop"], stop_src=lv["stop_src"], tp1=lv["tp1"],
        tp1_src=lv["tp1_src"], tp2=lv["tp2"], tp2_src=lv["tp2_src"], rr=lv["rr"], max_entry=lv["max_entry"],
        poc=round(vp.poc, 2) if vp else None, vah=round(vp.vah, 2) if vp else None,
        val=round(vp.val, 2) if vp else None, avwap=_r(row.get("avwap")),
        lrc_r2=_r(row.get("lrc_r2")), lrc_slope_pct_day=_r(row.get("lrc_slope_pct_day"), 3),
        lrc_lower=_r(row.get("lrc_lower")), lrc_upper=_r(row.get("lrc_upper")), lrc_pos_pct=_r(row.get("lrc_pos_pct"), 0),
        udv_ratio=_r(row.get("udv_ratio")), cmf=_r(row.get("cmf"), 3), obv_trend=_r(row.get("obv_trend")),
        flow_n=int(row.get("flow_n", 0)), adx=_r(row.get("adx"), 1), rs_spy=_r(row.get("rs_spy"), 3),
        sector=sector, sector_etf=etf, quadrant=quad, supports=lv["supports"], resistances=lv["resistances"],
    )
    s.score = rank_score(s)
    return s, []


def _r(v, nd=2):
    try:
        v = float(v)
        return round(v, nd) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# =========================================================================== 6. 15m entry

def _hhmm(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def prepare_intraday(intra: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Adds session VWAP (+ volume-weighted sigma), EMA9, time-of-day RVOL and the
    setup-independent part of the entry trigger. Vectorized form of
    intraday_strategy.add_vwap_bands (same maths, all sessions at once)."""
    d = intra.copy()
    idx = d.index
    d["session"] = idx.date
    d["slot"] = idx.hour * 60 + idx.minute
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    vol = d["Volume"].replace(0, np.nan)
    g = d["session"]
    cum_v = vol.groupby(g).cumsum()
    cum_pv = (tp * vol).groupby(g).cumsum()
    cum_p2v = ((tp ** 2) * vol).groupby(g).cumsum()
    d["vwap"] = (cum_pv / cum_v)
    d["vwap_std"] = np.sqrt(((cum_p2v / cum_v) - d["vwap"] ** 2).clip(lower=0))
    d["vwap"] = d.groupby("session")["vwap"].ffill()
    d["vwap_std"] = d.groupby("session")["vwap_std"].ffill()
    d["ema9"] = d["Close"].ewm(span=cfg["ENTRY_EMA"], adjust=False).mean()

    # time-of-day relative volume: bar volume / mean volume of the same slot over the previous N sessions
    piv = d.pivot_table(index="session", columns="slot", values="Volume", aggfunc="sum")
    base = piv.rolling(cfg["RVOL_LOOKBACK_SESSIONS"], min_periods=3).mean().shift(1)
    base_long = base.stack().rename("slot_avg_vol")
    d = d.join(base_long, on=["session", "slot"])
    d["rvol"] = d["Volume"] / d["slot_avg_vol"].replace(0, np.nan)

    prev_close = d.groupby("session")["Close"].shift(1)
    prev_vwap = d.groupby("session")["vwap"].shift(1)
    prev_low = d.groupby("session")["Low"].shift(1)
    touched = np.minimum(d["Low"], prev_low.fillna(d["Low"])) <= d["vwap"]
    reclaim = prev_close <= prev_vwap
    in_win = (d["slot"] >= _hhmm(cfg["ENTRY_START"])) & (d["slot"] <= _hhmm(cfg["ENTRY_END"]))
    common = in_win & (d["Close"] > d["Open"]) & (d["Close"] > d["ema9"]) & (d["rvol"] >= cfg["ENTRY_MIN_RVOL"])
    # entry type 1 -- VWAP reclaim/hold: back above session VWAP after touching it, not stretched
    d["base_trigger"] = (
        common & (d["Close"] > d["vwap"]) & (touched | reclaim.fillna(False))
        & (d["Close"] <= d["vwap"] + cfg["ENTRY_MAX_VWAP_SIGMA"] * d["vwap_std"])
    )
    # entry type 2 -- daily-level bounce (setup-specific part checked in scan_session): on a
    # pullback day price is usually still under session VWAP when it turns at the daily
    # POC / VAL / anchored VWAP / swing low, so VWAP only has to be "not collapsing"
    d["bounce_base"] = common & (d["Close"] >= d["vwap"] - cfg["BOUNCE_MAX_BELOW_VWAP_SIGMA"] * d["vwap_std"])
    d["prev_low"] = prev_low.fillna(d["Low"])
    return d


def split_by_session(prep: pd.DataFrame) -> dict:
    return {k: g for k, g in prep.groupby("session", sort=True)}


def find_entry(prep: pd.DataFrame | None, setup: Setup, cfg=CFG, min_rr: float | None = None,
               by_session: dict | None = None) -> dict:
    """Walk the sessions after the setup day, bar by bar.
    Returns {'status': 'triggered'|'invalidated'|'expired'|'pending', ...}.
    - invalidated: a bar traded at/below the stop before any trigger (structure broke)
    - a session that gaps > MAX_GAP_ATR above the setup close is skipped (no chasing)
    """
    setup_day = pd.Timestamp(setup.date).date()
    by_session = by_session if by_session is not None else split_by_session(prep)
    nxt = [s for s in by_session if s > setup_day][: cfg["SETUP_VALID_SESSIONS"]]
    if not nxt:
        return {"status": "pending"}
    for sess in nxt:
        res = scan_session(by_session[sess], setup, cfg, min_rr)
        if res is not None:
            return res
    return {"status": "expired" if len(nxt) >= cfg["SETUP_VALID_SESSIONS"] else "pending"}


def scan_session(bars: pd.DataFrame, setup: Setup, cfg=CFG, min_rr: float | None = None) -> dict | None:
    """One session of prepared 15m bars against one setup. Returns a
    triggered / invalidated dict, or None if nothing happened (setup still alive)."""
    min_rr = cfg["MIN_RR"] if min_rr is None else min_rr
    if bars is None or bars.empty:
        return None
    if bars["Open"].iloc[0] > setup.close + cfg["MAX_GAP_ATR"] * setup.atr:
        return None                                   # gapped too far above the setup -> don't chase today
    lows, closes = bars["Low"].to_numpy(), bars["Close"].to_numpy()
    reclaim = bars["base_trigger"].to_numpy(dtype=bool)
    bounce_ok = bars["bounce_base"].to_numpy(dtype=bool) if "bounce_base" in bars else np.zeros(len(bars), bool)
    tested = np.minimum(lows, bars["prev_low"].to_numpy()) if "prev_low" in bars else lows
    tol = cfg["LEVEL_TOUCH_ATR"] * setup.atr
    levels = [lv for lv in (setup.supports or []) if lv > setup.stop]
    for i in range(len(bars)):
        if lows[i] <= setup.stop:
            return {"status": "invalidated", "time": bars.index[i]}
        px = float(closes[i])
        kind, level = None, None
        types = cfg.get("ENTRY_TYPES", ("vwap_reclaim", "level_bounce"))
        if reclaim[i] and "vwap_reclaim" in types:
            kind = "vwap_reclaim"
        elif bounce_ok[i] and levels and "level_bounce" in types:
            # this bar or the previous one tagged a daily support and the bar closed back above it
            hit = [lv for lv in levels if tested[i] <= lv + tol and px > lv]
            if hit:
                kind, level = "level_bounce", max(hit)
        if kind is None:
            continue
        risk = px - setup.stop
        if risk < cfg["MIN_ENTRY_RISK_ATR"] * setup.atr or px >= setup.tp1:
            continue
        rr = (setup.tp1 - px) / risk
        if rr >= min_rr:
            b = bars.iloc[i]
            return {"status": "triggered", "time": bars.index[i], "trigger_close": round(px, 2), "rr": round(rr, 2),
                    "entry_type": kind, "level": level,
                    "vwap": round(float(b["vwap"]), 2), "rvol": round(float(b["rvol"]), 2)}
    return None


if __name__ == "__main__":
    # self-test: rolling LRC == enrich_html.linear_regression_channel on the last bar
    from enrich_html import linear_regression_channel
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2025-01-01", periods=300)
    close = pd.Series(100 + np.cumsum(rng.normal(0.1, 1, 300)), index=idx)
    df = pd.DataFrame({"Close": close})
    ref = linear_regression_channel(df, 50, 2.0)
    mine = rolling_lrc(close, 50, 2.0).iloc[-1]
    assert abs(ref["r2"] - round(mine.lrc_r2, 2)) < 1e-9, (ref["r2"], mine.lrc_r2)
    assert abs(ref["last_upper"] - round(mine.lrc_upper, 2)) < 0.011, (ref["last_upper"], mine.lrc_upper)
    assert abs(ref["last_lower"] - round(mine.lrc_lower, 2)) < 0.011
    print("rolling_lrc matches enrich_html.linear_regression_channel:", ref["r2"], ref["last_lower"], ref["last_upper"])
