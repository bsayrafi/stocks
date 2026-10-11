"""
The desktop HTML report: ticker cards, price / daily charts, the setups summary table.
Split out of enrich_html_v2.py, which re-exports every name (enrich_html_v2.X still works).
"""

import pandas as pd
import numpy as np
import html
import re
from setups_v2.config import _profiled
from setups_v2.data import _SESSION_NAMES
from setups_v2.engine import SETUP_EVIDENCE

def check_key(label: str) -> str:
    """Stable CSV column key for a check label: drops the (details), numbers and MA names
    so the same check always lands in the same column."""
    t = re.sub(r"\(.*?\)", "", label)
    t = re.sub(r"\b(EMA|SMA)\d*\b", "MA", t)
    t = re.sub(r"\d+(\.\d+)?", "", t)
    return _snake(t)[:40].strip("_")


def signal_badge_text(signal: dict) -> str:
    if signal["signal"] == "BUY":
        return f"BUY_SIGNAL - {signal['setup_label']}"
    if signal["signal"] == "WATCH":
        kind = {"price": "wait for price", "trigger": "wait for trigger"}.get(signal.get("watch_kind"), "")
        return f"WATCH - {signal['setup_label']}" + (f" ({kind})" if kind else "")
    return "WAIT"


def print_signal_block(signal: dict) -> None:
    ctx = signal["context"]
    print(f"\nSignal: {signal['signal']}"
          + (f"  -  {signal['setup_label']}" if signal["setup"] else ""))
    print(f"  Context: {ctx['state']}  (close {ctx['close']}, EMA50 {ctx['ema50']}, {ctx['ext_ema50_atr']:+.1f} ATR; "
          f"EMA50 slope {ctx['ema50_slope_pct']}%/5d; higher lows {ctx['higher_lows']}, "
          f"higher highs {ctx['higher_highs']}; {ctx['days_above_ema50']} close(s) above EMA50)")
    if not signal["setup"]:
        print(f"  No setup: {signal['wait_reason']}")
        if signal["watch_levels"]:
            print("  Levels to watch: " + ", ".join(f"{n} {v}" for n, v in signal["watch_levels"]))
        return
    print(f"  Setup: {signal['level_desc']}")
    print("  Required:")
    for lbl, ok in signal["required"]:
        print(f"      [{'PASS' if ok else 'FAIL'}] {lbl}")
    print("  Turn (daily AND 1h must have turned):")
    for lbl, ok in signal["turn"]:
        print(f"      [{'PASS' if ok else 'FAIL'}] {lbl}")
    print(f"  Confirmations ({signal['n_confirms']}/{len(signal['confirms'])}, need {signal['min_confirms']}):")
    for lbl, ok in signal["confirms"]:
        print(f"      [{'PASS' if ok else 'FAIL'}] {lbl}")
    print(f"  Plan: initial stop {signal['stop']} (1R = {signal['risk']:.2f}/share); no profit target. Once price trades at "
          f"{signal['arm_level']} (+2R) trail the stop under the daily EMA20 (now {signal['trail_now']}), raise only; "
          f"exit by the trail, the stop, or after 90 trading days.")
    print(f"        R:R filter: nearest real resistance {signal['target']} ({signal['target_src']}) gives R:R {signal['rr']}"
          + (f"; still worth buying up to {signal['buy_up_to']}" if signal.get("buy_up_to") else ""))
    if signal.get("setup") in SETUP_EVIDENCE:
        print("  Evidence: " + SETUP_EVIDENCE[signal["setup"]])
    if signal["fails"]:
        print("  Blocked by: " + "; ".join(signal["fails"]))
    if signal.get("buy_when"):
        print("  What would make it a BUY (in order, all must hold):")
        for item in signal["buy_when"]:
            print(f"      - {item}")
    if signal.get("buy_below"):
        print(f"  R:R reaches the minimum at or below {signal['buy_below']} (same stop/target) - a pullback entry zone")
    if signal.get("turn_above") and signal["signal"] != "BUY":
        print(f"  Turn watch: price back above {signal['turn_above']} (today's open / mid-range / session VWAP) would flip the turn checks")
    others = [s for s in signal["all_setups"] if s["type"] != signal["setup"]]
    if others:
        print("  Also detected: " + ", ".join(f"{s['label']} ({'BUY' if s['buy'] else 'watch' if s['near'] else 'weak'})" for s in others))


def render_signal_section(report: dict) -> str:
    sig, cfg = report["signal"], report["cfg"]
    ctx = sig["context"]
    good_state = ctx["state"] in ("UPTREND", "EARLY_UPTREND", "CORRECTION")

    def kv(label, value):
        return f"<tr><td>{html.escape(label)}</td><td>{value}</td></tr>"

    def chk(label, ok):
        return f"<tr><td>{html.escape(label)}</td><td>{_badge(ok)}</td></tr>"

    slope = ctx["ema50_slope_pct"]
    left = "".join([
        kv("Daily trend state", _badge(good_state, ctx["state"].replace("_", " "), ctx["state"].replace("_", " "))),
        kv("Close vs EMA50 / SMA50", f"{ctx['close']} vs {ctx['ema50']} / {ctx['sma50'] if ctx['sma50'] is not None else 'n/a'}"),
        kv("Extension above EMA50", f"{ctx['ext_ema50_atr']:+.1f} ATR "
           + _badge(ctx["ext_ema50_atr"] <= cfg["MAX_EXT_ATR"], "OK", "EXTENDED")),
        kv("EMA50 slope (5 days)", "n/a" if slope is None else f"{slope:+.2f}% " + _badge(slope > 0, "RISING", "FLAT/FALLING")),
        kv("Higher lows (major swings)", _badge(ctx["higher_lows"], "YES", "NO") + f" {html.escape(str(ctx['last_swing_lows']))}"
           + (f" <span class=\"note\">lowest since: {ctx['low_since_swing']}</span>" if ctx.get("low_since_swing") is not None else "")),
        kv("Higher highs (major swings)", _badge(ctx["higher_highs"], "YES", "NO") + f" {html.escape(str(ctx['last_swing_highs']))}"
           + (f" <span class=\"note\">highest since: {ctx['high_since_swing']}</span>" if ctx.get("high_since_swing") is not None else "")),
        kv("Closes above EMA50 / SMA50 in a row", f"{ctx['days_above_ema50']} / {ctx['days_above_sma50']}"),
        kv("Volume (live vs same time avg / last bars)",
           f"{sig['vol']['rvol_live'] if sig['vol']['rvol_live'] is not None else 'n/a'}x / "
           f"{html.escape(str(sig['vol']['done_ratios'][-3:]))}"),
    ])

    if sig["setup"]:
        plan_ok = sig["rr"] is not None and sig["rr"] >= cfg["SETUP_MIN_RR"]
        others = [s for s in sig["all_setups"] if s["type"] != sig["setup"]]
        right = (
            f"<h3>Setup: {html.escape(sig['setup_label'])}</h3>"
            f"<p class=\"note\">{html.escape(sig['level_desc'])}</p>"
            f"<p class=\"note\">{html.escape(SETUP_EVIDENCE.get(sig['setup'], ''))}</p>"
            "<h3>Required</h3><table class=\"detail-table\"><tbody>"
            + "".join(chk(l, ok) for l, ok in sig["required"]) + "</tbody></table>"
            "<h3>Turn (daily and 1h must both have turned)</h3><table class=\"detail-table\"><tbody>"
            + "".join(chk(l, ok) for l, ok in sig["turn"]) + "</tbody></table>"
            f"<h3>Confirmations ({sig['n_confirms']}/{len(sig['confirms'])}, need &ge; {sig['min_confirms']})</h3>"
            "<table class=\"detail-table\"><tbody>"
            + "".join(chk(l, ok) for l, ok in sig["confirms"]) + "</tbody></table>"
            "<h3>Trade plan</h3><table class=\"detail-table\"><tbody>"
            + kv("Entry (last price)", f"{report['current_price']:.2f}")
            + kv("Initial stop", f"{sig['stop']:.2f} <span class=\"note\">(1R = {sig['risk']:.2f} per share)</span>")
            + kv(f"+{cfg['TRAIL_ARM_R']:g}R level (arms the trail)", f"{sig['arm_level']:.2f}")
            + kv("Then trail the stop under", f"daily {cfg['TRAIL_MA']} <span class=\"note\">(now {sig['trail_now']:.2f}; raise only)</span>")
            + kv("Exit", f"trailing / initial stop, or after {cfg['EXIT_MAX_HOLD_DAYS']} trading days <span class=\"note\">(no profit target)</span>")
            + kv("Nearest real resistance (R:R filter)", f"{sig['target']:.2f} <span class=\"note\">({html.escape(sig['target_src'])})</span>")
            + kv(f"Reward : risk to it (min {cfg['SETUP_MIN_RR']:g})", f"{sig['rr']:.2f} " + _badge(plan_ok))
            + (kv(f"Buy up to (R:R stays >= {cfg['SETUP_MIN_RR']:g})",
                  f"{sig['buy_up_to']:.2f} <span class=\"note\">({(sig['buy_up_to'] / report['current_price'] - 1) * 100:+.1f}% from here)</span>")
               if sig.get("buy_up_to") else "")
            + (kv("Overhead resistance inside 1 ATR", html.escape(", ".join(str(x) for x in sig["overhead"])))
               if sig["overhead"] else "")
            + "</tbody></table>"
        )
        if sig["fails"]:
            right += f"<p class=\"note\"><strong>Blocked by:</strong> {html.escape('; '.join(sig['fails']))}</p>"
        if sig.get("buy_when"):
            right += ("<h3>What would make it a BUY (in order, all must hold)</h3><ul class=\"note\">"
                      + "".join(f"<li>{html.escape(x)}</li>" for x in sig["buy_when"]) + "</ul>")
        if sig.get("buy_below"):
            pct = (sig["buy_below"] / report["current_price"] - 1) * 100
            right += (f"<p class=\"note\"><strong>Entry zone:</strong> R:R to the resistance reaches {cfg['SETUP_MIN_RR']:g} at or below "
                      f"{sig['buy_below']:.2f} ({pct:.1f}% from here), keeping the same stop.</p>")
        if sig.get("turn_above") and sig["signal"] != "BUY":
            right += (f"<p class=\"note\"><strong>Turn watch:</strong> price back above {sig['turn_above']:.2f} "
                      "(today's open / mid-range / session VWAP) would flip the turn checks; MACD or %K must confirm.</p>")
        if others:
            right += ("<p class=\"note\"><strong>Also detected:</strong> "
                      + html.escape(", ".join(f"{s['label']} ({'BUY' if s['buy'] else 'watch' if s['near'] else 'weak'}, R:R {s['rr']})" for s in others))
                      + "</p>")
    else:
        watch = ", ".join(f"{n} {v}" for n, v in sig["watch_levels"]) or "none"
        right = ("<h3>Setup: none (traded setups)</h3>"
                 f"<p class=\"note\">{html.escape(sig['wait_reason'])}</p>"
                 f"<p class=\"note\"><strong>Levels to watch (below price):</strong> {html.escape(watch)}</p>")

    return (f'<div class="grid2"><div><h3>Trend context (daily)</h3>'
            f'<table class="detail-table"><tbody>{left}</tbody></table></div><div>{right}</div></div>')


# ---------------------------------------------------------------- HTML report

def _badge(ok: bool, true_text: str = "PASS", false_text: str = "FAIL") -> str:
    cls = "pass" if ok else "fail"
    text = true_text if ok else false_text
    return f'<span class="badge {cls}">{html.escape(str(text))}</span>'


def _four_col_table(items: list, extra_cls: str = "") -> str:
    """Render (label_html, value_html) pairs as a 4-column table
    (label | value | label | value). First half runs down the left pair, second
    half down the right pair, so it still reads top-to-bottom. Inputs must
    already be HTML-safe (so values can contain badges)."""
    if not items:
        return ""
    half = (len(items) + 1) // 2
    rows = ""
    for i in range(half):
        right = items[i + half] if i + half < len(items) else ("", "")
        rows += (f"<tr><td>{items[i][0]}</td><td>{items[i][1]}</td>"
                 f"<td>{right[0]}</td><td>{right[1]}</td></tr>")
    return f"""
      <table class="detail-table four-col {extra_cls}">
        <colgroup><col class="c-lbl"><col class="c-val"><col class="c-lbl"><col class="c-val"></colgroup>
        <tbody>{rows}
      </tbody></table>"""


def _hhmm_to_min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _series_xy(df_index, s, x, y):
    """[(x(i), y(v))] for the points of series `s` whose timestamp is on the chart index (in the order of `s`).
    Same result as looking every timestamp up in a {timestamp: position} dict, but vectorised (charts are drawn for
    every ticker, so this matters)."""
    if s is None or len(s) == 0:
        return []
    if df_index.is_unique:
        ip = df_index.get_indexer(s.index)
    else:
        pos = {ts: i for i, ts in enumerate(df_index)}
        ip = [pos.get(ts, -1) for ts in s.index]
    vals = s.to_numpy(dtype=float)
    return [(x(int(j)), y(float(v))) for j, v in zip(ip, vals) if j >= 0]


def render_price_chart_svg(report: dict) -> str:
    """Inline SVG candlestick chart of the last CHART_DAYS trading days (1h bars),
    with MA overlays, POC, 1h support, stop and take-profit levels, dotted
    market open/close markers, and a volume strip. Pure SVG - no JS or
    external libraries. Hover a candle for its OHLCV."""
    chart = report.get("chart")
    if not chart or chart["ohlcv"].empty:
        return ""
    df = chart["ohlcv"]
    overlays = chart.get("overlays") or {}
    cfg = report["cfg"]
    n = len(df)

    # ---- timestamps in the exchange's local time (for sessions + labels)
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert(cfg.get("MARKET_TZ", "America/New_York"))
    dates = idx.date
    mins = np.asarray(idx.hour * 60 + idx.minute)
    open_m = _hhmm_to_min(cfg.get("MARKET_OPEN", "09:30"))
    close_m = _hhmm_to_min(cfg.get("MARKET_CLOSE", "16:00"))

    # ---- geometry: a small gap between sessions so close/open lines don't overlap
    W = 1000  # ~ the card's rendered width on desktop, so text renders near 1:1
    pad_l, pad_r, pad_t = 56, 160, 10
    AXIS_W = 46  # room for right-axis tick labels; level labels sit to the right of it
    price_h, gap, vol_h, axis_h = 500, 12, 110, 22
    H = pad_t + price_h + gap + vol_h + axis_h
    plot_w = W - pad_l - pad_r
    new_day = [i == 0 or dates[i] != dates[i - 1] for i in range(n)]
    n_days = sum(new_day)
    SESSION_GAP = 0.8  # in candle widths
    step = plot_w / (n + SESSION_GAP * (n_days - 1))
    body_w = max(1.5, step * 0.62)

    left, cur = [], float(pad_l)
    for i in range(n):
        if new_day[i] and i > 0:
            cur += SESSION_GAP * step
        left.append(cur)
        cur += step

    def x(i):
        return left[i] + step / 2

    # ---- horizontal levels
    levels = [
        ("Target", report.get("take_profit"), "lvl-target"),
        (f"Current {((report.get('ext') or {}).get('current') or {}).get('session', '').upper()}",
         ((report.get("ext") or {}).get("current") or {}).get("price"), "lvl-premkt"),
        ("POC", report.get("poc"), "lvl-poc"),
        ("Support", report.get("entry_support"), "lvl-support"),
        ("Stop", report.get("stop"), "lvl-stop"),
    ]
    levels = [(lbl, float(v), cls) for lbl, v, cls in levels if v is not None and pd.notna(v)]

    overlays_clean = {name: s.dropna() for name, s in overlays.items()}
    vwap_s = chart.get("vwap")
    vwap_clean = vwap_s.dropna() if vwap_s is not None else pd.Series(dtype=float)
    if not vwap_clean.empty:
        overlays_clean_for_range = list(overlays_clean.values()) + [vwap_clean]
    else:
        overlays_clean_for_range = list(overlays_clean.values())
    lrc = chart.get("lrc")
    if lrc:
        overlays_clean_for_range += [lrc["upper"], lrc["lower"]]
    candidates = [float(df["Low"].min()), float(df["High"].max())] + [v for _, v, _ in levels]
    for s in overlays_clean_for_range:
        if not s.empty:
            candidates += [float(s.min()), float(s.max())]
    lo, hi = min(candidates), max(candidates)
    pad_p = (hi - lo) * 0.04 or 1.0
    lo, hi = lo - pad_p, hi + pad_p

    def y(p):
        return pad_t + (hi - p) / (hi - lo) * price_h

    vol_top = pad_t + price_h + gap
    vol_bot = vol_top + vol_h
    vmax = float(df["Volume"].max()) or 1.0

    def vy(v):
        return vol_bot - (v / vmax) * vol_h

    parts = []

    # ---- price grid: "nice" round-number ticks (…0.5, 1, 2, 2.5, 5…), ~10 across
    # the panel (~50px apart), dotted gridlines, labels on both axes + tick marks on the right
    raw_step = (hi - lo) / 10
    mag = 10 ** np.floor(np.log10(raw_step))
    tick_step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda st: abs(np.log(st / raw_step)))
    decimals = 2 if tick_step < 1 else (1 if tick_step % 1 else 0)
    right_x = pad_l + plot_w
    p = np.ceil(lo / tick_step) * tick_step
    while p <= hi:
        yy = y(p)
        label = f"{p:.{decimals}f}"
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{right_x}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<line class="tick" x1="{right_x}" x2="{right_x + 4}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="axis" x="{pad_l - 6}" y="{yy + 3:.1f}" text-anchor="end">{label}</text>')
        parts.append(f'<text class="axis" x="{right_x + 7}" y="{yy + 3:.1f}">{label}</text>')
        p += tick_step
    parts.append(f'<line class="tick" x1="{right_x}" x2="{right_x}" y1="{pad_t}" y2="{pad_t + price_h}"/>')

    hhhl_by_date = chart.get("hhhl_by_date") or {}   # daily HH/HL type per date (triangle after the date)

    # ---- market open / close markers (dotted white) + date labels, per session
    day_starts = [i for i in range(n) if new_day[i]] + [n]
    for d in range(n_days):
        ks = list(range(day_starts[d], day_starts[d + 1]))
        starts = mins[ks]
        kind = hhhl_by_date.get(idx[ks[0]].date())
        mark = ('<tspan class="hhhl-mark up"> &#9650;</tspan>' if kind == "up" else
                '<tspan class="hhhl-mark down"> &#9660;</tspan>' if kind == "down" else "")
        parts.append(f'<text class="axis" x="{left[ks[0]] + 2:.1f}" y="{H - 6}">'
                     f'{idx[ks[0]].strftime("%a %d %b")}{mark}</text>')

        # open: left edge of first bar if it starts at/after the open,
        # else interpolate inside the bar that contains the open (pre-market data)
        if starts[0] >= open_m:
            ox = left[ks[0]]
        else:
            j = max(j for j in range(len(ks)) if starts[j] <= open_m)
            nxt = starts[j + 1] if j + 1 < len(ks) else starts[j] + 60
            ox = left[ks[j]] + step * min(1.0, (open_m - starts[j]) / max(nxt - starts[j], 1))

        # close: right edge of the last bar if it's the closing bar, else
        # interpolate (after-hours data); skip if the session hasn't reached it yet
        cx = None
        if starts[-1] < close_m:
            if close_m - starts[-1] <= 60:
                cx = left[ks[-1]] + step
        else:
            j = max(j for j in range(len(ks)) if starts[j] <= close_m) if starts[0] <= close_m else None
            if j is not None:
                nxt = starts[j + 1] if j + 1 < len(ks) else starts[j] + 60
                cx = left[ks[j]] + step * min(1.0, (close_m - starts[j]) / max(nxt - starts[j], 1))

        for xx, what in ((ox, "Market open"), (cx, "Market close")):
            if xx is not None:
                parts.append(f'<line class="session" x1="{xx:.1f}" x2="{xx:.1f}" y1="{pad_t}" y2="{vol_bot}">'
                             f'<title>{what} {idx[ks[0]].strftime("%a %d %b")}</title></line>')

    # ---- standard filled candles: green if close >= open, red otherwise
    color_cls = ["up" if c >= o else "down" for o, c in zip(df["Open"], df["Close"])]

    # ---- linear regression channel (drawn behind candles)
    if lrc:
        def _pts(series):
            return _series_xy(df.index, series, x, y)
        up_pts, lo_pts, mid_pts = _pts(lrc["upper"]), _pts(lrc["lower"]), _pts(lrc["mid"])
        if up_pts:
            poly = " ".join(f"{a:.1f},{b:.1f}" for a, b in up_pts + lo_pts[::-1])
            parts.append(f'<polygon class="lrc-fill" points="{poly}"/>')
            for cls, pts in (("lrc-band", up_pts), ("lrc-band", lo_pts), ("lrc-mid", mid_pts)):
                parts.append(f'<polyline class="{cls}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')

    # ---- volume bars (columns read once as arrays - row-by-row pandas access is slow)
    O_, H_, L_, C_, V_ = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close", "Volume"))
    ts_py = idx.to_pydatetime()
    for i in range(n):
        cls = color_cls[i]
        top = vy(float(V_[i]))
        parts.append(f'<rect class="vol {cls}" x="{x(i) - body_w / 2:.1f}" y="{top:.1f}" '
                     f'width="{body_w:.1f}" height="{vol_bot - top:.1f}"/>')

    # ---- candles (with native hover tooltip)
    for i in range(n):
        o, h, l, c, v = float(O_[i]), float(H_[i]), float(L_[i]), float(C_[i]), float(V_[i])
        cls = color_cls[i]
        body_top, body_bot = y(max(o, c)), y(min(o, c))
        tip = f"{ts_py[i].strftime('%a %d %b %H:%M')}  O {o:.2f}  H {h:.2f}  L {l:.2f}  C {c:.2f}  Vol {v:,.0f}"
        parts.append(
            f'<g class="candle {cls}"><title>{html.escape(tip)}</title>'
            f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(h):.1f}" y2="{y(l):.1f}"/>'
            f'<rect x="{x(i) - body_w / 2:.1f}" y="{body_top:.1f}" width="{body_w:.1f}" '
            f'height="{max(body_bot - body_top, 1):.1f}"/></g>'
        )

    # ---- moving-average overlays
    pos = {ts: i for i, ts in enumerate(df.index)}
    legend_mas = []
    for k, (name, s) in enumerate(overlays_clean.items()):
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in _series_xy(df.index, s, x, y))
        if pts:
            parts.append(f'<polyline class="ma ma-{k % 4}" points="{pts}"><title>{html.escape(name)}</title></polyline>')
        legend_mas.append(f'<i class="sw ma-sw ma-{k % 4}"></i>{html.escape(name)}')

    # ---- session VWAP: one segment per session (it resets at each open)
    if not vwap_clean.empty:
        vdays = np.asarray((vwap_clean.index.tz_convert(cfg.get("MARKET_TZ", "America/New_York"))
                            if vwap_clean.index.tz is not None else vwap_clean.index).date)
        for dday in pd.unique(vdays):
            seg = vwap_clean[vdays == dday]
            pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in _series_xy(df.index, seg, x, y))
            if len(seg) == 1:
                xx, yy = x(pos[seg.index[0]]), y(float(seg.iloc[0]))
                pts = f"{xx - step / 2:.1f},{yy:.1f} {xx + step / 2:.1f},{yy:.1f}"
            parts.append(f'<polyline class="vwap" points="{pts}"><title>VWAP {dday}</title></polyline>')
        levels.append(("VWAP", float(vwap_clean.iloc[-1]), "lvl-vwap"))
        legend_mas.append('<i class="sw ma-sw vwap-sw"></i>VWAP')

    # ---- horizontal levels, labelled on the right (nudged apart if they collide)
    label_ys = []
    for lbl, val, cls in sorted(levels, key=lambda t: -t[1]):
        yy = y(val)
        ly = yy + 3
        while any(abs(ly - prev) < 12 for prev in label_ys):
            ly += 12
        label_ys.append(ly)
        if cls != "lvl-vwap":  # VWAP is a moving line - label only, no horizontal
            parts.append(f'<line class="lvl {cls}" x1="{pad_l}" x2="{pad_l + plot_w}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="lvl-label {cls}" x="{pad_l + plot_w + AXIS_W}" y="{ly:.1f}">{lbl} {val:.2f}</text>')

    # ---- last-price marker
    last_c = float(df["Close"].iloc[-1])
    parts.append(f'<circle class="last" cx="{x(n - 1):.1f}" cy="{y(last_c):.1f}" r="3"/>')

    lrc_legend = ' <i class="sw lrc-sw"></i>Reg. channel' if lrc else ""
    cur = (report.get("ext") or {}).get("current")
    if cur:
        lrc_legend = f' <i class="sw premkt-sw"></i>Current price ({_SESSION_NAMES[cur["session"]]})' + lrc_legend
    first_o = float(df["Open"].iloc[0])
    chg = (last_c / first_o - 1) * 100 if first_o else 0.0
    chg_cls = "pass" if chg >= 0 else "fail"

    return f"""
      <div class="chart-wrap">
        <div class="chart-head">
          <span>Last {n_days} trading days &middot; 1h candles</span>
          <span class="badge {chg_cls}">{chg:+.2f}%</span>
          <span class="legend">{' '.join(legend_mas)}
            <i class="sw lvl-poc"></i>POC <i class="sw lvl-support"></i>1h support
            <i class="sw lvl-stop"></i>Stop <i class="sw lvl-target"></i>Target
            <i class="sw session-sw"></i>Open/close{lrc_legend}</span>
        </div>
        <svg class="price-chart" viewBox="0 0 {W} {H}"
             role="img" aria-label="{html.escape(report['ticker'])} {n_days}-day price chart">
          {''.join(parts)}
        </svg>
      </div>"""


def render_daily_chart_svg(report: dict) -> str:
    """Inline SVG of the last DAILY_CHART_DAYS daily candles (built from the
    same hourly download - no extra data), with the MA overlays, daily
    support/resistance, stop and target, HH/HL markers (green triangle under an
    up day, red over a down day) and a volume strip."""
    daily = (report.get("chart") or {}).get("daily")
    if not daily or daily["ohlcv"].empty:
        return ""
    df = daily["ohlcv"]
    overlays = {k: v.dropna() for k, v in (daily.get("overlays") or {}).items()}
    cfg = report["cfg"]
    by_date = (report.get("chart") or {}).get("hhhl_by_date") or {}
    tz = cfg.get("MARKET_TZ", "America/New_York")
    idx = df.index.tz_convert(tz) if df.index.tz is not None else df.index
    n = len(df)
    in_progress = daily.get("today_in_progress", False)

    W = 1000
    pad_l, pad_r, pad_t = 56, 160, 16
    AXIS_W = 46
    price_h, gap, vol_h, axis_h = 320, 10, 70, 34
    H = pad_t + price_h + gap + vol_h + axis_h
    plot_w = W - pad_l - pad_r
    step = plot_w / n
    body_w = max(2.0, step * 0.6)

    def x(i):
        return pad_l + step * (i + 0.5)

    sr = (report.get("sr_levels") or {}).get("1D") or {}
    dsr = daily.get("sr") or {}
    levels = []
    for k, lvl in enumerate((dsr.get("resistances") or [])[:2]):
        rr = f" {lvl['r']:.1f}R" if lvl.get("r") is not None else ""
        levels.append((f"T{k + 1}{rr}", lvl["price"], "lvl-target" if k == 0 else "lvl-target2"))
    tgt = dsr.get("target") or {}
    if tgt and not dsr.get("resistances"):   # fallback target (no resistance above)
        levels.append((f"T {tgt['r']:g}R", tgt["price"], "lvl-target"))
    for k, lvl in enumerate((dsr.get("supports") or [])[:2]):
        levels.append((f"S{k + 1} x{lvl['touches']}", lvl["price"], "lvl-support" if k == 0 else "lvl-support2"))
    cur = (report.get("ext") or {}).get("current")
    if cur:   # same yellow dotted "current price" line as on the 1h chart (not during regular hours)
        levels.append((f"Current {cur['session'].upper()}", cur["price"], "lvl-premkt"))
    levels += [
              ("Stop", report.get("stop"), "lvl-stop")]
    levels = [(lbl, float(v), cls) for lbl, v, cls in levels if v is not None and pd.notna(v)]

    av = daily.get("avwap")
    av_s = av["series"].dropna() if av else pd.Series(dtype=float)
    cands = [float(df["Low"].min()), float(df["High"].max())] + [v for _, v, _ in levels]
    if not av_s.empty:
        cands += [float(av_s.min()), float(av_s.max())]
    for sser in overlays.values():
        if not sser.empty:
            cands += [float(sser.min()), float(sser.max())]
    lo, hi = min(cands), max(cands)
    pad_p = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad_p, hi + pad_p

    def y(p):
        return pad_t + (hi - p) / (hi - lo) * price_h

    vol_top = pad_t + price_h + gap
    vol_bot = vol_top + vol_h
    vmax = float(df["Volume"].max()) or 1.0
    parts = []

    # ---- grid (same "nice number" ticks as the hourly chart)
    raw_step = (hi - lo) / 8
    mag = 10 ** np.floor(np.log10(raw_step))
    tick_step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda st: abs(np.log(st / raw_step)))
    decimals = 2 if tick_step < 1 else (1 if tick_step % 1 else 0)
    right_x = pad_l + plot_w
    p = np.ceil(lo / tick_step) * tick_step
    while p <= hi:
        yy = y(p)
        label = f"{p:.{decimals}f}"
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{right_x}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<line class="tick" x1="{right_x}" x2="{right_x + 4}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="axis" x="{pad_l - 6}" y="{yy + 3:.1f}" text-anchor="end">{label}</text>')
        parts.append(f'<text class="axis" x="{right_x + 7}" y="{yy + 3:.1f}">{label}</text>')
        p += tick_step
    parts.append(f'<line class="tick" x1="{right_x}" x2="{right_x}" y1="{pad_t}" y2="{pad_t + price_h}"/>')

    # ---- date labels with the abbreviated day name underneath. Every bar is
    # labelled when there's room (~20 bars); with many bars, every other one.
    every = max(1, int(np.ceil(36 / step)))   # keep labels ~36px+ apart (20 bars: all, 50 bars: every 3rd)
    ts_py = idx.to_pydatetime()               # plain datetimes: much faster than indexing a DatetimeIndex per bar
    idx = ts_py
    for i in range(n):
        if (n - 1 - i) % every == 0:
            parts.append(f'<text class="axis" x="{x(i):.1f}" y="{vol_bot + 14}" text-anchor="middle">'
                         f'{idx[i].strftime("%d %b")}</text>')
            parts.append(f'<text class="axis dayname" x="{x(i):.1f}" y="{vol_bot + 27}" text-anchor="middle">'
                         f'{idx[i].strftime("%a")}</text>')

    # ---- dotted vertical line at the start of each new week (just before Monday,
    # or the first trading day of the week when Monday is a holiday)
    for i in range(1, n):
        if idx[i].isocalendar()[:2] != idx[i - 1].isocalendar()[:2]:
            wx = x(i) - step / 2
            parts.append(f'<line class="weeksep" x1="{wx:.1f}" x2="{wx:.1f}" y1="{pad_t}" y2="{vol_bot}">'
                         f'<title>week of {idx[i].strftime("%d %b")}</title></line>')

    # ---- volume, candles, HH/HL markers
    O_, H_, L_, C_, V_ = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close", "Volume"))
    for i in range(n):
        o, h, l, c, v = float(O_[i]), float(H_[i]), float(L_[i]), float(C_[i]), float(V_[i])
        cls = "up" if c >= o else "down"
        partial = in_progress and i == n - 1
        top = vol_bot - (v / vmax) * vol_h
        parts.append(f'<rect class="vol {cls}" x="{x(i) - body_w / 2:.1f}" y="{top:.1f}" '
                     f'width="{body_w:.1f}" height="{vol_bot - top:.1f}"/>')
        kind = by_date.get(idx[i].date())
        tip = (f"{idx[i].strftime('%a %d %b')}{' (in progress)' if partial else ''}  O {o:.2f}  H {h:.2f}  "
               f"L {l:.2f}  C {c:.2f}  Vol {v:,.0f}"
               + (f"  | {'higher high + higher low' if kind == 'up' else 'lower high + lower low' if kind == 'down' else 'mixed'}"
                  if kind else ""))
        body_top, body_bot = y(max(o, c)), y(min(o, c))
        parts.append(
            f'<g class="candle {cls}{" partial" if partial else ""}"><title>{html.escape(tip)}</title>'
            f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(h):.1f}" y2="{y(l):.1f}"/>'
            f'<rect x="{x(i) - body_w / 2:.1f}" y="{body_top:.1f}" width="{body_w:.1f}" '
            f'height="{max(body_bot - body_top, 1):.1f}"/></g>')
        if kind == "up":
            parts.append(f'<text class="hhhl-mark up" x="{x(i):.1f}" y="{y(l) + 13:.1f}" text-anchor="middle">&#9650;</text>')
        elif kind == "down":
            parts.append(f'<text class="hhhl-mark down" x="{x(i):.1f}" y="{y(h) - 5:.1f}" text-anchor="middle">&#9660;</text>')

    # ---- MA overlays
    pos = {ts: i for i, ts in enumerate(df.index)}
    legend = []
    for k, (name, sser) in enumerate(overlays.items()):
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in _series_xy(df.index, sser, x, y))
        if pts:
            parts.append(f'<polyline class="ma ma-{k % 4}" points="{pts}"><title>{html.escape(name)} (daily)</title></polyline>')
        legend.append(f'<i class="sw ma-sw ma-{k % 4}"></i>{html.escape(name)}')

    # ---- anchored VWAP (line from the anchor day on, dot on the anchor low/high)
    if not av_s.empty:
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in _series_xy(df.index, av_s, x, y))
        parts.append(f'<polyline class="avwap" points="{pts}"><title>Anchored VWAP from the '
                     f'{"low" if av["anchor_type"] == "low" else "high"} of '
                     f'{pd.Timestamp(av["anchor_date"]).strftime("%d %b")}</title></polyline>')
        if av["anchor_ts"] in pos:
            parts.append(f'<circle class="avwap-anchor" cx="{x(pos[av["anchor_ts"]]):.1f}" '
                         f'cy="{y(av["anchor_price"]):.1f}" r="3.5"/>')
        levels.append(("AVWAP", float(av_s.iloc[-1]), "lvl-avwap"))
        legend.append('<i class="sw ma-sw avwap-sw"></i>Anchored VWAP')

    # ---- horizontal levels with right-side labels
    label_ys = []
    for lbl, val, cls in sorted(levels, key=lambda t: -t[1]):
        yy = y(val)
        ly = yy + 3
        while any(abs(ly - prev) < 12 for prev in label_ys):
            ly += 12
        label_ys.append(ly)
        if cls == "lvl-avwap":   # moving line - label only
            parts.append(f'<text class="lvl-label {cls}" x="{right_x + AXIS_W}" y="{ly:.1f}">{lbl} {val:.2f}</text>')
            continue
        parts.append(f'<line class="lvl {cls}" x1="{pad_l}" x2="{right_x}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="lvl-label {cls}" x="{right_x + AXIS_W}" y="{ly:.1f}">{lbl} {val:.2f}</text>')

    first_o, last_c = float(df["Open"].iloc[0]), float(df["Close"].iloc[-1])
    chg = (last_c / first_o - 1) * 100 if first_o else 0.0
    st = report.get("hhhl") or {}
    return f"""
      <div class="chart-wrap">
        <div class="chart-head">
          <span>Last {n} trading days &middot; daily candles{' (today in progress)' if in_progress else ''}</span>
          <span class="badge {'pass' if chg >= 0 else 'fail'}">{chg:+.2f}%</span>
          <span class="legend">{' '.join(legend)}
            <i class="sw lvl-support"></i>Support S1/S2 <i class="sw lvl-target"></i>Target T1/T2
            <i class="sw lvl-stop"></i>Stop{f' <i class="sw premkt-sw"></i>Current price ({_SESSION_NAMES[cur["session"]]})' if cur else ""}
            <span class="hhhl-mark up">&#9650;</span>HH+HL <span class="hhhl-mark down">&#9660;</span>LH+LL</span>
        </div>
        <svg class="price-chart" viewBox="0 0 {W} {H}"
             role="img" aria-label="{html.escape(report['ticker'])} daily chart">
          {''.join(parts)}
        </svg>
      </div>"""


def _streak_text(streak) -> str:
    kind, length = streak if streak else (None, 0)
    if not kind:
        return "N/A"
    return f"{length} {kind} day{'s' if length != 1 else ''}"


def render_chart_toggles(report: dict) -> str:
    """Two independent show/hide links side by side ("1h chart", "Daily chart"),
    each opening its chart below. Pure CSS (hidden checkboxes) - no JavaScript."""
    cfg = report["cfg"]
    hourly, daily = render_price_chart_svg(report), render_daily_chart_svg(report)
    if not hourly and not daily:
        return ""
    uid = "".join(c if c.isalnum() else "_" for c in report["ticker"])
    h_chk = " checked" if cfg.get("CHART_OPEN", False) else ""
    d_chk = " checked" if cfg.get("DAILY_CHART_OPEN", False) else ""
    # daily chart first (link on the left, chart on top), then the hourly chart
    links, panels, inputs = [], [], []
    if daily:
        inputs.append(f'<input type="checkbox" class="tog tog-d" id="tog-d-{uid}"{d_chk}>')
        links.append(f'<label for="tog-d-{uid}" class="lnk lnk-d"><span class="when-closed">&#9656; Show daily chart</span>'
                     f'<span class="when-open">&#9662; Hide daily chart</span></label>')
        panels.append(f'<div class="panel panel-d">{daily}</div>')
    if hourly:
        inputs.append(f'<input type="checkbox" class="tog tog-h" id="tog-h-{uid}"{h_chk}>')
        links.append(f'<label for="tog-h-{uid}" class="lnk lnk-h"><span class="when-closed">&#9656; Show 1h chart</span>'
                     f'<span class="when-open">&#9662; Hide 1h chart</span></label>')
        panels.append(f'<div class="panel panel-h">{hourly}</div>')
    return (f'\n      <div class="charts">{"".join(inputs)}'
            f'<div class="tog-links">{"".join(links)}</div>{"".join(panels)}</div>')


def _sr_text(levels, with_r: bool = False) -> str:
    """'88.20 (-5.1%, 3 touches) / 84.10 (-9.5%, 1 touch)' for the two nearest levels."""
    if not levels:
        return "none in window"
    parts = []
    for lvl in levels[:2]:
        extra = f", {lvl['r']:.1f}R" if with_r and lvl.get("r") is not None else ""
        parts.append(f"{lvl['price']} ({lvl['dist_pct']:+.1f}%, {lvl['touches']} "
                     f"touch{'es' if lvl['touches'] != 1 else ''}{extra})")
    return " / ".join(parts)


def near_level(price: float, level: float | None, atr: float | None, max_atr: float = 0.5) -> dict | None:
    """Is `price` within max_atr daily ATRs of `level`? Returns
    {"near": bool, "dist_pct": %, "dist_atr": distance in ATRs (positive = price above level)}."""
    if level is None or not price:
        return None
    dist = price - level
    dist_atr = dist / atr if atr else None
    near = abs(dist_atr) <= max_atr if dist_atr is not None else abs(dist / price) <= 0.01
    return {"near": bool(near), "dist_pct": round(dist / price * 100, 2),
            "dist_atr": round(dist_atr, 2) if dist_atr is not None else None}


def premarket_near_support(report: dict) -> dict | None:
    """Is the pre-market price within NEAR_SR_ATR daily ATRs of S1 or S2?
    Returns {"near": bool, "level": "S1"/"S2", "price", "dist_pct", "dist_atr"} for the
    closest of the two, or None when there's no pre-market price or no support."""
    pm = report.get("pre_market")
    sups = ((report.get("day_sr") or {}).get("supports") or [])[:2]
    if pm is None or not sups:
        return None
    atr, max_atr = report.get("daily_atr"), report["cfg"].get("NEAR_SR_ATR", 0.5)
    best = None
    for k, lvl in enumerate(sups):
        nr = near_level(pm, lvl["price"], atr, max_atr)
        if nr and (best is None or abs(nr["dist_pct"]) < abs(best["dist_pct"])):
            best = {**nr, "level": f"S{k + 1}", "price": lvl["price"]}
    return best


def _premarket_near_row(report: dict) -> tuple:
    pns = premarket_near_support(report)
    if pns is None:
        why = "no pre-market price" if report.get("pre_market") is None else "no support in window"
        return ("Pre-market near support", f"N/A ({why})")
    side = "above" if pns["dist_pct"] >= 0 else "below"
    atr_txt = f", {abs(pns['dist_atr']):.2f} ATR" if pns["dist_atr"] is not None else ""
    return ("Pre-market near support",
            f"{_badge(pns['near'], 'YES', 'NO')} {abs(pns['dist_pct']):.2f}% {side} "
            f"{pns['level']} ({pns['price']}){atr_txt}")


def _near_sr_rows(report: dict) -> list:
    """'Near S1' / 'Near S2' rows for the Other levels table."""
    rows = []
    sups = (report.get("day_sr") or {}).get("supports") or []
    atr, max_atr = report.get("daily_atr"), report["cfg"].get("NEAR_SR_ATR", 0.5)
    for k in range(2):
        lvl = sups[k] if len(sups) > k else None
        nr = near_level(report["current_price"], lvl["price"] if lvl else None, atr, max_atr)
        if not nr:
            rows.append((f"Near S{k + 1}", "N/A (no support in window)"))
            continue
        atr_txt = f", {nr['dist_atr']:.2f} ATR" if nr["dist_atr"] is not None else ""
        rows.append((f"Near S{k + 1} ({lvl['price']})",
                     f"{_badge(nr['near'], 'YES', 'NO')} {nr['dist_pct']:.2f}% above{atr_txt}"))
    return rows


def _extended_hours_rows(report: dict) -> list:
    """Close / After-hours / Overnight / Pre-market rows, each vs the close."""
    ext = report.get("ext")
    if not ext:
        return [("Close / extended hours", "N/A (no Alpaca data)")]
    rows = [(f"Close ({pd.Timestamp(ext['close_date']).strftime('%a %d %b')})", f"{ext['close']:.2f}")]
    src_names = {"sip": "SIP (all exchanges, ~16 min delayed)", "iex": "IEX live", "boats": "Blue Ocean ATS",
                 "overnight": "Alpaca derived overnight feed"}
    for key, label in (("ah", "After-hours"), ("on", "Overnight"), ("pm", "Pre-market")):
        d = ext["sessions"].get(key) or {}
        if d.get("price") is None:
            rows.append((label, f'<span class="muted-small">{html.escape(d.get("status") or "N/A")}</span>'))
            continue
        pct_cls = "pass" if d.get("pct", 0) >= 0 else "fail"
        main_txt = (f"{d['price']:.2f} @ {d.get('time')} ET "
                    f"<span class=\"badge {pct_cls}\">{d.get('pct', 0):+.2f}%</span>")
        if key == "on":
            note = f"{d.get('trades', 0)} trades, {d.get('volume', 0):,.0f} shares - Blue Ocean ATS (thin)"
        else:
            note = f"{src_names.get(d.get('source'), d.get('source'))} - {d.get('reason', '')}"
            other = ("sip" if d.get("source") == "iex" else "iex")
            if d.get(f"{other}_price") is not None:
                note += f" | {other.upper()} {d[f'{other}_price']:.2f} @ {d[f'{other}_time']}"
        rows.append((label, main_txt + f'<br><span class="muted-small">{html.escape(note)}</span>'))
    return rows


def _target_r_html(report: dict) -> str:
    tgt = (report.get("day_sr") or {}).get("target") or {}
    if tgt.get("r") is None:
        return ""
    badge = " " + _badge(False, "", f"UNDER {report['cfg'].get('TARGET_MIN_R', 2.0):g}R") if tgt.get("below_min_r") else ""
    return f" ({tgt['r']:.1f}R){badge}"


@_profiled("HTML rendering (incl. chart)")
def render_ticker_html(report: dict) -> str:
    cfg = report["cfg"]
    results = report["results"]
    sr_levels = report["sr_levels"]
    today_range = report["today_range"]
    signal = report["signal"]
    signal_cls = {"BUY": "buy", "WATCH": "watch"}.get(signal["signal"], "wait")
    vwap_badge = "" if report.get("above_vwap") is None else _badge(report["above_vwap"], "ABOVE", "BELOW")

    # pre-market source line, e.g. "IEX live - moved +0.62% since the SIP snapshot (SIP 147.35 @ 08:05)"
    pm_info = report.get("pre_market_info") or {}
    pm_src_html = ""
    if pm_info.get("source"):
        src_lbl = {"sip": "SIP (all exchanges, ~16 min delayed)", "iex": "IEX live"}.get(pm_info["source"], pm_info["source"])
        other = ""
        if pm_info.get("source") == "iex" and pm_info.get("sip_price") is not None:
            other = f" | SIP {pm_info['sip_price']:.2f} @ {pm_info['sip_time']}"
        elif pm_info.get("source") == "sip" and pm_info.get("iex_price") is not None:
            other = f" | IEX {pm_info['iex_price']:.2f} @ {pm_info['iex_time']}"
        pm_src_html = html.escape(f"{src_lbl} - {pm_info.get('reason', '')}{other}")

    # ---- "Other levels" (own 4-column section)
    price = report["current_price"]
    other_items = []

    # 52-week range (from yfinance .info via fundamentals) + where price sits in it
    fund = report.get("fundamentals") or {}
    try:
        w52_lo, w52_hi = float(fund.get("52-Week Low")), float(fund.get("52-Week High"))
        pos52 = (price - w52_lo) / (w52_hi - w52_lo) * 100 if w52_hi > w52_lo else 50.0
        other_items.append(("52-week range", f"{w52_lo:.2f} - {w52_hi:.2f} (at {pos52:.0f}%)"))
    except (TypeError, ValueError):
        other_items.append(("52-week range", "N/A"))

    # ATR range: daily ATR, how much of it today has used, and the ATR projection
    # (today's low + ATR = projected high, today's high - ATR = projected low)
    atr_d = report.get("daily_atr")
    if atr_d:
        used = today_range["day_range"] / atr_d * 100
        other_items.append((f"Daily ATR({cfg['ATR_PERIOD']})", f"{atr_d:.2f} (today used {used:.0f}%)"))
        other_items.append(("ATR projected range",
                            f"{today_range['day_high'] - atr_d:.2f} - {today_range['day_low'] + atr_d:.2f}"))

    other_items += [
        *_extended_hours_rows(report),
        _premarket_near_row(report),
        *([(f"Anchored VWAP (from {report['avwap']['anchor_type']} of {report['avwap']['anchor_date']})",
             f"{report['avwap']['value']} " + ("" if report['avwap']['above'] is None
                                               else _badge(report['avwap']['above'], "ABOVE", "BELOW")))]
          if report.get("avwap") and report["avwap"].get("value") is not None else []),
        *([(f"HH/HL days (last {report['hhhl']['window']} completed)",
             f"{report['hhhl']['up_days']} up / {report['hhhl']['down_days']} down / "
             f"{report['hhhl']['mixed_days']} mixed (net {report['hhhl']['net']:+d})"),
            ("Current streak", _streak_text(report["hhhl"]["current_streak"])),
            ("Previous streak", _streak_text(report["hhhl"]["previous_streak"]))]
          if report.get("hhhl") else []),
        ("Today's range", f"{today_range['day_low']} - {today_range['day_high']} (range {today_range['day_range']})"),
        ("1h entry support", f"{report['entry_support']}"),
        ("Day support S1 / S2", _sr_text((report.get("day_sr") or {}).get("supports"))),
        ("Day resistance T1 / T2", _sr_text((report.get("day_sr") or {}).get("resistances"), with_r=True)),
        *_near_sr_rows(report),
        (f"POC ({cfg.get('CHART_DAYS', 7)}d volume profile)", f"{report.get('poc', 'N/A')}"),
        ("Session VWAP", f"{report.get('vwap') if report.get('vwap') is not None else 'N/A'} {vwap_badge}"),
    ]
    lrc = report.get("lrc")
    if lrc:
        pos = lrc["position_pct"]
        pos_txt = f"{pos:.0f}%" + (" (below)" if pos < 0 else " (above)" if pos > 100 else "")
        other_items += [
            (f"Reg. channel ({lrc['bars']} bars, {lrc['dev']}&sigma;)",
             f"{lrc['last_lower']} / {lrc['last_mid']} / {lrc['last_upper']}"),
            ("Channel slope", f"{lrc['slope_pct_per_day']:+.2f}%/day"),
            ("Channel fit (R&sup2;)", f"{lrc['r2']}"),
            ("Price in channel", pos_txt),
        ]
    if report.get("stop") is not None:          # only a setup has a trade plan
        other_items += [
            ("Initial stop (from the setup)", f"{report['stop']}"),
            ("<strong>Nearest real resistance (R:R filter; no profit target)</strong>",
             f"<strong>{report['take_profit']}</strong>" + _target_r_html(report)),
        ]
    other_levels_block = "\n      <h3>Other levels</h3>" + _four_col_table(other_items)

    fundamentals = report.get("fundamentals") or {}
    fundamentals_block = ""
    if fundamentals:
        fundamentals_block = "\n      <h3>Fundamentals</h3>" + _four_col_table(
            [(html.escape(label), html.escape(str(value))) for label, value in fundamentals.items()],
            "fundamentals-table")

    analyst = report.get("analyst") or {}
    analyst_block = ""
    if analyst:
        counts = analyst.get("recommendation_counts") or {}
        eps_improving = analyst.get("eps_improving")
        if eps_improving is None:
            eps_trend_html = '<span class="badge">N/A</span>'
        else:
            eps_trend_html = _badge(eps_improving, "IMPROVING", "DETERIORATING")

        rec_items = [
            ("Consensus", html.escape(analyst["recommendation_key"])),
            ("Mean Score", html.escape(analyst["recommendation_mean"])),
            ("# Analyst Opinions", html.escape(analyst["num_analyst_opinions"])),
        ] + [(html.escape(label), str(n)) for label, n in counts.items()]
        eps_items = [
            ("Current Consensus EPS", html.escape(analyst["eps_current"])),
            ("EPS 30 Days Ago", html.escape(analyst["eps_30d_ago"])),
            ("# Analysts (EPS)", html.escape(analyst["eps_num_analysts"])),
            ("EPS Trend", eps_trend_html),
        ]
        analyst_block = (
            "\n      <h3>Analyst Recommendations</h3>" + _four_col_table(rec_items)
            + "\n\n      <h3>EPS Estimate Revisions (current quarter)</h3>" + _four_col_table(eps_items)
        )

    news = report.get("news") or []
    if news:
        tz_lbl = {"America/New_York": "ET", "Asia/Jerusalem": "IL", "UTC": "UTC"}.get(
            cfg.get("NEWS_TZ", "America/New_York"), "")
        li = ""
        for n in news:
            url = n["url"] if n["url"].lower().startswith(("http://", "https://")) else ""
            title = html.escape(n["headline"])
            link = (f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                    if url else title)
            li += f'<li><span class="news-time">{html.escape(n["time"])} {tz_lbl}</span>{link}</li>'
        plural = "s" if len(news) != 1 else ""
        news_cell = (f'<details class="news-toggle"><summary>'
                     f'<span class="when-closed">Show {len(news)} article{plural}</span>'
                     f'<span class="when-open">Hide article{plural}</span></summary>'
                     f'<ul class="news-list">{li}</ul></details>')
    else:
        skipped = (report.get("catalysts") or {}).get("news_skipped")
        news_cell = f'<span class="muted-small">{html.escape(skipped)}</span>' if skipped else "none"
    news_days = cfg.get("NEWS_DAYS", 0)
    news_label = "today" if news_days == 0 else f"last {news_days + 1} days"
    news_row = f"<tr><td>News ({len(news)}, {news_label})</td><td>{news_cell}</td></tr>"

    catalysts = report.get("catalysts") or {}
    catalysts_block = ""
    if not catalysts and news:
        catalysts_block = f"""
      <h3>Event Catalysts</h3>
      <table class="detail-table"><tbody>{news_row}</tbody></table>"""
    if catalysts:
        earnings_date = catalysts.get("next_earnings_date") or "N/A"
        days_to_earnings = catalysts.get("days_to_earnings")
        earnings_str = earnings_date
        if days_to_earnings is not None:
            earnings_str = f"{earnings_date} ({days_to_earnings:+d}d)"
        earnings_window_html = _badge(
            not catalysts.get("in_earnings_window", False), "CLEAR", "IN WINDOW"
        )

        sp500_member = catalysts.get("in_sp500")
        if sp500_member is None:
            sp500_html = '<span class="badge">N/A</span>'
        else:
            sp500_extra = ""
            if catalysts.get("sp500_added_date"):
                sp500_extra = (
                    f" (added {catalysts['sp500_added_date']}, "
                    f"{catalysts.get('days_since_index_addition')}d ago)"
                )
            sp500_html = _badge(sp500_member, "MEMBER", "NOT MEMBER") + html.escape(sp500_extra)
        recent_addition_html = ""
        if catalysts.get("recent_index_addition"):
            recent_addition_html = f"<tr><td>Recent index addition</td><td>{_badge(True, 'RECENT ADD')}</td></tr>"

        buyback_headlines = catalysts.get("buyback_headlines") or []
        guidance_headlines = catalysts.get("guidance_headlines") or []
        def _head_list(heads, limit=5):
            shown = "<br>".join(html.escape(h) for h in heads[:limit])
            more = f'<br><span class="muted-small">... and {len(heads) - limit} more</span>' if len(heads) > limit else ""
            return (shown + more) or "none"
        buyback_html = _head_list(buyback_headlines)
        guidance_html = _head_list(guidance_headlines)
        if catalysts.get("news_skipped"):   # no headlines were scanned for this ticker
            buyback_html = guidance_html = '<span class="muted-small">not checked (no news fetched)</span>'


        rating_actions = catalysts.get("rating_actions") or []
        rating_html = "<br>".join(html.escape(a) for a in rating_actions) or "none"

        social_rank = catalysts.get("social_rank", "N/A")
        if social_rank != "N/A":
            mom = catalysts.get("social_momentum_pct", "N/A")
            mom_str = f"+{mom}%" if isinstance(mom, (int, float)) and mom > 0 else f"{mom}%"
            social_html = (
                f"Rank <strong>#{social_rank}</strong> | "
                f"Mentions: {catalysts.get('social_mentions', 0)} ({mom_str} 24h) | "
                f"Upvotes: {catalysts.get('social_upvotes', 0)}"
            )
        else:
            social_html = '<span class="badge">Not in Top 50</span>'
        catalysts_block = f"""
      <h3>Event Catalysts</h3>
      <table class="detail-table"><tbody>
        <tr><td>Next earnings</td><td>{html.escape(str(earnings_str))} {earnings_window_html}</td></tr>
        <tr><td>S&amp;P 500 membership</td><td>{sp500_html}</td></tr>
        {recent_addition_html}
        <tr><td>Buyback headlines ({len(buyback_headlines)})</td><td>{buyback_html}</td></tr>
        <tr><td>Guidance headlines ({len(guidance_headlines)})</td><td>{guidance_html}</td></tr>
        <tr><td>Analyst actions ({catalysts.get('upgrades', 0)} up / {catalysts.get('downgrades', 0)} down)</td><td>{rating_html}</td></tr>
        <tr><td>Reddit Sentiment (ApeWisdom)</td><td>{social_html}</td></tr>
        {news_row}
      </tbody></table>"""


    tf_rows = ""
    for tf in ["1D", "4h", "1h"]:
        r = results[tf]
        sr = sr_levels[tf]
        patterns = ", ".join(r["patterns_detected"]) if r["patterns_detected"] else "-"
        tf_rows += f"""
        <tr>
          <td>{tf}</td>
          <td>{r['close']}</td>
          <td>{r['MA']}</td>
          <td>{_badge(r['trend_up'], 'UP', 'DOWN')}</td>
          <td>{r['%K']}</td>
          <td>{r['%D']}</td>
          <td>{_badge(r['k_above_d'], 'K>D', 'K<D')}</td>
          <td>{_badge(r['stoch_bullish'])}</td>
          <td>{_badge(not r['overbought'], 'OK', 'HOT')}</td>
          <td>{html.escape(patterns)}</td>
          <td>{sr['support']}</td>
          <td>{sr['resistance']}</td>
        </tr>"""

    signal_section = render_signal_section(report)

    card = f"""
    <section class="card" id="{_card_id(report['ticker'])}">
      <div class="card-header">
        <h2>{html.escape(report['ticker'])}</h2>
        <div class="price">${report['current_price']:.2f}</div>
        <div class="signal-badge {signal_cls}">{html.escape(signal_badge_text(signal))}</div>
      </div>
      {render_chart_toggles(report)}

      <table class="tf-table">
        <thead>
          <tr>
            <th>TF</th><th>Close</th><th>{cfg['MA_TYPE'].upper()}{cfg['MA_PERIOD']}</th><th>Trend</th>
            <th>%K</th><th>%D</th><th>K vs D</th><th>Stoch</th><th>Overbought</th>
            <th>Candles</th><th>Support</th><th>Resistance</th>
          </tr>
        </thead>
        <tbody>{tf_rows}
        </tbody>
      </table>

      {signal_section}
      {other_levels_block}
      {catalysts_block}
      {fundamentals_block}
      {analyst_block}
    </section>"""
    # every section heading ends with the ticker, e.g. "Other levels - AMD"
    tk = html.escape(report["ticker"])
    return re.sub(r"<h3>(.*?)</h3>", lambda m: f"<h3>{m.group(1)} - {tk}</h3>", card, flags=re.S)


def build_html_report(title: str, ticker_sections_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #0f1115;
    --card: #171a21;
    --border: #2a2f3a;
    --text: #e6e6e6;
    --muted: #9aa4b2;
    --accent: #9cdcfe;
    --pass: #2ecc71;
    --fail: #e74c3c;
    --buy: #2ecc71;
    --wait: #556070;
    --ma0: #c39bd3;  /* EMA9  */
    --ma1: #f5b041;
    --ma2: #48c9b0;
    --ma3: #85c1e9;
    --poc: #ff79c6;
    --vwap: #ffffff;
    --premkt: #ffe600;
    --avwap: #4dd0e1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text); margin: 0; padding: 32px;
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }}
  h1 {{ font-size: 20px; color: var(--accent); margin: 0 0 4px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 28px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px 24px; margin-bottom: 24px;
  }}
  .card-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
  .card-header h2 {{ margin: 0; font-size: 18px; }}
  .price {{ color: var(--muted); font-size: 15px; }}
  .signal-badge {{
    margin-left: auto; padding: 6px 14px; border-radius: 999px; font-weight: 700;
    font-size: 13px; letter-spacing: 0.04em;
  }}
  .signal-badge.buy {{ background: rgba(46,204,113,0.15); color: var(--buy); border: 1px solid var(--buy); }}
  .signal-badge.wait {{ background: rgba(85,96,112,0.25); color: var(--muted); border: 1px solid var(--border); }}
  .signal-badge.watch {{ background: rgba(245,176,65,0.15); color: #f5b041; border: 1px solid #f5b041; }}
  html {{ scroll-behavior: smooth; }}
  .card {{ scroll-margin-top: 12px; }}
  a.tk {{ color: var(--accent); font-weight: 700; text-decoration: none; }}
  a.tk:hover {{ text-decoration: underline; }}
  #tk-tip {{ position: fixed; z-index: 9999; display: none; pointer-events: none; background: #0d1017; border: 1px solid #3a4560; border-radius: 8px; padding: 10px 14px; font-size: 12px; line-height: 1.65; color: #dfe6f0; box-shadow: 0 8px 24px rgba(0,0,0,0.55); max-width: 520px; }}
  #tk-tip b {{ color: #ffffff; font-size: 13px; }}
  #tk-tip .k {{ color: #8b97ab; }}
  #tk-tip .pos {{ color: var(--pass); font-weight: 700; }}
  #tk-tip .neg {{ color: var(--fail); font-weight: 700; }}
  #tk-tip .badge {{ margin-left: 4px; }}
  #tk-tip .sep {{ border-top: 1px solid #2c3550; margin: 6px 0; }}
  .summary-table th[data-tip] {{ cursor: help; text-decoration: underline dotted rgba(156,220,254,0.45); text-underline-offset: 3px; }}
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }}
  .summary-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  .summary-table th {{ background: #232a3a; color: var(--accent); padding: 10px 12px; white-space: nowrap; border-bottom: 2px solid #33405a; }}
  .summary-table td {{ padding: 9px 12px; vertical-align: top; border-bottom: 1px solid #262c39; }}
  .summary-table tbody tr:nth-child(odd) td {{ background: #171a21; }}
  .summary-table tbody tr:nth-child(even) td {{ background: #212838; }}
  .summary-table tbody tr:hover td {{ background: #2d3a52; }}
  .summary-table tr.buy-row td:first-child {{ border-left: 4px solid var(--buy); }}
  .summary-table tr.watch-row td:first-child {{ border-left: 4px solid #f5b041; }}
  .summary-table tr.wait-row td:first-child {{ border-left: 4px solid #556070; }}
  .summary-table tr.turn-row td:first-child {{ border-left: 4px solid #4dd0e1; }}
  .card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
  .card-head h2 {{ margin: 0; }}
  .runbar {{ display: none; align-items: center; gap: 8px; }}
  .runinfo {{ font-size: 13px; font-weight: 400; color: var(--text); margin: 6px 0 12px; }}
  .runinfo .t {{ color: var(--pass); font-weight: 600; }}
  .runbtn {{ display: inline-block; padding: 6px 14px; border-radius: 8px; background: #233a5c; color: #e6e6e6; font-size: 13px;
            font-weight: 600; text-decoration: none; border: 1px solid #33507a; }}
  .runbtn:hover {{ background: #2d4a75; }}
  .summary-table tr.wait-row td, .summary-table tr.wait-row a.tk, .summary-table tr.wait-row .px-closed {{ color: #6b7585; }}
  .summary-table td.num {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
  .summary-table th:nth-child(n+4):nth-child(-n+10) {{ text-align: right; }}
  .summary-table td.buywhen {{ line-height: 1.55; min-width: 360px; color: #cfd6e0; }}
  .gap {{ display: inline-block; min-width: 34px; text-align: center; padding: 2px 9px; border-radius: 999px; font-weight: 700; font-size: 12px; }}
  .gap.g0 {{ background: rgba(46,204,113,0.22); color: var(--pass); }}
  .gap.g1 {{ background: rgba(46,204,113,0.12); color: #7bdc9c; }}
  .gap.g2 {{ background: rgba(245,176,65,0.15); color: #f5b041; }}
  .gap.g3 {{ background: rgba(231,76,60,0.14); color: #ee8a80; }}
  .ext {{ font-size: 11px; font-weight: 500; color: var(--muted); white-space: nowrap; margin-top: 2px; }}
  .ext .pos {{ color: var(--pass); font-weight: 600; }}
  .ext .neg {{ color: var(--fail); font-weight: 600; }}
  .px-closed {{ color: var(--muted); }}
  .badge.watch {{ background: rgba(245,176,65,0.15); color: #f5b041; }}
  .badge.wait {{ background: rgba(85,96,112,0.25); color: #8a93a0; }}

  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.03em; }}
  .tf-table {{ margin-bottom: 20px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 800px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  h3 {{ font-size: 13px; color: var(--accent); margin: 16px 0 6px; text-transform: uppercase; letter-spacing: 0.03em; }}
  .detail-table td:first-child {{ color: var(--text); }}
  .fundamentals-table {{ margin-bottom: 18px; }}
  .four-col {{ table-layout: fixed; margin-bottom: 18px; }}
  .four-col col.c-lbl {{ width: 30%; }}
  .four-col col.c-val {{ width: 20%; }}
  .four-col td:nth-child(odd) {{ color: var(--muted); }}
  .four-col td:nth-child(2) {{ border-right: 1px solid var(--border); padding-right: 16px; }}
  .four-col td:nth-child(3) {{ padding-left: 16px; }}
  .four-col td {{ overflow-wrap: anywhere; }}
  .note {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}

  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px;
    font-weight: 700; letter-spacing: 0.03em;
  }}
  .badge.pass {{ background: rgba(46,204,113,0.15); color: var(--pass); }}
  .badge.fail {{ background: rgba(231,76,60,0.15); color: var(--fail); }}

  /* ---- 7-day price chart */
  .chart-wrap {{ margin: 4px 0 20px; }}
  .charts {{ margin: 0 0 12px; }}
  .charts .tog, .charts .panel {{ display: none; }}
  .charts .tog-links {{ display: flex; gap: 22px; margin-bottom: 6px; }}
  .charts .lnk {{ cursor: pointer; color: var(--accent); font-size: 12px; user-select: none; }}
  .charts .lnk:hover {{ text-decoration: underline; }}
  .charts .lnk .when-open {{ display: none; }}
  .charts .tog-h:checked ~ .panel-h, .charts .tog-d:checked ~ .panel-d {{ display: block; }}
  .charts .tog-h:checked ~ .tog-links .lnk-h .when-open,
  .charts .tog-d:checked ~ .tog-links .lnk-d .when-open {{ display: inline; }}
  .charts .tog-h:checked ~ .tog-links .lnk-h .when-closed,
  .charts .tog-d:checked ~ .tog-links .lnk-d .when-closed {{ display: none; }}
  .hhhl-mark {{ font-size: 10px; }}
  .price-chart .axis.dayname {{ fill: var(--muted); opacity: 0.75; }}
  .price-chart .weeksep {{ stroke: #ffffff; stroke-opacity: 0.4; stroke-width: 1; stroke-dasharray: 1 3; }}
  .price-chart .avwap {{ fill: none; stroke: var(--avwap); stroke-width: 1.8; }}
  .price-chart .avwap-anchor {{ fill: var(--avwap); }}
  .price-chart .lvl-label.lvl-avwap {{ fill: var(--avwap); }}
  .sw.avwap-sw {{ border-color: var(--avwap); }}
  .hhhl-mark.up {{ fill: var(--pass); color: var(--pass); }}
  .hhhl-mark.down {{ fill: var(--fail); color: var(--fail); }}
  .price-chart .candle.partial {{ opacity: 0.55; }}
  .price-chart .lvl.lvl-res {{ stroke: var(--muted); }}
  .price-chart .lvl-label.lvl-res {{ fill: var(--muted); }}
  .sw.lvl-res {{ border-color: var(--muted); }}
  .price-chart .lvl.lvl-support2 {{ stroke: var(--accent); stroke-opacity: 0.5; stroke-dasharray: 2 4; }}
  .price-chart .lvl-label.lvl-support2 {{ fill: var(--accent); opacity: 0.7; }}
  .price-chart .lvl.lvl-target2 {{ stroke: var(--pass); stroke-opacity: 0.5; stroke-dasharray: 2 4; }}
  .price-chart .lvl-label.lvl-target2 {{ fill: var(--pass); opacity: 0.7; }}
  .chart-toggle {{ margin: 0 0 12px; }}
  .chart-toggle summary {{ list-style: none; cursor: pointer; color: var(--accent);
                           font-size: 12px; display: inline-block; margin-bottom: 6px; user-select: none; }}
  .muted-small {{ color: var(--muted); font-size: 11px; }}
  .news-toggle summary {{ cursor: pointer; color: var(--accent); user-select: none; }}
  .news-toggle summary:hover {{ text-decoration: underline; }}
  .news-list {{ list-style: none; margin: 8px 0 2px; padding: 0; }}
  .news-list li {{ padding: 4px 0; border-bottom: 1px solid var(--border); line-height: 1.4; }}
  .news-list li:last-child {{ border-bottom: none; }}
  .news-time {{ color: var(--muted); font-variant-numeric: tabular-nums; margin-right: 10px; white-space: nowrap; }}
  .news-list a {{ color: var(--text); text-decoration: none; }}
  .news-list a:hover {{ color: var(--accent); text-decoration: underline; }}
  .chart-toggle summary::-webkit-details-marker {{ display: none; }}
  .chart-toggle summary:hover {{ text-decoration: underline; }}
  .chart-toggle .when-open, .news-toggle .when-open {{ display: none; }}
  .chart-toggle[open] .when-open, .news-toggle[open] .when-open {{ display: inline; }}
  .chart-toggle[open] .when-closed, .news-toggle[open] .when-closed {{ display: none; }}
  .chart-head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
                 color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
  .chart-head .legend {{ margin-left: auto; display: flex; align-items: center; gap: 6px; }}
  .sw {{ display: inline-block; width: 14px; height: 0; border-top: 2px dashed; margin-left: 8px; }}
  .sw.ma-sw {{ border-top-style: solid; }}
  .sw.ma-0 {{ border-color: var(--ma0); }} .sw.ma-1 {{ border-color: var(--ma1); }}
  .sw.ma-2 {{ border-color: var(--ma2); }} .sw.ma-3 {{ border-color: var(--ma3); }}
  .sw.lvl-poc {{ border-color: var(--poc); }}
  .sw.session-sw {{ border-top: 2px dotted var(--text); }}
  .sw.lvl-support {{ border-color: var(--accent); }}
  .sw.lvl-stop {{ border-color: var(--fail); }}
  .sw.lvl-target {{ border-color: var(--pass); }}
  .price-chart {{ width: 100%; height: auto; display: block; }}  /* keeps aspect ratio, so text isn't stretched */
  .price-chart .grid {{ stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 1; stroke-dasharray: 1 4; }}
  .price-chart .tick {{ stroke: var(--muted); stroke-width: 1; }}
  .price-chart .axis {{ fill: var(--muted); font-size: 10px; }}
  .price-chart .candle line {{ stroke-width: 1; }}
  .price-chart .candle.up line {{ stroke: var(--pass); }}
  .price-chart .candle.down line {{ stroke: var(--fail); }}
  .price-chart .candle.up rect {{ fill: var(--pass); }}
  .price-chart .candle.down rect {{ fill: var(--fail); }}
  .price-chart .candle:hover rect {{ opacity: 0.7; }}
  .price-chart .vol.up {{ fill: rgba(46,204,113,0.35); }}
  .price-chart .vol.down {{ fill: rgba(231,76,60,0.35); }}
  .price-chart .ma {{ fill: none; stroke-width: 1.5; }}
  .price-chart .ma.ma-0 {{ stroke: var(--ma0); }} .price-chart .ma.ma-1 {{ stroke: var(--ma1); }}
  .price-chart .ma.ma-2 {{ stroke: var(--ma2); }} .price-chart .ma.ma-3 {{ stroke: var(--ma3); }}
  .price-chart .session {{ stroke: #ffffff; stroke-opacity: 0.55; stroke-width: 1; stroke-dasharray: 1 3; }}
  .price-chart .lvl.lvl-poc {{ stroke: var(--poc); }}
  .price-chart .lvl-label.lvl-poc {{ fill: var(--poc); }}
  .price-chart .vwap {{ fill: none; stroke: var(--vwap); stroke-width: 1.8; stroke-opacity: 0.9; }}
  .price-chart .lvl-label.lvl-vwap {{ fill: var(--vwap); }}
  .sw.vwap-sw {{ border-color: var(--vwap); }}
  .sw.premkt-sw {{ border-top: 2px dotted var(--premkt); }}
  .price-chart .lvl.lvl-premkt {{ stroke: var(--premkt); stroke-width: 1.5; stroke-dasharray: 2 4; }}
  .price-chart .lvl-label.lvl-premkt {{ fill: var(--premkt); }}
  .sw.lrc-sw {{ border-top: 6px solid rgba(154,164,178,0.25); height: 0; }}
  .price-chart .lrc-fill {{ fill: var(--muted); fill-opacity: 0.07; }}
  .price-chart .lrc-band {{ fill: none; stroke: var(--muted); stroke-opacity: 0.6; stroke-width: 1; }}
  .price-chart .lrc-mid {{ fill: none; stroke: var(--muted); stroke-opacity: 0.8; stroke-width: 1; stroke-dasharray: 6 4; }}
  .price-chart .lvl {{ stroke-width: 1; stroke-dasharray: 5 4; }}
  .price-chart .lvl.lvl-support {{ stroke: var(--accent); }}
  .price-chart .lvl.lvl-stop {{ stroke: var(--fail); }}
  .price-chart .lvl.lvl-target {{ stroke: var(--pass); }}
  .price-chart .lvl-label {{ font-size: 11px; font-weight: 600; }}
  .price-chart .lvl-label.lvl-support {{ fill: var(--accent); }}
  .price-chart .lvl-label.lvl-stop {{ fill: var(--fail); }}
  .price-chart .lvl-label.lvl-target {{ fill: var(--pass); }}
  .price-chart .last {{ fill: var(--text); }}
  .fab {{ position: fixed; right: 24px; bottom: 24px; z-index: 50; display: none; width: 48px; height: 48px;
          align-items: center; justify-content: center; border-radius: 50%; background: rgba(35,42,58,.94);
          border: 1px solid #33405a; color: var(--text); font-size: 22px; font-weight: 700; text-decoration: none;
          box-shadow: 0 6px 18px rgba(0,0,0,.6); }}
  .fab:hover {{ background: #2d3a52; }}
  @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
</style>
</head>
<body id="top">
<h1>{html.escape(title)}</h1>
<div class="subtitle">Multi-Timeframe Buy Signal Screener</div>
{ticker_sections_html}
<a class="fab" href="#top" aria-label="Back to top" title="Back to top">&uarr;</a>
<script>(function(){{var f=document.querySelector('.fab');if(!f)return;
function u(){{f.style.display=(window.scrollY>500)?'flex':'none'}}u();
addEventListener('scroll',u,{{passive:true}})}})();</script>
</body>
</html>
"""


def _snake(label: str) -> str:
    """'Price/Sales (TTM)' -> 'price_sales_ttm'"""
    out = "".join(c.lower() if c.isalnum() else "_" for c in label.replace("%", " pct "))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def report_filter_checks(report: dict) -> dict:
    """The four report-inclusion checks. Each value is True if the ticker FAILS
    that check. A ticker is left out of the HTML only when it fails ALL four."""
    signal = report["signal"]
    eps_improving = (report.get("analyst") or {}).get("eps_improving")
    return {
        "no_setup": signal["setup"] is None and not signal.get("turning_up"),      # turning up near the EMA50 counts as a setup here
        "weak_context": signal["context"]["state"] in ("DOWNTREND", "RANGE"),
        "wait_signal": signal["signal"] == "WAIT",
        "eps_deteriorating": eps_improving is False,                     # N/A does not count as failing
    }


def _skipped_card(skipped: list) -> str:
    if not skipped:
        return ""
    items = "".join(f"<li><strong>{html.escape(t)}</strong>: {html.escape(why)}</li>" for t, why in skipped)
    return (f'<section class="card"><h3>Not shown ({len(skipped)})</h3>'
            f'<ul class="note">{items}</ul></section>')



_EXT_ABBR = {"after-hours": "AH", "overnight": "ON", "pre-market": "PRE"}


def _ext_line(r: dict, up_cls: str = "pos", dn_cls: str = "neg", tag: str = "div") -> str:
    """The most recent extended-hours price (AH / ON / PRE) as a small line to put under the regular price; '' when the
    market is open or no extended-hours price is available."""
    price, sess = r.get("current_ext_price"), r.get("current_ext_session")
    if price is None or (isinstance(price, float) and price != price) or not sess:
        return ""
    pct = r.get("current_ext_pct")
    pc = ""
    if pct is not None and pct == pct:
        pc = f' <span class="{up_cls if pct >= 0 else dn_cls}">{pct:+.2f}%</span>'
    return f'<{tag} class="ext">{_EXT_ABBR.get(sess, "EXT")} {float(price):,.2f}{pc}</{tag}>'


def _summary_price(r: dict) -> str:
    """Price cell of the 'Setups at a glance' table: the regular price (grey once the market is closed, i.e. it is the
    closing price), and under it the latest extended-hours price (AH / ON / PRE) without label or %, green when it is
    at or above the regular-session close and red when below."""
    def num(v):
        return None if v is None or (isinstance(v, float) and v != v) else float(v)
    px = num(r.get("current_price"))
    closed = r.get("market_open") is False
    out = "" if px is None else (f'<span class="px-closed">{px:.2f}</span>' if closed else f"{px:.2f}")
    ext, ref = num(r.get("current_ext_price")), num(r.get("close_price"))
    if ext is not None and r.get("current_ext_session"):
        if ref is None:                                   # no close on record: fall back to the session's % vs close
            pct = num(r.get("current_ext_pct"))
            cls = "" if pct is None else ("pos" if pct >= 0 else "neg")
        else:
            cls = "pos" if ext >= ref else "neg"
        out += f'<div class="ext"><span class="{cls}">{ext:,.2f}</span></div>'
    return out


_TIP_SESSIONS = {"after-hours": "After-hours", "overnight": "Overnight", "pre-market": "Pre-market"}


def _ticker_tip(r: dict) -> str:
    """HTML for the hover card on a ticker: company / sector / industry, then the live price
    (labelled with the session it comes from) and the previous session's O/H/L/C."""
    def g(k):
        v = r.get(k)
        return None if v is None or (isinstance(v, float) and v != v) else v

    def line(label, value):
        return f'<div><span class="k">{label}</span> {html.escape(str(value))}</div>'

    out = [f"<div><b>{html.escape(str(g('name') or r.get('ticker')))}</b></div>",
           line("Sector:", g("sector") or "N/A"),
           line("Industry:", g("industry") or "N/A"),
           '<div class="sep"></div>']
    price = g("current_price")
    if g("market_open") is False and g("current_ext_price") is not None:
        label = _TIP_SESSIONS.get(g("current_ext_session"), "Extended hours")
        pct = g("current_ext_pct")
        badge = (f' <span class="badge {"pass" if pct >= 0 else "fail"}">{pct:+.2f}%</span> vs close'
                 if pct is not None else "")
        out.append(f'<div><span class="k">{label}:</span> {float(g("current_ext_price")):,.2f}{badge}</div>')
    elif price is not None:
        out.append(line("Market open:" if g("market_open") else "Last price:", f"{float(price):,.2f}"))
    if g("prev_day_open") is not None:
        c, prior = g("prev_day_close"), g("prev_day_prior_close")
        cls = "" if prior is None or c == prior else ("pos" if c > prior else "neg")   # closed higher / lower than the day before
        out.append(f'<div style="white-space:nowrap"><span class="k">Prev day ({html.escape(str(g("prev_day_date")))}):</span> '
                   f'O {g("prev_day_open"):,.2f} &middot; H {g("prev_day_high"):,.2f} &middot; '
                   f'L {g("prev_day_low"):,.2f} &middot; C <span class="{cls}">{c:,.2f}</span></div>')
    return "".join(out)


_TIP_SCRIPT = (
    "<script>(function(){if(window.__tkTip)return;window.__tkTip=1;var tip=document.getElementById('tk-tip');"
    "if(!tip){tip=document.createElement('div');tip.id='tk-tip';document.body.appendChild(tip);}"
    "function place(e){var x=e.clientX+16,y=e.clientY+16,w=tip.offsetWidth,h=tip.offsetHeight;"
    "if(x+w>window.innerWidth-8)x=e.clientX-w-16;if(y+h>window.innerHeight-8)y=e.clientY-h-16;"
    "tip.style.left=Math.max(4,x)+'px';tip.style.top=Math.max(4,y)+'px';}"
    "document.addEventListener('mouseover',function(e){var a=e.target.closest&&e.target.closest('[data-tip]');"
    "if(!a)return;tip.innerHTML=a.getAttribute('data-tip');tip.style.display='block';place(e);});"
    "document.addEventListener('mousemove',function(e){if(tip.style.display==='block')place(e);});"
    "document.addEventListener('mouseout',function(e){var a=e.target.closest&&e.target.closest('[data-tip]');"
    "if(a)tip.style.display='none';});})();</script>")


def _card_id(ticker) -> str:
    return "card-" + re.sub(r"[^A-Za-z0-9_.-]", "_", str(ticker))


# Column descriptions for the 'Setups at a glance' table (shown when you hover a column name).
_SUMMARY_COLS = [
    ("Ticker", "Hover a ticker for company, sector, industry, price session and the previous day. "
               "Click it to jump to its full card."),
    ("Signal", "<b>BUY</b> = everything passes: daily uptrend, pullback structure, turn on the daily and 1h charts, "
               "3 of 4 confirmations, R:R 1.5+ and not extended.<br>"
               "<b>WATCH</b> = valid setup, but something is still missing (listed on the right).<br>"
               "<b>WAIT</b> (grey) = a pullback setup is forming; too much is missing for a WATCH yet."),
    ("Setup", "The setup type. Only pullbacks in a confirmed daily uptrend are traded - the only setup "
              "with a measurable backtest edge."),
    ("Price", "Last price; grey = the market is closed, so it is the closing price.<br>The small line below is the "
              "latest after-hours / overnight / pre-market price: green at or above the close, red below."),
    ("R:R", "Reward-to-risk at the current price (often just called risk/reward): "
            "(resistance - price) / (price - stop). Higher is better: 1.5 means the possible gain is 1.5 times the "
            "risk (1 : 1.5 written as risk : reward). A BUY needs at least 1.5."),
    ("Gap", "Distance from a BUY (0 = BUY; smaller = closer). A missing turn or confirmation counts 1 each, "
            "a price / R:R problem 2-5, an extended stock 2, a failed structural requirement 3."),
    ("Buy up to", "The highest entry that still gives R:R of at least 1.5 with the same stop and resistance. "
                  "Above it the trade no longer pays enough - don't chase."),
    ("Buy at/below", "The price a pullback must reach for R:R to get to the minimum - the entry zone when the "
                     "current price is too high."),
    ("Stop", "Initial stop = 1R of risk: just under the pullback's low (a quarter ATR below it). Once price trades "
             "+2R, the stop trails under the daily EMA20 (raise only)."),
    ("Resistance (R:R)", "Nearest real resistance above the price (a swing level with 2+ touches, or a 20-bar / "
                         "window / 52-week high). Used only to measure R:R - it is not a profit target."),
    ("What would make it a BUY", "The conditions still missing, in order; all of them must hold. Turn levels "
                                 "(today's open / mid-range, session VWAP) move during the day."),
]


# ... and for the 'Turning up near the EMA50' table
_TURN_COLS = [
    _SUMMARY_COLS[0],
    ("Price", _SUMMARY_COLS[3][1]),
    ("Today", "Change since the previous close, during the session (live) - the biggest gain is listed first."),
    ("EMA50", "The daily 50-day exponential moving average. Closing back above it is the first step back to an "
              "uptrend; a pullback can only become a BUY in a confirmed uptrend."),
    ("From EMA50 (ATR)", "How far the close is below the EMA50, in daily ATRs (average daily ranges): -0.2 = a fifth "
                         "of a normal day's range below. This list only includes stocks at most 1.5 ATR below."),
    ("EMA50 slope (5d)", "Change of the EMA50 over the last 5 trading days. Negative = still falling; a slope moving "
                         "towards zero means the decline is losing steam."),
    ("Trend", "Daily trend state. DOWNTREND = below a falling EMA50. CORRECTION = a shallow dip under the EMA50 "
              "(at most 0.75 ATR) with the higher lows intact. RANGE = below an EMA50 that is not falling, but too far "
              "below it or with the higher lows broken."),
]


def _summary_head_html(cols=None) -> str:
    return "".join(f'<th data-tip="{html.escape(tip, quote=True)}">{html.escape(name)}</th>'
                   for name, tip in (cols or _SUMMARY_COLS))


def _setup_summary_card(rows: list, down_file: str | None = None, header_right: str = "", info: str = "") -> str:
    """One table at the top of the _up report: every BUY and WATCH with its entry zone,
    breakout level, stop, target and the concrete conditions that would turn it into a BUY."""
    # BUY and WATCH first, then traded setups that are still forming (WAIT with a setup), greyed out at the bottom
    picks = [r for r in rows if r.get("signal") in ("BUY", "WATCH")
             or (r.get("signal") == "WAIT" and r.get("setup") is not None)]
    def head(count):        # no str.format here: the buttons' script contains { }
        return f'<div class="card-head"><h2>Setups at a glance ({count})</h2>' + header_right + '</div>' + info
    if not picks:
        # keep the card (and its buttons / run info) even when there is nothing to list
        return ('<section class="card">' + head(0)
                + '<p class="note">No BUY, WATCH or forming setups in this report.</p>' + _TIP_SCRIPT + '</section>'
                ) if (header_right or info) else ""
    n_forming = sum(1 for r in picks if r["signal"] == "WAIT")
    def gap(r):
        v = r.get("setup_distance")
        return 99.0 if v is None or v != v else float(v)
    # closest to a BUY first: BUYs, then WATCH, then forming (WAIT); inside each the smallest gap, ties -> better R:R
    rank = {"BUY": 0, "WATCH": 1, "WAIT": 2}
    picks.sort(key=lambda r: (rank[r["signal"]], gap(r), -(r.get("setup_rr") or 0)))

    def f(v):
        return "" if v is None or (isinstance(v, float) and v != v) else f"{v:.2f}"

    body = ""
    for r in picks:
        if r["signal"] == "BUY":
            waiting = "-"
        else:
            items = [x for x in str(r.get("setup_buy_when") or "").split(" | ") if x]
            waiting = "<br>".join("&bull; " + html.escape(x) for x in items) or html.escape(str(r.get("setup_fails") or ""))
        cls = {"BUY": "pass", "WATCH": "watch"}.get(r["signal"], "wait")
        g = gap(r)
        gcls = "g0" if g == 0 else "g1" if g <= 1.5 else "g2" if g <= 3.5 else "g3"
        gap_html = "" if g == 99.0 else f'<span class="gap {gcls}">{g:.1f}</span>'
        # the card is on this page (#id) unless the ticker was sent to the _down report
        href = f"#{_card_id(r['ticker'])}" if r.get("report") == "up" else (
            f"{down_file}#{_card_id(r['ticker'])}" if down_file else None)
        tk = html.escape(str(r["ticker"]))
        tip = html.escape(_ticker_tip(r), quote=True)
        tk_html = (f'<a class="tk" href="{html.escape(href)}" data-tip="{tip}">{tk}</a>' if href
                   else f'<strong data-tip="{tip}">{tk}</strong>')
        body += (f'<tr class="{ {"BUY": "buy-row", "WATCH": "watch-row"}.get(r["signal"], "wait-row") }">'
                 f"<td>{tk_html}</td>"
                 f"<td><span class=\"badge {cls}\">{r['signal']}</span>"
                 + (f'<div class="ext">was {html.escape(str(r["prev_signal"]))}</div>' if r.get("prev_signal") else "")
                 + "</td>"
                 f"<td style=\"white-space:nowrap\">{html.escape(str(r.get('setup') or '').replace('_', ' ').capitalize())}</td>"
                 f'<td class="num">{_summary_price(r)}</td><td class="num">{f(r.get("setup_rr"))}</td>'
                 f'<td class="num">{gap_html}</td>'
                 f'<td class="num">{f(r.get("setup_buy_up_to"))}</td>'
                 f'<td class="num">{f(r.get("setup_buy_below"))}</td>'
                 f'<td class="num">{f(r.get("setup_stop"))}</td><td class="num">{f(r.get("setup_target"))}</td>'
                 f'<td class="buywhen">{waiting}</td></tr>')
    n_active = len(picks) - n_forming
    count = f"{n_active}" + (f" + {n_forming} forming" if n_forming else "")
    return ('<section class="card">' + head(count) +
            '<div class="table-wrap"><table class="summary-table"><thead><tr>' + _summary_head_html() + '</tr></thead>'
            f'<tbody>{body}</tbody></table></div>'
            '<p class="note"><strong>Exit plan:</strong> initial stop, no profit target. Once price trades at the <strong>+2R</strong> level, '
            'trail the stop under the daily EMA20 (raise only); leave after 90 trading days at the latest. '
            '<strong>Resistance (R:R)</strong> is the nearest real resistance, used only to filter entries (R:R at least 1.5).<br>'
            '<strong>Buy up to</strong> = the highest entry that still gives R:R at or above the minimum '
            'with the same stop; <strong>Buy at/below</strong> = the price a pullback must reach to get there. '
            'Only pullbacks in an uptrend are traded (the only setup with a measurable backtest edge); other setups are not shown as BUY.<br>'
            '<strong>Greyed rows (WAIT)</strong> = a traded setup that is still forming: too many items fail for a WATCH yet.<br>'
            'Sorted by signal (BUY, WATCH, WAIT), then by Gap = distance from a BUY (0 = BUY). A missing turn or confirmation counts 1 each, '
            'a price/R:R problem 2-5 depending on how far the entry zone is, an extended stock 2, a failed structural '
            'requirement 3. Conditions are listed in order and must all hold. Turn levels (prior close, session VWAP) '
            'move during the day. Hover a ticker for company, sector, industry, price and the previous day.</p>'
            + _TIP_SCRIPT + '</section>')


def _turning_up_card(rows: list) -> str:
    """Watch list under the setups table: tickers with no setup that are turning up just under the EMA50."""
    picks = [r for r in rows if r.get("turning_up") and r.get("report") == "up"]
    if not picks:
        return ""

    def num(v):
        return None if v is None or (isinstance(v, float) and v != v) else float(v)

    def f(v, fmt="{:.2f}"):
        v = num(v)
        return "" if v is None else fmt.format(v)

    picks.sort(key=lambda r: -(num(r.get("day_chg_pct")) if num(r.get("day_chg_pct")) is not None else -999))   # biggest gain today first
    body = ""
    for r in picks:
        tk = html.escape(str(r["ticker"]))
        tip = html.escape(_ticker_tip(r), quote=True)
        chg = num(r.get("day_chg_pct"))
        chg_html = "" if chg is None else f'<span class="{"pos" if chg >= 0 else "neg"}">{chg:+.2f}%</span>'
        body += (f'<tr class="turn-row"><td><a class="tk" href="#{_card_id(r["ticker"])}" data-tip="{tip}">{tk}</a></td>'
                 f'<td class="num">{_summary_price(r)}</td><td class="num">{chg_html}</td>'
                 f'<td class="num">{f(r.get("ctx_ema50"))}</td><td class="num">{f(r.get("ctx_ext_ema50_atr"))}</td>'
                 f'<td class="num">{f(r.get("ctx_ema50_slope_pct"), "{:+.2f}%")}</td>'
                 f'<td>{html.escape(str(r.get("ctx_state") or ""))}</td></tr>')
    return (f'<section class="card"><h2>Turning up near the EMA50 ({len(picks)})</h2>'
            '<div class="table-wrap"><table class="summary-table"><thead><tr>' + _summary_head_html(_TURN_COLS)
            + '</tr></thead>'
            f'<tbody>{body}</tbody></table></div>'
            '<p class="note"><strong>Watch only - not a traded setup and not backtested.</strong> No setup yet, the close is at most '
            '1.5 ATR under the daily EMA50, and both turn checks pass (daily: green candle closing in the upper half; '
            '1h: above session VWAP with MACD bullish or %K above %D). A Pullback only becomes possible after the stock '
            'reclaims the EMA50 and rebuilds an uptrend. During the session the turn checks can still change.</p>'
            + _TIP_SCRIPT + '</section>')
