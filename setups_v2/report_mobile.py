"""
The phone ("lite") report: the BUY / WATCH list and their cards.
Split out of enrich_html_v2.py, which re-exports every name (enrich_html_v2.X still works).
"""

import pandas as pd
import numpy as np
import html
import re
from setups_v2.config import CONFIGH
from setups_v2.engine import SETUP_EVIDENCE
from setups_v2.report_html import _TIP_SESSIONS, _badge, _ext_line, _extended_hours_rows, _near_sr_rows, _premarket_near_row, _sr_text, _streak_text

# ---------------------------------------------------------------- mobile "lite" report
# A phone-first version of the BUY / WATCH report: one column, big tap targets, no wide tables, no hover tooltips.
# The list at the top links to a compact card per ticker; detail checks are collapsed.

_M_CSS = """
:root{color-scheme:dark;--bg:#000;--card:#0c0f14;--card2:#12161d;--border:#232a35;--text:#e8eaed;--muted:#98a2b0;--accent:#9cdcfe;
--pass:#2ecc71;--fail:#ef5b4d;--warn:#f5b041}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%;scroll-behavior:smooth;background:#000}
body{margin:0 auto;background:var(--bg);color:var(--text);
font:16.5px/1.45 system-ui,Roboto,"Segoe UI",Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums;
max-width:760px;overflow-wrap:anywhere;
padding:max(10px,env(safe-area-inset-top)) max(14px,env(safe-area-inset-right)) calc(76px + env(safe-area-inset-bottom)) max(14px,env(safe-area-inset-left))}
h1{font-size:20px;margin:8px 2px 2px}
.sub{color:var(--muted);font-size:14px;margin:0 2px 12px}
a{color:var(--accent)}
.legend{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:12px 14px;margin:6px 0 12px;font-size:14px;color:var(--muted)}
.legend b{color:var(--text)}
a.row{display:block;min-height:64px;text-decoration:none;color:inherit;background:var(--card);border:1px solid var(--border);
border-left:6px solid var(--warn);border-radius:14px;padding:14px 14px 13px;margin:10px 0;touch-action:manipulation;
-webkit-tap-highlight-color:rgba(156,220,254,.18)}
a.row:active{background:var(--card2)}
a.row.buy{border-left-color:var(--pass)}
.r1{display:flex;align-items:flex-start;gap:8px;flex-wrap:wrap}
a.row .r1{flex-wrap:nowrap}
.lhs{display:flex;align-items:center;flex-wrap:wrap;gap:4px 8px;min-width:0;flex:1 1 auto}
a.row .px{flex:none}
.tk{font-weight:800;font-size:22px;color:var(--accent);letter-spacing:.2px}
.px{margin-left:auto;font-weight:700;font-size:19px;text-align:right}
.px .ext{display:block;font-size:12.5px;font-weight:600;color:var(--muted);line-height:1.3;white-space:nowrap}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:13px;font-weight:700;white-space:nowrap}
.pill.buy{background:rgba(46,204,113,.18);color:var(--pass)}
.pill.watch{background:rgba(245,176,65,.16);color:var(--warn)}
.pill.g0,.pill.g1{background:rgba(46,204,113,.14);color:#7bdc9c}
.pill.g2{background:rgba(245,176,65,.15);color:var(--warn)}
.pill.g3{background:rgba(239,91,77,.15);color:#ee8a80}
.r2{color:var(--muted);font-size:15px;margin-top:5px}
.r3{font-size:15px;margin-top:7px;color:#d9dde3}
.r3 .need{color:var(--warn)}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:16px 14px;margin:18px 0;scroll-margin-top:10px}
.card.buy{border-top:4px solid var(--pass)}
.card.watch{border-top:4px solid var(--warn)}
.ch{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.name{color:var(--muted);font-size:15px;margin:3px 0 0}
.sess{font-size:15px;margin:9px 0 0;line-height:1.75}
.up{color:var(--pass);font-weight:700}.dn{color:var(--fail);font-weight:700}
h3{font-size:12.5px;letter-spacing:.7px;text-transform:uppercase;color:var(--accent);margin:18px 0 6px}
.kv{display:flex;justify-content:space-between;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);font-size:16px}
.kv:last-child{border-bottom:0}
.kv span:first-child{color:var(--muted)}
.kv span:last-child{text-align:right;font-weight:600}
.kv.stack{flex-direction:column;gap:4px}
.kv.stack span:last-child{text-align:left;font-weight:400;color:#d9dde3}
.kv.stack small{display:block}
.kv small{display:block;color:var(--muted);font-weight:400;font-size:12.5px}
.need-list{margin:4px 0 0;padding-left:22px}
.need-list li{margin:7px 0;font-size:16px}
.note{color:var(--muted);font-size:14px;margin:6px 0}
details{background:var(--card2);border:1px solid var(--border);border-radius:12px;margin:10px 0;padding:0 14px}
summary{cursor:pointer;min-height:52px;padding:14px 0;font-size:15.5px;color:var(--text);list-style:none;display:flex;
align-items:center;justify-content:space-between;gap:8px;touch-action:manipulation}
summary span:first-child{flex:none;white-space:nowrap}
summary span:nth-child(2){flex:1 1 auto;min-width:0;margin-left:auto;text-align:right;color:var(--muted);font-size:14.5px}
summary::-webkit-details-marker{display:none}
summary::after{content:"+";color:var(--muted);font-weight:700;font-size:18px}
details[open] summary::after{content:"-"}
.chk{display:flex;gap:10px;padding:9px 0;border-top:1px solid var(--border);font-size:15px}
.chk i{font-style:normal;font-weight:800;width:18px;flex:none}
.chk.ok i{color:var(--pass)}.chk.no i{color:var(--fail)}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12.5px;font-weight:700;background:#222a35;color:var(--muted);white-space:nowrap}
.badge.pass{background:rgba(46,204,113,.16);color:var(--pass)}
.badge.fail{background:rgba(239,91,77,.16);color:var(--fail)}
.muted-small{color:var(--muted);font-size:12.5px;font-weight:400}
.kv span:last-child .muted-small{display:inline-block;margin-top:2px}
.tfb{padding:10px 0;border-top:1px solid var(--border)}
.tfh{display:flex;gap:6px;flex-wrap:wrap;align-items:center;font-size:16px}
.fgrid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin:4px 0 10px}
.fcell{background:var(--card);padding:9px 10px;min-width:0}
.fcell small{display:block;color:var(--muted);font-size:12px}
.fcell b{font-size:15.5px;font-weight:600}
.news{list-style:none;margin:4px 0 6px;padding:0}
.news li{padding:10px 0;border-top:1px solid var(--border);font-size:15px;line-height:1.4}
.recbar{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:10px 0 6px}
.recbar span{display:block;height:100%}
details.chartbox{padding:0 10px 8px;margin:12px 0}
details.chartbox summary{padding-left:2px;padding-right:2px}
details.chartbox .note{margin:6px 2px 2px;line-height:1.9}
.top{display:inline-block;margin-top:10px;padding:10px 0;font-size:14px;color:var(--muted);text-decoration:none}
.fab{position:fixed;right:max(14px,env(safe-area-inset-right));bottom:calc(16px + env(safe-area-inset-bottom));z-index:5;display:none;width:48px;height:48px;align-items:center;justify-content:center;border-radius:50%;background:rgba(26,31,40,.94);border:1px solid var(--border);color:var(--text);font-size:22px;font-weight:700;text-decoration:none;box-shadow:0 6px 18px rgba(0,0,0,.7);touch-action:manipulation}
@media (orientation:landscape) and (min-width:760px){
 body{max-width:1100px}
 .plan{display:grid;grid-template-columns:1fr 1fr;column-gap:28px}
 .plan .kv:nth-last-child(2){border-bottom:0}
}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""


def build_mobile_report(title: str, body: str) -> str:
    head, _, tail = title.partition(" - ")
    h1, sub = (tail, head) if tail else (title, "")
    script = ("<script>(function(){var f=document.querySelector('.fab');if(!f)return;"
              "function u(){f.style.display=(window.scrollY>500)?'flex':'none'}u();"
              "addEventListener('scroll',u,{passive:true})})();</script>")
    return ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<meta name="color-scheme" content="dark"><meta name="theme-color" content="#000000">'
            f"<title>{html.escape(title)}</title><style>{_M_CSS}</style></head>"
            f'<body id="top"><h1>{html.escape(h1)}</h1>'
            + (f'<p class="sub" style="margin-bottom:4px">{html.escape(sub)}</p>' if sub else "")
            + f'{body}<a class="fab" href="#top" aria-label="Back to list">&uarr;</a>{script}</body></html>')


def _m_num(v, nd=2):
    return "" if v is None or (isinstance(v, float) and v != v) else f"{float(v):,.{nd}f}"


def _m_sessions(row: dict) -> str:
    """Price line(s): live / extended-hours price, then the previous session's O/H/L/C (close coloured vs the day before)."""
    def g(k):
        v = row.get(k)
        return None if v is None or (isinstance(v, float) and v != v) else v
    lines = []
    if g("market_open") is False and g("current_ext_price") is not None:
        label = _TIP_SESSIONS.get(g("current_ext_session"), "Extended hours")
        pct = g("current_ext_pct")
        pc = ""
        if pct is not None:
            pc = f' <span class="{"up" if pct >= 0 else "dn"}">{pct:+.2f}%</span> vs close'
        lines.append(f"{label}: <b>{float(g('current_ext_price')):,.2f}</b>{pc}")
    elif g("market_open"):
        lines.append("Market open")
    if g("prev_day_open") is not None:
        c, prior = g("prev_day_close"), g("prev_day_prior_close")
        cls = "" if prior is None or c == prior else ("up" if c > prior else "dn")
        nb = lambda t: f'<span style="white-space:nowrap">{t}</span>'
        lines.append(f'<span class="note">Prev day {html.escape(str(g("prev_day_date")))}:</span> '
                     + " ".join([nb(f"O {_m_num(g('prev_day_open'))}"), nb(f"H {_m_num(g('prev_day_high'))}"),
                                 nb(f"L {_m_num(g('prev_day_low'))}"), nb(f'C <span class="{cls}">{_m_num(c)}</span>')]))
    return "<br>".join(lines)


def _m_checks(items) -> str:
    return "".join(f'<div class="chk {"ok" if ok else "no"}"><i>{"✓" if ok else "✗"}</i><span>{html.escape(str(lbl))}</span></div>'
                   for lbl, ok in items)



def _m_chart_svg(df, lines, levels, in_progress=False, date_mode="day", label="chart", W=360, price_h=215, vol_h=36):
    """Phone-sized candlestick chart. One SVG user unit = one CSS pixel at ~360 px width, so 10.5-unit text is readable.
    lines  : [(name, series, colour)]   moving averages / VWAP, drawn over the candles
    levels : [(label, price, colour, dash)]   horizontal levels with their label in the right gutter (only if near the candles)"""
    n = len(df)
    if n < 5:
        return ""
    pad_l, pad_r, pad_t, gap, axis_h = 4, 84, 8, 6, 18
    H = pad_t + price_h + gap + vol_h + axis_h
    plot_w = W - pad_l - pad_r
    step = plot_w / n
    body = max(2.0, min(8.0, step * 0.62))
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    lo, hi = float(np.nanmin(l)), float(np.nanmax(h))
    rng = (hi - lo) or 1.0
    keep = [(lb, float(v), col, dash) for lb, v, col, dash in levels
            if v is not None and not pd.isna(v) and lo - 0.6 * rng <= float(v) <= hi + 0.6 * rng]
    vals = [lo, hi] + [v for _, v, _, _ in keep]
    vmin, vmax = min(vals), max(vals)
    padv = (vmax - vmin) * 0.05 or 0.5
    vmin, vmax = vmin - padv, vmax + padv

    def y(v):
        return pad_t + (vmax - v) / (vmax - vmin) * price_h

    def x(i):
        return pad_l + step * (i + 0.5)

    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York")
    dates = np.asarray(idx.date)
    t = ['<text font-family="system-ui,Roboto,sans-serif" font-size="10" fill="#98a2b0">']  # placeholder (kept simple)
    parts = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#000" rx="8"/>']
    for k in range(1, 4):                                     # faint gridlines with a tiny price at the left
        gy = pad_t + price_h * k / 4
        gv = vmax - (vmax - vmin) * k / 4
        parts.append(f'<line x1="{pad_l}" x2="{pad_l + plot_w}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="#1a2029" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l + 2}" y="{gy - 2:.1f}" font-size="9.5" fill="#566070">{gv:,.2f}</text>')
    vol_arr = df["Volume"].to_numpy(dtype=float)
    vmaxv = float(np.nanmax(vol_arr)) or 1.0
    vol0 = pad_t + price_h + gap
    for i in range(n):
        up = c[i] >= o[i]
        col = "#2ecc71" if up else "#ef5b4d"
        op = ' opacity=".55"' if (in_progress and i == n - 1) else ""
        yt, yb = y(max(o[i], c[i])), y(min(o[i], c[i]))
        parts.append(f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(h[i]):.1f}" y2="{y(l[i]):.1f}" stroke="{col}" stroke-width="1"{op}/>')
        parts.append(f'<rect x="{x(i) - body / 2:.1f}" y="{yt:.1f}" width="{body:.1f}" height="{max(1.0, yb - yt):.1f}" fill="{col}"{op}/>')
        vh = float(vol_arr[i]) / vmaxv * vol_h
        parts.append(f'<rect x="{x(i) - body / 2:.1f}" y="{vol0 + vol_h - vh:.1f}" width="{body:.1f}" height="{max(0.5, vh):.1f}" fill="{col}" opacity=".45"/>')
    for name, series, col in lines:                          # MAs / VWAP (broken at day boundaries for session VWAP)
        try:
            ser = series.reindex(df.index).to_numpy(dtype=float)
        except Exception:
            continue
        seg = []
        for i in range(n):
            new_day = date_mode == "hour" and name == "VWAP" and i > 0 and dates[i] != dates[i - 1]
            if np.isnan(ser[i]) or new_day:
                if len(seg) > 1:
                    parts.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{col}" stroke-width="1.6"/>')
                seg = []
            if not np.isnan(ser[i]):
                seg.append(f"{x(i):.1f},{y(ser[i]):.1f}")
        if len(seg) > 1:
            parts.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{col}" stroke-width="1.6"/>')
    ys = sorted(((y(v), lb, v, col, dash) for lb, v, col, dash in keep), key=lambda r: r[0])
    last_label_y = -99.0
    for yy, lb, v, col, dash in ys:
        parts.append(f'<line x1="{pad_l}" x2="{pad_l + plot_w}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="1.2" stroke-dasharray="{dash}"/>')
        ly = max(yy + 3.5, last_label_y + 12)
        last_label_y = ly
        parts.append(f'<text x="{pad_l + plot_w + 4}" y="{ly:.1f}" font-size="10" font-weight="600" fill="{col}">{html.escape(lb)} {v:,.2f}</text>')
    last_x_label = -99.0                                     # x-axis labels
    for i in range(n):
        show = ((date_mode == "day" and (n - 1 - i) % 10 == 0) or (date_mode == "hour" and (i == 0 or dates[i] != dates[i - 1])))
        if show and x(i) - last_x_label >= 38 and 18 <= x(i) <= pad_l + plot_w - 8:
            d = dates[i]
            parts.append(f'<text x="{x(i):.1f}" y="{H - 4}" font-size="9.5" fill="#6b7686" text-anchor="middle">{d.month:02d}/{d.day:02d}</text>')
            last_x_label = x(i)
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block;width:100%;height:auto" role="img" '
            f'aria-label="{html.escape(label)}">{"".join(parts)}</svg>')


def _m_swatch(col, text, dash=False):
    return (f'<span style="white-space:nowrap;margin-right:10px"><span style="display:inline-block;width:14px;height:0;'
            f'border-top:2px {"dashed" if dash else "solid"} {col};vertical-align:middle;margin-right:4px"></span>{html.escape(text)}</span>')


def _m_charts(report: dict, sig: dict, px: float) -> str:
    """Daily chart (open) and 1h chart (collapsed) for a card, or '' when the chart data is missing."""
    chart = report.get("chart") or {}
    out = []
    dm = chart.get("daily_m")
    if dm is not None and not dm["ohlcv"].empty:
        lv = [("Stop", sig.get("stop"), "#ef5b4d", "5 4"), ("+2R", sig.get("arm_level"), "#2ecc71", "2 4"),
              ("Res", sig.get("target"), "#c9d1d9", "6 4"), ("Max", sig.get("buy_up_to"), "#4dd0e1", "3 3"),
              ("Buy<=", sig.get("buy_below"), "#4dd0e1", "3 3"), ("Now", px, "#ffd166", "1 3")]
        svg = _m_chart_svg(dm["ohlcv"], [("EMA20", dm["ema20"], "#f5b041"), ("EMA50", dm["ema50"], "#b388ff")], lv,
                           dm.get("today_in_progress", False), "day", f"{report['ticker']} daily chart")
        if svg:
            leg = (_m_swatch("#f5b041", "EMA20") + _m_swatch("#b388ff", "EMA50") + _m_swatch("#ef5b4d", "stop", True)
                   + _m_swatch("#2ecc71", "+2R", True) + _m_swatch("#c9d1d9", "resistance", True) + _m_swatch("#4dd0e1", "max buy", True) + _m_swatch("#ffd166", "now", True))
            out.append(f'<details class="chartbox" open><summary><span>Daily chart</span><span>{len(dm["ohlcv"])} days</span></summary>'
                       f'{svg}<div class="note">{leg}</div></details>')
    h = chart.get("ohlcv")
    if h is not None and not h.empty:
        lines = []
        if chart.get("vwap") is not None:
            lines.append(("VWAP", chart["vwap"], "#6ec1ff"))
        for nm, ser in list((chart.get("overlays") or {}).items())[:1]:
            lines.append((nm, ser, "#b388ff"))
        lv = [("Now", px, "#ffd166", "1 3"), ("Stop", sig.get("stop"), "#ef5b4d", "5 4"), ("+2R", sig.get("arm_level"), "#2ecc71", "2 4")]
        svg = _m_chart_svg(h, lines, lv, False, "hour", f"{report['ticker']} 1h chart")
        if svg:
            leg = "".join(_m_swatch(col, nm) for nm, _, col in lines) + _m_swatch("#ffd166", "now", True)
            out.append(f'<details class="chartbox"><summary><span>1h chart</span><span>{h.index.normalize().nunique()} days</span></summary>'
                       f'{svg}<div class="note">{leg}</div></details>')
    return "".join(out)



# ---- the rest of the desktop card, minimised: each block is a collapsed section with a one-line teaser
def _m_kv(items) -> str:
    """items: [(label_html, value_html)]  (both already HTML-safe, as built for the desktop card)"""
    def stacked(vl: str) -> bool:
        return "<br>" in vl or len(re.sub(r"<[^>]+>", "", vl)) > 34
    return "".join(f'<div class="kv{" stack" if stacked(vl) else ""}"><span>{lb}</span><span>{vl}</span></div>' for lb, vl in items)


def _m_details(title: str, teaser: str, inner: str) -> str:
    return (f'<details><summary><span>{html.escape(title)}</span><span>{teaser}</span></summary>{inner}</details>')


def _m_sec_timeframes(report: dict) -> str:
    cfg, results, sr_levels = report["cfg"], report["results"], report["sr_levels"]
    blocks, teaser = [], []
    for tf in ("1D", "4h", "1h"):
        r, sr = results[tf], sr_levels[tf]
        pats = ", ".join(r["patterns_detected"]) if r["patterns_detected"] else "-"
        blocks.append(
            f'<div class="tfb"><div class="tfh"><b>{tf}</b> {_badge(r["trend_up"], "UP", "DOWN")} '
            f'{_badge(r["k_above_d"], "K>D", "K<D")} {_badge(not r["overbought"], "OK", "HOT")}</div>'
            f'<div class="note">close {r["close"]} &middot; {cfg["MA_TYPE"].upper()}{cfg["MA_PERIOD"]} {r["MA"]} &middot; '
            f'%K {r["%K"]} / %D {r["%D"]}<br>support {sr["support"]} &middot; resistance {sr["resistance"]}'
            f'<br>candles: {html.escape(pats)}</div></div>')
        teaser.append(f'{tf} {"up" if r["trend_up"] else "down"}')
    return _m_details("Timeframes", " &middot; ".join(teaser), "".join(blocks))


def _m_sec_levels(report: dict) -> str:
    cfg, price, tr = report["cfg"], report["current_price"], report["today_range"]
    items = []
    fund = report.get("fundamentals") or {}
    try:
        lo, hi = float(fund.get("52-Week Low")), float(fund.get("52-Week High"))
        pos = (price - lo) / (hi - lo) * 100 if hi > lo else 50.0
        items.append(("52-week range", f"{lo:.2f} - {hi:.2f}<small>price at {pos:.0f}% of the range</small>"))
        teaser = f"52w at {pos:.0f}%"
    except (TypeError, ValueError):
        teaser = ""
    atr_d = report.get("daily_atr")
    if atr_d:
        items.append((f"Daily ATR({cfg['ATR_PERIOD']})", f"{atr_d:.2f}<small>today used {tr['day_range'] / atr_d * 100:.0f}%</small>"))
        items.append(("ATR projected range", f"{tr['day_high'] - atr_d:.2f} - {tr['day_low'] + atr_d:.2f}"))
    items += _extended_hours_rows(report)
    items.append(_premarket_near_row(report))
    av = report.get("avwap")
    if av and av.get("value") is not None:
        items.append((f"Anchored VWAP<small>from {av['anchor_type']} of {av['anchor_date']}</small>",
                      f"{av['value']} " + ("" if av["above"] is None else _badge(av["above"], "ABOVE", "BELOW"))))
    st = report.get("hhhl")
    if st:
        items += [(f"HH/HL days<small>last {st['window']} completed</small>",
                   f"{st['up_days']} up / {st['down_days']} down / {st['mixed_days']} mixed<small>net {st['net']:+d}</small>"),
                  ("Current streak", _streak_text(st["current_streak"])), ("Previous streak", _streak_text(st["previous_streak"]))]
    items += [("Today's range", f"{tr['day_low']} - {tr['day_high']}<small>range {tr['day_range']}</small>"),
              ("1h entry support", f"{report['entry_support']}"),
              ("Day support S1 / S2", _sr_text((report.get("day_sr") or {}).get("supports"))),
              ("Day resistance T1 / T2", _sr_text((report.get("day_sr") or {}).get("resistances"), with_r=True))]
    items += _near_sr_rows(report)
    vb = "" if report.get("above_vwap") is None else _badge(report["above_vwap"], "ABOVE", "BELOW")
    items += [(f"POC<small>{cfg.get('CHART_DAYS', 7)}d volume profile</small>", f"{report.get('poc', 'N/A')}"),
              ("Session VWAP", f"{report.get('vwap') if report.get('vwap') is not None else 'N/A'} {vb}")]
    lrc = report.get("lrc")
    if lrc:
        p = lrc["position_pct"]
        items += [(f"Regression channel<small>{lrc['bars']} bars, {lrc['dev']}&sigma;</small>", f"{lrc['last_lower']} / {lrc['last_mid']} / {lrc['last_upper']}"),
                  ("Channel slope / fit", f"{lrc['slope_pct_per_day']:+.2f}%/day &middot; R&sup2; {lrc['r2']}"),
                  ("Price in channel", f"{p:.0f}%" + (" (below)" if p < 0 else " (above)" if p > 100 else ""))]
    return _m_details("Other levels", html.escape(teaser), _m_kv(items))


def _m_sec_catalysts(report: dict) -> str:
    cat, news = report.get("catalysts") or {}, report.get("news") or []
    if not cat and not news:
        return ""
    cfg = report["cfg"]
    items, teaser = [], ""
    if cat:
        ed, d = cat.get("next_earnings_date") or "N/A", cat.get("days_to_earnings")
        items.append(("Next earnings", html.escape(str(ed)) + (f" ({d:+d}d)" if d is not None else "") + " "
                      + _badge(not cat.get("in_earnings_window", False), "CLEAR", "IN WINDOW")))
        teaser = f"earnings {d:+d}d" if d is not None else ""
        sp = cat.get("in_sp500")
        extra = f"<small>added {cat['sp500_added_date']}, {cat.get('days_since_index_addition')}d ago</small>" if cat.get("sp500_added_date") else ""
        items.append(("S&amp;P 500", ('<span class="badge">N/A</span>' if sp is None else _badge(sp, "MEMBER", "NOT MEMBER")) + extra))
        if cat.get("recent_index_addition"):
            items.append(("Recent index addition", _badge(True, "RECENT ADD")))
        for key, label in (("buyback_headlines", "Buyback headlines"), ("guidance_headlines", "Guidance headlines")):
            heads = cat.get(key) or []
            body = ('<span class="muted-small">not checked (no news fetched)</span>' if cat.get("news_skipped")
                    else "<br>".join(html.escape(x) for x in heads[:4]) or "none")
            items.append((f"{label} ({len(heads)})", body))
        acts = cat.get("rating_actions") or []
        items.append((f"Analyst actions<small>{cat.get('upgrades', 0)} up / {cat.get('downgrades', 0)} down</small>",
                      "<br>".join(html.escape(a) for a in acts[:5]) or "none"))
        rank = cat.get("social_rank", "N/A")
        if rank != "N/A":
            mom = cat.get("social_momentum_pct", "N/A")
            mom_s = f"+{mom}%" if isinstance(mom, (int, float)) and mom > 0 else f"{mom}%"
            items.append(("Reddit (ApeWisdom)", f"rank #{rank}<small>{cat.get('social_mentions', 0)} mentions ({mom_s} 24h), {cat.get('social_upvotes', 0)} upvotes</small>"))
        else:
            items.append(("Reddit (ApeWisdom)", '<span class="badge">Not in Top 50</span>'))
    inner = _m_kv(items)
    if news:
        tz = {"America/New_York": "ET", "Asia/Jerusalem": "IL", "UTC": "UTC"}.get(cfg.get("NEWS_TZ", "America/New_York"), "")
        li = ""
        for n in news[:8]:
            url = n["url"] if str(n["url"]).lower().startswith(("http://", "https://")) else ""
            ttl = html.escape(n["headline"])
            link = f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{ttl}</a>' if url else ttl
            li += f'<li><span class="muted-small">{html.escape(n["time"])} {tz}</span><br>{link}</li>'
        inner += f'<h3>News ({len(news)})</h3><ul class="news">{li}</ul>'
        teaser = (teaser + " &middot; " if teaser else "") + f"{len(news)} news"
    elif cat.get("news_skipped"):
        inner += f'<div class="note">News: {html.escape(str(cat["news_skipped"]))}</div>'
    return _m_details("Event catalysts", teaser, inner)


def _m_sec_analysts(report: dict) -> str:
    an = report.get("analyst") or {}
    if not an:
        return ""
    counts = an.get("recommendation_counts") or {}
    total = sum(v for v in counts.values() if isinstance(v, (int, float))) or 0
    palette = {"strong buy": "#1faa59", "buy": "#2ecc71", "hold": "#f5b041", "sell": "#ef8a4d", "strong sell": "#ef5b4d"}
    bar = leg = ""
    if total:
        segs = "".join(f'<span style="width:{n / total * 100:.1f}%;background:{palette.get(str(lb).lower(), "#6b7686")}"></span>'
                       for lb, n in counts.items() if n)
        bar = f'<div class="recbar">{segs}</div>'
        leg = '<div class="note">' + " &middot; ".join(f"{html.escape(str(lb))} <b>{n}</b>" for lb, n in counts.items()) + "</div>"
    head = _m_kv([("Consensus", html.escape(str(an.get("recommendation_key", "N/A")))),
                  ("Mean score", html.escape(str(an.get("recommendation_mean", "N/A")))),
                  ("# analysts", html.escape(str(an.get("num_analyst_opinions", "N/A"))))])
    imp = an.get("eps_improving")
    eps = _m_kv([("Consensus EPS (current qtr)", html.escape(str(an.get("eps_current", "N/A")))),
                 ("30 days ago", html.escape(str(an.get("eps_30d_ago", "N/A")))),
                 ("# analysts (EPS)", html.escape(str(an.get("eps_num_analysts", "N/A")))),
                 ("EPS trend", '<span class="badge">N/A</span>' if imp is None else _badge(imp, "IMPROVING", "DETERIORATING"))])
    teaser = f'{html.escape(str(an.get("recommendation_key", "")))} ({html.escape(str(an.get("num_analyst_opinions", "")))})'
    return _m_details("Analysts & EPS revisions", teaser, head + bar + leg + "<h3>EPS estimate revisions</h3>" + eps)


def _m_sec_fundamentals(report: dict) -> str:
    fund = report.get("fundamentals") or {}
    if not fund:
        return ""
    cells = "".join(f'<div class="fcell"><small>{html.escape(str(k))}</small><b>{html.escape(str(v))}</b></div>' for k, v in fund.items())
    keys = [("Market Cap", "cap"), ("Forward P/E", "fwd P/E")]
    teaser = " &middot; ".join(f"{lb} {html.escape(str(fund[k]))}" for k, lb in keys if k in fund)
    return _m_details("Fundamentals", teaser, f'<div class="fgrid">{cells}</div>')


def render_mobile_card(report: dict, row: dict) -> str:
    """Compact phone card for one BUY / WATCH ticker."""
    sig = report["signal"]
    tk = report["ticker"]
    kind = "buy" if sig["signal"] == "BUY" else "watch"
    px = float(report.get("current_price") or 0)
    ctx = sig["context"]
    cfg = report.get("cfg") or CONFIGH
    name = row.get("name") or ""
    sector_industry = " · ".join(x for x in (row.get("sector"), row.get("industry")) if x)

    head = (f'<div class="ch"><span class="tk">{html.escape(tk)}</span>'
            f'<span class="pill {kind}">{sig["signal"]}</span>'
            + (f'<span class="pill g{0 if (sig.get("distance") or 0) == 0 else 1 if sig["distance"] <= 1.5 else 2 if sig["distance"] <= 3.5 else 3}">'
               f'gap {sig["distance"]:.1f}</span>' if sig.get("distance") is not None and kind == "watch" else "")
            + f'<span class="px">{px:,.2f}</span></div>')
    sub = f'<div class="name">{html.escape(name)}' + (f"<br>{html.escape(sector_industry)}" if sector_industry else "") + "</div>"
    sess = _m_sessions(row)
    sess_html = f'<div class="sess">{sess}</div>' if sess else ""

    out = [f'<section class="card {kind}" id="m-{html.escape(tk)}">', head, sub, sess_html]
    try:
        out.append(_m_charts(report, sig, px))
    except Exception as e_:                                   # a chart problem must never hide the card
        out.append(f'<div class="note">(chart unavailable: {html.escape(type(e_).__name__)})</div>')

    out.append(f'<h3>Setup: {html.escape(sig.get("setup_label") or "")}</h3>')
    out.append(f'<div class="note">{html.escape(sig.get("level_desc") or "")}</div>')
    ev = SETUP_EVIDENCE.get(sig.get("setup"), "")
    if ev:
        out.append(f'<div class="note">{html.escape(ev)}</div>')

    if sig.get("buy_when"):
        out.append("<h3>What would make it a BUY</h3><ol class=\"need-list\">"
                   + "".join(f"<li>{html.escape(x)}</li>" for x in sig["buy_when"]) + "</ol>")

    def kv(label, value, small=""):
        return f'<div class="kv"><span>{label}</span><span>{value}{f"<small>{small}</small>" if small else ""}</span></div>'

    plan = [kv("Initial stop", f"{sig['stop']:.2f}", f"1R = {sig['risk']:.2f} per share"),
            kv(f"+{cfg['TRAIL_ARM_R']:g}R level (arms the trail)", f"{sig['arm_level']:.2f}"),
            kv(f"Then trail under daily {cfg['TRAIL_MA']}", f"now {sig['trail_now']:.2f}", "raise only; no profit target"),
            kv("Exit", f"trail / stop / {cfg['EXIT_MAX_HOLD_DAYS']} days"),
            kv("Resistance (R:R filter)", f"{sig['target']:.2f}", html.escape(sig.get("target_src") or "")),
            kv(f"R:R (min {cfg['SETUP_MIN_RR']:g})", f"{sig['rr']:.2f}")]
    if sig.get("buy_up_to"):
        plan.append(kv("Buy up to", f"{sig['buy_up_to']:.2f}", f"{(sig['buy_up_to'] / px - 1) * 100:+.1f}% from here" if px else ""))
    if sig.get("buy_below"):
        plan.append(kv("Buy at / below", f"{sig['buy_below']:.2f}", f"{(sig['buy_below'] / px - 1) * 100:+.1f}% from here" if px else ""))
    out.append('<h3>Trade plan</h3><div class="plan">' + "".join(plan) + "</div>")

    n_ok = sum(1 for _, ok in sig.get("turn", []) if ok)
    out.append(f'<details><summary><span>Turn (daily + 1h)</span><span>{n_ok}/{len(sig.get("turn", []))}</span></summary>'
               + _m_checks(sig.get("turn", [])) + "</details>")
    out.append(f'<details><summary><span>Required checks</span><span>{sum(1 for _, ok in sig["required"] if ok)}/{len(sig["required"])}</span></summary>'
               + _m_checks(sig["required"]) + "</details>")
    out.append(f'<details><summary><span>Confirmations (need {sig["min_confirms"]})</span>'
               f'<span>{sig["n_confirms"]}/{len(sig["confirms"])}</span></summary>' + _m_checks(sig["confirms"]) + "</details>")

    lows, highs = ctx.get("last_swing_lows"), ctx.get("last_swing_highs")
    out.append('<details><summary><span>Trend context</span><span>' + html.escape(ctx["state"].replace("_", " ").title()) + "</span></summary>"
               + kv("Close vs EMA50", f"{ctx['close']} vs {ctx['ema50']}", f"{ctx['ext_ema50_atr']:+.1f} ATR")
               + kv("EMA50 slope (5d)", "n/a" if ctx.get("ema50_slope_pct") is None else f"{ctx['ema50_slope_pct']:+.2f}%")
               + kv("Higher lows", "yes" if ctx["higher_lows"] else "no", html.escape(str(lows)) if lows else "")
               + kv("Higher highs", "yes" if ctx["higher_highs"] else "no", html.escape(str(highs)) if highs else "")
               + "</details>")

    for sec in (_m_sec_timeframes, _m_sec_levels, _m_sec_catalysts, _m_sec_analysts, _m_sec_fundamentals):
        try:
            out.append(sec(report))
        except Exception as e_:                               # one broken block must not hide the others
            out.append(f'<div class="note">({sec.__name__[7:]} unavailable: {html.escape(type(e_).__name__)})</div>')
    out.append('<a class="top" href="#top">&uarr; back to list</a></section>')
    return "".join(out)


def mobile_summary(rows: list) -> str:
    """Top of the phone report: one tappable row per BUY / WATCH ticker, closest to a BUY first."""
    picks = [r for r in rows if r.get("signal") in ("BUY", "WATCH") and r.get("report") == "up"]
    if not picks:
        return '<div class="legend">No BUY or WATCH setups right now.</div>'

    def gap(r):
        v = r.get("setup_distance")
        return 99.0 if v is None or v != v else float(v)
    picks.sort(key=lambda r: (r["signal"] != "BUY", gap(r), -(r.get("setup_rr") or 0)))
    n_buy = sum(1 for r in picks if r["signal"] == "BUY")
    out = [f'<p class="sub">{n_buy} BUY &middot; {len(picks) - n_buy} WATCH &middot; closest to a BUY first. Tap a row for the full card.</p>',
           '<div class="legend"><b>Exit plan:</b> initial stop, no profit target. After +2R trail the stop under the daily EMA20 '
           '(raise only), leave after 90 trading days. <b>Gap</b> = distance from a BUY (0 = BUY). Only pullbacks in an uptrend are traded.</div>']
    for r in picks:
        kind = "buy" if r["signal"] == "BUY" else "watch"
        g = gap(r)
        gcls = "g0" if g == 0 else "g1" if g <= 1.5 else "g2" if g <= 3.5 else "g3"
        items = [x for x in str(r.get("setup_buy_when") or "").split(" | ") if x]
        r3 = ""
        if kind == "buy":
            r3 = (f"Buy up to <b>{_m_num(r.get('setup_buy_up_to'))}</b> &middot; +2R {_m_num(r.get('setup_arm_level'))}"
                  if r.get("setup_buy_up_to") else f"+2R {_m_num(r.get('setup_arm_level'))}")
        elif items:
            more = f' <span class="note">(+{len(items) - 1} more)</span>' if len(items) > 1 else ""
            r3 = f'<span class="need">Needs:</span> {html.escape(items[0])}{more}'
        out.append(
            f'<a class="row {kind}" href="#m-{html.escape(str(r["ticker"]))}">'
            f'<div class="r1"><span class="lhs"><span class="tk">{html.escape(str(r["ticker"]))}</span>'
            f'<span class="pill {kind}">{r["signal"]}</span>'
            + (f'<span class="pill {gcls}">gap {g:.1f}</span>' if kind == "watch" and g != 99.0 else "")
            + f'</span><span class="px">{_m_num(r.get("current_price"))}{_ext_line(r, "up", "dn", "span")}</span></div>'
            f'<div class="r2">{html.escape(str(r.get("setup_label") or ""))} &middot; R:R {_m_num(r.get("setup_rr"))} '
            f'&middot; stop {_m_num(r.get("setup_stop"))}</div>'
            + (f'<div class="r3">{r3}</div>' if r3 else "") + "</a>")
    return "".join(out)
