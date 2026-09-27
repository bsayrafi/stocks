"""
mds_report.py
=============
Self-contained HTML report for the Multi-Day Swing scanner:
  - market regime + funnel (how many names survived each gate)
  - sector rotation map (RRG scatter) + table
  - one card per setup: levels (stop / TP1 / TP2 / R:R), trend, flow,
    fundamentals, and a daily chart with the regression channel, anchored VWAP,
    POC / value area, stop and targets
  - near misses (names that failed exactly one gate)
Light/dark aware, no external assets (opens offline on the phone via ntfy).
"""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from mds_config import MDS_CONFIG as CFG

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--card:#ffffff;--border:#e4e3df;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;
--grid:#ecebe7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0ca30c;--crit:#d03b3b;--candle:#52514e;--va:rgba(27,175,122,.10);
--chip:#f1f0ec}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface:#1a1a19;--card:#232322;
--border:#3a3936;--text:#fff;--text2:#c3c2b7;--muted:#8f8e86;--grid:#2e2d2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--candle:#c3c2b7;
--va:rgba(25,158,112,.16);--chip:#2e2d2b}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--card:#232322;--border:#3a3936;--text:#fff;--text2:#c3c2b7;--muted:#8f8e86;
--grid:#2e2d2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--candle:#c3c2b7;--va:rgba(25,158,112,.16);--chip:#2e2d2b}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:22px;margin:4px 0}h2{font-size:17px;margin:28px 0 10px}
.sub{color:var(--text2)}.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;margin:12px 0}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--text2)}
.ok{color:var(--good);font-weight:600}.bad{color:var(--crit);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{color:var(--text2);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin-top:10px}
.kv div{display:flex;justify-content:space-between;border-bottom:1px dashed var(--grid);padding:2px 0}.kv span:first-child{color:var(--text2)}
.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}.lv{background:var(--chip);border-radius:8px;padding:8px}
.lv b{display:block;font-size:17px;font-variant-numeric:tabular-nums}.lv small{color:var(--text2)}
.scroll{overflow-x:auto}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--text2);margin:8px 0 2px}
.legend i{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
svg{display:block;width:100%;height:auto}svg.pc{min-width:640px}svg text{fill:var(--text2);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:12px;
box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9;font-variant-numeric:tabular-nums}
details summary{cursor:pointer;color:var(--text2)}
@media (max-width:600px){.levels{grid-template-columns:repeat(2,1fr)}main{padding:12px}}
"""

JS = """
const tip=document.querySelector('.tip');
document.querySelectorAll('svg[data-bars]').forEach(svg=>{
  const d=JSON.parse(svg.dataset.bars), g=JSON.parse(svg.dataset.geom), hl=svg.querySelector('.hl');
  svg.addEventListener('mousemove',e=>{
    const r=svg.getBoundingClientRect(), x=(e.clientX-r.left)*g.w/r.width;
    const i=Math.max(0,Math.min(d.length-1,Math.round((x-g.l-g.bw/2)/g.bw)));
    const b=d[i]; hl.setAttribute('x',g.l+i*g.bw); hl.style.display='block';
    tip.innerHTML=`<b>${b[0]}</b><br>O ${b[1]} &nbsp;H ${b[2]}<br>L ${b[3]} &nbsp;C ${b[4]}<br>Vol ${b[5]}`;
    tip.style.display='block'; tip.style.left=Math.min(e.clientX+14,innerWidth-150)+'px'; tip.style.top=(e.clientY+14)+'px';});
  svg.addEventListener('mouseleave',()=>{tip.style.display='none';hl.style.display='none';});
});
document.querySelectorAll('svg[data-pts] circle').forEach(c=>{
  c.addEventListener('mousemove',e=>{tip.innerHTML=c.dataset.t;tip.style.display='block';
    tip.style.left=Math.min(e.clientX+14,innerWidth-190)+'px';tip.style.top=(e.clientY+14)+'px';});
  c.addEventListener('mouseleave',()=>tip.style.display='none');
});
"""


def _e(v) -> str:
    return html.escape(str(v))


def _f(v, nd=2, pct=False, suffix=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v * 100:.{nd}f}%" if pct else f"{v:,.{nd}f}{suffix}"


def max_entry(s: dict) -> float:
    """Highest entry price that still gives R:R >= MIN_RR to TP1: (TP1 + RR*stop) / (1 + RR)."""
    rr = CFG["MIN_RR"]
    return (s["tp1"] + rr * s["stop"]) / (1 + rr)


def _vol(v):
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"


# --------------------------------------------------------------------------- charts

def price_chart(daily: pd.DataFrame, s: dict, days: int = 90) -> str:
    """Daily candles + regression channel (last LRC_LENGTH bars), anchored VWAP,
    value area / POC, stop and targets. One y-axis (price)."""
    df = daily.iloc[-days:]
    n = len(df)
    W, H, L, R, T, B = 900, 360, 8, 92, 12, 22
    bw = (W - L - R) / n
    lvls = [s.get(k) for k in ("stop", "tp1", "tp2", "poc", "vah", "val", "lrc_upper", "lrc_lower") if s.get(k)]
    lo = min(df["Low"].min(), *lvls) if lvls else df["Low"].min()
    hi = max(df["High"].max(), *lvls) if lvls else df["High"].max()
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    y = lambda p: T + (hi - p) / (hi - lo) * (H - T - B)
    x = lambda i: L + i * bw + bw / 2
    out = [f'<rect class="hl" y="{T}" width="{bw:.2f}" height="{H - T - B}" fill="var(--grid)" style="display:none"/>']

    # grid (recessive)
    for k in range(5):
        p = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(p):.1f}" y2="{y(p):.1f}" stroke="var(--grid)"/>')
    # value area band + POC
    if s.get("vah") and s.get("val"):
        out.append(f'<rect x="{L}" width="{W - L - R}" y="{y(s["vah"]):.1f}" height="{max(y(s["val"]) - y(s["vah"]), 1):.1f}" fill="var(--va)"/>')
    # candles: up hollow, down filled (neutral ink -- colour is reserved for levels)
    for i, (ts, r) in enumerate(df.iterrows()):
        cx = x(i)
        up = r.Close >= r.Open
        top, bot = y(max(r.Open, r.Close)), y(min(r.Open, r.Close))
        out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(r.High):.1f}" y2="{y(r.Low):.1f}" stroke="var(--candle)" stroke-width="1"/>')
        out.append(f'<rect x="{cx - bw * 0.32:.1f}" y="{top:.1f}" width="{bw * 0.64:.1f}" height="{max(bot - top, 1):.1f}" '
                   f'fill="{"var(--card)" if up else "var(--candle)"}" stroke="var(--candle)" stroke-width="1"/>')
    # regression channel over the last LRC_LENGTH bars (fit recomputed here for drawing)
    m = min(CFG["LRC_LENGTH"], n)
    yy = df["Close"].to_numpy(float)[-m:]
    xx = np.arange(m, dtype=float)
    slope, icpt = np.polyfit(xx, yy, 1)
    sd = float((yy - (icpt + slope * xx)).std())
    i0 = n - m
    for off, dash in ((0, ""), (CFG["LRC_DEV"], ' stroke-dasharray="5 4"'), (-CFG["LRC_DEV"], ' stroke-dasharray="5 4"')):
        p0, p1 = icpt + off * sd, icpt + slope * (m - 1) + off * sd
        out.append(f'<line x1="{x(i0):.1f}" x2="{x(n - 1):.1f}" y1="{y(p0):.1f}" y2="{y(p1):.1f}" stroke="var(--s1)" stroke-width="2"{dash}/>')
    # anchored VWAP from the channel's lowest low
    win = df.iloc[i0:]
    a = int(np.argmin(win["Low"].to_numpy()))
    seg = win.iloc[a:]
    tp = (seg["High"] + seg["Low"] + seg["Close"]) / 3
    av = (tp * seg["Volume"]).cumsum() / seg["Volume"].cumsum()
    pts = " ".join(f"{x(i0 + a + k):.1f},{y(v):.1f}" for k, v in enumerate(av.to_numpy()))
    out.append(f'<polyline points="{pts}" fill="none" stroke="var(--s2)" stroke-width="2"/>')
    # horizontal levels with direct labels on the right
    labels = []
    for key, name, col, dash in (("poc", "POC", "var(--s3)", ""), ("tp2", "TP2", "var(--good)", ' stroke-dasharray="2 3"'),
                                 ("tp1", "TP1", "var(--good)", ""), ("stop", "Stop", "var(--crit)", "")):
        v = s.get(key)
        if not v:
            continue
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{col}" stroke-width="{1.5 if key != "poc" else 1}"{dash}/>')
        labels.append([y(v), f"{name} {v:,.2f}"])
    labels.sort()
    for k in range(1, len(labels)):                       # nudge overlapping labels apart
        labels[k][0] = max(labels[k][0], labels[k - 1][0] + 13)
    for yv, t in labels:
        out.append(f'<text x="{W - R + 6}" y="{yv + 4:.1f}">{_e(t)}</text>')
    # x-axis month ticks
    last_m = None
    for i, ts in enumerate(df.index):
        if ts.month != last_m:
            last_m = ts.month
            out.append(f'<text x="{x(i):.1f}" y="{H - 6}">{ts.strftime("%b")}</text>')
    bars = [[ts.strftime("%Y-%m-%d"), f"{r.Open:.2f}", f"{r.High:.2f}", f"{r.Low:.2f}", f"{r.Close:.2f}", _vol(r.Volume)]
            for ts, r in df.iterrows()]
    geom = {"w": W, "l": L, "bw": bw}
    return (f'<svg class="pc" viewBox="0 0 {W} {H}" role="img" aria-label="{_e(s["symbol"])} daily chart" '
            f"data-bars='{_e(json.dumps(bars))}' data-geom='{json.dumps(geom)}'>{''.join(out)}</svg>")


def rrg_chart(table: pd.DataFrame) -> str:
    """Sector rotation scatter: x = RS-Ratio, y = RS-Momentum, 100/100 crosshair."""
    W, H, P = 520, 340, 34
    t = table.dropna(subset=["rs_ratio", "rs_momentum"])
    if t.empty:
        return ""
    xr = max(abs(t["rs_ratio"] - 100).max(), 1) * 1.25
    yr = max(abs(t["rs_momentum"] - 100).max(), 0.5) * 1.25
    X = lambda v: P + (v - (100 - xr)) / (2 * xr) * (W - 2 * P)
    Y = lambda v: H - P - (v - (100 - yr)) / (2 * yr) * (H - 2 * P)
    o = [f'<rect x="{X(100):.1f}" y="{P}" width="{W - P - X(100):.1f}" height="{Y(100) - P:.1f}" fill="var(--va)"/>',
         f'<rect x="{P}" y="{P}" width="{X(100) - P:.1f}" height="{Y(100) - P:.1f}" fill="var(--va)" opacity=".5"/>',
         f'<line x1="{P}" x2="{W - P}" y1="{Y(100):.1f}" y2="{Y(100):.1f}" stroke="var(--border)"/>',
         f'<line y1="{P}" y2="{H - P}" x1="{X(100):.1f}" x2="{X(100):.1f}" stroke="var(--border)"/>',
         f'<text x="{W - P - 4}" y="{P + 14}" text-anchor="end">Leading</text>',
         f'<text x="{P + 4}" y="{P + 14}">Improving</text>',
         f'<text x="{P + 4}" y="{H - P - 6}">Lagging</text>',
         f'<text x="{W - P - 4}" y="{H - P - 6}" text-anchor="end">Weakening</text>',
         f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle">RS-Ratio →</text>',
         f'<text x="12" y="{H / 2}" transform="rotate(-90 12 {H / 2})" text-anchor="middle">RS-Momentum →</text>']
    for _, r in t.iterrows():
        cx, cy = X(r.rs_ratio), Y(r.rs_momentum)
        tipt = _e(f"<b>{r.etf}</b> {r.sector}<br>RS-Ratio {r.rs_ratio:.2f}<br>RS-Mom {r.rs_momentum:.2f}<br>{r.quadrant}")
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="transparent" data-t="{tipt}"/>')
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="var(--s1)" stroke="var(--card)" stroke-width="2" pointer-events="none"/>')
        o.append(f'<text x="{cx + 8:.1f}" y="{cy + 4:.1f}" style="fill:var(--text)">{_e(r.etf)}</text>')
    return f'<svg viewBox="0 0 {W} {H}" data-pts="1" role="img" aria-label="Sector rotation map">{"".join(o)}</svg>'


# --------------------------------------------------------------------------- sections

def setup_card(s: dict, daily: pd.DataFrame | None) -> str:
    f = s.get("fundamentals") or {}
    q = f.get("quality") or {}
    risk = s["close"] - s["stop"]
    head = (f'<div class="row"><h2 style="margin:0">{_e(s["symbol"])}</h2><span class="sub">{_e(f.get("name") or "")}</span>'
            f'<span class="chip">{_e(s.get("sector") or "?")} · {_e(s.get("sector_etf") or "")} {_e(s.get("quadrant") or "")}</span>'
            f'<span class="chip">score {s["score"]:.0f}</span></div>')
    levels = (
        '<div class="levels">'
        f'<div class="lv"><small>Buy zone (15m trigger)</small><b>≤ {max_entry(s):,.2f}</b><small>close {s["close"]:,.2f} · ATR {s["atr"]:.2f}</small></div>'
        f'<div class="lv"><small>Stop · {_e(s["stop_src"])}</small><b class="bad">{s["stop"]:,.2f}</b><small>−{risk / s["close"] * 100:.1f}% · {risk / s["atr"]:.1f} ATR</small></div>'
        f'<div class="lv"><small>TP1 · {_e(s["tp1_src"])}</small><b class="ok">{s["tp1"]:,.2f}</b><small>+{(s["tp1"] / s["close"] - 1) * 100:.1f}% · R:R {s["rr"]:.2f} at close</small></div>'
        f'<div class="lv"><small>TP2 · {_e(s["tp2_src"])}</small><b class="ok">{s["tp2"]:,.2f}</b><small>+{(s["tp2"] / s["close"] - 1) * 100:.1f}%</small></div>'
        '</div>')
    trend = (f'<div class="kv"><b>Trend</b><div><span>LRC R²</span><span>{_f(s.get("lrc_r2"))}</span></div>'
             f'<div><span>Slope</span><span>{_f(s.get("lrc_slope_pct_day"), 3, suffix="%/day")}</span></div>'
             f'<div><span>Channel position</span><span>{_f(s.get("lrc_pos_pct"), 0, suffix="%")}</span></div>'
             f'<div><span>ADX</span><span>{_f(s.get("adx"), 1)}</span></div>'
             f'<div><span>RS vs SPY (3m)</span><span>{_f(s.get("rs_spy"), 1, pct=True)}</span></div></div>')
    flow = (f'<div class="kv"><b>Institutional flow ({s.get("flow_n")}/3)</b>'
            f'<div><span>Anchored VWAP</span><span>{_f(s.get("avwap"))}</span></div>'
            f'<div><span>Up/Down volume</span><span>{_f(s.get("udv_ratio"))}</span></div>'
            f'<div><span>CMF(20)</span><span>{_f(s.get("cmf"), 3)}</span></div>'
            f'<div><span>OBV trend</span><span>{_f(s.get("obv_trend"))}</span></div>'
            f'<div><span>POC / VAH / VAL</span><span>{_f(s.get("poc"))} / {_f(s.get("vah"))} / {_f(s.get("val"))}</span></div></div>')
    fund = (f'<div class="kv"><b>Fundamentals</b>'
            f'<div><span>Rating</span><span>{_e((f.get("recommendation") or "–").replace("_", " "))} ({f.get("analysts", "–")} an.)</span></div>'
            f'<div><span>Beta · PEG</span><span>{_f(f.get("beta"))} · {_f(f.get("peg"))}</span></div>'
            f'<div><span>Fwd P/E · Target</span><span>{_f(f.get("fwd_pe"), 1)} · {_f(f.get("target_mean"))}</span></div>'
            f'<div><span>Quality</span><span>{f.get("quality_n", "–")}/5 {" ".join("✓" if v else "✗" for v in q.values())}</span></div>'
            f'<div><span>Next earnings</span><span>{_e(f.get("next_earnings") or "unknown")}</span></div></div>')
    chart = ""
    if daily is not None and len(daily) > 30:
        chart = ('<div class="legend"><span><i style="border-color:var(--s1)"></i>Regression channel</span>'
                 '<span><i style="border-color:var(--s2)"></i>Anchored VWAP</span>'
                 '<span><i style="border-color:var(--s3)"></i>POC (shaded = value area)</span>'
                 '<span><i style="border-color:var(--crit)"></i>Stop</span><span><i style="border-color:var(--good)"></i>Targets</span></div>'
                 f'<div class="scroll">{price_chart(daily, s, CFG["CHART_DAYS"])}</div>')
    entry = ('<p class="sub" style="margin:8px 0 0">15m entry: green bar closing back above session VWAP and EMA9, '
             f'relative volume ≥ {CFG["ENTRY_MIN_RVOL"]}, not above VWAP+{CFG["ENTRY_MAX_VWAP_SIGMA"]}σ, '
             f'R:R ≥ {CFG["MIN_RR"]} at the entry price → only below <b>{max_entry(s):,.2f}</b>. '
             f'Valid {CFG["SETUP_VALID_SESSIONS"]} sessions; void if price trades ≤ stop first.</p>')
    return f'<section class="card">{head}{levels}{chart}<div class="grid">{trend}{flow}{fund}</div>{entry}</section>'


def build_html(title: str, meta: dict, setups: list[dict], sectors: pd.DataFrame, near: pd.DataFrame,
               daily: dict[str, pd.DataFrame], funnel: dict) -> str:
    regime = meta.get("market_ok")
    reg = (f'<span class="{"ok" if regime else "bad"}">{"SPY above SMA200 – longs allowed" if regime else "SPY below SMA200 – no new longs"}</span>')
    fun = " → ".join(f"{_e(k)} <b>{v}</b>" for k, v in funnel.items())
    sec_rows = "".join(
        f'<tr><td>{_e(r.etf)}</td><td>{_e(r.sector)}</td><td class="n">{r.rs_ratio:.2f}</td><td class="n">{r.rs_momentum:.2f}</td>'
        f'<td class="{"ok" if r.quadrant in CFG["ALLOWED_QUADRANTS"] else ""}">{_e(r.quadrant)}</td></tr>' for r in sectors.itertuples())
    summary = "".join(
        f'<tr><td><b>{_e(s["symbol"])}</b></td><td>{_e(s.get("sector_etf") or "")}</td><td class="n">{s["close"]:,.2f}</td>'
        f'<td class="n"><b>{max_entry(s):,.2f}</b></td><td class="n">{s["stop"]:,.2f}</td><td class="n">{s["tp1"]:,.2f}</td><td class="n">{s["tp2"]:,.2f}</td>'
        f'<td class="n">{s["rr"]:.2f}</td><td class="n">{s["score"]:.0f}</td></tr>' for s in setups)
    near_html = ""
    if near is not None and not near.empty:
        near_html = ('<h2>Near misses (one gate short)</h2><div class="card scroll"><table><tr><th>Ticker</th><th>Sector</th><th>Failed</th></tr>'
                     + "".join(f'<tr><td>{_e(r.symbol)}</td><td>{_e(r.sector or "")}</td><td>{_e(r.why)}</td></tr>' for r in near.itertuples())
                     + '</table></div>')
    cards = "".join(setup_card(s, daily.get(s["symbol"])) for s in setups) or '<div class="card sub">No setups today.</div>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{CSS}</style></head><body><main>
<h1>{_e(title)}</h1><div class="sub">{_e(meta.get("generated", ""))} · universe {_e(meta.get("universe", ""))} · data through {_e(meta.get("asof", ""))}</div>
<div class="card"><div>{reg}</div><div class="sub" style="margin-top:6px">{fun}</div></div>
<h2>Setups ({len(setups)})</h2>
<div class="card scroll"><table><tr><th>Ticker</th><th>Sector</th><th class="n">Close</th><th class="n">Buy ≤</th><th class="n">Stop</th><th class="n">TP1</th><th class="n">TP2</th><th class="n">R:R</th><th class="n">Score</th></tr>{summary}</table></div>
{cards}
<h2>Sector rotation</h2>
<div class="card"><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))"><div>{rrg_chart(sectors)}</div>
<div class="scroll"><table><tr><th>ETF</th><th>Sector</th><th class="n">RS-Ratio</th><th class="n">RS-Mom</th><th>Quadrant</th></tr>{sec_rows}</table></div></div></div>
{near_html}
<details class="card"><summary>Rules</summary><p class="sub">Market: SPY &gt; SMA{CFG["MARKET_SMA"]}. Sector ETF in {", ".join(CFG["ALLOWED_QUADRANTS"])}.
Fundamentals: rating in {", ".join(CFG["ALLOWED_RECOMMENDATIONS"])} (≥{CFG["MIN_ANALYSTS"]} analysts), beta ≤ {CFG["MAX_BETA"]}, {CFG["MIN_PEG"]} &lt; PEG &lt; {CFG["MAX_PEG"]},
quality ≥ {CFG["MIN_QUALITY_CHECKS"]}/5, no earnings within {CFG["EARNINGS_BLACKOUT_DAYS"]} days.
Trend: {CFG["LRC_LENGTH"]}-day regression slope &gt; 0{f', R² ≥ {CFG["LRC_MIN_R2"]}' if CFG["LRC_MIN_R2"] > 0 else ''}, channel position ≤ {CFG["LRC_MAX_POSITION_PCT"]}%{f', EMA{CFG["EMA_FAST"]} &gt; EMA{CFG["EMA_SLOW"]}' if CFG["REQUIRE_EMA_STACK"] else ''}{f', ADX ≥ {CFG["ADX_MIN"]}, +DI &gt; −DI' if CFG["REQUIRE_ADX"] else ''}.
Flow: close &gt; anchored VWAP and ≥ {CFG["MIN_FLOW_CHECKS"]}/3 of up/down volume &gt; 1, CMF &gt; 0, OBV rising.
Levels: stop = nearest structural support ≥ {CFG["MIN_STOP_ATR"]} ATR below − {CFG["STOP_ATR_CUSHION"]} ATR (max {CFG["MAX_STOP_ATR"]} ATR); TP1 = nearest resistance ≥ {CFG["MIN_TARGET_ATR"]} ATR above; TP2 = next resistance ≥ {CFG["TP2_MIN_GAP_ATR"]} ATR beyond TP1.
Buy zone = highest price giving R:R ≥ {CFG["MIN_RR"]} to TP1; it must lie within {CFG["SETUP_MAX_PULLBACK_ATR"]} ATR of the close.
Management: sell {int(CFG["TP1_FRACTION"] * 100)}% at TP1 and move stop to breakeven; rest at TP2, on a daily close below the channel's lower band, or after {CFG["MAX_HOLD_SESSIONS"]} sessions.</p></details>
</main><div class="tip"></div><script>{JS}</script></body></html>"""
