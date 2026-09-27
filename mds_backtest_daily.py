"""
mds_backtest_daily.py
=====================
Daily-bar backtest of the MDS setup -- no 15m data needed, so it can run on
years of daily bars (e.g. the Alpaca SIP cache from the TSM project).

Same rules as mds_strategy (features, sector rotation, levels, exits); the 15m
trigger is replaced by a LIMIT order at max_entry (filled at the open if the
stock opens below it). Conservative: a day that touches both a fill/target and
the stop is resolved against us. Fundamentals are NOT applied (no point-in-time
history for free).

If SPY / sector ETFs are missing from the daily folder, stand-ins are built:
SPY = equal-weight index of the universe, each sector ETF = equal-weight basket
of its names (GICS from the S&P constituents file; names no longer in the index
get the basket they correlate with most).

Usage:
    python3 mds_backtest_daily.py                                  # cache/tsm_daily
    python3 mds_backtest_daily.py --daily-dir cache/mds/backtest/1d
    python3 mds_backtest_daily.py --mode no_filters                # control: levels+exits only
    python3 mds_backtest_daily.py --quad Leading,Improving --set LRC_MIN_R2=0.5 REQUIRE_EMA_STACK=1
"""
import sys, os, json, time, argparse
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import mds_data as D, mds_strategy as S
from mds_config import MDS_CONFIG as CFG0
from intraday_strategy import compute_volume_profile


ETF_BY_GICS = {'Information Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV', 'Consumer Discretionary': 'XLY',
               'Consumer Staples': 'XLP', 'Energy': 'XLE', 'Industrials': 'XLI', 'Materials': 'XLB', 'Real Estate': 'XLRE',
               'Utilities': 'XLU', 'Communication Services': 'XLC'}


def _ew_index(frames):
    r = pd.concat({k: v['Close'].pct_change() for k, v in frames.items()}, axis=1).mean(axis=1).fillna(0)
    c = 100 * (1 + r).cumprod()
    return pd.DataFrame({'Open': c, 'High': c, 'Low': c, 'Close': c, 'Volume': 1e6})


def load(daily_dir):
    daily = {}
    for fn in sorted(os.listdir(daily_dir)):
        if fn.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(daily_dir, fn))
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = D.to_daily_index(df)
            if df is not None and len(df) > 300:
                daily[fn[:-8]] = df
    spy = daily.pop('SPY', None)
    etf = {e: daily.pop(e) for e in list(ETF_BY_GICS.values()) if e in daily}
    try:
        from survivorship_free_universe import _load_current_sector_map
        gics = {k.replace('.', '-'): v for k, v in _load_current_sector_map().items()}
    except Exception:
        gics = {}
    sectors = {s: ETF_BY_GICS.get(gics.get(s)) for s in daily}
    if spy is None:
        print('SPY not in folder -> equal-weight stand-in')
        spy = _ew_index(daily)
    if not etf:
        print('sector ETFs not in folder -> equal-weight GICS baskets')
        groups = {}
        for s, e in sectors.items():
            if e:
                groups.setdefault(e, {})[s] = daily[s]
        etf = {e: _ew_index(g) for e, g in groups.items() if len(g) >= 10}
    rets = {e: etf[e]['Close'].pct_change() for e in etf}
    for s, e in sectors.items():
        if e not in etf:                      # no longer in the index / unknown -> most correlated sector
            r = daily[s]['Close'].pct_change()
            cors = {k: r.corr(v.reindex(r.index)) for k, v in rets.items()}
            sectors[s] = max(cors, key=lambda k: cors[k] if np.isfinite(cors[k]) else -9)
    return daily, spy, etf, sectors


def sim_symbol(args):
    sym, df, spy, rot, etfsym, cfg, mode = args
    trades, stats = [], {'cand': 0, 'setups': 0, 'filled': 0}
    f = S.daily_features(df, spy['Close'], cfg)
    mkt = S.market_regime(spy, cfg).reindex(f.index).fillna(False)
    idx = f.index
    O, H, L, Cl = (df[c].to_numpy(float) for c in ['Open', 'High', 'Low', 'Close'])
    lower = f['lrc_lower'].to_numpy(float)
    mom = f['tsm_mom'].to_numpy(float)
    if mode == 'full':
        ok = f['tech_ok'] & mkt
    elif mode == 'no_filters':          # control: same levels & exits, no trend/flow filter
        ok = mkt & f['atr'].notna() & f['lrc_r2'].notna()
    elif mode == 'trend_only':
        ok = f['trend_ok'] & f['position_ok'] & mkt
    ok = ok.to_numpy(bool)
    start = 260
    i = start; busy_until = -1
    active = None; age = 0
    while i < len(idx):
        # 1) try to fill the active setup today
        if active is not None and i > busy_until:
            age += 1
            s = active
            nxt_open = cfg.get('XP_ENTRY', 'limit') == 'next_open'
            if O[i] > s['close'] + cfg['MAX_GAP_ATR'] * s['atr']:
                fill = None
            elif nxt_open and O[i] > s['stop']:
                fill = O[i]
            elif O[i] <= s['max_entry'] and O[i] > s['stop']:
                fill = O[i]
            elif L[i] <= s['max_entry'] and O[i] > s['max_entry']:
                fill = s['max_entry']
            else:
                fill = None
            if O[i] <= s['stop'] or (fill is None and L[i] <= s['stop']):
                active = None                     # structure broke before we got in
            elif fill is not None and (nxt_open or (fill - s['stop'] >= cfg['MIN_ENTRY_RISK_ATR'] * s['atr'] and (s['tp1'] - fill) / (fill - s['stop']) >= cfg['MIN_RR'] - 1e-9)):
                stats['filled'] += 1
                tr = manage(s, fill, i, O, H, L, Cl, lower, cfg, idx, mom)
                tr.update(symbol=sym, sector_etf=etfsym, quadrant=s['quadrant'], score=s['score'], setup_date=s['date'])
                trades.append(tr); busy_until = tr['_exit_i']; active = None
            if active is not None and age >= cfg['SETUP_VALID_SESSIONS']:
                active = None
        # 2) new setup at today's close
        if ok[i] and i > busy_until:
            stats['cand'] += 1
            row = f.iloc[i]
            if mode == 'full':
                sok, quad = S.sector_ok(rot, etfsym, idx[i], cfg)
            else:
                sok, quad = True, S.sector_ok(rot, etfsym, idx[i], cfg)[1]
            if sok:
                vp = compute_volume_profile(df.iloc[i - cfg['VP_SESSIONS'] + 1:i + 1], bins=cfg['VP_BINS'])
                lv = S.build_levels(df.iloc[:i + 1], row, vp, cfg)
                if lv is not None:
                    a = float(row['atr'])
                    good = lv['rr'] >= cfg['MIN_RR'] or lv['max_entry'] >= row['close'] - cfg['SETUP_MAX_PULLBACK_ATR'] * a
                    if cfg.get('XP_ENTRY', 'limit') == 'next_open':
                        good = True
                    if good and (cfg.get('XP_ENTRY', 'limit') == 'next_open' or lv['max_entry'] - lv['stop'] >= cfg['MIN_ENTRY_RISK_ATR'] * a):
                        stats['setups'] += 1
                        st = S.Setup(symbol=sym, date=str(idx[i].date()), close=float(row['close']), atr=a, stop=lv['stop'],
                                     stop_src=lv['stop_src'], tp1=lv['tp1'], tp1_src=lv['tp1_src'], tp2=lv['tp2'], tp2_src=lv['tp2_src'],
                                     rr=lv['rr'], max_entry=lv['max_entry'], lrc_r2=row['lrc_r2'], flow_n=int(row['flow_n']),
                                     rs_spy=row.get('rs_spy'), quadrant=quad)
                        d = st.to_dict(); d['score'] = S.rank_score(st)
                        d.update(pos=float(row['lrc_pos_pct']), slope=float(row['lrc_slope_pct_day']), r2=float(row['lrc_r2']),
                                 adx=float(row['adx']), cmf=float(row['cmf']), udv=float(row['udv_ratio']), obv=float(row['obv_trend']),
                                 above_avwap=bool(row['close'] > row['avwap']), ema_ok=bool(row['ema_fast'] > row['ema_slow']),
                                 rs=float(row.get('rs_spy', np.nan)), trend_ok=bool(row['trend_ok']), flow_ok=bool(row['flow_ok']))
                        active, age = d, 0
        i += 1
    return sym, trades, stats

def manage(s, entry, i0, O, H, L, Cl, lower, cfg, idx, mom=None):
    stop0 = s['stop']
    if cfg.get('XP_STOP_ATR', 0) > 0:                      # experiment: fixed ATR stop instead of structure
        stop0 = entry - cfg['XP_STOP_ATR'] * s['atr']
    stop, rem, tp1_hit, fills = stop0, 1.0, False, []
    risk = entry - stop0
    use_tp = cfg.get('XP_TP', 1) in (1, 2)
    tp2_on = cfg.get('XP_TP', 1) == 1                      # XP_TP=2: partial at TP1, then trail only
    trail = cfg.get('XP_TRAIL_ATR', 0)
    hi_close = entry
    # fill day: if it also traded at/below the stop, assume stopped (conservative)
    held = 0
    for k in range(i0, len(Cl)):
        first = k == i0
        o = entry if first else O[k]
        if L[k] <= stop:
            fills.append((k, rem, min(o, stop), 'trail' if stop > stop0 + 1e-9 and not tp1_hit else ('breakeven' if tp1_hit else 'stop'))); rem = 0; break
        if not first and use_tp:
            if not tp1_hit and H[k] >= s['tp1']:
                fills.append((k, cfg['TP1_FRACTION'], max(o, s['tp1']), 'tp1')); rem -= cfg['TP1_FRACTION']; tp1_hit = True
                stop = max(stop, entry)
            if tp2_on and tp1_hit and rem > 0 and H[k] >= s['tp2']:
                fills.append((k, rem, max(o, s['tp2']), 'tp2')); rem = 0; break
        held += 1
        if trail > 0:                                      # chandelier: only ratchets up, applied from next day
            hi_close = max(hi_close, Cl[k])
            stop = max(stop, hi_close - trail * s['atr'])
        if cfg.get('XP_TSM_EXIT', 0) and mom is not None and np.isfinite(mom[k]) and mom[k] < 0:
            fills.append((k, rem, Cl[k], 'tsm_off')); rem = 0; break
        if cfg['EXIT_ON_CLOSE_BELOW_LRC_LOWER'] and np.isfinite(lower[k]) and Cl[k] < lower[k]:
            fills.append((k, rem, Cl[k], 'lrc_break')); rem = 0; break
        if held >= cfg['MAX_HOLD_SESSIONS']:
            fills.append((k, rem, Cl[k], 'time')); rem = 0; break
    if rem > 0:
        fills.append((len(Cl) - 1, rem, Cl[-1], 'open_at_end'))
    c = cfg['COST_BPS_PER_SIDE'] / 1e4
    net = sum(f * (p - entry) for _, f, p, _ in fills) - c * (entry + sum(f * p for _, f, p, _ in fills))
    ts = lambda k, hh: idx[k] + pd.Timedelta(hours=hh)
    return {'entry_time': ts(i0, 10), 'entry': entry, 'stop': stop0, 'tp1': s['tp1'], 'tp2': s['tp2'],
            'exit_time': ts(fills[-1][0], 15.9), 'exit_reason': fills[-1][3], '_exit_i': fills[-1][0],
            'fills': [(str(ts(k, 15.5 if r in ('lrc_break', 'time', 'open_at_end') else 12)), f, float(p), r) for k, f, p, r in fills],
            'sessions_held': held, 'r_multiple': net / risk, 'pct_return': net / entry, 'rr_at_trigger': (s['tp1'] - entry) / risk,
            'lrc_r2': s['lrc_r2'], 'flow_n': s['flow_n'], 'risk_atr': risk / s['atr'],
            **{k: s[k] for k in ('pos', 'slope', 'r2', 'adx', 'cmf', 'udv', 'obv', 'above_avwap', 'ema_ok', 'rs', 'trend_ok', 'flow_ok')}, 'stop_src': s['stop_src'], 'tp1_src': s['tp1_src'], 'entry_type': 'limit'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='full', choices=['full', 'no_filters', 'trend_only']); ap.add_argument('--daily-dir', default='cache/tsm_daily'); ap.add_argument('--no-sector', action='store_true')
    ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--tag', default=''); ap.add_argument('--quad', default='')
    a = ap.parse_args()
    cfg = dict(CFG0)
    if a.no_sector: cfg['REQUIRE_SECTOR_ROTATION'] = False
    if a.quad: cfg['ALLOWED_QUADRANTS'] = tuple(a.quad.split(','))
    for kv in a.set:
        k, v = kv.split('=')
        if k not in CFG0:
            cfg[k] = v if not v.replace('.', '', 1).isdigit() else float(v)
        else:
            cfg[k] = (v == '1') if isinstance(CFG0[k], bool) else type(CFG0[k])(float(v))
    t0 = time.time()
    daily, spy, etf, sectors = load(a.daily_dir)
    rot = S.sector_rotation({e: etf[e]['Close'] for e in etf}, spy['Close'], cfg)
    jobs = [(s, df, spy, rot, sectors[s], cfg, a.mode) for s, df in daily.items() if len(df) > 300]
    trades, st = [], {'cand': 0, 'setups': 0, 'filled': 0}
    with ProcessPoolExecutor(4) as ex:
        for sym, tr, s in ex.map(sim_symbol, jobs, chunksize=4):
            trades += tr
            for k in st: st[k] += s[k]
    t = pd.DataFrame(trades)
    import mds_backtest as B
    daily_all = dict(daily); daily_all['SPY'] = spy
    eq, acc = B.simulate_portfolio(t, daily_all, cfg, start=spy.index[260], end=spy.index[-1])
    def summ(x):
        r = x['r_multiple']; w = r[r > 0].sum(); l = -r[r <= 0].sum()
        return f"n={len(x):5d} win={100*(r>0).mean():3.0f}% avgR={r.mean():+.3f} PF={w/l if l else float('nan'):.2f}"
    yr = pd.to_datetime(t['entry_time']).dt.year
    ret = eq['equity'].pct_change().dropna(); yrs = (eq.index[-1]-eq.index[0]).days/365.25
    cagr = (eq['equity'].iloc[-1]/cfg['STARTING_CAPITAL'])**(1/yrs)-1; dd = (eq['equity']/eq['equity'].cummax()-1).min()
    print(f"SUMMARY {a.mode:10s} q={','.join(cfg['ALLOWED_QUADRANTS']):18s} sec={'off' if a.no_sector else 'on '} {' '.join(a.set):45s} | ALL {summ(t)} | IS19-23 {summ(t[yr<=2023])} | OOS24-26 {summ(t[yr>=2024])} | port CAGR={cagr:+.1%} DD={dd:.0%} Sh={ret.mean()/ret.std()*252**.5:.2f} exp={eq['n_positions'].mean():.1f}")

    print(f"mode={a.mode} sector={'off' if a.no_sector else 'on'} {a.set}  symbols={len(jobs)}  {time.time()-t0:.0f}s")
    print('funnel:', st)
    print(B.stats_report(eq, acc, t, spy['Close'], cfg))
    os.makedirs('results', exist_ok=True)
    tag = f"{a.mode}{'_nosec' if a.no_sector else ''}{a.tag}"
    t.drop(columns=['fills']).to_csv(f'results/mds_daily_trades_{tag}.csv', index=False)
    eq.to_csv(f'results/mds_daily_equity_{tag}.csv')
    acc.drop(columns=['fills'], errors='ignore').to_csv(f'results/mds_daily_portfolio_{tag}.csv', index=False)
