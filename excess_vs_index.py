"""excess_vs_index.py -- pass/fail test for any strategy: trade return minus the index return over the same holding window.
Usage: python3 excess_vs_index.py <daily-dir with SPY> <results/trades csv> [...]   (csv needs entry_time, exit_time, pct_return)"""
import os, sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from mds_backtest_daily import load
daily, spy, etf, sec = load(sys.argv[1])
sc = spy['Close']; pc = sc.shift(1)
for f in sys.argv[2:]:
    t=pd.read_csv(f if os.path.exists(f) else 'results/'+f)
    ed=pd.to_datetime(t.entry_time).dt.normalize(); xd=pd.to_datetime(t.exit_time).dt.normalize()
    t['idx']=sc.reindex(xd).values/pc.reindex(ed).values-1
    t['ex']=t.pct_return-t['idx']; t['yr']=ed.dt.year
    print(f"{f}: n={len(t)} stock {t.pct_return.mean()*100:+.3f}%  index {t['idx'].mean()*100:+.3f}%  EXCESS {t.ex.mean()*100:+.3f}% per trade (t={t.ex.mean()/t.ex.std()*len(t)**.5:.1f}), win vs index {(t.ex>0).mean():.0%}")
    g=t.groupby('yr')['ex'].agg(['size','mean']); print('   excess by year: '+' '.join(f"{y}:{m*100:+.2f}%({n})" for y,(n,m) in g.iterrows()))
