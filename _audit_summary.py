import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
from src.data import fetch_index_hist
from src.regime import classify_regime

bench = fetch_index_hist('000300', '2021-01-01', None, 'data')['close']
print('CSI300 yearly:')
for y, g in bench.groupby(bench.index.year):
    print(' ', y, round(float(g.iloc[-1] / g.iloc[0] - 1), 4))
print('CSI300 total 2021-09/24:', round(float(bench.iloc[-1] / bench.iloc[0] - 1), 4))

for name in ['E0', 'E2']:
    df = pd.read_csv(f'output_audit/nav_{name}.csv', parse_dates=['date']).set_index('date')
    nav = df['nav']
    print(f'\n{name} total {nav.iloc[-1]-1:.4f}')
    for a, b in [('2021-07-01', '2023-12-31'), ('2024-01-01', '2026-09-24')]:
        s = nav.loc[a:b]
        print(f'  {a}~{b}: {s.iloc[-1]/s.iloc[0]-1:+.2%}')
    r = nav.pct_change().dropna()
    years = len(r) / 252
    sharpe = (nav.iloc[-1] ** (1 / years) - 1) / (r.std() * np.sqrt(252))
    print(f'  Sharpe {sharpe:.3f}  t≈{sharpe*np.sqrt(years):.2f}  years {years:.2f}')
    print(f'  in-market days {(df["inv"]>0).sum()} / {len(df)} = {(df["inv"]>0).mean():.1%}')

# E2 持仓日 vs 指数同期
df = pd.read_csv('output_audit/nav_E2.csv', parse_dates=['date']).set_index('date')
inm = df['inv'] > 0
r = df['nav'].pct_change().fillna(0)
ir = bench.reindex(df.index).ffill().pct_change().fillna(0)
port = (1 + r[inm]).prod() - 1
idx = (1 + ir[inm]).prod() - 1
print(f'\nE2 持仓日 {(inm).sum()} 天: 组合 {port:+.2%}  指数同期 {idx:+.2%}  差 {port-idx:+.2%}')
