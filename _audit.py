import pandas as pd, numpy as np

nav = pd.read_csv('output/nav.csv', parse_dates=['date']).set_index('date')
print(nav.head(3))
print(nav.tail(3))
print('rows', len(nav), 'range', nav.index.min().date(), nav.index.max().date())
r = nav['nav'].pct_change().dropna()
print('exact zero ret days %', round((r == 0).mean()*100, 1))
print('pos ret days %', round((r > 0).mean()*100, 2))
print('neg ret days %', round((r < 0).mean()*100, 2))
print('mean invested', round(nav['invested'].mean(), 3))
print('\nby year:')
for y, d in nav.groupby(nav.index.year):
    inv = (d['invested'] > 0).mean()
    ret = d['nav'].iloc[-1] / d['nav'].iloc[0] - 1 if len(d) > 1 else np.nan
    print(y, 'days', len(d), 'inv%', round(inv*100, 1), 'ret', round(ret, 4), 'maxdd', round(d['drawdown'].max(), 4), 'n_pos_avg', round(d['n_pos'].mean(),1))

# yearly in-market days count
print('\nmonths with positions:')
pos = nav[nav['invested'] > 0]
print(pos.groupby([pos.index.year, pos.index.month]).size().to_string())
