import pandas as pd, sys
sys.path.insert(0, '.')
from src.regime import classify_regime

idx = pd.read_parquet('data/index_000300_20210101_today.parquet')
rg = classify_regime(idx['close'])
print('overall:', {k: round(v, 3) for k, v in rg['regime'].value_counts(normalize=True).items()})
print(rg.groupby([rg.index.year, 'regime']).size().unstack(fill_value=0).to_string())

# month-end signal days and next-day exec labels
dates = rg.index
sig = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).tail(1)
for s in sig:
    nxt = dates[dates > s]
    e = nxt[0] if len(nxt) else None
    print(s.date(), '->', e.date() if e is not None else None,
          'sig_regime=', rg.loc[s, 'regime'],
          'exec_regime=', rg.loc[e, 'regime'] if e is not None else None)
