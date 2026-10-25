import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
from src.regime import classify_regime
import glob

f = glob.glob('data/index_000300_*_today.parquet')
print(f)
idx = pd.read_parquet('data/index_000300_20210101_today.parquet')
print(idx.tail())
rg = classify_regime(idx['close'])
print('\nregime share overall:', rg['regime'].value_counts(normalize=True).round(3).to_dict())
print('\nby year (trading days):')
print(rg.groupby([rg.index.year, 'regime']).size().unstack(fill_value=0))
print('\n2024 monthly:')
sub = rg.loc['2024']
print(sub[['close','ret60','vol20','regime']].resample('ME').last())
print('\n2026 monthly:')
print(rg.loc['2026'][['close','ret60','vol20','regime']].resample('ME').last())
