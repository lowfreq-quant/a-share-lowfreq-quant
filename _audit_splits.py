import pandas as pd, numpy as np
from pathlib import Path

files = sorted(Path('data/hist').glob('*.parquet'))
rows = []
for f in files:
    code = f.name.split('_')[0]
    try:
        h = pd.read_parquet(f, columns=['close', 'outstanding_share'])
    except Exception:
        continue
    if len(h) < 30:
        continue
    r = h['close'].pct_change()
    ds = h['outstanding_share'].pct_change()
    # 疑似送转/拆股：价格大跌 + 股本不变或增加
    mask = (r < -0.15) & (ds > 0.05)
    if mask.any():
        for d in h.index[mask]:
            rows.append((code, d, float(r.loc[d]), float(ds.loc[d])))
    # 反向：股本大减（缩股）
    mask2 = (r > 0.15) & (ds < -0.05)
    if mask2.any():
        for d in h.index[mask2]:
            rows.append((code, d, float(r.loc[d]), float(ds.loc[d])))

df = pd.DataFrame(rows, columns=['code', 'date', 'ret', 'dshare'])
print('suspected split/shrink events:', len(df), 'stocks:', df['code'].nunique())
print(df.groupby(df['date'].dt.year).size().to_string())
print('\nsample:')
print(df.head(20).to_string())
df.to_csv('output_audit/split_events.csv', index=False, encoding='utf-8-sig')

# 极端日收益（可能是数据错误）
print('\n--- extreme daily returns in cached data ---')
cnt = 0
for f in files[:2000]:
    code = f.name.split('_')[0]
    try:
        h = pd.read_parquet(f, columns=['close'])
    except Exception:
        continue
    r = h['close'].pct_change().dropna()
    if len(r) and (r.abs() > 0.21).sum() > 0:
        cnt += 1
print('files with |ret|>21% (first 2000):', cnt)
