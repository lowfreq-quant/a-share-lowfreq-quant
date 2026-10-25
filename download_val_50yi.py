"""对市值>50亿子集下载 PE/PB，供估值因子实验"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import download_valuation_map, fetch_stock_list, load_config

cfg = load_config(ROOT / "config_choppy.yaml")
cache = ROOT / "data"
names = fetch_stock_list(cache)
codes = names["code"].tolist()
end_tag = "20210101_today"
hist = cache / "hist"

# 用最近一日 total_mv 粗筛 >50亿
todo = []
for c in codes:
    p = hist / f"{c}_{end_tag}.parquet"
    if not p.exists() and not p.with_suffix(".pkl").exists():
        continue
    try:
        df = pd.read_parquet(p) if p.exists() else pd.read_pickle(p.with_suffix(".pkl"))
    except Exception:  # noqa: BLE001
        continue
    if df.empty or "total_mv" not in df.columns:
        continue
    mv = df["total_mv"].dropna()
    if len(mv) == 0:
        continue
    if float(mv.iloc[-1]) >= 5e9:
        todo.append(c)

print(f"市值>50亿: {len(todo)}")
val_dir = cache / "valuation"
val_dir.mkdir(parents=True, exist_ok=True)
have = sum(1 for c in todo if (val_dir / f"{c}.parquet").exists())
print(f"已有估值缓存 {have}/{len(todo)}")

need = [c for c in todo if not (val_dir / f"{c}.parquet").exists()]
print(f"待下载 {len(need)}")
if need:
    download_valuation_map(need, cache, sleep=0.08, progress_every=50)
print("done")
