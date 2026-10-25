"""按牛/熊/震荡拆解已有回测结果"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_index_hist, load_config

cfg = load_config(ROOT / "config.yaml")
nav = pd.read_csv(ROOT / "output" / "nav.csv", parse_dates=["date"]).set_index("date")
bench = fetch_index_hist(cfg["backtest"]["benchmark"], cfg["data"]["start_date"], None, ROOT / "data")
b = bench["close"].reindex(nav.index).ffill()
common = nav.index[b.notna()]
s_ret = nav.loc[common, "nav"].pct_change().fillna(0)
b_ret = b.loc[common].pct_change().fillna(0)

# 市况：用基准 60 日收益 + 波动
b60 = b.loc[common].pct_change(60)
vol20 = b_ret.rolling(20).std() * np.sqrt(252)

regime = pd.Series("震荡", index=common)
regime[b60 > 0.12] = "牛市"
regime[b60 < -0.12] = "熊市"

print("=== 市况分布（交易日） ===")
print(regime.value_counts())

print("\n=== 分市况：策略 vs 沪深300 ===")
rows = []
for name in ["牛市", "熊市", "震荡"]:
    m = regime == name
    if m.sum() < 20:
        continue
    sr = s_ret[m]
    br = b_ret[m]
    # 期间累计
    s_cum = (1 + sr).prod() - 1
    b_cum = (1 + br).prod() - 1
    # 年化波动
    s_vol = sr.std() * np.sqrt(252)
    # 胜率
    win = (sr > 0).mean()
    rows.append(
        {
            "市况": name,
            "天数": int(m.sum()),
            "策略累计": s_cum,
            "基准累计": b_cum,
            "超额": s_cum - b_cum,
            "日均超额": float((sr - br).mean()),
            "策略日胜率": win,
            "策略年化波动": s_vol,
            "策略日均收益": float(sr.mean()),
        }
    )

df = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
pd.set_option("display.width", 120)
print(df.to_string(index=False))

print("\n=== 分年度 + 该年基准 ===")
nav2 = nav.loc[common].copy()
nav2["b"] = b.loc[common]
nav2["regime"] = regime
for y, g in nav2.groupby(nav2.index.year):
    s = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
    bb = g["b"].iloc[-1] / g["b"].iloc[0] - 1
    rdom = g["regime"].mode().iloc[0]
    print(f"{y}  策略 {s:+.2%}  基准 {bb:+.2%}  超额 {s-bb:+.2%}  主导市况 {rdom}  最大回撤 {(1-g['nav']/g['nav'].cummax()).max():.2%}")

print("\n=== 关键提醒 ===")
print("回撤风险主要发生在熊市与震荡；牛市容易赚钱但也会被回撤风控减仓。")
