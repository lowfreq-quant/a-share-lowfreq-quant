"""熊市组合审计：修正 512890 拆分 + 估算分红漏计影响"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_index_hist, load_config
from src.etf_portfolio import backtest_etf_portfolio, load_etf_map
from src.regime import classify_regime

cfg = load_config(ROOT / "config_bear.yaml")
etf_cfg = {str(k): str(v) for k, v in cfg["etf"].items()}
weights = {str(k): float(v) for k, v in cfg["weights"].items()}
px = load_etf_map(etf_cfg, cfg["data"]["start_date"], None, ROOT / "data")
bench = fetch_index_hist("000300", cfg["data"]["start_date"], None, ROOT / "data")["close"]
regime = classify_regime(bench)["regime"]


def stats(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0]
    cagr = total ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    return dict(total=total - 1, cagr=cagr, vol=vol, sharpe=cagr / vol, mdd=(1 - nav / nav.cummax()).max())


# 1) 原样
nav0 = backtest_etf_portfolio(px, weights, cfg, regime)["nav"]
print("reported:", {k: round(v, 4) for k, v in stats(nav0).items()})

# 2) 修正 512890 拆分
px2 = {k: v.copy() for k, v in px.items()}
d = px2["512890"]
split_date = pd.Timestamp("2021-10-25")
ratio = float(d.close.loc["2021-10-21"]) / float(d.close.loc[split_date])  # 2.046
d.loc[d.index < split_date, "close"] *= 1 / ratio
print(f"512890 split ratio {ratio:.4f}; adjusted pre-split prices")
nav1 = backtest_etf_portfolio(px2, weights, cfg, regime)["nav"]
print("split-adjusted:", {k: round(v, 4) for k, v in stats(nav1).items()})

# 3) 分红漏计估算：找 510880 每年最大单日跌幅
e = px["510880"].close
r = e.pct_change()
print("\n510880 yearly largest drops (likely dividends):")
for y, g in r.groupby(r.index.year):
    print(" ", y, [(str(i.date()), round(float(v), 4)) for i, v in g.nsmallest(2).items()])

e2 = px["511260"].close
r2 = e2.pct_change()
print("\n511260 yearly largest drops:")
for y, g in r2.groupby(r2.index.year):
    print(" ", y, [(str(i.date()), round(float(v), 4)) for i, v in g.nsmallest(2).items()])

# 分年（修正后）
nav1 = nav1.to_frame("nav")
print("\nsplit-adjusted yearly:")
for y, g in nav1.groupby(nav1.index.year):
    print(" ", y, round(float(g.nav.iloc[-1] / g.nav.iloc[0] - 1), 4))
