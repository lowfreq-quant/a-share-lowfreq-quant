"""熊市融券做空测试：只在 regime=熊市 时做空沪深300

成本假设：
- 融券费率 6% / 8% / 10% 年化
- 分红补偿 2% 年化（做空要还给券商分红）
- 每次多空切换 0.05% 交易成本
- 信号用前一日收盘的市况，次日生效（无前视）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.regime import classify_regime

idx = pd.read_parquet(ROOT / "data" / "index_000300_live.parquet")["close"].astype(float)
reg = classify_regime(idx)["regime"]
ret = idx.pct_change().fillna(0.0)
pos = (reg.shift(1) == "熊市").astype(float) * -1.0  # 熊市做空 100%，其余空仓


def run(borrow: float, div: float = 0.02, switch: float = 0.0005) -> pd.Series:
    cost = (borrow + div) / 252.0 * pos.abs()
    trade = pos.diff().abs().fillna(0.0) * switch
    r = pos * ret - cost - trade
    return (1 + r).cumprod()


def stats(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    yrs = len(r) / 252
    total = nav.iloc[-1] / nav.iloc[0]
    cagr = total ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    return dict(total=total - 1, cagr=cagr, vol=vol, sharpe=cagr / vol if vol else np.nan,
                mdd=(1 - nav / nav.cummax()).max(), years=yrs)


for start in ["2005-01-01", "2021-01-01"]:
    s = idx.loc[start:]
    print(f"\n=== 样本 {s.index.min().date()} → {s.index.max().date()} ===")
    print(f"沪深300 同期: {s.iloc[-1] / s.iloc[0] - 1:+.2%}")
    for borrow in [0.06, 0.08, 0.10]:
        nav = run(borrow).loc[start:]
        st = stats(nav)
        print(f"  融券 {borrow:.0%}: 总收益 {st['total']:+.1%}  年化 {st['cagr']:+.1%}  "
              f"波动 {st['vol']:.1%}  夏普 {st['sharpe']:.2f}  回撤 {st['mdd']:.1%}")
        # 分年
        yr = {y: float(g.iloc[-1] / g.iloc[0] - 1) for y, g in nav.groupby(nav.index.year) if len(g) > 1}
        print("    分年:", {y: f"{v:+.0%}" for y, v in yr.items()})

# 做空只在「熊市 + 指数低于 MA120」时
ma120 = idx.rolling(120).mean()
pos2 = ((reg.shift(1) == "熊市") & (idx.shift(1) < ma120.shift(1))).astype(float) * -1.0
print("\n=== 加强过滤：熊市 + 收盘<MA120 ===")
for borrow in [0.08]:
    cost = (borrow + 0.02) / 252 * pos2.abs()
    trade = pos2.diff().abs().fillna(0) * 0.0005
    nav = (1 + pos2 * ret - cost - trade).cumprod().loc["2005-01-01":]
    st = stats(nav)
    print(f"  融券 {borrow:.0%}: 总收益 {st['total']:+.1%}  年化 {st['cagr']:+.1%}  "
          f"夏普 {st['sharpe']:.2f}  回撤 {st['mdd']:.1%}")
