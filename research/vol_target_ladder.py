"""风险梯度 + 波动率目标：防守/平衡/进取"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from backtest_etf import load_prices, run_strategy, stats, yearly  # noqa: E402

PX = load_prices(["510880", "511260", "518880", "513100", "510300"], start="2017-08-24")


def vol_target(nav, target, lookback=60, cash=0.015, floor=0.5, cost=0.0005):
    ret = nav.pct_change().fillna(0)
    vol = ret.rolling(lookback).std() * np.sqrt(252)
    dates = nav.index
    me = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).tail(1).index
    exp = pd.Series(1.0, index=dates)
    cur = 1.0
    for dt in dates:
        if dt in me:
            v = vol.loc[dt]
            cur = float(np.clip(target / v, floor, 1.0)) if pd.notna(v) and v > 0 else 1.0
        exp.loc[dt] = cur
    exp_lag = exp.shift(1).fillna(1.0)
    r = exp_lag * ret + (1 - exp_lag) * cash / 252
    r = r - exp_lag.diff().abs().fillna(0) * cost
    return (1 + r).cumprod()


CASES = {
    "防守 30/50/20": {"510880": 0.30, "511260": 0.50, "518880": 0.20},
    "平衡 30/40/20/10": {"510880": 0.30, "511260": 0.40, "518880": 0.20, "513100": 0.10},
    "进取 30/30/20/20": {"510880": 0.30, "511260": 0.30, "518880": 0.20, "513100": 0.20},
    "进取 40/30/20/10": {"510880": 0.40, "511260": 0.30, "518880": 0.20, "513100": 0.10},
}

for name, w in CASES.items():
    base = run_strategy(PX, name, "fixed", fixed_weights=w, rebalance="Q").nav
    s = stats(base)
    print(f"\n{name}  基准  年化 {s['cagr']:.2%} 波动 {s['vol']:.2%} 夏普 {s['sharpe']:.2f} 回撤 {s['mdd']:.2%}")
    for tv in [0.04, 0.05, 0.06]:
        nav = vol_target(base, tv)
        s2 = stats(nav)
        print(f"   + 波动目标 {tv:.0%}  年化 {s2['cagr']:.2%} 波动 {s2['vol']:.2%} "
              f"夏普 {s2['sharpe']:.2f} 回撤 {s2['mdd']:.2%}  Calmar {s2['calmar']:.2f}")
    print("   分年(基准/5%):", {y: f"{v:+.1%}" for y, v in yearly(base).items()})
    nav5 = vol_target(base, 0.05)
    print("   分年(5%):   ", {y: f"{v:+.1%}" for y, v in yearly(nav5).items()})
