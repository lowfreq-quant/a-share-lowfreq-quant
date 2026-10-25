"""波动率目标叠加层：跨样本验证

规则：月末按过去 L 日实现波动，把仓位缩到 min(1, target/vol)，不加杠杆；
     缩减部分按 cash 收益率计；月度调仓，换手成本 0.05%/单位。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from backtest_etf import load_prices, run_strategy, stats, yearly  # noqa: E402

SAMPLES = {
    "2013_5年国债": (["510880", "511010", "518880"], "511010", "2013-07-29"),
    "2017_十年国债": (["510880", "511260", "518880"], "511260", "2017-08-24"),
    "2019_十年国债": (["510880", "511260", "518880"], "511260", "2019-01-18"),
}


def base_nav(codes, bond, start):
    px = load_prices(codes, start=start)
    return run_strategy(px, "b", "fixed", fixed_weights={"510880": 0.3, bond: 0.5, "518880": 0.2},
                        rebalance="Q").nav


def vol_target(nav, target, lookback=60, cash=0.0, floor=0.0, cost=0.0005):
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
    return (1 + r).cumprod(), exp.mean()


if __name__ == "__main__":
    for sname, (codes, bond, start) in SAMPLES.items():
        base = base_nav(codes, bond, start)
        sb = stats(base)
        print(f"\n=== {sname}  {base.index.min().date()}→{base.index.max().date()} ===")
        print(f"基准 30/50/20       年化 {sb['cagr']:+.2%} 波动 {sb['vol']:.2%} 夏普 {sb['sharpe']:.2f} 回撤 {sb['mdd']:.2%}")
        for lb in [20, 60]:
            for tv in [0.04, 0.05, 0.06]:
                nav, avg_exp = vol_target(base, tv, lb)
                s = stats(nav)
                print(f"  目标{tv:.0%} 回看{lb:>2}  年化 {s['cagr']:+.2%} 波动 {s['vol']:.2%} "
                      f"夏普 {s['sharpe']:.2f} 回撤 {s['mdd']:.2%} 平均仓位 {avg_exp:.0%}")
        # 边界：floor 0.3 / cash 1.5% / 季度
        for floor in [0.3, 0.5]:
            nav, avg_exp = vol_target(base, 0.05, 60, floor=floor)
            s = stats(nav)
            print(f"  5% 60日 floor={floor:.0%}  年化 {s['cagr']:+.2%} 夏普 {s['sharpe']:.2f} 回撤 {s['mdd']:.2%} 平均仓位 {avg_exp:.0%}")
        nav, avg_exp = vol_target(base, 0.05, 60, cash=0.015)
        s = stats(nav)
        print(f"  5% 60日 现金1.5%     年化 {s['cagr']:+.2%} 夏普 {s['sharpe']:.2f} 回撤 {s['mdd']:.2%} 平均仓位 {avg_exp:.0%}")
