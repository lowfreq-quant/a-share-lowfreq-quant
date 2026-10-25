"""30/50/20 的风险叠加层测试

1) 回撤触发式 IF 对冲：组合回撤 >= 4/6/8% 才开对冲，回到一半就关
2) 波动率目标：按 20/60 日实现波动把仓位缩到目标波动（不加杠杆）
3) 趋势倾斜：沪深300 在 MA200 上方用 30/50/20，下方用 20/60/20
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
from hedge_if import load_if  # noqa: E402

SPOT = pd.read_parquet(ROOT / "data" / "index_000300_live.parquet")["close"].astype(float)
PX = load_prices(["510880", "511260", "518880", "510300"], start="2017-08-24")


def base_nav(w_eq: float = 0.3, w_bond: float = 0.5, w_gold: float = 0.2) -> pd.Series:
    return run_strategy(
        PX[["510880", "511260", "518880"]], "p", "fixed",
        fixed_weights={"510880": w_eq, "511260": w_bond, "518880": w_gold}, rebalance="Q",
    ).nav


BASE = base_nav()


def show(name: str, nav: pd.Series, extra: str = "") -> None:
    s = stats(nav)
    yr = yearly(nav)
    print(f"{name:30s} 年化 {s['cagr']:+.2%}  波动 {s['vol']:.2%}  夏普 {s['sharpe']:.2f}  "
          f"回撤 {s['mdd']:.2%}  累计 {s['total']:+.1%}  {extra}")
    print("   分年:", {y: f"{v:+.1%}" for y, v in yr.items()})


# ---------------- 1) 回撤触发式 IF 对冲 ----------------
def hedge_dd(nav0: pd.Series, trigger: float, h: float, margin: float = 0.13, funding: float = 0.04,
             trade: float = 0.0002) -> tuple[pd.Series, dict]:
    fut = load_if()
    dates = nav0.index.intersection(fut.index)
    port = nav0.reindex(dates).ffill()
    fut = fut.reindex(dates).ffill()
    spot = SPOT.reindex(dates).ffill()
    ret = port.pct_change().fillna(0)
    peak = port.cummax()
    dd = 1 - port / peak

    # 触发信号：回撤到 trigger 开，回到 release 关（带迟滞）
    release = trigger / 2
    sig = pd.Series(False, index=dates)
    on = False
    for i in range(len(dates)):
        if not on and dd.iloc[i] >= trigger:
            on = True
        elif on and dd.iloc[i] <= release:
            on = False
        sig.iloc[i] = on

    nav = 1.0
    hedged = False
    contracts = 0.0
    basis = margin_cost = trade_cost = 0.0
    switches = 0
    out_rows = []
    for i, dt in enumerate(dates):
        if i > 0:
            nav *= 1 + ret.iloc[i]
        if hedged and i > 0:
            pnl = -contracts * 300 * (fut.iloc[i] - fut.iloc[i - 1])
            nav += pnl
            basis += pnl - (-contracts * 300 * (spot.iloc[i] - spot.iloc[i - 1]))
            mc = contracts * 300 * fut.iloc[i - 1] * margin * funding / 252
            nav -= mc
            margin_cost += mc
        want = bool(sig.iloc[i])
        if want and not hedged:
            contracts = h * nav / (fut.iloc[i] * 300)
            hedged = True
            tc = abs(contracts * 300 * fut.iloc[i]) * trade
            nav -= tc
            trade_cost += tc
            switches += 1
        elif not want and hedged:
            tc = abs(contracts * 300 * fut.iloc[i]) * trade
            nav -= tc
            trade_cost += tc
            contracts = 0
            hedged = False
            switches += 1
        out_rows.append((dt, nav))
    out = pd.Series({d: v for d, v in out_rows})
    return out, dict(active=float(sig.mean()), switches=switches, basis=basis, margin=margin_cost, trade=trade_cost)


# ---------------- 2) 波动率目标 ----------------
def vol_target(nav0: pd.Series, target: float, lookback: int = 20, cash: float = 0.0,
               cost: float = 0.0005) -> pd.Series:
    ret = nav0.pct_change().fillna(0)
    vol = ret.rolling(lookback).std() * np.sqrt(252)
    dates = nav0.index
    me = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).tail(1).index
    exp = pd.Series(1.0, index=dates)
    cur = 1.0
    for dt in dates:
        if dt in me:
            v = vol.loc[dt]
            cur = min(1.0, target / v) if pd.notna(v) and v > 0 else 1.0
        exp.loc[dt] = cur
    exp_lag = exp.shift(1).fillna(1.0)
    r = exp_lag * ret + (1 - exp_lag) * cash / 252
    r = r - exp_lag.diff().abs().fillna(0) * cost
    return (1 + r).cumprod()


# ---------------- 3) 趋势倾斜 ----------------
def trend_tilt(ma: int = 200, risk_on=(0.3, 0.5, 0.2), risk_off=(0.2, 0.6, 0.2),
               cost: float = 0.0005) -> pd.Series:
    px300 = PX["510300"]
    ma_s = px300.rolling(ma).mean()
    on_sig = (px300 > ma_s)
    p_on = base_nav(*risk_on)
    p_off = base_nav(*risk_off)
    r_on = p_on.pct_change().fillna(0)
    r_off = p_off.pct_change().fillna(0)
    dates = p_on.index
    me = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).tail(1).index
    sel = pd.Series(True, index=dates)
    cur = True
    for dt in dates:
        if dt in me:
            cur = bool(on_sig.loc[:dt].iloc[-1]) if len(on_sig.loc[:dt]) else True
        sel.loc[dt] = cur
    sel_lag = sel.shift(1).fillna(True)
    r = np.where(sel_lag, r_on, r_off)
    switch = (sel_lag != sel_lag.shift(1)).astype(float)
    r = r - switch * cost * 0.2  # 10% 权重切换，约 20% 换手
    return pd.Series((1 + pd.Series(r, index=dates)).cumprod(), index=dates)


if __name__ == "__main__":
    show("30/50/20 基准", BASE)

    print("\n--- 1) 回撤触发式对冲 ---")
    for trig in [0.04, 0.06, 0.08]:
        for h in [0.2, 0.3]:
            nav, m = hedge_dd(BASE, trig, h)
            show(f"回撤>{trig:.0%} 对冲{h:.0%}", nav,
                 f"对冲占比{m['active']:.0%} 切换{m['switches']} 基差{m['basis']:+.2%} 保证金{m['margin']:.2%} 交易{m['trade']:.2%}")

    print("\n--- 2) 波动率目标（月度调仓） ---")
    for lb in [20, 60]:
        for tv in [0.04, 0.05, 0.06]:
            show(f"目标波动{tv:.0%} 回看{lb}", vol_target(BASE, tv, lb))

    print("\n--- 3) 趋势倾斜（沪深300 MA200） ---")
    for ma in [120, 200]:
        show(f"MA{ma} 上30/50/20 下20/60/20", trend_tilt(ma))
