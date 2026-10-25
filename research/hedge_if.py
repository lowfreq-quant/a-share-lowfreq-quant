"""30/50/20 + 熊市 IF 对冲回测（真实 IF0 连续合约）

成本：
- 保证金比例 13%，资金/机会成本 4%/年（可调）
- 交易成本 0.02%/边（开、平）
- 基差/展期成本已包含在 IF0 的日收益里；另拆出「期货-现货」差作为基差拖累
- 信号：前一日收盘 regime=熊市，次日收盘建/平（无前视）
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
from src.regime import classify_regime  # noqa: E402

IF_CACHE = ROOT / "data" / "if0.parquet"
SPOT = pd.read_parquet(ROOT / "data" / "index_000300_live.parquet")["close"].astype(float)


def load_if() -> pd.Series:
    if IF_CACHE.exists():
        df = pd.read_parquet(IF_CACHE)
        return df["fut"].astype(float)
    import akshare as ak

    f = ak.futures_main_sina(symbol="IF0", start_date="20170101", end_date="20261016")
    f["date"] = pd.to_datetime(f["日期"])
    f = f.set_index("date")["收盘价"].astype(float).rename("fut")
    f.to_frame().to_parquet(IF_CACHE)
    return f


def backtest(hedge_ratio: float, margin_ratio: float = 0.13, funding: float = 0.04,
             trade_cost: float = 0.0002, use_ma: bool = False,
             port_nav: pd.Series | None = None) -> tuple[pd.Series, dict]:
    fut = load_if()
    if port_nav is None:
        px = load_prices(["510880", "511260", "518880"], start="2017-08-24")
        port = run_strategy(
            px, "def", "fixed", fixed_weights={"510880": 0.3, "511260": 0.5, "518880": 0.2},
            rebalance="Q",
        ).nav
    else:
        port = port_nav

    dates = port.index.intersection(fut.index)
    port = port.reindex(dates).ffill()
    fut = fut.reindex(dates).ffill()
    spot = SPOT.reindex(dates).ffill()
    ret = port.pct_change().fillna(0.0)
    regime = classify_regime(spot)["regime"]
    signal = regime.shift(1) == "熊市"
    if use_ma:
        ma120 = spot.rolling(120).mean()
        signal = signal & (spot.shift(1) < ma120.shift(1))

    nav = 1.0
    on = False
    contracts = 0.0
    basis_pnl = 0.0
    margin_cost_total = 0.0
    trade_cost_total = 0.0
    rows = []

    for i, dt in enumerate(dates):
        if i > 0:
            nav *= 1.0 + ret.iloc[i]
        if on and i > 0:
            pnl = -contracts * 300.0 * (fut.iloc[i] - fut.iloc[i - 1])
            nav += pnl
            basis_pnl += pnl - (-contracts * 300.0 * (spot.iloc[i] - spot.iloc[i - 1]))
            mc = contracts * 300.0 * fut.iloc[i - 1] * margin_ratio * funding / 252.0
            nav -= mc
            margin_cost_total += mc

        want = bool(signal.iloc[i])
        if want and not on:
            contracts = hedge_ratio * nav / (fut.iloc[i] * 300.0)
            on = True
            tc = abs(contracts * 300.0 * fut.iloc[i]) * trade_cost
            nav -= tc
            trade_cost_total += tc
        elif not want and on:
            tc = abs(contracts * 300.0 * fut.iloc[i]) * trade_cost
            nav -= tc
            trade_cost_total += tc
            contracts = 0.0
            on = False

        rows.append((dt, nav, 1 if on else 0))

    out = pd.DataFrame(rows, columns=["date", "nav", "on"]).set_index("date")["nav"]
    meta = dict(
        basis_pnl=basis_pnl,
        margin_cost=margin_cost_total,
        trade_cost=trade_cost_total,
        active=float(pd.DataFrame(rows, columns=["date", "nav", "on"]).set_index("date")["on"].mean()),
        switches=int(pd.DataFrame(rows, columns=["date", "nav", "on"]).set_index("date")["on"].diff().abs().sum()),
    )
    return out, meta


def show(name: str, nav: pd.Series, meta: dict | None = None) -> None:
    s = stats(nav)
    yr = yearly(nav)
    print(f"{name:26s} 年化 {s['cagr']:+.2%}  波动 {s['vol']:.2%}  夏普 {s['sharpe']:.2f}  "
          f"回撤 {s['mdd']:.2%}  累计 {s['total']:+.1%}")
    print("   分年:", {y: f"{v:+.1%}" for y, v in yr.items()})
    if meta:
        print(f"   对冲占比 {meta['active']:.0%}  切换 {meta['switches']} 次  "
              f"基差/展期 {meta['basis_pnl']:+.2%}  保证金 {meta['margin_cost']:.2%}  交易 {meta['trade_cost']:.3%}")


if __name__ == "__main__":
    unhedged, _ = backtest(0.0)
    show("30/50/20 不对冲", unhedged)
    for h in [0.10, 0.15, 0.20, 0.30, 0.45]:
        nav, meta = backtest(h)
        show(f"熊市 IF 对冲 {h:.0%}", nav, meta)
    nav, meta = backtest(0.30, use_ma=True)
    show("对冲30% + 熊市&<MA120", nav, meta)
    for f in [0.02, 0.06]:
        nav, meta = backtest(0.30, funding=f)
        show(f"对冲30% 资金成本{f:.0%}", nav, meta)

    # 对照：高 beta 的沪深300 才可能是对冲的主场
    px300 = load_prices(["510300"], start="2017-08-24")
    nav300 = run_strategy(px300, "300", "fixed", fixed_weights={"510300": 1.0}).nav
    show("沪深300 买入持有", nav300)
    for h in [0.5, 1.0]:
        nav, meta = backtest(h, port_nav=nav300)
        show(f"300 + 熊市对冲 {h:.0%}", nav, meta)
