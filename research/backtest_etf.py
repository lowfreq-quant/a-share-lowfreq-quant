"""ETF 组合回测核心：多资产动量 / 趋势 / 固定权重

原则：
- 月末收盘出信号，次一交易日收盘成交（不用开盘跳空）
- 权重按市值漂移，只有调仓日才交易
- 成本：佣金 + 滑点，双边
- 不做空、不加杠杆
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "etf_hfq"


def load_prices(codes: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out = {}
    for c in codes:
        p = DATA / f"{c}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_parquet(p)
        out[c] = df["close"].astype(float)
    px = pd.DataFrame(out).sort_index()
    if start:
        px = px.loc[pd.to_datetime(start):]
    if end:
        px = px.loc[:pd.to_datetime(end)]
    return px.ffill()


@dataclass
class Result:
    name: str
    nav: pd.Series
    weights: pd.DataFrame
    turnover: float
    cost: float
    meta: dict = field(default_factory=dict)


def month_end_exec(dates: pd.DatetimeIndex, freq: str = "M") -> dict[pd.Timestamp, pd.Timestamp]:
    """返回 {信号日: 执行日}；执行日 = 次一交易日

    freq: M 月度 / Q 季度 / A 年度
    """
    s = pd.Series(dates, index=dates)
    sig = pd.DatetimeIndex(s.groupby([dates.year, dates.month]).tail(1))
    if freq == "Q":
        sig = pd.DatetimeIndex([d for d in sig if d.month in (3, 6, 9, 12)])
    elif freq == "A":
        sig = pd.DatetimeIndex([d for d in sig if d.month == 12])
    out = {}
    for d in sig:
        later = dates[dates > d]
        if len(later):
            out[d] = later[0]
    return out


def stats(nav: pd.Series, bench: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return {}
    yrs = len(r) / 252
    total = float(nav.iloc[-1] / nav.iloc[0])
    cagr = total ** (1 / yrs) - 1
    vol = float(r.std() * np.sqrt(252))
    mdd = float((1 - nav / nav.cummax()).max())
    out = {
        "total": total - 1,
        "cagr": cagr,
        "vol": vol,
        "sharpe": cagr / vol if vol else np.nan,
        "mdd": mdd,
        "calmar": cagr / mdd if mdd else np.nan,
        "win": float((r > 0).mean()),
        "years": yrs,
    }
    if bench is not None:
        b = bench.reindex(nav.index).ffill()
        ex = (nav / nav.iloc[0]) / (b / b.iloc[0]) - 1
        out["excess"] = float(ex.iloc[-1])
    return out


def yearly(nav: pd.Series) -> dict[int, float]:
    out = {}
    for y, g in nav.groupby(nav.index.year):
        if len(g) > 1:
            out[int(y)] = float(g.iloc[-1] / g.iloc[0] - 1)
    return out


def run_strategy(
    px: pd.DataFrame,
    name: str,
    kind: str,
    lookback: int = 250,
    top_n: int = 2,
    defensive: str | None = None,
    fixed_weights: dict[str, float] | None = None,
    trend_ma: int = 200,
    cost_side: float = 0.0004,
    rebalance: str = "M",
) -> Result:
    """kind: momentum | trend | fixed

    momentum: 每月按过去 lookback 日收益排名，持有前 top_n；若第一名收益<=0：
              defensive=None → 空仓；否则全仓 defensive
    trend:    沪深300 收盘 > MA(trend_ma) → 持有 510300；否则持有 defensive（默认 511260）
    fixed:    固定权重，月末再平衡
    """
    dates = px.index
    sched = month_end_exec(dates, freq=rebalance)
    rets = px.pct_change().fillna(0.0)

    value = 1.0
    cash = 1.0
    pos: dict[str, float] = {}
    rows = []
    w_rows = []
    total_turnover = 0.0
    total_cost = 0.0

    for dt in dates:
        # 1) 盯市
        if pos:
            for c in list(pos):
                pos[c] *= 1.0 + float(rets.at[dt, c])
            value = cash + sum(pos.values())

        # 2) 执行日调仓（收盘成交，从次日起生效）
        if dt in sched.values():
            sig_day = [s for s, e in sched.items() if e == dt][0]
            target: dict[str, float] = {}

            if kind == "fixed":
                target = dict(fixed_weights or {})

            elif kind == "trend":
                ma = px["510300"].loc[:sig_day].tail(trend_ma).mean()
                close = float(px.at[sig_day, "510300"])
                if close > ma:
                    target = {"510300": 1.0}
                elif defensive:
                    target = {defensive: 1.0}

            elif kind == "momentum":
                hist = px.loc[:sig_day]
                if len(hist) > lookback:
                    mom = hist.iloc[-1] / hist.iloc[-1 - lookback] - 1
                    mom = mom.dropna().sort_values(ascending=False)
                    if len(mom) >= top_n:
                        top = list(mom.index[:top_n])
                        if float(mom.iloc[0]) <= 0:
                            if defensive:
                                target = {defensive: 1.0}
                        else:
                            w = 1.0 / top_n
                            target = {c: w for c in top}

            # 交易：按当前市值计算换手
            cur = {c: (pos.get(c, 0.0) / value if value else 0.0) for c in px.columns}
            turnover = sum(abs(target.get(c, 0.0) - cur.get(c, 0.0)) for c in px.columns)
            cost = turnover * cost_side
            total_turnover += turnover
            total_cost += cost

            investable = value * (1.0 - cost)
            pos = {c: investable * w for c, w in target.items() if w > 0}
            cash = investable - sum(pos.values())
            value = investable

        rows.append((dt, value))
        w_rows.append((dt, {c: (pos.get(c, 0.0) / value if value else 0.0) for c in px.columns}))

    nav = pd.Series({d: v for d, v in rows}).sort_index()
    wdf = pd.DataFrame([{**{"date": d}, **w} for d, w in w_rows]).set_index("date")
    return Result(name=name, nav=nav, weights=wdf, turnover=total_turnover, cost=total_cost)


def regime_series() -> pd.Series:
    from src.regime import classify_regime
    import akshare as ak

    p = ROOT / "data" / "index_000300_live.parquet"
    idx = pd.read_parquet(p)["close"]
    return classify_regime(idx)["regime"]
