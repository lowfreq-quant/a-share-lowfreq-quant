"""ETF 策略网格：动量 / 趋势 / 固定权重

输出：research/grid_results.csv、research/GRID.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from backtest_etf import (  # noqa: E402
    load_prices,
    month_end_exec,
    regime_series,
    run_strategy,
    stats,
    yearly,
)

OUT = ROOT / "research"
OUT.mkdir(exist_ok=True)

UNIVERSES = {
    "U1_2019": dict(codes=["510880", "512890", "511260", "518880", "510300", "512100"], start="2019-01-18"),
    "U2_2013": dict(codes=["510880", "511010", "518880", "510300", "510500"], start="2013-07-29"),
    "U3_2017": dict(codes=["510880", "511260", "518880", "510300"], start="2017-08-24"),
}

BENCH = ROOT / "data" / "etf_hfq" / "510300.parquet"


def monthly_returns(nav: pd.Series) -> pd.Series:
    return nav.resample("ME").last().pct_change().dropna()


def regime_monthly(nav: pd.Series, reg: pd.Series) -> dict:
    """按月分组：用月初（上月末）市况标签"""
    m = nav.resample("ME").last().pct_change().dropna()
    labels = reg.resample("ME").last().reindex(m.index, method="ffill")
    out = {}
    for name in ["牛市", "熊市", "震荡"]:
        sel = m[labels == name]
        if len(sel):
            out[name] = dict(n=len(sel), mean=float(sel.mean()), ann=float((1 + sel).prod() ** (12 / len(sel)) - 1))
    return out


def main() -> None:
    reg = regime_series()
    rows = []
    navs = {}

    for uname, u in UNIVERSES.items():
        px = load_prices(u["codes"], start=u["start"])
        defensive = "511260" if "511260" in px.columns else "511010"
        bench = px["510300"]
        print(f"\n== {uname} {u['codes']} {px.index.min().date()}→{px.index.max().date()} ==", flush=True)

        # 基准
        bh300 = run_strategy(px[["510300"]], f"{uname}_BH_300", "fixed", fixed_weights={"510300": 1.0})
        rows.append(dict(universe=uname, name="BH_300", kind="bench", lookback=np.nan, top_n=np.nan, **stats(bh300.nav, bench), turnover=bh300.turnover, cost=bh300.cost))
        navs[f"{uname}_BH_300"] = bh300.nav

        bhdiv = run_strategy(px[["510880"]], f"{uname}_BH_红利", "fixed", fixed_weights={"510880": 1.0})
        rows.append(dict(universe=uname, name="BH_红利", kind="bench", lookback=np.nan, top_n=np.nan, **stats(bhdiv.nav, bench), turnover=bhdiv.turnover, cost=bhdiv.cost))
        navs[f"{uname}_BH_红利"] = bhdiv.nav

        fixed = run_strategy(px, f"{uname}_固定红利50/国债50", "fixed", fixed_weights={"510880": 0.5, defensive: 0.5})
        rows.append(dict(universe=uname, name="固定红利50/国债50", kind="fixed", lookback=np.nan, top_n=np.nan, **stats(fixed.nav, bench), turnover=fixed.turnover, cost=fixed.cost))
        navs[f"{uname}_固定红利50/国债50"] = fixed.nav

        trend = run_strategy(px, f"{uname}_趋势300>MA200", "trend", trend_ma=200, defensive=defensive)
        rows.append(dict(universe=uname, name="趋势300>MA200", kind="trend", lookback=200, top_n=np.nan, **stats(trend.nav, bench), turnover=trend.turnover, cost=trend.cost))
        navs[f"{uname}_趋势300>MA200"] = trend.nav

        for lb in [60, 120, 250]:
            for n in [1, 2, 3]:
                if n > len(px.columns):
                    continue
                for dname, d in [("防守", defensive), ("空仓", None)]:
                    nm = f"{uname}_动量{lb}_top{n}_{dname}"
                    r = run_strategy(px, nm, "momentum", lookback=lb, top_n=n, defensive=d)
                    s = stats(r.nav, bench)
                    s.update(dict(universe=uname, name=f"动量{lb}_top{n}_{dname}", kind="momentum", lookback=lb, top_n=n, turnover=r.turnover, cost=r.cost))
                    rows.append(s)
                    navs[nm] = r.nav

    df = pd.DataFrame(rows)
    df["turnover_yr"] = df["turnover"] / df["years"]
    df["cost_yr"] = df["cost"] / df["years"]
    # 分年
    for y in [2013, 2019, 2022, 2023, 2024, 2025, 2026]:
        df[f"y{y}"] = np.nan
    for i, row in df.iterrows():
        key = [k for k in navs if k.endswith(row["name"]) and k.startswith(row["universe"])]
        if key:
            yr = yearly(navs[key[0]])
            for y in [2013, 2019, 2022, 2023, 2024, 2025, 2026]:
                df.at[i, f"y{y}"] = yr.get(y, np.nan)

    df.to_csv(OUT / "grid_results.csv", index=False, encoding="utf-8-sig")

    # 汇总表：每个 universe 按夏普排序
    lines = ["# ETF 策略网格结果", ""]
    for uname in UNIVERSES:
        sub = df[df["universe"] == uname].sort_values("sharpe", ascending=False)
        lines += [f"## {uname}", "", "| 策略 | 年化 | 波动 | 夏普 | 回撤 | Calmar | 2022 | 2023 | 2026 | 换手/年 | 成本/年 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['name']} | {r['cagr']:.2%} | {r['vol']:.2%} | {r['sharpe']:.2f} | {r['mdd']:.2%} | {r['calmar']:.2f} | "
                f"{r.get('y2022', np.nan):.1%} | {r.get('y2023', np.nan):.1%} | {r.get('y2026', np.nan):.1%} | {r['turnover_yr']:.1f} | {r['cost_yr']:.2%} |"
            )
        lines.append("")

    # 市况分组（用 U1 的代表策略）
    px = load_prices(UNIVERSES["U1_2019"]["codes"], start=UNIVERSES["U1_2019"]["start"])
    for nm, kw in [
        ("动量250_top2_防守", dict(kind="momentum", lookback=250, top_n=2, defensive="511260")),
        ("动量120_top2_防守", dict(kind="momentum", lookback=120, top_n=2, defensive="511260")),
        ("趋势300>MA200", dict(kind="trend", trend_ma=200, defensive="511260")),
        ("固定红利50/国债50", dict(kind="fixed", fixed_weights={"510880": 0.5, "511260": 0.5})),
    ]:
        r = run_strategy(px, nm, **kw)
        rm = regime_monthly(r.nav, reg)
        lines.append(f"- **{nm}** 分市况年化：" + "；".join(f"{k} {v['ann']:+.1%}（{v['n']}月）" for k, v in rm.items()))

    (OUT / "GRID.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n写出 {OUT / 'grid_results.csv'} / {OUT / 'GRID.md'}")


if __name__ == "__main__":
    main()
