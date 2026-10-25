"""固定权重配置网格：红利 / 国债 / 黄金 / 纳指 / 现金

目的：找一个不依赖择时、参数少、在震荡/熊市也能正收益的配置。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from backtest_etf import load_prices, regime_series, run_strategy, stats, yearly  # noqa: E402

OUT = ROOT / "research"
reg = regime_series()

# U2：2013 起可用的长样本
PX2 = load_prices(["510880", "511010", "518880", "510300", "510500", "513100"], start="2013-07-29")
# U1：2019 起，含红利低波/十年国债/中证1000
PX1 = load_prices(["510880", "512890", "511260", "518880", "510300", "512100", "513100"], start="2019-01-18")

CANDS = {
    "长样本(2013)": {
        "px": PX2,
        "cases": {
            "红利100": {"510880": 1.0},
            "红利70/国债30": {"510880": 0.7, "511010": 0.3},
            "红利60/国债40": {"510880": 0.6, "511010": 0.4},
            "红利50/国债50": {"510880": 0.5, "511010": 0.5},
            "红利40/国债60": {"510880": 0.4, "511010": 0.6},
            "红利30/国债70": {"510880": 0.3, "511010": 0.7},
            "红利40/国债40/黄金20": {"510880": 0.4, "511010": 0.4, "518880": 0.2},
            "红利30/国债50/黄金20": {"510880": 0.3, "511010": 0.5, "518880": 0.2},
            "红利30/国债40/黄金30": {"510880": 0.3, "511010": 0.4, "518880": 0.3},
            "红利25/国债25/黄金25/现金25": {"510880": 0.25, "511010": 0.25, "518880": 0.25},
            "红利20/国债40/黄金20/纳指20": {"510880": 0.2, "511010": 0.4, "518880": 0.2},
        },
    },
    "近年(2019)": {
        "px": PX1,
        "cases": {
            "红利100": {"510880": 1.0},
            "红利低波100": {"512890": 1.0},
            "红利50/十年国债50": {"510880": 0.5, "511260": 0.5},
            "红利低波50/十年国债50": {"512890": 0.5, "511260": 0.5},
            "红利30/十年国债50/黄金20": {"510880": 0.3, "511260": 0.5, "518880": 0.2},
            "红利低波30/十年国债50/黄金20": {"512890": 0.3, "511260": 0.5, "518880": 0.2},
            "红利30/十年国债40/黄金20/纳指10": {"510880": 0.3, "511260": 0.4, "518880": 0.2, "513100": 0.1},
            "现有防守15/15/70": {"510880": 0.15, "512890": 0.15, "511260": 0.70},
            "现有防守5/5/90": {"510880": 0.05, "512890": 0.05, "511260": 0.90},
            "红利25/十年国债25/黄金25/现金25": {"510880": 0.25, "511260": 0.25, "518880": 0.25},
        },
    },
}


def reg_stats(nav: pd.Series) -> dict:
    m = nav.resample("ME").last().pct_change().dropna()
    labels = reg.resample("ME").last().reindex(m.index, method="ffill")
    out = {}
    for name in ["牛市", "熊市", "震荡"]:
        sel = m[labels == name]
        if len(sel):
            out[name] = (len(sel), float((1 + sel).prod() ** (12 / len(sel)) - 1))
    return out


rows = []
for label, spec in CANDS.items():
    px = spec["px"]
    for name, w in spec["cases"].items():
        # 只保留有数据的资产；不重新归一化（权重和<1 的部分留现金）
        w = {k: v for k, v in w.items() if k in px.columns}
        r = run_strategy(px, name, "fixed", fixed_weights=w, cost_side=0.0004)
        s = stats(r.nav, px["510300"])
        s.update(dict(group=label, name=name, weights=str(w), turnover_yr=r.turnover / s["years"], cost_yr=r.cost / s["years"]))
        rows.append(s)
        yr = yearly(r.nav)
        for y in [2013, 2014, 2015, 2019, 2021, 2022, 2023, 2024, 2025, 2026]:
            s[f"y{y}"] = yr.get(y, np.nan)
        for k, (n, a) in reg_stats(r.nav).items():
            s[f"{k}_ann"] = a
            s[f"{k}_n"] = n

df = pd.DataFrame(rows)
df.to_csv(OUT / "fixed_results.csv", index=False, encoding="utf-8-sig")

lines = ["# 固定权重配置结果（后复权，含分红）", ""]
for label in CANDS:
    sub = df[df["group"] == label].sort_values("sharpe", ascending=False)
    lines += [f"## {label}", "", "| 配置 | 年化 | 波动 | 夏普 | 回撤 | Calmar | 2022 | 2023 | 2026 | 熊市年化 | 震荡年化 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in sub.iterrows():
        yr = yearly(pd.Series(dtype=float)) if False else {}
        lines.append(
            f"| {r['name']} | {r['cagr']:.2%} | {r['vol']:.2%} | {r['sharpe']:.2f} | {r['mdd']:.2%} | {r['calmar']:.2f} | "
            f"{r.get('y2022', float('nan')):.1%} | {r.get('y2023', float('nan')):.1%} | {r.get('y2026', float('nan')):.1%} | "
            f"{r.get('熊市_ann', float('nan')):+.1%} | {r.get('震荡_ann', float('nan')):+.1%} |"
        )
    lines.append("")

(OUT / "FIXED.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
