"""稳健性检验：权重邻域、起点、调仓频率、成本"""
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


def reg_ann(nav: pd.Series) -> dict:
    m = nav.resample("ME").last().pct_change().dropna()
    labels = reg.resample("ME").last().reindex(m.index, method="ffill")
    out = {}
    for name in ["牛市", "熊市", "震荡"]:
        sel = m[labels == name]
        if len(sel):
            out[name] = float((1 + sel).prod() ** (12 / len(sel)) - 1)
    return out


def one(px, w, rebalance="M", cost=0.0004, name="x"):
    r = run_strategy(px, name, "fixed", fixed_weights=w, cost_side=cost, rebalance=rebalance)
    s = stats(r.nav, px["510300"])
    s["turnover_yr"] = r.turnover / s["years"]
    s["cost_yr"] = r.cost / s["years"]
    yr = yearly(r.nav)
    for y in [2013, 2015, 2018, 2019, 2022, 2023, 2024, 2025, 2026]:
        s[f"y{y}"] = yr.get(y, np.nan)
    s.update({f"reg_{k}": v for k, v in reg_ann(r.nav).items()})
    return s, r.nav


samples = {
    "A_2013_5年国债": (load_prices(["510880", "511010", "518880", "510300"], start="2013-07-29"), "511010"),
    "B_2017_十年国债": (load_prices(["510880", "511260", "518880", "510300"], start="2017-08-24"), "511260"),
    "C_2019_十年国债": (load_prices(["510880", "511260", "518880", "510300"], start="2019-01-18"), "511260"),
}

# ---------- 1. 权重邻域 ----------
print("=== 1. 权重邻域（红利/国债/黄金） ===", flush=True)
rows = []
for sname, (px, bond) in samples.items():
    for eq in range(10, 61, 10):
        for bd in range(20, 81, 10):
            gd = 100 - eq - bd
            if gd < 0 or gd > 40 or gd % 10:
                continue
            w = {"510880": eq / 100, bond: bd / 100, "518880": gd / 100}
            if sum(w.values()) > 1.0001:
                continue
            s, _ = one(px, w, name=f"{eq}/{bd}/{gd}")
            s.update(dict(sample=sname, eq=eq, bond=bd, gold=gd))
            rows.append(s)

df = pd.DataFrame(rows)
df.to_csv(OUT / "robust_weights.csv", index=False, encoding="utf-8-sig")
for sname in samples:
    sub = df[df["sample"] == sname].sort_values("sharpe", ascending=False)
    print(f"\n-- {sname} top10 by Sharpe --")
    print(sub[["eq", "bond", "gold", "cagr", "vol", "sharpe", "mdd", "y2022", "reg_熊市", "reg_震荡"]].head(10).round(4).to_string(index=False))
    base = sub[(sub.eq == 30) & (sub.bond == 50) & (sub.gold == 20)]
    if len(base):
        print("30/50/20:", base[["cagr", "vol", "sharpe", "mdd", "y2022", "y2023", "reg_熊市", "reg_震荡"]].round(4).to_dict("records"))

# ---------- 2. 频率 / 成本 / 起点 ----------
print("\n=== 2. 频率/成本/起点（30/50/20） ===", flush=True)
rows2 = []
for sname, (px, bond) in samples.items():
    for reb in ["M", "Q", "A"]:
        for cost in [0.0002, 0.0004, 0.0010]:
            s, _ = one(px, {"510880": 0.3, bond: 0.5, "518880": 0.2}, rebalance=reb, cost=cost, name="30/50/20")
            s.update(dict(sample=sname, rebalance=reb, cost_side=cost))
            rows2.append(s)
df2 = pd.DataFrame(rows2)
df2.to_csv(OUT / "robust_freq.csv", index=False, encoding="utf-8-sig")
print(df2[["sample", "rebalance", "cost_side", "cagr", "vol", "sharpe", "mdd", "turnover_yr"]].round(4).to_string(index=False))

# ---------- 3. 起点敏感性 ----------
print("\n=== 3. 起点敏感性（30/50/20，十年国债） ===", flush=True)
px_full = load_prices(["510880", "511260", "518880", "510300"], start="2017-08-24")
rows3 = []
for start in ["2017-08-24", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]:
    px = px_full.loc[pd.to_datetime(start):]
    s, _ = one(px, {"510880": 0.3, "511260": 0.5, "518880": 0.2}, name="30/50/20")
    s["start"] = start
    rows3.append(s)
df3 = pd.DataFrame(rows3)
print(df3[["start", "cagr", "vol", "sharpe", "mdd", "y2022", "y2023", "y2026"]].round(4).to_string(index=False))
df3.to_csv(OUT / "robust_start.csv", index=False, encoding="utf-8-sig")

# ---------- 4. 与基准对比 ----------
print("\n=== 4. 基准对比（2019 起） ===", flush=True)
px = samples["C_2019_十年国债"][0]
cases = {
    "沪深300": (px, {"510300": 1.0}),
    "红利100": (px, {"510880": 1.0}),
    "红利50/国债50": (px, {"510880": 0.5, "511260": 0.5}),
    "30/50/20": (px, {"510880": 0.3, "511260": 0.5, "518880": 0.2}),
    "20/40/20/纳指20": (load_prices(["510880", "511260", "518880", "513100", "510300"], start="2019-01-18"),
                    {"510880": 0.2, "511260": 0.4, "518880": 0.2, "513100": 0.2}),
    "现有防守15/15/70": (load_prices(["510880", "512890", "511260", "510300"], start="2019-01-18"),
                    {"510880": 0.15, "512890": 0.15, "511260": 0.7}),
}
for nm, (pxx, w) in cases.items():
    w = {k: v for k, v in w.items() if k in pxx.columns}
    s, nav = one(pxx, w, name=nm)
    print(f"{nm:22s} cagr={s['cagr']:.2%} vol={s['vol']:.2%} sharpe={s['sharpe']:.2f} mdd={s['mdd']:.2%} "
          f"2022={s.get('y2022', np.nan):+.1%} 2023={s.get('y2023', np.nan):+.1%} 2026={s.get('y2026', np.nan):+.1%} "
          f"熊市={s.get('reg_熊市', np.nan):+.1%} 震荡={s.get('reg_震荡', np.nan):+.1%}")
