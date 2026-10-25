"""逐项拆解 v2 回测的执行偏差

V0 原版
V1 建仓日不赚当日涨幅（次日才计收益）——去掉前视
V2 V1 + 转出日仍吃当日涨跌（收盘卖出）——去掉清仓日跳过
V3 V2 + 建仓判断用信号日市况（不用执行日收盘）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_index_hist, load_config  # noqa: E402
from src.regime import classify_regime  # noqa: E402

OUT = ROOT / "output_audit"
cfg = load_config(ROOT / "config_choppy_v2.yaml")
picks = {pd.Timestamp(k): v for k, v in json.loads((OUT / "picks.json").read_text(encoding="utf-8")).items()}

bench = fetch_index_hist("000300", cfg["data"]["start_date"], None, ROOT / "data")["close"]

files = sorted((ROOT / "data" / "hist").glob("*.parquet"))
hm = {}
for f in files:
    try:
        hm[f.name.split("_")[0]] = pd.read_parquet(f, columns=["close"])
    except Exception:
        pass
all_dates = sorted({d for h in hm.values() for d in h.index})
start = pd.to_datetime(cfg["backtest"]["start_date"])
all_dates = pd.DatetimeIndex([d for d in all_dates if d >= start])
close = pd.DataFrame({c: h["close"] for c, h in hm.items()}).reindex(all_dates)
ret = close.pct_change(fill_method=None)
regime = classify_regime(bench)["regime"].reindex(all_dates).ffill()

sig = pd.Series(all_dates, index=all_dates).groupby([all_dates.year, all_dates.month]).tail(1)
sig_days = pd.DatetimeIndex(sig)
exec_of = {}
for s in sig_days:
    later = all_dates[all_dates > s]
    exec_of[s] = later[0] if len(later) else None

ccfg = cfg["cost"]
fee = ccfg["commission"] + ccfg["slippage"]
stamp = ccfg["stamp_duty"]


def r_at(c, dt):
    v = ret.at[dt, c] if c in ret.columns else np.nan
    return float(v) if pd.notna(v) else 0.0


def run(v1=False, v2=False, v3=False):
    holdings, nav, peak, scale, halted = {}, 1.0, 1.0, 1.0, False
    rows, in_mkt = [], []
    for dt in all_dates:
        ex = [s for s, e in exec_of.items() if e == dt]
        s = ex[0] if ex else None

        # 当日收益：V1 起用“调仓前持仓”
        if v1:
            dr = sum(w * r_at(c, dt) for c, w in holdings.items())
            nav *= 1 + dr
        # 转出判断
        signal_regime_ok = bool(regime.loc[s] == "震荡") if (v3 and s is not None) else None
        if holdings and not bool(regime.loc[dt] == "震荡"):
            if not v2:
                sell_w = sum(holdings.values())
                nav *= 1 - sell_w * (fee + stamp)
                holdings = {}
            else:
                sell_w = sum(holdings.values())
                nav *= 1 - sell_w * (fee + stamp)
                holdings = {}
        # 调仓
        if s is not None and not halted:
            ok = bool(regime.loc[dt] == "震荡") if not v3 else bool(signal_regime_ok)
            if ok:
                top = picks.get(s, [])
                if top:
                    w = scale / len(top)
                    new = {c: w for c in top}
                    allc = set(new) | set(holdings)
                    turnover = sum(abs(new.get(c, 0) - holdings.get(c, 0)) for c in allc)
                    sell_w = sum(holdings.get(c, 0) for c in set(holdings) - set(new))
                    nav *= 1 - (turnover * fee + sell_w * stamp)
                    holdings = new
        if not v1:
            dr = sum(w * r_at(c, dt) for c, w in holdings.items())
            nav *= 1 + dr
        if holdings:
            in_mkt.append((dt, dr))
        peak = max(peak, nav)
        dd = 1 - nav / peak
        if dd >= 0.20 and not halted:
            holdings, halted = {}, True
        elif dd >= 0.15 and scale == 1.0 and not halted:
            scale = 0.5
            holdings = {c: w * 0.5 for c, w in holdings.items()}
        rows.append((dt, nav, dd, sum(holdings.values())))
    df = pd.DataFrame(rows, columns=["date", "nav", "dd", "inv"]).set_index("date")
    return df, in_mkt


def st(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0]
    cagr = total ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    b = bench.reindex(nav.index).ffill()
    ex = (nav / nav.iloc[0]) / (b / b.iloc[0]) - 1
    return dict(total=total - 1, cagr=cagr, vol=vol, sharpe=cagr / vol, mdd=(1 - nav / nav.cummax()).max(), excess=ex.iloc[-1])


res = {}
for name, kw in [("V0_original", {}), ("V1_no_entry_day", dict(v1=True)), ("V2_plus_exit_day", dict(v1=True, v2=True)), ("V3_regime_at_signal", dict(v1=True, v2=True, v3=True))]:
    nav, in_mkt = run(**kw)
    s = st(nav["nav"])
    s["exposure"] = float((nav["inv"] > 0).mean())
    res[name] = s
    nav.to_csv(OUT / f"nav_{name}.csv")
    print(name, {k: round(float(v), 4) for k, v in s.items()}, flush=True)

# 持仓期间 vs 指数（选股 alpha 与择时的拆分）
nav0, in_mkt = run()
idx = bench.reindex(all_dates).ffill()
d = pd.DataFrame(in_mkt, columns=["date", "r"]).set_index("date")
ip = idx.pct_change().reindex(d.index)
print("\n持仓日数:", len(d), " 组合累计:", round(float((1 + d.r).prod() - 1), 4), " 指数同期:", round(float((1 + ip.fillna(0)).prod() - 1), 4))
print("年化 alpha(持仓期):", round(float(((1 + d.r).prod() / (1 + ip.fillna(0)).prod()) ** (252 / len(d)) - 1), 4))
(OUT / "variants.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
