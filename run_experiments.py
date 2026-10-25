"""震荡盘稳健性实验：一次加载行情，多组参数对比（防过拟合）

用法: python run_experiments.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.data import download_universe_history, fetch_index_hist, fetch_stock_list, load_config
from src.report import performance_stats

# 行业约束需要在 backtest 内支持；这里用“选股时跳过已超限行业”实现
from src.factors import composite_score, compute_point_in_time
from src.universe import build_universe_at, is_st_name
import numpy as np


def load_all():
    cfg = load_config(ROOT / "config_choppy.yaml")
    cache = ROOT / "data"
    names = fetch_stock_list(cache)
    codes = names["code"].tolist()
    print(f"loading {len(codes)} ...", flush=True)
    t0 = time.time()
    hist = download_universe_history(codes, "2021-01-01", None, cache, sleep=0, progress_every=1000)
    print(f"loaded {len(hist)} in {time.time()-t0:.0f}s", flush=True)
    bench = fetch_index_hist("000300", "2021-01-01", None, cache)["close"]
    return cfg, names, hist, bench


def pick_portfolio(
    signal_day,
    hist_map,
    names,
    cfg,
    n_stocks,
    weights,
    ind_map: dict[str, str] | None,
    max_ind_w: float | None,
    min_mv: float,
    min_amt: float,
    exclude_prefix: tuple[str, ...] = (),
):
    name_map = names.set_index("code")["name"].to_dict() if isinstance(names, pd.DataFrame) else names
    codes = []
    for c, h in hist_map.items():
        if any(c.startswith(p) for p in exclude_prefix):
            continue
        hh = h.loc[:signal_day]
        if len(hh) < cfg["universe"]["min_list_days"]:
            continue
        if "amount" in hh.columns and hh["amount"].tail(20).notna().any():
            if float(hh["amount"].tail(20).mean()) < min_amt:
                continue
        if "total_mv" in hh.columns and hh["total_mv"].notna().any():
            if float(hh["total_mv"].iloc[-1]) < min_mv:
                continue
        if c in name_map and is_st_name(name_map[c]):
            continue
        codes.append(c)
    if len(codes) < n_stocks:
        return []
    fac = compute_point_in_time(signal_day, hist_map, codes, None)
    fac["score"] = composite_score(fac, weights, cfg["factors"].get("winsorize", 0.02))
    ranked = fac.sort_values("score", ascending=False)

    picked = []
    ind_count: dict[str, int] = {}
    max_per = None
    if ind_map and max_ind_w:
        max_per = max(1, int(round(max_ind_w * n_stocks)))
    for code in ranked.index:
        ind = (ind_map or {}).get(code)
        if max_per is not None and ind is not None:
            if ind_count.get(ind, 0) >= max_per:
                continue
        picked.append(code)
        if ind is not None:
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(picked) >= n_stocks:
            break
    return picked


def quick_backtest(hist_map, names, bench, cfg, *, n_stocks, weights, ind_map=None, max_ind_w=None,
                   min_mv=2e9, min_amt=5e6, exclude_prefix=(), only_regime=True):
    """简化月度回测，用于实验对比"""
    from src.regime import classify_regime

    cc = cfg["cost"]
    start = pd.to_datetime(cfg["backtest"]["start_date"])
    all_dates = sorted({pd.Timestamp(d) for h in hist_map.values() for d in h.index})
    all_dates = pd.DatetimeIndex([d for d in all_dates if d >= start])
    regime = classify_regime(bench.reindex(all_dates).ffill())["regime"]

    # 月末
    sig = pd.DatetimeIndex(pd.Series(all_dates, index=all_dates).groupby([all_dates.year, all_dates.month]).tail(1))
    exec_of = {}
    for s in sig:
        later = all_dates[all_dates > s]
        exec_of[s] = later[0] if len(later) else None

    holdings = {}
    nav = 1.0
    peak = 1.0
    rows = []
    for dt in all_dates:
        ok_regime = (not only_regime) or (dt in regime.index and regime.loc[dt] == "震荡") or (
            dt not in regime.index and regime.loc[:dt].iloc[-1] == "震荡" if len(regime.loc[:dt]) else True
        )
        if only_regime and not ok_regime and holdings:
            cost = sum(holdings.values()) * (cc["commission"] + cc["slippage"] + cc["stamp_duty"])
            nav *= 1 - cost
            holdings = {}
        if dt in exec_of.values() and ok_regime:
            # 找对应 signal day
            sday = [s for s, e in exec_of.items() if e == dt][0]
            top = pick_portfolio(sday, hist_map, names, cfg, n_stocks, weights, ind_map, max_ind_w, min_mv, min_amt, exclude_prefix)
            if top:
                w = 1.0 / len(top)
                new = {c: w for c in top}
                turn = sum(abs(new.get(c, 0) - holdings.get(c, 0)) for c in set(new) | set(holdings))
                sell = sum(holdings.get(c, 0) for c in set(holdings) - set(new))
                cost = turn * (cc["commission"] + cc["slippage"]) + sell * cc["stamp_duty"]
                nav *= 1 - cost
                holdings = new
        day_ret = 0.0
        for c, w in holdings.items():
            h = hist_map[c]
            if dt not in h.index:
                continue
            i = h.index.get_loc(dt)
            if i == 0:
                continue
            day_ret += w * (h["close"].iloc[i] / h["close"].iloc[i - 1] - 1)
        nav *= 1 + day_ret
        peak = max(peak, nav)
        rows.append({"date": dt, "nav": nav, "dd": 1 - nav / peak})
    s = pd.Series([r["nav"] for r in rows], index=[r["date"] for r in rows])
    st = performance_stats(s, bench)
    return st, s


def main():
    cfg, names, hist, bench = load_all()
    ind_map = None
    ind_path = ROOT / "data" / "industry_map.parquet"
    if ind_path.exists():
        ind_map = pd.read_parquet(ind_path).set_index("code")["industry"].to_dict()
        print(f"industry map: {len(ind_map)}")
    else:
        print("no industry map; run build_industry_map.py first")

    base_w = dict(cfg["factors"]["weights"])
    # 关闭估值
    for k in ("ep", "roe"):
        base_w[k] = 0.0

    experiments = [
        ("基线 n25 全市场", dict(n_stocks=25, weights=base_w, min_mv=2e9, min_amt=5e6, exclude_prefix=())),
        ("n40 分散", dict(n_stocks=40, weights=base_w, min_mv=2e9, min_amt=5e6, exclude_prefix=())),
        ("剔除北交所", dict(n_stocks=25, weights=base_w, min_mv=2e9, min_amt=5e6, exclude_prefix=("92", "83", "43", "87", "88"))),
        ("市值下限50亿", dict(n_stocks=25, weights=base_w, min_mv=5e9, min_amt=5e6, exclude_prefix=("92", "83", "43", "87", "88"))),
        ("成交额下限2000万", dict(n_stocks=25, weights=base_w, min_mv=2e9, min_amt=2e7, exclude_prefix=("92", "83", "43", "87", "88"))),
        ("动量加强 w+1.0", dict(n_stocks=25, weights={**base_w, "momentum_20": 1.0}, min_mv=2e9, min_amt=5e6, exclude_prefix=("92", "83", "43", "87", "88"))),
        ("低波加强 w-0.8", dict(n_stocks=25, weights={**base_w, "volatility_20": -0.8}, min_mv=2e9, min_amt=5e6, exclude_prefix=("92", "83", "43", "87", "88"))),
        ("小市值减弱 -0.6", dict(n_stocks=25, weights={**base_w, "ln_mv": -0.6}, min_mv=2e9, min_amt=5e6, exclude_prefix=("92", "83", "43", "87", "88"))),
        ("n30+剔北交所+50亿", dict(n_stocks=30, weights=base_w, min_mv=5e9, min_amt=5e6, exclude_prefix=("92", "83", "43", "87", "88"))),
    ]
    if ind_map:
        experiments.append(
            ("行业约束15% n30", dict(n_stocks=30, weights=base_w, min_mv=2e9, min_amt=5e6, exclude_prefix=("92", "83", "43", "87", "88"), max_ind_w=0.15, ind_map=ind_map))
        )

    print(f"\n{'实验':<22} {'总收益':>8} {'年化':>8} {'最大回撤':>8} {'夏普':>7} {'超额':>8}")
    print("-" * 72)
    results = []
    for name, kw in experiments:
        st, _ = quick_backtest(hist, names, bench, cfg, **kw)
        print(
            f"{name:<22} {st.get('总收益',0):>7.2%} {st.get('年化收益',0):>7.2%} "
            f"{st.get('最大回撤',0):>7.2%} {st.get('夏普比率',0):>7.2f} {st.get('累计超额',0):>7.2%}"
        )
        results.append((name, st))

    out = pd.DataFrame([{ "实验": n, **{k: st.get(k) for k in ["总收益","年化收益","最大回撤","夏普比率","累计超额","年化超额"]} } for n, st in results])
    out.to_csv(ROOT / "output" / "experiments.csv", index=False, encoding="utf-8-sig")
    print(f"\nsaved output/experiments.csv")


if __name__ == "__main__":
    main()
