"""C3：用修正后的数据重跑 choppy v2（hfq + 退市股 + 不用当前 ST 过滤）

用法：python research/corrected_experiment.py [--quick]
输出：output_audit/corrected_results.json、output_audit/corrected_report.md
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest  # noqa: E402
from src.data import (  # noqa: E402
    download_universe_history,
    fetch_index_hist,
    fetch_stock_list,
    load_config,
    load_delisted_histories,
)
from src.regime import classify_regime  # noqa: E402
from src.report import performance_stats, yearly_returns  # noqa: E402

OUT = ROOT / "output_audit"
OUT.mkdir(exist_ok=True)


def split_stats(nav: pd.Series) -> dict:
    out = {}
    for label, a, b in [("2021-07_2023", "2021-07-01", "2023-12-31"),
                        ("2024_2026", "2024-01-01", "2026-12-31")]:
        s = nav.loc[a:b]
        if len(s) > 1:
            out[label] = float(s.iloc[-1] / s.iloc[0] - 1)
    return out


def main() -> None:
    quick = "--quick" in sys.argv
    no_hfq = "--no-hfq" in sys.argv
    if no_hfq:
        import src.data as _d

        _d._with_hfq_or_split_adj = lambda df, code, cache: _d.add_adjusted_close(df)
        print(">> 使用送转复权（--no-hfq），隔离退市/ST 修正的影响", flush=True)
    cfg = load_config(ROOT / "config_choppy_v2.yaml")
    cfg["backtest"]["include_delisted"] = True
    cfg["backtest"]["point_in_time_st"] = False
    cache = ROOT / cfg["data"]["cache_dir"]

    names = fetch_stock_list(cache)
    codes = names["code"].tolist()
    start = cfg["data"]["start_date"]
    print(f">> 加载当前 {len(codes)} 只 …", flush=True)
    hist = download_universe_history(codes, start, None, cache, sleep=0, progress_every=1000)
    delisted = load_delisted_histories(cache)
    hist.update(delisted)
    hfq_n = len(list((cache / "hist_hfq").glob("*.parquet")))
    print(f"   当前 {len(hist) - len(delisted)} + 退市 {len(delisted)}；hfq 覆盖 {hfq_n} 只", flush=True)

    bench = fetch_index_hist(cfg["backtest"]["benchmark"], start, None, cache)["close"]
    regime = classify_regime(bench)["regime"]

    variants = [
        ("v2_50亿_n25", 5e9, 25),
        ("50亿_n40", 5e9, 40),
        ("30亿_n30", 3e9, 30),
    ]
    if quick:
        variants = variants[:1]

    results = {}
    for name, mv, n in variants:
        c = copy.deepcopy(cfg)
        c["universe"]["min_total_mv"] = mv
        c["portfolio"]["n_stocks"] = n
        print(f"\n>> 回测 {name} (mv>={mv/1e8:.0f}亿, n={n}) …", flush=True)
        nav, trades = run_backtest(hist, names, c, val_map=None, regime_series=regime)
        s = performance_stats(nav["nav"], bench)
        s["split"] = split_stats(nav["nav"])
        s["yearly"] = {int(y): float(v) for y, v in yearly_returns(nav["nav"]).items()}
        s["cost"] = float(trades["cost"].sum()) if len(trades) else 0.0
        results[name] = s
        nav.to_csv(OUT / f"nav_{name}.csv")
        print(f"   总收益 {s['总收益']:+.2%} 年化 {s['年化收益']:+.2%} 夏普 {s['夏普比率']:.2f} "
              f"回撤 {s['最大回撤']:.2%} 2024后 {s['split'].get('2024_2026', float('nan')):+.2%}", flush=True)

    (OUT / "corrected_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n写出 {OUT / 'corrected_results.json'}")


if __name__ == "__main__":
    main()
