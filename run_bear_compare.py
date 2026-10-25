"""熊市防守组合：几种权重对比"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_index_hist, load_config
from src.etf_portfolio import backtest_etf_portfolio, load_etf_map
from src.regime import classify_regime
from src.report import performance_stats

cfg = load_config(ROOT / "config_bear.yaml")
cache_dir = ROOT / cfg["data"]["cache_dir"]
etf_cfg = {str(k): str(v) for k, v in cfg["etf"].items()}
px = load_etf_map(etf_cfg, cfg["data"]["start_date"], None, cache_dir)
bench = fetch_index_hist("000300", cfg["data"]["start_date"], None, cache_dir)["close"]
regime = classify_regime(bench)["regime"]

mixes = {
    "60%红利+40%国债(全程)": {"510880": 0.30, "512890": 0.30, "511260": 0.40},
    "30%红利+70%国债(全程)": {"510880": 0.15, "512890": 0.15, "511260": 0.70},
    "60%红利+40%国债(仅熊市)": {"510880": 0.30, "512890": 0.30, "511260": 0.40},
    "30%红利+70%国债(仅熊市)": {"510880": 0.15, "512890": 0.15, "511260": 0.70},
    "纯国债(全程)": {"511260": 1.0},
}

print(f"{'组合':<28} {'总收益':>8} {'年化':>8} {'最大回撤':>8} {'夏普':>7} {'仓位天数%':>10}")
print("-" * 80)
for name, w in mixes.items():
    only = "熊市" if "仅熊市" in name else None
    cfg2 = dict(cfg)
    cfg2["backtest"] = dict(cfg["backtest"])
    cfg2["backtest"]["only_regime"] = only
    nav = backtest_etf_portfolio(px, w, cfg2, regime)
    st = performance_stats(nav["nav"], bench)
    inv = float(nav["invested"].mean())
    print(
        f"{name:<28} {st.get('总收益',0):>7.2%} {st.get('年化收益',0):>7.2%} "
        f"{st.get('最大回撤',0):>7.2%} {st.get('夏普比率',0):>7.2f} {inv:>9.0%}"
    )
