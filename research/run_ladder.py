import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from backtest_etf import load_prices, run_strategy, stats, yearly

px = load_prices(["510880", "511260", "518880", "513100", "510300", "512890"], start="2017-08-24")
cases = {
    "防守 30/50/20": {"510880": 0.30, "511260": 0.50, "518880": 0.20},
    "平衡 30/40/20/10": {"510880": 0.30, "511260": 0.40, "518880": 0.20, "513100": 0.10},
    "进取 30/30/20/20": {"510880": 0.30, "511260": 0.30, "518880": 0.20, "513100": 0.20},
    "进取 40/30/20/10": {"510880": 0.40, "511260": 0.30, "518880": 0.20, "513100": 0.10},
    "现有 15/15/70": {"510880": 0.15, "512890": 0.15, "511260": 0.70},
}
for nm, w in cases.items():
    w = {k: v for k, v in w.items() if k in px.columns}
    r = run_strategy(px, nm, "fixed", fixed_weights=w, cost_side=0.0004, rebalance="Q")
    s = stats(r.nav, px["510300"])
    yr = yearly(r.nav)
    print(
        f"{nm:20s} cagr={s['cagr']:.2%} vol={s['vol']:.2%} sharpe={s['sharpe']:.2f} "
        f"mdd={s['mdd']:.2%} calmar={s['calmar']:.2f} cost/yr={r.cost / s['years']:.3%}"
    )
    print("    ", {y: f"{v:+.1%}" for y, v in yr.items()})
