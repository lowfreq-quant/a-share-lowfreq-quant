"""小样本冒烟测试：验证新执行口径不崩、成本/换手方向正确"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest

rng = np.random.default_rng(0)
dates = pd.bdate_range("2023-01-02", periods=90)
hist = {}
for i in range(30):
    code = f"{i:06d}"
    ret = rng.normal(0.0005, 0.015, len(dates))
    close = 10 * np.cumprod(1 + ret)
    openp = close * (1 + rng.normal(0, 0.003, len(dates)))
    df = pd.DataFrame(
        {
            "open": openp,
            "close": close,
            "amount": np.full(len(dates), 1e8),
            "outstanding_share": np.full(len(dates), 1e8),
            "turnover": np.full(len(dates), 0.01),
        },
        index=dates,
    )
    df["total_mv"] = df["close"] * df["outstanding_share"]
    df["adj_close"] = df["close"]
    hist[code] = df

names = pd.DataFrame({"code": [f"{i:06d}" for i in range(30)], "name": ["x"] * 30})
cfg = {
    "universe": {"min_list_days": 20, "min_avg_amount": 1e6, "min_total_mv": 1e8, "exclude_prefixes": []},
    "factors": {"weights": {"ln_mv": -1, "momentum_20": 0.5, "volatility_20": -0.4, "turnover_20": -0.4}, "winsorize": 0.02},
    "portfolio": {"n_stocks": 5, "weight": "equal"},
    "cost": {"commission": 0.00025, "stamp_duty": 0.0005, "slippage": 0.0015},
    "risk": {"drawdown_reduce": 0.15, "drawdown_halt": 0.20},
    "backtest": {"start_date": "2023-01-02", "max_stocks": None, "only_regime": None},
    "data": {"cache_dir": "data"},
}
nav, trades = run_backtest(hist, names, cfg)
print(nav.tail(3))
print(trades.head(10))
print("nav rows", len(nav), "trades", len(trades))
assert nav["nav"].iloc[-1] > 0
assert trades["cost"].sum() >= 0
print("OK")
