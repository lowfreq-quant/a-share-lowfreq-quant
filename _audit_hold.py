"""估算持仓期内「送转/拆股」未复权造成的假亏损"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data import fetch_index_hist, load_config  # noqa: E402
from src.regime import classify_regime  # noqa: E402

cfg = load_config(ROOT / "config_choppy_v2.yaml")
picks = {pd.Timestamp(k): v for k, v in json.loads((ROOT / "output_audit" / "picks.json").read_text(encoding="utf-8")).items()}
bench = fetch_index_hist("000300", cfg["data"]["start_date"], None, ROOT / "data")["close"]

hm = {}
for f in sorted((ROOT / "data" / "hist").glob("*.parquet")):
    try:
        hm[f.name.split("_")[0]] = pd.read_parquet(f, columns=["close", "outstanding_share"])
    except Exception:
        pass
all_dates = sorted({d for h in hm.values() for d in h.index})
start = pd.to_datetime(cfg["backtest"]["start_date"])
all_dates = pd.DatetimeIndex([d for d in all_dates if d >= start])
regime = classify_regime(bench)["regime"].reindex(all_dates).ffill()

sig = pd.Series(all_dates, index=all_dates).groupby([all_dates.year, all_dates.month]).tail(1)
exec_of = {}
for s in pd.DatetimeIndex(sig):
    later = all_dates[all_dates > s]
    exec_of[s] = later[0] if len(later) else None
exec_days = {v: k for k, v in exec_of.items() if v is not None}

# 复刻 E0 持仓时间线
holdings = {}
timeline = []
for dt in all_dates:
    if not bool(regime.loc[dt] == "震荡"):
        holdings = {}
    s = exec_days.get(dt)
    if s is not None and bool(regime.loc[dt] == "震荡") and picks.get(s):
        holdings = {c: 1 / len(picks[s]) for c in picks[s]}
    if holdings:
        timeline.append((dt, dict(holdings)))

print("持仓日", len(timeline))
events = []
for dt, hold in timeline:
    for c in hold:
        h = hm.get(c)
        if h is None or dt not in h.index:
            continue
        pos = h.index.get_loc(dt)
        if pos == 0:
            continue
        r = float(h["close"].iloc[pos] / h["close"].iloc[pos - 1] - 1)
        ds = h["outstanding_share"]
        dshare = float(ds.iloc[pos] / ds.iloc[pos - 1] - 1) if pd.notna(ds.iloc[pos - 1]) and ds.iloc[pos - 1] else 0.0
        if r < -0.15 and dshare > 0.05:
            events.append((dt, c, r, dshare, hold[c], hold[c] * (-r)))
print("持仓期内送转事件:", len(events))
for e in events:
    dt, c, r, dshare, w, loss = e
    print(" ", dt.date(), c, f"价格{r:+.1%} 股本{dshare:+.0%} 权重{w:.2%} 假亏{loss:+.2%}")
print("合计假亏损约:", f"{sum(e[5] for e in events):.2%}")
