"""执行口径拆解（open 价）

E0 原版：建仓日按「昨收→今收」全额计收益；转出日不计收益
E2 真实：信号日收盘定市况 → 次日开盘建仓/调仓 → 市况转出后次日开盘卖出
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

hm = {}
for f in sorted((ROOT / "data" / "hist").glob("*.parquet")):
    try:
        hm[f.name.split("_")[0]] = pd.read_parquet(f, columns=["open", "close", "turnover", "outstanding_share", "amount"])
    except Exception:
        pass
all_dates = sorted({d for h in hm.values() for d in h.index})
start = pd.to_datetime(cfg["backtest"]["start_date"])
all_dates = pd.DatetimeIndex([d for d in all_dates if d >= start])
close = pd.DataFrame({c: h["close"] for c, h in hm.items()}).reindex(all_dates)
openp = pd.DataFrame({c: h["open"] for c, h in hm.items()}).reindex(all_dates)
prev_close = close.shift(1)
ret = close.pct_change()
regime = classify_regime(bench)["regime"].reindex(all_dates).ffill()

sig = pd.Series(all_dates, index=all_dates).groupby([all_dates.year, all_dates.month]).tail(1)
exec_of = {}
for s in pd.DatetimeIndex(sig):
    later = all_dates[all_dates > s]
    exec_of[s] = later[0] if len(later) else None
exec_days = {v: k for k, v in exec_of.items() if v is not None}

ccfg = cfg["cost"]
fee = ccfg["commission"] + ccfg["slippage"]
stamp = ccfg["stamp_duty"]


def val(df, c, dt):
    try:
        v = df.at[dt, c]
    except Exception:
        return np.nan
    return float(v) if pd.notna(v) else np.nan


def st(df):
    nav = df["nav"]
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0]
    cagr = total ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    b = bench.reindex(nav.index).ffill()
    ex = (nav / nav.iloc[0]) / (b / b.iloc[0]) - 1
    return dict(total=total - 1, cagr=cagr, vol=vol, sharpe=cagr / vol,
                mdd=(1 - nav / nav.cummax()).max(), excess=ex.iloc[-1],
                exposure=float((df["inv"] > 0).mean()))


# ---------------- E0 原版 ----------------
def run_e0():
    holdings, nav, peak, scale, halted = {}, 1.0, 1.0, 1.0, False
    rows = []
    for dt in all_dates:
        if not bool(regime.loc[dt] == "震荡") and holdings:
            nav *= 1 - sum(holdings.values()) * (fee + stamp)
            holdings = {}
        s = exec_days.get(dt)
        if s is not None and not halted and bool(regime.loc[dt] == "震荡"):
            top = picks.get(s, [])
            if top:
                w = scale / len(top)
                new = {c: w for c in top}
                allc = set(new) | set(holdings)
                turnover = sum(abs(new.get(c, 0) - holdings.get(c, 0)) for c in allc)
                sell_w = sum(holdings.get(c, 0) for c in set(holdings) - set(new))
                nav *= 1 - (turnover * fee + sell_w * stamp)
                holdings = new
        dr = sum(w * (float(ret.at[dt, c]) if pd.notna(ret.at[dt, c]) else 0.0) for c, w in holdings.items())
        nav *= 1 + dr
        peak = max(peak, nav)
        dd = 1 - nav / peak
        if dd >= 0.20 and not halted:
            holdings, halted = {}, True
        elif dd >= 0.15 and scale == 1.0 and not halted:
            scale = 0.5
            holdings = {c: w * 0.5 for c, w in holdings.items()}
        rows.append((dt, nav, dd, sum(holdings.values())))
    return pd.DataFrame(rows, columns=["date", "nav", "dd", "inv"]).set_index("date")


# ---------------- E2 真实 ----------------
def run_e2():
    holdings, nav, peak, scale, halted = {}, 1.0, 1.0, 1.0, False
    pending_exit = False
    rows = []
    tot_gap = tot_intra = 0.0
    for dt in all_dates:
        # 昨日收盘确认转出 → 今开卖出
        if pending_exit and holdings:
            gap = 0.0
            for c, w in holdings.items():
                o, pc = val(openp, c, dt), val(prev_close, c, dt)
                if pd.notna(o) and pd.notna(pc) and pc:
                    gap += w * (o / pc - 1)
            nav *= 1 + gap
            nav *= 1 - sum(holdings.values()) * (fee + stamp)
            holdings = {}
            pending_exit = False

        s = exec_days.get(dt)
        if s is not None and not halted and holdings:
            # 调仓：今开换仓
            new = picks.get(s, []) if bool(regime.loc[s] == "震荡") else []
            gap = 0.0
            intra = 0.0
            if new:
                w = scale / len(new)
                newh = {c: w for c in new}
                for c, w0 in holdings.items():
                    o, pc = val(openp, c, dt), val(prev_close, c, dt)
                    if pd.notna(o) and pd.notna(pc) and pc:
                        gap += w0 * (o / pc - 1)
                for c, w1 in newh.items():
                    o, cl = val(openp, c, dt), val(close, c, dt)
                    if pd.notna(o) and pd.notna(cl) and o:
                        intra += w1 * (cl / o - 1)
                allc = set(newh) | set(holdings)
                turnover = sum(abs(newh.get(c, 0) - holdings.get(c, 0)) for c in allc)
                sell_w = sum(holdings.get(c, 0) for c in set(holdings) - set(newh))
                nav *= 1 + gap
                nav *= 1 - (turnover * fee + sell_w * stamp)
                nav *= 1 + intra
                tot_gap += gap
                tot_intra += intra
                holdings = newh
            else:
                # 不再持有：今开清仓
                for c, w0 in holdings.items():
                    o, pc = val(openp, c, dt), val(prev_close, c, dt)
                    if pd.notna(o) and pd.notna(pc) and pc:
                        gap += w0 * (o / pc - 1)
                nav *= 1 + gap
                nav *= 1 - sum(holdings.values()) * (fee + stamp)
                holdings = {}
        elif s is not None and not halted and not holdings:
            # 空仓建仓：今开买入
            if bool(regime.loc[s] == "震荡"):
                top = picks.get(s, [])
                if top:
                    w = scale / len(top)
                    hold = {c: w for c in top}
                    intra = 0.0
                    for c, w1 in hold.items():
                        o, cl = val(openp, c, dt), val(close, c, dt)
                        if pd.notna(o) and pd.notna(cl) and o:
                            intra += w1 * (cl / o - 1)
                    nav *= 1 - sum(hold.values()) * fee
                    nav *= 1 + intra
                    tot_intra += intra
                    holdings = hold
        else:
            dr = sum(w * (float(ret.at[dt, c]) if pd.notna(ret.at[dt, c]) else 0.0) for c, w in holdings.items())
            nav *= 1 + dr

        if holdings and not bool(regime.loc[dt] == "震荡"):
            pending_exit = True

        peak = max(peak, nav)
        dd = 1 - nav / peak
        if dd >= 0.20 and not halted:
            holdings, halted = {}, True
        elif dd >= 0.15 and scale == 1.0 and not halted:
            scale = 0.5
            holdings = {c: w * 0.5 for c, w in holdings.items()}
        rows.append((dt, nav, dd, sum(holdings.values())))
    return pd.DataFrame(rows, columns=["date", "nav", "dd", "inv"]).set_index("date"), tot_gap, tot_intra


e0 = run_e0()
e2, gap, intra = run_e2()
print("E0", {k: round(float(v), 4) for k, v in st(e0).items()})
print("E2", {k: round(float(v), 4) for k, v in st(e2).items()})
print(f"E2 建仓日分解：隔夜跳空 {gap:+.4f} / 日内 {intra:+.4f}")
e0.to_csv(OUT / "nav_E0.csv")
e2.to_csv(OUT / "nav_E2.csv")

for name, df in [("E0", e0), ("E2", e2)]:
    print(name, "yearly:", {int(y): round(float(g.nav.iloc[-1] / g.nav.iloc[0] - 1), 4) for y, g in df.groupby(df.index.year) if len(g) > 1})

(OUT / "exec_result.json").write_text(json.dumps({"E0": st(e0), "E2": st(e2), "gap": gap, "intra": intra}, ensure_ascii=False, indent=2), encoding="utf-8")

