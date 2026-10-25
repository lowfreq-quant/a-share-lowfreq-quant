"""审计：复现 v2 回测 + 量化执行偏差

用法: python _audit_run.py
输出: output_audit/，不覆盖原 output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_index_hist, fetch_stock_list, load_config  # noqa: E402
from src.portfolio import load_industry_map, select_top  # noqa: E402
from src.regime import classify_regime  # noqa: E402

OUT = ROOT / "output_audit"
OUT.mkdir(exist_ok=True)


def month_end_signals(all_dates):
    s = pd.Series(all_dates, index=all_dates)
    return pd.DatetimeIndex(s.groupby([all_dates.year, all_dates.month]).tail(1))


def load_hist_map(cache_dir: Path):
    hm = {}
    for f in sorted((cache_dir / "hist").glob("*.parquet")):
        code = f.name.split("_")[0]
        try:
            hm[code] = pd.read_parquet(f, columns=["open", "close", "amount", "outstanding_share", "turnover"])
        except Exception:
            pass
    return hm


def stats(nav: pd.Series, bench: pd.Series | None = None):
    r = nav.pct_change().dropna()
    n_years = len(r) / 252
    total = nav.iloc[-1] / nav.iloc[0]
    cagr = total ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(252)
    out = {
        "total": total - 1,
        "cagr": cagr,
        "vol": vol,
        "sharpe": cagr / vol if vol else np.nan,
        "mdd": (1 - nav / nav.cummax()).max(),
        "win": (r > 0).mean(),
    }
    if bench is not None:
        b = bench.reindex(nav.index).ffill()
        ex = nav / nav.iloc[0] / (b / b.iloc[0]) - 1
        out["excess"] = ex.iloc[-1]
    return out


def main():
    cfg = load_config(ROOT / "config_choppy_v2.yaml")
    names = fetch_stock_list(ROOT / "data")
    hm = load_hist_map(ROOT / "data")
    print(f"hist {len(hm)}", flush=True)

    start = pd.to_datetime(cfg["backtest"]["start_date"])
    all_dates = sorted({d for h in hm.values() for d in h.index})
    all_dates = pd.DatetimeIndex([d for d in all_dates if d >= start])
    print(f"dates {len(all_dates)}", flush=True)

    bench = fetch_index_hist("000300", cfg["data"]["start_date"], None, ROOT / "data")["close"]
    regime = classify_regime(bench)["regime"].reindex(all_dates).ffill()
    print("regime share:", {k: round(v, 3) for k, v in regime.value_counts(normalize=True).items()}, flush=True)

    close = pd.DataFrame({c: h["close"] for c, h in hm.items()}).reindex(all_dates)
    ret = close.pct_change()

    sig_days = month_end_signals(all_dates)
    exec_of = {}
    for s in sig_days:
        later = all_dates[all_dates > s]
        exec_of[s] = later[0] if len(later) else None

    # ---- 选股缓存 ----
    cache_path = OUT / "picks.json"
    if cache_path.exists():
        picks = {pd.Timestamp(k): v for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
        print(f"picks loaded {len(picks)}", flush=True)
    else:
        industry_map = load_industry_map(ROOT / "data")
        picks = {}
        for i, s in enumerate(sig_days, 1):
            p = select_top(s, hm, names, cfg, industry_map=industry_map)
            picks[s] = p
            print(f"  pick {i}/{len(sig_days)} {s.date()} n={len(p)}", flush=True)
        cache_path.write_text(json.dumps({str(k): v for k, v in picks.items()}, ensure_ascii=False))

    ccfg = cfg["cost"]
    fee_side = ccfg["commission"] + ccfg["slippage"]
    stamp = ccfg["stamp_duty"]

    def r_at(c, dt):
        v = ret.at[dt, c] if c in ret.columns else np.nan
        return float(v) if pd.notna(v) else 0.0

    # ================= 基线：完全照原代码 =================
    holdings, nav, peak, scale, halted = {}, 1.0, 1.0, 1.0, False
    rows = []
    entry_contrib = []   # 新买入标的在执行日“昨收→今收”的收益（实盘吃不到）
    exit_skip = []       # 清仓日被跳过的当日收益
    for dt in all_dates:
        if not bool(regime.loc[dt] == "震荡") and holdings:
            # 原代码：先按当日收盘清仓、记成本，之后才计算当日收益 → 当日收益被跳过
            sell_w = sum(holdings.values())
            nav *= 1 - sell_w * (fee_side + stamp)
            exit_skip.append(sum(w * r_at(c, dt) for c, w in holdings.items()))
            holdings = {}
        ex = [s for s, e in exec_of.items() if e == dt]
        if ex and not halted and bool(regime.loc[dt] == "震荡"):
            top = picks.get(ex[0], [])
            if top:
                w = scale / len(top)
                new = {c: w for c in top}
                old = holdings
                new_codes = set(new) - set(old)
                allc = set(new) | set(old)
                turnover = sum(abs(new.get(c, 0) - old.get(c, 0)) for c in allc)
                sell_w = sum(old.get(c, 0) for c in set(old) - set(new))
                nav *= 1 - (turnover * fee_side + sell_w * stamp)
                entry_contrib.append(sum(new[c] * r_at(c, dt) for c in new_codes))
                holdings = new
        dr = sum(w * r_at(c, dt) for c, w in holdings.items())
        nav *= 1 + dr
        peak = max(peak, nav)
        dd = 1 - nav / peak
        if dd >= 0.20 and not halted:
            holdings, halted = {}, True
        elif dd >= 0.15 and scale == 1.0 and not halted:
            scale = 0.5
            holdings = {c: w * 0.5 for c, w in holdings.items()}
        rows.append((dt, nav, dd, sum(holdings.values()), len(holdings)))
    base = pd.DataFrame(rows, columns=["date", "nav", "dd", "invested", "n"]).set_index("date")
    base.to_csv(OUT / "nav_baseline.csv")

    # ================= 修正版：无前视 =================
    # 1) 建仓/持有判断用信号日 s 的市况（s 收盘后已知），不用执行日收盘
    # 2) 执行日收盘成交：新买入标的从次日起才计收益
    # 3) 转出在 t 日收盘确认，按 t 日收盘卖出：当日涨跌照吃
    holdings, nav, peak, scale, halted = {}, 1.0, 1.0, 1.0, False
    rows2 = []
    for dt in all_dates:
        # 当日收益先按「昨日持仓」结算
        dr = sum(w * r_at(c, dt) for c, w in holdings.items())
        nav *= 1 + dr
        # 收盘后：市况转出 → 清仓
        if holdings and not bool(regime.loc[dt] == "震荡"):
            sell_w = sum(holdings.values())
            nav *= 1 - sell_w * (fee_side + stamp)
            holdings = {}
        # 收盘后：调仓（用信号日 s 的市况决定是否持有）
        ex = [s for s, e in exec_of.items() if e == dt]
        if ex and not halted:
            s = ex[0]
            if bool(regime.loc[s] == "震荡"):
                top = picks.get(s, [])
                if top:
                    w = scale / len(top)
                    new = {c: w for c in top}
                    allc = set(new) | set(holdings)
                    turnover = sum(abs(new.get(c, 0) - holdings.get(c, 0)) for c in allc)
                    sell_w = sum(holdings.get(c, 0) for c in set(holdings) - set(new))
                    nav *= 1 - (turnover * fee_side + sell_w * stamp)
                    holdings = new
        peak = max(peak, nav)
        dd = 1 - nav / peak
        if dd >= 0.20 and not halted:
            holdings, halted = {}, True
        elif dd >= 0.15 and scale == 1.0 and not halted:
            scale = 0.5
            holdings = {c: w * 0.5 for c, w in holdings.items()}
        rows2.append((dt, nav, dd, sum(holdings.values()), len(holdings)))
    corr = pd.DataFrame(rows2, columns=["date", "nav", "dd", "invested", "n"]).set_index("date")
    corr.to_csv(OUT / "nav_corrected.csv")

    res = {
        "baseline": stats(base["nav"], bench),
        "corrected": stats(corr["nav"], bench),
        "baseline_yearly": {int(y): float(g["nav"].iloc[-1] / g["nav"].iloc[0] - 1) for y, g in base.groupby(base.index.year) if len(g) > 1},
        "corrected_yearly": {int(y): float(g["nav"].iloc[-1] / g["nav"].iloc[0] - 1) for y, g in corr.groupby(corr.index.year) if len(g) > 1},
        "baseline_exposure": float((base["invested"] > 0).mean()),
        "corrected_exposure": float((corr["invested"] > 0).mean()),
        "n_rebalances": int(len(entry_contrib)),
        "sum_entry_day_ret": float(np.sum(entry_contrib)),
        "mean_entry_day_ret": float(np.mean(entry_contrib)) if entry_contrib else None,
        "sum_skipped_exit_day_ret": float(np.sum(exit_skip)),
        "n_exit_skips": int(len(exit_skip)),
    }
    (OUT / "audit_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)

    inv = base["invested"] > 0
    eps, st, last = [], None, None
    for d, v in inv.items():
        if v and st is None:
            st = d
        if not v and st is not None:
            eps.append((st, last))
            st = None
        last = d
    if st is not None:
        eps.append((st, last))
    print(f"\n{len(eps)} episodes; avg calendar days", round(np.mean([(b - a).days for a, b in eps]), 1) if eps else None)
    for a, b in eps:
        print(" ", a.date(), "→", b.date(), (b - a).days, "days")


if __name__ == "__main__":
    main()

