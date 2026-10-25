"""防守型多资产组合：回测 / 建仓 / 估值 / 再平衡

策略（见 config_defensive.yaml）：
  红利ETF 30% + 十年国债ETF 50% + 黄金ETF 20%
  每季度检查，偏离目标 5 个百分点才再平衡。
  不做市况择时，任何市况都持有——牛市中会跑输指数，这是设计选择。

用法：
  python run_defensive.py backtest
  python run_defensive.py init 1000000
  python run_defensive.py mark
  python run_defensive.py signal
  python run_defensive.py rebalance
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from src.data import load_config  # noqa: E402

BOOK = ROOT / "output" / "防守组合_台账.csv"
STATE = ROOT / "output" / "防守组合_状态.json"


# ---------------- 数据 ----------------
def live_px(code: str) -> tuple[float, pd.Timestamp, pd.DataFrame]:
    """腾讯不复权日线：最新价 + 全部历史（用于判断新鲜度）"""
    import akshare as ak

    sym = ("sh" if str(code)[0] in "56" else "sz") + str(code).zfill(6)
    end = pd.Timestamp.today().strftime("%Y%m%d")
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()):  # 屏蔽 tqdm 进度条
        df = ak.stock_zh_a_hist_tx(symbol=sym, start_date="20260101", end_date=end, adjust="")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return float(df["close"].iloc[-1]), df.index[-1], df


def check_fresh(dates: list[pd.Timestamp]) -> bool:
    last = max(dates)
    gap = (pd.Timestamp.today().normalize() - last.normalize()).days
    return gap <= 7


def target_weights(cfg: dict) -> dict[str, float]:
    w = {str(k): float(v) for k, v in cfg["weights"].items()}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    limit = (cfg.get("risk") or {}).get("max_single_weight")
    if limit and max(w.values()) > float(limit) + 1e-9:
        raise SystemExit(f"单一资产权重 {max(w.values()):.0%} 超过上限 {float(limit):.0%}")
    return w


def build_portfolio(cfg: dict):
    """用后复权 ETF 回测 30/50/20，返回 Result + 价格矩阵"""
    from backtest_etf import load_prices, run_strategy

    codes = [str(c) for c in cfg["etf"].keys()]
    px = load_prices(codes, start=cfg["data"]["start_date"])
    res = run_strategy(px, "def", "fixed", fixed_weights=target_weights(cfg),
                       rebalance="Q", cost_side=0.0004)
    return res, px


def target_exposure(cfg: dict) -> tuple[float, float]:
    """波动率目标：min(1, target_vol / 近 lookback 日组合波动)，有下限。"""
    ov = cfg.get("risk_overlay") or {}
    if not ov.get("enabled"):
        return 1.0, float("nan")
    res, _ = build_portfolio(cfg)
    ret = res.nav.pct_change().dropna()
    lb = int(ov.get("vol_lookback", 60))
    vol = float(ret.tail(lb).std() * np.sqrt(252))
    if not np.isfinite(vol) or vol <= 0:
        return 1.0, float("nan")
    exp = float(np.clip(float(ov["vol_target"]) / vol, float(ov.get("min_exposure", 0.5)), 1.0))
    return exp, vol


def _vol_target_nav(nav: pd.Series, target: float, lookback: int, cash: float,
                    floor: float, cost: float = 0.0005) -> pd.Series:
    ret = nav.pct_change().fillna(0.0)
    vol = ret.rolling(lookback).std() * np.sqrt(252)
    dates = nav.index
    me = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).tail(1).index
    exp = pd.Series(1.0, index=dates)
    cur = 1.0
    for dt in dates:
        if dt in me:
            v = vol.loc[dt]
            cur = float(np.clip(target / v, floor, 1.0)) if pd.notna(v) and v > 0 else 1.0
        exp.loc[dt] = cur
    exp_lag = exp.shift(1).fillna(1.0)
    r = exp_lag * ret + (1 - exp_lag) * cash / 252.0
    r = r - exp_lag.diff().abs().fillna(0.0) * cost
    return (1 + r).cumprod()


# ---------------- 回测 ----------------
def backtest() -> None:
    from backtest_etf import load_prices, stats, yearly

    cfg = load_config(ROOT / "config_defensive.yaml")
    res, px = build_portfolio(cfg)
    bench = load_prices(["510300"], start=cfg["data"]["start_date"])["510300"]
    s = stats(res.nav, bench)
    print("=" * 60)
    print(f"{cfg['name']}  {px.index.min().date()} → {px.index.max().date()}")
    print("=" * 60)
    print(f"基准（30/50/20，季度）  年化 {s['cagr']:.2%}  波动 {s['vol']:.2%}  夏普 {s['sharpe']:.2f}  最大回撤 {s['mdd']:.2%}")
    print(f"累计 {s['total']:.2%}   Calmar {s['calmar']:.2f}   年换手 {res.turnover / s['years']:.2f}   年成本 {res.cost / s['years']:.3%}")

    ov = cfg.get("risk_overlay") or {}
    if ov.get("enabled"):
        nav2 = _vol_target_nav(res.nav, float(ov["vol_target"]), int(ov["vol_lookback"]),
                               float(ov.get("cash_yield", 0.0)), float(ov.get("min_exposure", 0.5)))
        s2 = stats(nav2, bench)
        print(f"\n+ 波动率目标 {float(ov['vol_target']):.0%}/{int(ov['vol_lookback'])}日  "
              f"年化 {s2['cagr']:.2%}  波动 {s2['vol']:.2%}  夏普 {s2['sharpe']:.2f}  最大回撤 {s2['mdd']:.2%}")
        print(f"累计 {s2['total']:.2%}   Calmar {s2['calmar']:.2f}")

    print("\n分年（基准 / 叠加层）：")
    y1 = yearly(res.nav)
    y2 = yearly(nav2) if ov.get("enabled") else {}
    for y, v in y1.items():
        extra = f"   {y2.get(y, float('nan')):+.2%}" if y2 else ""
        print(f"  {y}  {v:+.2%}{extra}")
    print("\n当前权重（按最新净值漂移）：")
    last = res.weights.iloc[-1]
    for c, v in last.items():
        if v > 0.001:
            print(f"  {c}  {v:.1%}")
    if ov.get("enabled"):
        exp, vol = target_exposure(cfg)
        print(f"\n当前组合波动 {vol:.1%}  →  目标仓位 {exp:.0%}（其余现金/货基）")


# ---------------- 台账 ----------------
def load_book() -> pd.DataFrame | None:
    if not BOOK.exists():
        return None
    df = pd.read_csv(BOOK, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def load_state() -> dict:
    if not STATE.exists():
        return {"init_capital": 0.0, "cash": 0.0, "last_rebalance": None}
    return json.loads(STATE.read_text(encoding="utf-8"))


def save(book: pd.DataFrame, state: dict) -> None:
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    book.to_csv(BOOK, index=False, encoding="utf-8-sig")
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def init(capital: float) -> None:
    cfg = load_config(ROOT / "config_defensive.yaml")
    w = target_weights(cfg)
    exp, vol = target_exposure(cfg)
    names = {str(k): str(v) for k, v in cfg["etf"].items()}
    cost_rate = float(cfg["cost"]["commission"]) + float(cfg["cost"]["slippage"])

    px, dates = {}, {}
    for c in w:
        p, d, _ = live_px(c)
        px[c], dates[c] = p, d
    if not check_fresh(list(dates.values())):
        print(f"⚠ 最新行情只到 {max(dates.values()).date()}，超过 7 天，拒绝建仓。")
        return

    if exp < 1.0:
        print(f"波动率目标：组合波动 {vol:.1%} → 总仓位 {exp:.0%}，其余 {1-exp:.0%} 现金")
    rows, invested, fees = [], 0.0, 0.0
    for c, weight in w.items():
        amount = capital * exp * weight
        lots = int(amount / (px[c] * 100))
        shares = lots * 100
        cost = shares * px[c]
        fee = cost * cost_rate
        invested += cost
        fees += fee
        rows.append(
            dict(code=c, name=names.get(c, c), shares=shares, cost_price=px[c],
                 cost_amount=cost, target_weight=exp * weight, px_date=str(dates[c].date()))
        )
        print(f"  {c} {names.get(c, c)}: {shares} 份 @ {px[c]:.3f} = {cost:,.0f}  (目标 {exp*weight:.0%})")

    cash = capital - invested - fees
    book = pd.DataFrame(rows)
    state = dict(init_capital=capital, cash=cash, last_rebalance=str(max(dates.values()).date()),
                 fees=fees, exposure=exp, vol=vol)
    save(book, state)
    print(f"\n现金 {cash:,.0f}（含费用 {fees:,.0f}）  合计 {invested + cash:,.0f}")
    print(f"台账: {BOOK}")


def mark(write: bool = True) -> pd.DataFrame | None:
    book = load_book()
    if book is None:
        print("尚无台账，请先 init。")
        return None
    cfg = load_config(ROOT / "config_defensive.yaml")
    state = load_state()
    px, dates = {}, []
    for c in book["code"]:
        p, d, _ = live_px(c)
        px[c], _ = p, d
        dates.append(d)

    book["mark_price"] = book["code"].map(px)
    book["mv"] = book["shares"] * book["mark_price"]
    book["pnl"] = book["mv"] - book["cost_amount"]
    total = float(book["mv"].sum()) + float(state.get("cash", 0.0))
    exp, vol = target_exposure(cfg)
    w = target_weights(cfg)
    targets = {c: exp * wt for c, wt in w.items()}
    book["cur_weight"] = book["mv"] / total
    book["target_weight"] = book["code"].map(targets)
    book["drift"] = book["cur_weight"] - book["target_weight"]
    cash = float(state.get("cash", 0.0))
    cash_w = cash / total if total else 0.0
    cash_target = 1.0 - exp

    print(f"{'代码':<8}{'名称':<12}{'份额':>8}{'成本':>9}{'现价':>9}{'市值':>12}{'目标':>8}{'当前':>8}{'偏离':>8}")
    for _, r in book.iterrows():
        print(f"{r['code']:<8}{str(r['name'])[:10]:<12}{int(r['shares']):>8}{r['cost_price']:>9.3f}"
              f"{r['mark_price']:>9.3f}{r['mv']:>12,.0f}{r['target_weight']:>8.1%}{r['cur_weight']:>8.1%}{r['drift']:>+8.1%}")
    print(f"{'现金':<8}{'':<12}{'':>8}{'':>9}{'':>9}{cash:>12,.0f}{cash_target:>8.1%}{cash_w:>8.1%}{cash_w-cash_target:>+8.1%}")
    print(f"\n持仓市值 {book['mv'].sum():,.0f} + 现金 {cash:,.0f} = 总净值 {total:,.0f}")
    print(f"累计盈亏 {book['pnl'].sum():+,.0f}")
    if exp < 1.0:
        print(f"波动率目标：组合波动 {vol:.1%} → 总仓位 {exp:.0%}，现金 {cash_target:.0%}")
    if not check_fresh(dates):
        print(f"⚠ 最新行情只到 {max(dates).date()}，超过 7 天，仅供估值参考。")

    band = float(cfg["rebalance"]["band"])
    need = max(float(book["drift"].abs().max()), abs(cash_w - cash_target)) >= band
    print(f"\n再平衡阈值 {band:.0%}：{'需要再平衡' if need else '暂不需要'}")
    if write:
        save(book, state)
    return book


def rebalance() -> None:
    book = load_book()
    if book is None:
        print("尚无台账，请先 init。")
        return
    cfg = load_config(ROOT / "config_defensive.yaml")
    w = target_weights(cfg)
    exp, vol = target_exposure(cfg)
    targets = {c: exp * wt for c, wt in w.items()}
    state = load_state()
    cost_rate = float(cfg["cost"]["commission"]) + float(cfg["cost"]["slippage"])

    px = {}
    dates = []
    for c in w:
        p, d, _ = live_px(c)
        px[c] = p
        dates.append(d)
    if not check_fresh(dates):
        print(f"⚠ 最新行情只到 {max(dates).date()}，超过 7 天，拒绝再平衡。")
        return

    total = float((book["shares"] * book["code"].map(px)).sum()) + float(state.get("cash", 0.0))
    cur = {r["code"]: int(r["shares"]) for _, r in book.iterrows()}
    band = float(cfg["rebalance"]["band"])
    cur_w = {c: (cur.get(c, 0) * px[c] / total if total else 0.0) for c in w}
    cash_w = float(state.get("cash", 0.0)) / total if total else 0.0
    drift = max(max(abs(cur_w[c] - targets[c]) for c in w), abs(cash_w - (1 - exp)))
    if drift < band and "--force" not in sys.argv:
        print(f"最大偏离 {drift:.1%} < 阈值 {band:.0%}，暂不需要再平衡（要强制请加 --force）。")
        return

    print(f"再平衡前总净值: {total:,.0f}   目标仓位 {exp:.0%}（组合波动 {vol:.1%}）")
    trades, fees = [], 0.0
    new_shares = {}
    for c, weight in targets.items():
        target_amount = total * weight
        tgt_lots = int(target_amount / (px[c] * 100))
        tgt = tgt_lots * 100
        delta = tgt - cur.get(c, 0)
        new_shares[c] = tgt
        if delta:
            amt = abs(delta) * px[c]
            fees += amt * cost_rate
            trades.append((c, delta, px[c], amt))
    for c, delta, p, amt in trades:
        print(f"  {c} {'买入' if delta > 0 else '卖出'} {abs(delta):>6} 份 @ {p:.3f} = {amt:,.0f}")

    # 更新成本
    rows = []
    for _, r in book.iterrows():
        c = r["code"]
        sh = new_shares.get(c, int(r["shares"]))
        cost_amt = float(r["cost_amount"])
        if sh > int(r["shares"]):  # 买入
            add = (sh - int(r["shares"])) * px[c]
            cost_amt += add
        elif sh < int(r["shares"]):  # 卖出，按平均成本减
            cost_amt *= sh / max(int(r["shares"]), 1)
        rows.append(dict(code=c, name=r["name"], shares=sh,
                         cost_price=(cost_amt / sh if sh else 0.0), cost_amount=cost_amt,
                         target_weight=targets.get(c, 0.0), px_date=str(pd.Timestamp.today().date())))
    book = pd.DataFrame(rows)
    cash = total - float(book["shares"].mul(book["code"].map(px)).sum()) - fees
    state.update(dict(cash=cash, last_rebalance=str(pd.Timestamp.today().date()), fees=fees,
                      exposure=exp, vol=vol))
    save(book, state)
    print(f"\n费用 {fees:,.0f}  剩余现金 {cash:,.0f}")
    mark(write=False)


def sheet() -> None:
    """生成月度操作单（只写 markdown，不动台账）"""
    book = load_book()
    if book is None:
        print("尚无台账，请先 init。")
        return
    cfg = load_config(ROOT / "config_defensive.yaml")
    w = target_weights(cfg)
    exp, vol = target_exposure(cfg)
    targets = {c: exp * wt for c, wt in w.items()}
    state = load_state()
    cost_rate = float(cfg["cost"]["commission"]) + float(cfg["cost"]["slippage"])

    px, dates = {}, []
    for c in w:
        p, d, _ = live_px(c)
        px[c], _ = p, d
        dates.append(d)
    asof = max(dates)
    total = float((book["shares"] * book["code"].map(px)).sum()) + float(state.get("cash", 0.0))
    cur = {r["code"]: int(r["shares"]) for _, r in book.iterrows()}
    cash = float(state.get("cash", 0.0))
    cash_w = cash / total if total else 0.0

    rows, trades, fees = [], [], 0.0
    for c, target_w in targets.items():
        target_amt = total * target_w
        tgt = int(target_amt / (px[c] * 100)) * 100
        delta = tgt - cur.get(c, 0)
        if delta:
            amt = abs(delta) * px[c]
            fees += amt * cost_rate
            action = "买入" if delta > 0 else "卖出"
            qty = abs(delta)
            trades.append((c, action, qty, px[c], amt))
        else:
            action, qty, amt = "持有", 0, 0.0
        rows.append((c, book.loc[book["code"] == c, "name"].iloc[0] if (book["code"] == c).any() else c,
                     cur.get(c, 0), tgt, action, qty, px[c], amt, target_w))

    band = float(cfg["rebalance"]["band"])
    max_drift = max(max(abs(cur.get(c, 0) * px[c] / total - targets[c]) for c in w),
                    abs(cash_w - (1 - exp))) if total else 0.0
    need = max_drift >= band

    path = ROOT / "output" / f"操作单_防守组合_{asof:%Y%m%d}.md"
    lines = [
        f"# 防守组合操作单 {asof:%Y-%m-%d}",
        "",
        f"> 数据截至 **{asof:%Y-%m-%d}** ｜ 配置：红利30/国债50/黄金20 + 5%/60d 波动率目标",
        f"> 组合波动 **{vol:.1%}** ｜ 目标总仓位 **{exp:.0%}** ｜ 现金目标 **{1-exp:.0%}**",
        "",
        "## 1. 当前状态",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 总净值 | {total:,.0f} |",
        f"| 持仓市值 | {total - cash:,.0f} |",
        f"| 现金 | {cash:,.0f}（{cash_w:.1%}） |",
        f"| 最大偏离 | {max_drift:.1%} |",
        f"| 再平衡阈值 | {band:.0%} |",
        f"| 结论 | {'需要再平衡' if need else '暂不需要交易'} |",
        "",
        "## 2. 目标持仓与交易",
        "",
        "| 代码 | 名称 | 当前份额 | 目标份额 | 动作 | 数量 | 价格 | 金额 | 目标权重 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for c, name, sh, tgt, action, qty, p, amt, tw in rows:
        lines.append(f"| {c} | {name} | {sh:,} | {tgt:,} | {action} | {qty:,} | {p:.3f} | {amt:,.0f} | {tw:.1%} |")
    lines += [
        f"| 现金 | — | {cash:,.0f} | — | — | — | — | — | {1-exp:.1%} |",
        "",
        f"**费用估算：{fees:,.0f} 元**（万2 佣金 + 2bp 滑点）。",
        "",
        "## 3. 执行说明",
        "",
        "1. **先卖后买**，卖出资金到账后再买；不要先买后卖。",
        "2. ETF 按 100 份一手，目标份额已向下取整，剩余留现金。",
        "3. 单只金额较大时分 2–3 笔，间隔几分钟，避免一次打穿盘口。",
        "4. 当日成交后把实际成交价/数量记到 `output/模拟盘台账.csv`。",
        "5. 若当天价格偏离本单超过 1%，以实际价格重算，不要机械照抄金额。",
        "",
        "## 4. 备注",
        "",
        f"- 本次生成于 {pd.Timestamp.today():%Y-%m-%d %H:%M}；数据日期 {asof:%Y-%m-%d}。",
        "- 目标仓位随 60 日波动率每月变化；建议每月最后交易日收盘后跑一次 `sheet`。",
        "- 这是配置 + 风险预算，不是择时；不要因为当天涨跌临时改单。",
        "- 本操作单为纸面/研究用途，不构成投资建议。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"操作单: {path}")
    print(f"目标仓位 {exp:.0%}（波动 {vol:.1%}）  最大偏离 {max_drift:.1%}  "
          f"{'需要再平衡' if need else '暂不需要交易'}  费用估算 {fees:,.0f}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mark"
    if cmd == "backtest":
        backtest()
    elif cmd == "init":
        init(float(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000.0)
    elif cmd == "mark":
        mark(write=True)
    elif cmd == "signal":
        mark(write=False)
    elif cmd == "rebalance":
        rebalance()
    elif cmd == "sheet":
        sheet()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
