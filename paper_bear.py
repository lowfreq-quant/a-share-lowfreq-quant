"""熊市防守虚拟盘：初始化 / 盯市 / 看持仓

用法:
  python paper_bear.py init 1000000   # 按最新价建仓（默认权重 weights_bear）
  python paper_bear.py mark           # 按最新收盘估值
  python paper_bear.py show
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_index_hist, load_config
from src.etf_portfolio import fetch_etf_hist
from src.regime import current_regime

BOOK = ROOT / "output" / "虚拟盘_熊市防守.csv"
STATE = ROOT / "output" / "虚拟盘_熊市防守_状态.json"


def target_weights() -> dict[str, float]:
    cfg = load_config(ROOT / "config_bear.yaml")
    bench = fetch_index_hist("000300", cfg["data"]["start_date"], None, ROOT / "data")
    snap = current_regime(bench["close"])
    regime = snap["regime"]
    if regime == "熊市":
        w = cfg.get("weights_bear") or cfg["weights"]
    else:
        w = cfg["weights"]
    w = {str(k): float(v) for k, v in w.items()}
    return w, regime, snap


def latest_px(code: str) -> tuple[float, object]:
    df = fetch_etf_hist(code, "2024-01-01", None, adjust="")
    return float(df["close"].iloc[-1]), df.index[-1]


def init(capital: float = 1_000_000.0) -> None:
    w, regime, snap = target_weights()
    names = {str(k): str(v) for k, v in load_config(ROOT / "config_bear.yaml")["etf"].items()}
    rows = []
    cash = capital
    print(f"市况={regime}  资金={capital:,.0f}")
    for code, weight in w.items():
        name = str(names.get(code, code))
        px, dt = latest_px(code)
        amount = capital * weight
        # 场内 ETF 100份一手
        lots = int(amount / (px * 100))
        shares = lots * 100
        cost = shares * px
        cash -= cost
        rows.append(
            {
                "code": code,
                "name": name,
                "shares": shares,
                "cost_price": px,
                "cost_amount": cost,
                "target_weight": weight,
                "px_date": str(pd.Timestamp(dt).date()),
            }
        )
        print(f"  {code} {name}: {shares}份 @ {px:.3f} = {cost:,.0f}  (目标{weight:.0%})")
    print(f"  现金: {cash:,.0f}")
    df = pd.DataFrame(rows)
    df.to_csv(BOOK, index=False, encoding="utf-8-sig")
    STATE.write_text(
        pd.Series({"init_capital": capital, "cash": cash, "regime": regime, "px_date": snap["date"]}).to_json(force_ascii=False),
        encoding="utf-8",
    )
    print(f"\n台账: {BOOK}")


def mark() -> None:
    if not BOOK.exists():
        print("请先: python paper_bear.py init")
        return
    df = pd.read_csv(BOOK, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    total_mv = 0.0
    print(f"{'代码':<8} {'名称':<10} {'份额':>8} {'成本':>8} {'现价':>8} {'市值':>12} {'浮动盈亏':>10}")
    for _, r in df.iterrows():
        px, _ = latest_px(r["code"])
        mv = float(r["shares"]) * px
        pnl = mv - float(r["cost_amount"])
        total_mv += mv
        print(f"{r['code']:<8} {str(r['name'])[:8]:<10} {int(r['shares']):>8} {float(r['cost_price']):>8.3f} {px:>8.3f} {mv:>12,.0f} {pnl:>+10,.0f}")
        df.loc[df["code"] == r["code"], "mark_price"] = px
        df.loc[df["code"] == r["code"], "mv"] = mv
        df.loc[df["code"] == r["code"], "pnl"] = pnl
    cash = 0.0
    if STATE.exists():
        import json

        cash = float(json.loads(STATE.read_text(encoding="utf-8")).get("cash", 0))
    nav = total_mv + cash
    print(f"\n持仓市值 {total_mv:,.0f} + 现金 {cash:,.0f} = 总净值 {nav:,.0f}")
    df.to_csv(BOOK, index=False, encoding="utf-8-sig")


def show() -> None:
    if not BOOK.exists():
        print("尚无虚拟盘，请先 init")
        return
    df = pd.read_csv(BOOK, dtype={"code": str})
    print(df.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "init":
        cap = float(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000.0
        init(cap)
    elif cmd == "mark":
        mark()
    else:
        show()
