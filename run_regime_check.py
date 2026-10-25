"""一键看当前市况：python run_regime_check.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_index_hist, load_config  # noqa: E402
from src.regime import classify_regime, current_regime  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    start = cfg["data"]["start_date"]
    bench = fetch_index_hist(cfg["backtest"]["benchmark"], start, None, ROOT / "data")
    close = bench["close"]

    snap = current_regime(close)
    df = classify_regime(close)

    print("=" * 48)
    print("市况判定（沪深300）")
    print("=" * 48)
    print(f"日期:     {snap['date']}")
    print(f"当前市况: {snap['regime']}")
    print(f"收盘:     {snap['close']:.2f}")
    if snap["ret60"] is not None:
        print(f"近60日:   {snap['ret60']:+.2%}")
    if snap["vol20"] is not None:
        print(f"20日波动: {snap['vol20']:.1%}")
    print(f"MA20/60/120: {snap['ma20']:.2f} / {snap['ma60']:.2f} / {snap['ma120']:.2f}")
    print(f"规则:     {snap['rule']}")
    print("-" * 48)

    # 近 10 日标签
    recent = df.tail(10)[["close", "ret60", "vol20", "regime"]]
    print("近10个交易日：")
    for dt, r in recent.iterrows():
        ret = f"{r['ret60']:+.1%}" if pd_notna(r["ret60"]) else "  n/a"
        vol = f"{r['vol20']:.1%}" if pd_notna(r["vol20"]) else "  n/a"
        print(f"  {dt.date() if hasattr(dt,'date') else dt}  {r['regime']:<2}  ret60={ret}  vol20={vol}")

    print("-" * 48)
    action = {
        "震荡": "→ 核心仓仍用 30/50/20；choppy 只做卫星：python run_signals.py config_choppy_v2.yaml",
        "牛市": "→ 用你自己的牛市策略；核心仓 30/50/20 继续持有（牛市会跑输指数）",
        "熊市": "→ 核心仓 30/50/20：python run_defensive.py mark；震荡盘空仓",
    }
    print(action.get(snap["regime"], ""))
    print("=" * 48)


def pd_notna(x) -> bool:
    try:
        import pandas as pd

        return bool(pd.notna(x))
    except Exception:  # noqa: BLE001
        return x is not None


if __name__ == "__main__":
    main()
