"""市况判定：牛市 / 熊市 / 震荡

规则（对沪深300，日频，可复现）：
  1) 近 60 个交易日收益率
     ret60 >  +12%  → 偏牛
     ret60 <  -12%  → 偏熊
     否则           → 震荡
  2) 辅助确认（提高稳健性）：
     - 熊市加深：收盘 < MA120 且 MA20 < MA60
     - 牛市加深：收盘 > MA120 且 MA20 > MA60
     - 震荡确认：|ret60| <= 12% 且 20 日年化波动 < 30%（波动太大更接近单边）

使用：
  from src.regime import classify_regime, current_regime
  print(current_regime(bench_close))
"""
from __future__ import annotations

import pandas as pd

BULL_RET = 0.12
BEAR_RET = -0.12
VOL_CAP = 0.30  # 20日年化波动超过 30% 时，即使 |ret60| 不大也不叫震荡


def _ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def classify_regime(bench_close: pd.Series) -> pd.DataFrame:
    """返回每日期市况标签（DataFrame: close, ret60, vol20, ma20, ma60, ma120, regime）"""
    s = bench_close.dropna().astype(float).copy()
    df = pd.DataFrame(index=s.index)
    df["close"] = s
    df["ret60"] = s.pct_change(60)
    df["vol20"] = s.pct_change().rolling(20).std() * (252 ** 0.5)
    df["ma20"] = _ma(s, 20)
    df["ma60"] = _ma(s, 60)
    df["ma120"] = _ma(s, 120)

    regime = pd.Series("震荡", index=df.index)
    # 先按 60 日收益切
    regime[df["ret60"] > BULL_RET] = "牛市"
    regime[df["ret60"] < BEAR_RET] = "熊市"

    # 趋势加深覆盖
    bull_deep = (df["close"] > df["ma120"]) & (df["ma20"] > df["ma60"])
    bear_deep = (df["close"] < df["ma120"]) & (df["ma20"] < df["ma60"])
    regime[bull_deep & (df["ret60"] > 0)] = "牛市"
    regime[bear_deep & (df["ret60"] < 0)] = "熊市"

    # 高波动且 |ret60| 中等 → 不叫震荡，按方向归类
    high_vol = df["vol20"] > VOL_CAP
    mid = (df["ret60"].abs() > 0.06) & (df["ret60"].abs() <= abs(BULL_RET))
    regime[high_vol & mid & (df["ret60"] > 0)] = "牛市"
    regime[high_vol & mid & (df["ret60"] < 0)] = "熊市"

    # |ret60| 大但波动极高时，仍以收益方向为准（已在上面 BULL/BEAR 覆盖）
    df["regime"] = regime
    return df


def current_regime(bench_close: pd.Series) -> dict:
    """最新一日市况快照，给实盘用"""
    df = classify_regime(bench_close)
    last = df.iloc[-1]
    return {
        "date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        "regime": last["regime"],
        "close": float(last["close"]),
        "ret60": float(last["ret60"]) if pd.notna(last["ret60"]) else None,
        "vol20": float(last["vol20"]) if pd.notna(last["vol20"]) else None,
        "ma20": float(last["ma20"]) if pd.notna(last["ma20"]) else None,
        "ma60": float(last["ma60"]) if pd.notna(last["ma60"]) else None,
        "ma120": float(last["ma120"]) if pd.notna(last["ma120"]) else None,
        "rule": (
            f"ret60>{BULL_RET:.0%}=牛 / ret60<{BEAR_RET:.0%}=熊 / 其余=震荡；"
            f"MA120+均线组加深；vol20>{VOL_CAP:.0%} 时不计震荡"
        ),
    }
