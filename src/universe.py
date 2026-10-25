"""股票池过滤"""
from __future__ import annotations

import pandas as pd


def is_st_name(name: str) -> bool:
    s = str(name).upper()
    return "ST" in s or "*ST" in s or "退" in s


def build_universe_at(
    date: pd.Timestamp,
    hist_map: dict[str, pd.DataFrame],
    names: dict[str, str] | pd.DataFrame,
    min_list_days: int = 120,
    min_avg_amount: float = 5e6,
    min_total_mv: float = 20e8,
    exclude_prefixes: tuple[str, ...] | list[str] = (),
    apply_st: bool = True,
) -> list[str]:
    """历史某日可用股票池：用当日起股本估算市值

    apply_st=False 用于回测：当前 ST 名单是「未来信息」，不能拿来过滤历史。
    """
    if isinstance(names, pd.DataFrame):
        name_map = names.set_index("code")["name"].to_dict()
    else:
        name_map = names

    valid = []
    for code, hist in hist_map.items():
        if exclude_prefixes and any(str(code).startswith(p) for p in exclude_prefixes):
            continue
        h = hist.loc[:date]
        if len(h) < min_list_days:
            continue
        # 退市/长期停牌：信号日之前 15 天内没有交易，不能选（否则退市后还会被“买入”）
        if len(h) == 0 or (pd.Timestamp(date) - pd.Timestamp(h.index[-1])).days > 15:
            continue

        if "amount" in h.columns and h["amount"].tail(20).notna().any():
            avg_amt = h["amount"].tail(20).mean()
            if pd.isna(avg_amt) or avg_amt < float(min_avg_amount):
                continue

        if "total_mv" in h.columns and h["total_mv"].notna().any():
            mv = h["total_mv"].iloc[-1]
        elif "outstanding_share" in h.columns and h["outstanding_share"].notna().any():
            mv = float(h["close"].iloc[-1]) * float(h["outstanding_share"].iloc[-1])
        else:
            mv = None
        if mv is not None and (pd.isna(mv) or mv < float(min_total_mv)):
            continue

        if apply_st and code in name_map and is_st_name(name_map[code]):
            continue

        valid.append(code)
    return valid
