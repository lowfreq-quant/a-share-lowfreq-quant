"""因子计算与合成（价量 + 估值）"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, limit: float = 0.02) -> pd.Series:
    s = s.astype(float)
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(limit), s.quantile(1 - limit)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sigma = s.mean(), s.std()
    if pd.isna(sigma) or sigma == 0:
        return s * 0.0
    return (s - mu) / sigma


def _val_asof(val: pd.DataFrame, date: pd.Timestamp) -> dict:
    """估值截面：取 <= date 的最近一行"""
    if val is None or val.empty:
        return {}
    v = val.loc[:date]
    if v.empty:
        return {}
    row = v.iloc[-1]
    out = {}
    for k in ["pe_ttm", "pb", "ep", "roe", "total_mv"]:
        if k in row.index and pd.notna(row[k]):
            out[k] = float(row[k])
    return out


def compute_point_in_time(
    date: pd.Timestamp,
    hist_map: dict[str, pd.DataFrame],
    codes: list[str],
    val_map: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """截面因子表（index=code）

    价量：ln_mv / momentum_20 / volatility_20 / turnover_20
    估值：ep / roe（来自 val_map，无数据则 NaN，合成时自动跳过）
    """
    rows = []
    for code in codes:
        h = hist_map.get(code)
        if h is None or h.empty:
            continue
        h = h.loc[:date]
        if len(h) < 25:
            continue

        close = float(h["close"].iloc[-1])
        # 动量/波动用后复权序列（送转/拆股不会造成假暴跌）；展示价仍用原始 close
        px = pd.to_numeric(h["adj_close"], errors="coerce") if "adj_close" in h.columns else pd.to_numeric(h["close"], errors="coerce")
        c20 = float(px.iloc[-21]) if len(px) >= 21 else float(px.iloc[0])
        mom20 = float(px.iloc[-1]) / c20 - 1.0 if c20 else np.nan

        ret = px.pct_change().tail(20)
        vol20 = float(ret.std()) if len(ret) >= 5 else np.nan

        if "turnover" in h.columns:
            to20 = float(h["turnover"].tail(20).mean())
        else:
            to20 = np.nan

        if "total_mv" in h.columns and pd.notna(h["total_mv"].iloc[-1]):
            mv = float(h["total_mv"].iloc[-1])
        elif "outstanding_share" in h.columns and pd.notna(h["outstanding_share"].iloc[-1]):
            mv = close * float(h["outstanding_share"].iloc[-1])
        else:
            mv = np.nan

        val = _val_asof((val_map or {}).get(code), date)
        # 优先用东财估值市值（更准）
        if val.get("total_mv"):
            mv = val["total_mv"]
        ln_mv = float(np.log(mv)) if mv and mv > 0 else np.nan

        ep = val.get("ep", np.nan)
        roe = val.get("roe", np.nan)
        if pd.isna(ep) and val.get("pe_ttm"):
            ep = 1.0 / val["pe_ttm"] if val["pe_ttm"] > 0 else np.nan
        if pd.isna(roe) and val.get("pb") and val.get("pe_ttm") and val["pe_ttm"] > 0 and val["pb"] > 0:
            roe = val["pb"] / val["pe_ttm"]

        rows.append(
            {
                "code": code,
                "close": close,
                "ln_mv": ln_mv,
                "ep": ep,
                "roe": roe,
                "momentum_20": mom20,
                "volatility_20": vol20,
                "turnover_20": to20,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["close", "ln_mv", "ep", "roe", "momentum_20", "volatility_20", "turnover_20"]
        ).rename_axis("code")
    return pd.DataFrame(rows).set_index("code")


def composite_score(
    factors: pd.DataFrame,
    weights: dict[str, float],
    winsor: float = 0.02,
) -> pd.Series:
    score = pd.Series(0.0, index=factors.index, dtype=float)
    for name, w in weights.items():
        if name not in factors.columns or not w:
            continue
        s = factors[name]
        if s.notna().sum() < max(10, 0.15 * len(s)):
            continue
        s = winsorize(s.fillna(s.median()), winsor)
        s = zscore(s)
        score = score + float(w) * s.fillna(0.0)
    return score
