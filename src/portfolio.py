"""选股组合构建（含行业约束）"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .factors import composite_score, compute_point_in_time
from .universe import build_universe_at


def load_industry_map(cache_dir: str | Path) -> dict[str, str]:
    path = Path(cache_dir) / "industry_map.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["code"].astype(str), df["industry"].astype(str)))


def select_top(
    signal_day,
    hist_map: dict[str, pd.DataFrame],
    names,
    cfg: dict,
    val_map: dict[str, pd.DataFrame] | None = None,
    industry_map: dict[str, str] | None = None,
    apply_st: bool = True,
) -> list[str]:
    """按综合得分选 n_stocks 只，可选单行业权重上限（按只数近似）"""
    ucfg = cfg["universe"]
    fcfg = cfg["factors"]
    pcfg = cfg["portfolio"]
    n = int(pcfg["n_stocks"])
    max_ind_w = pcfg.get("max_industry_weight")

    codes = build_universe_at(
        signal_day,
        hist_map,
        names,
        min_list_days=ucfg["min_list_days"],
        min_avg_amount=ucfg["min_avg_amount"],
        min_total_mv=ucfg["min_total_mv"],
        exclude_prefixes=tuple(ucfg.get("exclude_prefixes") or ()),
        apply_st=apply_st,
    )
    if len(codes) < n:
        return []

    fac = compute_point_in_time(signal_day, hist_map, codes, val_map)
    fac["score"] = composite_score(fac, fcfg["weights"], fcfg.get("winsorize", 0.02))
    ranked = fac.sort_values("score", ascending=False)

    ind_map = industry_map or {}
    max_per = None
    if ind_map and max_ind_w:
        max_per = max(1, int(round(float(max_ind_w) * n)))

    picked: list[str] = []
    ind_count: dict[str, int] = {}
    for code in ranked.index:
        ind = ind_map.get(code)
        if max_per is not None and ind is not None and ind_count.get(ind, 0) >= max_per:
            continue
        picked.append(code)
        if ind is not None:
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(picked) >= n:
            break

    # 行业约束导致不够时，按得分补齐
    if len(picked) < n:
        for code in ranked.index:
            if code not in picked:
                picked.append(code)
            if len(picked) >= n:
                break
    return picked
