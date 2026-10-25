"""生成最新一期持仓信号（供实盘执行）"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import fetch_stock_list
from .portfolio import load_industry_map, select_top


def generate_signals(
    hist_map: dict[str, pd.DataFrame],
    cfg: dict,
    as_of: pd.Timestamp | None = None,
    val_map: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    ucfg = cfg["universe"]
    fcfg = cfg["factors"]
    pcfg = cfg["portfolio"]

    names_df = fetch_stock_list(cfg.get("data", {}).get("cache_dir", "data"))
    name_map = names_df.set_index("code")["name"].to_dict()

    if as_of is None:
        as_of = max(h.index.max() for h in hist_map.values())
    as_of = pd.Timestamp(as_of)

    ind_map = load_industry_map(cfg.get("data", {}).get("cache_dir", "data"))
    codes = select_top(as_of, hist_map, name_map, cfg, val_map=val_map, industry_map=ind_map)
    if not codes:
        raise RuntimeError("选股失败：股票池不足或市况/过滤过严")
    if len(codes) < pcfg["n_stocks"]:
        print(f"  警告: 仅选出 {len(codes)} 只 < 目标 {pcfg['n_stocks']}")

    from .factors import composite_score, compute_point_in_time

    fac = compute_point_in_time(as_of, hist_map, codes, val_map)
    fac["score"] = composite_score(fac, fcfg["weights"], fcfg.get("winsorize", 0.02))
    fac["industry"] = [ind_map.get(c, "") for c in fac.index]
    top = fac.loc[codes].copy()

    n = len(top)
    out = top.reset_index()
    out["name"] = out["code"].map(name_map)
    out["target_weight"] = 1.0 / n
    out["target_amount_万"] = out["target_weight"] * 100  # 按100万资金示意
    cols = [
        "code",
        "name",
        "industry",
        "score",
        "close",
        "ln_mv",
        "momentum_20",
        "volatility_20",
        "turnover_20",
        "target_weight",
        "target_amount_万",
    ]
    for c in ["ep", "roe"]:
        if c in out.columns:
            cols.insert(3, c)
    out = out[[c for c in cols if c in out.columns]]
    return out, as_of


def write_signals(df: pd.DataFrame, out_dir: str | Path, as_of: pd.Timestamp) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = as_of.strftime("%Y%m%d")
    path = out / f"signals_{tag}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    lines = [
        f"# 持仓调仓信号  信号日 {as_of.strftime('%Y-%m-%d')}",
        "",
        "次一交易日上午按下列目标仓位执行（先卖不在表中的，再买新入的）。",
        "",
        "| 代码 | 名称 | 目标权重 | 金额(万元) | 得分 | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['code']} | {r['name']} | {r['target_weight']:.1%} | {r['target_amount_万']:.1f} | {r['score']:.3f} | 买入/持有 |"
        )
    op_path = out / f"操作单_{tag}.md"
    op_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path
