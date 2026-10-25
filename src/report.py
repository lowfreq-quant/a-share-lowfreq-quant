"""绩效指标与报告"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def performance_stats(nav: pd.Series, bench: pd.Series | None = None, freq: int = 252) -> dict:
    nav = nav.dropna()
    rets = nav.pct_change().dropna()
    if len(rets) < 2:
        return {}
    n_years = len(rets) / freq
    total = float(nav.iloc[-1] / nav.iloc[0])
    cagr = total ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = float(rets.std() * np.sqrt(freq))
    sharpe = float(cagr / vol) if vol > 0 else np.nan
    mdd = float((1 - nav / nav.cummax()).max())
    win = float((rets > 0).mean())

    out = {
        "总收益": total - 1,
        "年化收益": cagr,
        "年化波动": vol,
        "夏普比率": sharpe,
        "最大回撤": mdd,
        "日胜率": win,
        "年化换手估计": np.nan,
    }
    if bench is not None:
        b = bench.reindex(nav.index).ffill().dropna()
        common = nav.index.intersection(b.index)
        if len(common) > 20:
            nb = nav.loc[common]
            bb = b.loc[common]
            excess = nb / nb.iloc[0] / (bb / bb.iloc[0]) - 1
            out["累计超额"] = float(excess.iloc[-1])
            out["年化超额"] = float((1 + excess.iloc[-1]) ** (1 / max(n_years, 1e-6)) - 1)
    return out


def yearly_returns(nav: pd.Series) -> pd.Series:
    nav = nav.dropna()
    by_year = nav.groupby(nav.index.year)
    return by_year.apply(lambda s: float(s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 else np.nan)


def write_report(
    nav_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    bench: pd.Series | None,
    out_dir: str | Path,
    cfg: dict | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = performance_stats(nav_df["nav"], bench)
    yr = yearly_returns(nav_df["nav"])

    lines = ["# 回测绩效报告", ""]
    if cfg:
        pf = cfg.get("portfolio") or {}
        cc = cfg.get("cost") or {}
        if pf:
            lines.append(f"- 股票数: {pf.get('n_stocks', '-')}")
            lines.append(f"- 调仓: {pf.get('rebalance', '-')}")
        if "etf" in cfg:
            lines.append(f"- 标的: {cfg.get('etf')}")
            lines.append(f"- 权重: {cfg.get('weights')}")
        if cc:
            lines.append(
                f"- 成本: 佣金 {float(cc.get('commission', 0)):.4%} + 印花 {float(cc.get('stamp_duty', 0)):.4%} + 滑点 {float(cc.get('slippage', 0)):.4%}"
            )
        lines.append("")
    lines += ["## 核心指标", "", "| 指标 | 数值 |", "|---|---|"]
    for k, v in stats.items():
        if isinstance(v, float):
            if "夏普" in k:
                lines.append(f"| {k} | {v:.2f} |")
            elif any(x in k for x in ["率", "收益", "回撤", "波动", "超额"]):
                lines.append(f"| {k} | {v:.2%} |")
            else:
                lines.append(f"| {k} | {v:.3f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += ["", "## 分年收益", "", "| 年份 | 收益 |", "|---|---|"]
    for y, r in yr.items():
        lines.append(f"| {y} | {r:.2%} |")

    if trade_df is not None and len(trade_df):
        lines += ["", "## 调仓摘要", ""]
        lines.append(f"- 调仓次数: {len(trade_df)}")
        if "turnover" in trade_df.columns:
            lines.append(f"- 平均单次换手: {trade_df['turnover'].mean():.2f}")
            lines.append(f"- 累计成本: {trade_df['cost'].sum():.2%}")

    report_path = out / "回测报告.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    nav_df.to_csv(out / "nav.csv", encoding="utf-8-sig")
    if trade_df is not None and len(trade_df):
        trade_df.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")

    # 图
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(nav_df.index, nav_df["nav"], label="策略净值", linewidth=1.2)
        if bench is not None:
            b = bench.reindex(nav_df.index).ffill()
            if b.notna().sum() > 10:
                axes[0].plot(b.index, b / b.iloc[0], label="基准净值", linewidth=1.0, alpha=0.8)
        axes[0].set_title("策略净值曲线")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].fill_between(nav_df.index, nav_df["drawdown"] * 100, 0, color="crimson", alpha=0.35)
        axes[1].set_ylabel("回撤 %")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "nav_curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"绘图失败: {e}")

    return report_path
