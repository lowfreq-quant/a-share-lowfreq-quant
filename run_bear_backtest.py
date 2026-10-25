"""熊市防守组合回测：python run_bear_backtest.py [config_bear.yaml]"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_index_hist, load_config  # noqa: E402
from src.etf_portfolio import backtest_etf_portfolio, load_etf_map  # noqa: E402
from src.regime import classify_regime, current_regime  # noqa: E402
from src.report import performance_stats, write_report  # noqa: E402


def main() -> None:
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "config_bear.yaml"
    cfg_path = ROOT / cfg_name
    cfg = load_config(cfg_path)
    cache_dir = ROOT / cfg["data"]["cache_dir"]
    out_dir = ROOT / "output"
    print(f">> 配置: {cfg_path.name}")

    etf_cfg: dict[str, str] = {str(k): str(v) for k, v in cfg["etf"].items()}
    weights: dict[str, float] = {str(k): float(v) for k, v in cfg["weights"].items()}

    print(">> 1/4 下载 ETF 行情 …")
    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date")
    adjust = cfg.get("data", {}).get("adjust", "hfq")
    px = load_etf_map(etf_cfg, start, end, cache_dir, adjust=adjust)
    print(f"   有效 ETF: {list(px.keys())}  复权: {adjust or 'raw'}")

    print(">> 2/4 基准与市况 …")
    bench_close = None
    regime_series = None
    try:
        bench = fetch_index_hist(cfg["backtest"]["benchmark"], start, end, cache_dir)
        bench_close = bench["close"]
        regime_series = classify_regime(bench_close)["regime"]
        snap = current_regime(bench_close)
        print(f"   当前市况: {snap['regime']}  ret60={snap['ret60']}")
    except Exception as e:  # noqa: BLE001
        print(f"   基准失败: {e}")

    print(">> 3/4 回测防守组合 …")
    nav_df = backtest_etf_portfolio(px, weights, cfg, regime_series)
    trade_df = pd.DataFrame()  # 本组合调仓简单，可从 invested 变化看

    print(">> 4/4 写报告 …")
    report_path = write_report(nav_df, trade_df, bench_close, out_dir, cfg)
    stats = performance_stats(nav_df["nav"], bench_close)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n## 熊市防守组合说明\n\n")
        f.write(f"- 标的: {etf_cfg}\n")
        f.write(f"- 权重: {weights}\n")
        f.write(f"- 市况门控: only_regime={cfg['backtest'].get('only_regime')}\n")
        f.write(f"- 当前市况: {snap['regime'] if bench_close is not None else '未知'}\n\n")
        f.write("### 关键指标\n\n")
        for k, v in stats.items():
            f.write(f"- {k}: {v:.2%}\n" if isinstance(v, float) and abs(v) < 5 else f"- {k}: {v}\n")
        f.write("\n**说明**：防守组合目标是熊市少亏，不是高收益。\n")

    # 拷贝一份
    (out_dir / "回测报告_熊市防守.md").write_text(open(report_path, encoding="utf-8").read(), encoding="utf-8")
    print(f"\n完成: {report_path}")
    print(f"副本: {out_dir / '回测报告_熊市防守.md'}")
    print(f"净值图: {out_dir / 'nav_curve.png'}")
    print(f"\n末日净值 {nav_df['nav'].iloc[-1]:.4f}  仓位 {nav_df['invested'].iloc[-1]:.0%}  最大回撤 {nav_df['drawdown'].max():.2%}")


if __name__ == "__main__":
    main()
