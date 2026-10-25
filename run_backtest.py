"""一键回测：python run_backtest.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest  # noqa: E402
from src.data import (  # noqa: E402
    download_universe_history,
    download_valuation_map,
    fetch_index_hist,
    fetch_stock_list,
    load_config,
    load_delisted_histories,
)
from src.report import write_report  # noqa: E402


def main() -> None:
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg_path = ROOT / cfg_name if not Path(cfg_name).is_absolute() else Path(cfg_name)
    print(f">> 配置: {cfg_path.name}")
    cfg = load_config(cfg_path)
    cache_dir = ROOT / cfg["data"]["cache_dir"]
    out_dir = ROOT / "output"

    print(">> 1/5 拉取股票列表 …")
    names = fetch_stock_list(cache_dir)
    codes = names["code"].tolist()
    print(f"   全 A {len(codes)} 只")

    max_stocks = cfg.get("backtest", {}).get("max_stocks")
    if max_stocks:
        codes = codes[: int(max_stocks)]
        print(f"   本次下载 {len(codes)} 只（config: backtest.max_stocks）")

    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date")
    print(f">> 2/5 下载/读取日线（新浪） {start} → {end or 'today'} …")
    hist_map = download_universe_history(codes, start, end, cache_dir)
    print(f"   有效行情 {len(hist_map)} 只")
    if cfg.get("backtest", {}).get("include_delisted"):
        extra = load_delisted_histories(cache_dir)
        hist_map.update(extra)
        print(f"   加入退市股 {len(extra)} 只（修正幸存者偏差）")

    use_val = cfg.get("factors", {}).get("use_valuation", True)
    val_map = {}
    if use_val:
        print(">> 2b/5 下载估值 PE/PB …")
        val_map = download_valuation_map(list(hist_map.keys()), cache_dir)
        print(f"   有效估值 {len(val_map)} 只")
    else:
        print(">> 2b/5 跳过估值（config factors.use_valuation=false）")

    print(">> 3/5 读取基准指数 …")
    bench_close = None
    regime_series = None
    try:
        bench = fetch_index_hist(cfg["backtest"]["benchmark"], start, end, cache_dir)
        bench_close = bench["close"]
        if cfg.get("backtest", {}).get("only_regime"):
            from src.regime import classify_regime

            regime_series = classify_regime(bench_close)["regime"]
            only = cfg["backtest"]["only_regime"]
            frac = (regime_series == only).mean()
            print(f"   市况过滤: only={only}  历史占比 {frac:.0%}")
    except Exception as e:  # noqa: BLE001
        print(f"   基准指数失败: {e}")

    print(">> 4/5 运行月度调仓回测 …")
    nav_df, trade_df = run_backtest(hist_map, names, cfg, val_map, regime_series)
    print(nav_df.tail(3))
    if len(trade_df):
        print(trade_df.tail(3))

    print(">> 5/5 写报告 …")
    report_path = write_report(nav_df, trade_df, bench_close, out_dir, cfg)
    # 追加样本说明
    note = [
        "",
        "## 样本说明（必读）",
        "",
        f"- 本次股票池规模: **{len(hist_map)} 只**（config `backtest.max_stocks`）",
        f"- 全 A 约 5500 只；把 `max_stocks` 改为 `null` 后重跑才是正式全市场回测",
        f"- 估值覆盖: {len(val_map)}/{len(hist_map)} 只",
        f"- 收益用 adj_close（优先后复权，含分红/拆分；未覆盖的退回送转复权）；执行口径为信号日定市况、次日开盘成交",
        "- **这不是实盘预测，只是流程验证**",
    ]
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(note) + "\n")
    print(f"\n完成！报告: {report_path}")
    print(f"净值曲线: {out_dir / 'nav_curve.png'}")
    print(f"交易明细: {out_dir / 'trades.csv'}")


if __name__ == "__main__":
    main()
