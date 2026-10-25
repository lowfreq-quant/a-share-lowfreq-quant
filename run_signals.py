"""生成最新实盘持仓信号：python run_signals.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    assert_data_fresh,
    assert_hist_fresh,
    download_universe_history,
    download_valuation_map,
    fetch_stock_list,
    load_config,
)
from src.signals import generate_signals, write_signals  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    cfg_name = args[0] if args else "config.yaml"
    cfg_path = ROOT / cfg_name if not Path(cfg_name).is_absolute() else Path(cfg_name)
    print(f">> 配置: {cfg_path.name}")
    cfg = load_config(cfg_path)
    cache_dir = ROOT / cfg["data"]["cache_dir"]

    if not force:
        asof = assert_data_fresh(cache_dir)
        print(f">> 数据截至 {asof.date()}")

    print(">> 拉取股票列表 …")
    names = fetch_stock_list(cache_dir)
    codes = names["code"].tolist()
    max_stocks = cfg.get("backtest", {}).get("max_stocks")
    if max_stocks:
        codes = codes[: int(max_stocks)]

    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date")
    print(f">> 下载/读取日线 {start} → {end or 'today'} …")
    hist_map = download_universe_history(codes, start, end, cache_dir)
    if not force:
        asof_h = assert_hist_fresh(hist_map)
        print(f">> 个股行情截至 {asof_h.date()}")
    use_val = cfg.get("factors", {}).get("use_valuation", True)
    val_map = {}
    if use_val:
        print(">> 下载估值 PE/PB …")
        val_map = download_valuation_map(list(hist_map.keys()), cache_dir)
    else:
        print(">> 跳过估值（use_valuation=false）")

    print(">> 生成信号 …")
    sig, as_of = generate_signals(hist_map, cfg, val_map=val_map)
    path = write_signals(sig, ROOT / "output", as_of)
    print(f"\n信号表: {path}")
    print(f"操作单: {ROOT / 'output' / ('操作单_' + as_of.strftime('%Y%m%d') + '.md')}")
    print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
