"""每月一键操作：python run_monthly.py [config]

市况 → 给出本月该做的事（震荡出信号 / 牛熊切换 / 防守）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import assert_data_fresh, assert_hist_fresh, download_universe_history, fetch_index_hist, fetch_stock_list, load_config  # noqa: E402
from src.regime import current_regime  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    cfg_name = args[0] if args else "config_choppy_v2.yaml"
    cfg_path = ROOT / cfg_name
    cfg = load_config(cfg_path)
    cache = ROOT / cfg["data"]["cache_dir"]

    if not force:
        asof = assert_data_fresh(cache)
        print(f">> 数据截至 {asof.date()}")

    print("=" * 52)
    print("月度操作台")
    print("=" * 52)

    # 1) 市况
    bench = fetch_index_hist(cfg["backtest"]["benchmark"], cfg["data"]["start_date"], None, cache)
    snap = current_regime(bench["close"])
    print(f"\n[1] 市况: {snap['regime']}  ({snap['date']})")
    print(f"    ret60={snap['ret60']:+.1%}  vol20={snap['vol20']:.1%}")

    regime = snap["regime"]
    if regime == "牛市":
        print("\n[2] 动作: 使用你的牛市策略。震荡盘 v2 / 熊市防守均停开新仓。")
        print("    防守仓可保持 config_bear 常持部分，或按你的牛市体系整体调配。")
        return

    if regime == "熊市":
        print("\n[2] 动作: 熊市防守")
        print("    核心底仓: 红利30/国债50/黄金20（config_defensive.yaml）")
        print("              python run_defensive.py mark   # 季度检查，偏离5%再平衡")
        print("    更防守可选: 5% 红利 + 5% 红利低波 + 90% 十年国债（config_bear.weights_bear）")
        print("    震荡盘策略空仓，勿买小市值组合。")
        return

    # 震荡
    print("\n[2] 动作: 震荡盘策略 config_choppy_v2")
    print("    正在生成持仓操作单 ...")
    names = fetch_stock_list(cache)
    codes = names["code"].tolist()
    hist = download_universe_history(codes, cfg["data"]["start_date"], None, cache, sleep=0, progress_every=1000)
    if not force:
        asof_h = assert_hist_fresh(hist)
        print(f">> 个股行情截至 {asof_h.date()}")

    from src.signals import generate_signals, write_signals

    sig, as_of = generate_signals(hist, cfg)
    path = write_signals(sig, ROOT / "output", as_of)
    print(f"    操作单: {path}")
    print(f"    行业分布:")
    if "industry" in sig.columns:
        print(sig["industry"].value_counts().to_string())
    print("\n[3] 次日执行: 先卖不在表中的，再买新入的；每只约 4%。")
    print("    记录到 output/模拟盘台账.csv")


if __name__ == "__main__":
    main()
