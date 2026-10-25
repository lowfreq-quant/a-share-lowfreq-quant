"""全市场日线批量下载（断点续传）：python download_all.py [batch_size] [max_batches]

已有缓存会跳过。按代码分批打印进度，Ctrl+C 可中断，下次续跑。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_stock_list, load_config, load_or_fetch_history  # noqa: E402


def main() -> None:
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9

    cfg = load_config(ROOT / "config_choppy.yaml")
    cache_dir = ROOT / cfg["data"]["cache_dir"]
    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date")

    names = fetch_stock_list(cache_dir)
    codes = names["code"].tolist()
    print(f"全 A {len(codes)} 只 | 区间 {start} → {end or 'today'}")

    # 先算已有缓存
    hist_dir = cache_dir / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    end_tag = (end or "today").replace("-", "")
    suffix = f"_{start.replace('-', '')}_{end_tag}.parquet"

    def cached(code: str) -> bool:
        p = hist_dir / f"{code}{suffix}"
        return p.exists() or p.with_suffix(".pkl").exists()

    done0 = sum(1 for c in codes if cached(c))
    print(f"已有缓存 {done0}/{len(codes)}")

    todo = [c for c in codes if not cached(c)]
    print(f"待下载 {len(todo)}")

    ok = 0
    fail = 0
    batches = 0
    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        if batches >= max_batches:
            print(f"已达 max_batches={max_batches}，停。")
            break
        chunk = todo[i : i + batch_size]
        for code in chunk:
            success = False
            for attempt in range(3):
                try:
                    df = load_or_fetch_history(code, start, end, cache_dir)
                    if not df.empty:
                        ok += 1
                        success = True
                        break
                    # 空数据也算跳过，避免死循环
                    success = True
                    ok += 1
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        fail += 1
                        if fail % 20 == 0:
                            print(f"  fail {code}: {e}")
                    time.sleep(1.5 * (attempt + 1))
            if not success:
                fail += 1
            time.sleep(0.05)
        batches += 1
        elapsed = time.time() - t0
        left = max(0, len(todo) - i - len(chunk))
        print(
            f"[batch {batches}] 本轮 {i + len(chunk)}/{len(todo)}  "
            f"ok {ok} fail {fail}  剩 {left}  耗时 {elapsed / 60:.1f}min",
            flush=True,
        )

    print(f"\n完成. 缓存总数 {sum(1 for c in codes if cached(c))}/{len(codes)}  fail {fail}")


if __name__ == "__main__":
    main()
