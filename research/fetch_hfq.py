"""抓取股票后复权数据（Sina hfq），只抓 50 亿以上（策略相关）

缓存：data/hist_hfq/{code}.parquet
用法：python research/fetch_hfq.py [--limit N] [--min-mv 5e9]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hist_hfq"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import akshare as ak

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-mv", type=float, default=5e9)
    a = p.parse_args()

    files = sorted((ROOT / "data" / "hist").glob("*.parquet"))
    codes = []
    for f in files:
        try:
            mv = pd.read_parquet(f, columns=["total_mv"])["total_mv"].dropna()
            if len(mv) and float(mv.iloc[-1]) >= a.min_mv:
                codes.append(f.name.split("_")[0])
        except Exception:
            pass
    if a.limit:
        codes = codes[: a.limit]
    print(f"目标 {len(codes)} 只（市值 >= {a.min_mv/1e8:.0f} 亿）", flush=True)

    ok = skip = fail = 0
    for i, code in enumerate(codes, 1):
        path = OUT / f"{code}.parquet"
        if path.exists():
            skip += 1
            continue
        sym = ("sh" if code[0] in "56" else "sz") + code
        try:
            df = ak.stock_zh_a_daily(symbol=sym, start_date="20210101", end_date=None, adjust="hfq")
            if df is None or df.empty:
                skip += 1
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            cols = [c for c in ["open", "close", "volume", "amount", "turnover"] if c in df.columns]
            df[cols].to_parquet(path)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            if fail <= 5:
                print(f"  FAIL {code}: {type(e).__name__} {str(e)[:80]}", flush=True)
        if i % 100 == 0:
            print(f"  {i}/{len(codes)} ok={ok} skip={skip} fail={fail}", flush=True)
        time.sleep(0.05)
    print(f"完成 ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
