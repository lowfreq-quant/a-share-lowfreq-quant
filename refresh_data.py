"""增量刷新本地数据（不改缓存文件名，只补最新一段）

用法：
  python refresh_data.py index          # 沪深300 指数（快，几秒）
  python refresh_data.py etf            # 3 只 ETF 后复权（快）
  python refresh_data.py stocks         # 全市场股票增量（约 30-90 分钟）
  python refresh_data.py stocks --limit 50   # 先试 50 只
  python refresh_data.py all            # index + etf + stocks
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import add_adjusted_close, fetch_history  # noqa: E402


def refresh_index() -> None:
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()[["close"]].astype(float)
    df.to_parquet(ROOT / "data" / "index_000300_live.parquet")
    # 覆盖 run_regime_check / run_monthly 使用的缓存
    out = df.loc["2021-01-01":]
    out.to_parquet(ROOT / "data" / "index_000300_20210101_today.parquet")
    print(f"index 000300: {len(df)} rows, 截至 {df.index.max().date()}")


def refresh_etf() -> None:
    import subprocess

    subprocess.run([sys.executable, str(ROOT / "research" / "fetch_data.py")], check=False)


def refresh_stocks(limit: int | None = None) -> None:
    files = sorted((ROOT / "data" / "hist").glob("*.parquet"))
    if limit:
        files = files[:limit]
    today = pd.Timestamp.today().strftime("%Y%m%d")
    ok = skip = fail = 0
    for i, f in enumerate(files, 1):
        code = f.name.split("_")[0]
        try:
            old = pd.read_parquet(f)
            last = pd.Timestamp(old.index.max())
            # 从最后一天前 5 个自然日开始抓，覆盖可能的修订
            start = (last - pd.Timedelta(days=5)).strftime("%Y%m%d")
            new = fetch_history(code, start, today)
            if new.empty:
                skip += 1
            else:
                old = old.drop(columns=[c for c in ["adj_close"] if c in old.columns])
                merged = pd.concat([old, new])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                merged = add_adjusted_close(merged)
                merged.to_parquet(f)
                ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            if fail <= 5:
                print(f"  FAIL {code}: {type(e).__name__} {str(e)[:80]}")
        if i % 200 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  ok={ok} skip={skip} fail={fail}", flush=True)
        time.sleep(0.05)
    print(f"stocks 完成: ok={ok} skip={skip} fail={fail}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target", nargs="?", default="all", choices=["index", "etf", "stocks", "all"])
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    if a.target in ("index", "all"):
        refresh_index()
    if a.target in ("etf", "all"):
        refresh_etf()
    if a.target in ("stocks", "all"):
        refresh_stocks(a.limit)


if __name__ == "__main__":
    main()
