"""抓取研究用 ETF 后复权数据 + 最新沪深300

数据源：腾讯（stock_zh_a_hist_tx, adjust=hfq）
缓存：data/etf_hfq/{code}.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "etf_hfq"
OUT.mkdir(parents=True, exist_ok=True)

ETFS = {
    "510880": "红利ETF",
    "512890": "红利低波ETF",
    "511260": "十年国债ETF",
    "511010": "国债ETF",
    "518880": "黄金ETF",
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "512100": "中证1000ETF",
    "513100": "纳指ETF",
}


def fetch(code: str, start: str = "20120101", end: str | None = None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y%m%d")
    sym = ("sh" if code[0] in "56" else "sz") + code
    df = ak.stock_zh_a_hist_tx(symbol=sym, start_date=start, end_date=end, adjust="hfq")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open", "close", "volume", "amount"]]


def main() -> None:
    for code, name in ETFS.items():
        try:
            df = fetch(code)
            if df.empty:
                print(f"{code} {name}: EMPTY")
                continue
            df.to_parquet(OUT / f"{code}.parquet")
            print(f"{code} {name}: {len(df)} rows  {df.index.min().date()} → {df.index.max().date()}")
        except Exception as e:  # noqa: BLE001
            print(f"{code} {name}: FAIL {type(e).__name__} {str(e)[:120]}")
        time.sleep(0.5)

    # 沪深300 最新（Sina 指数）
    try:
        idx = ak.stock_zh_index_daily(symbol="sh000300")
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.set_index("date").sort_index()
        idx[["close"]].astype(float).to_parquet(ROOT / "data" / "index_000300_live.parquet")
        print(f"000300: {len(idx)} rows  {idx.index.min().date()} → {idx.index.max().date()}")
    except Exception as e:  # noqa: BLE001
        print(f"000300 FAIL {type(e).__name__} {str(e)[:120]}")


if __name__ == "__main__":
    main()
