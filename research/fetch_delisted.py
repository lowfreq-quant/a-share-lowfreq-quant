"""抓取 2021+ 退市股票行情（腾讯原始价），用于修正幸存者偏差

输出：
  data/delisted_2021.csv          退市清单
  data/hist_delisted/{code}.parquet
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hist_delisted"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import akshare as ak

    sh = ak.stock_info_sh_delist()
    sz = ak.stock_info_sz_delist()
    sh["d"] = pd.to_datetime(sh["暂停上市日期"], errors="coerce")
    sz["d"] = pd.to_datetime(sz["终止上市日期"], errors="coerce")
    rows = []
    for _, r in sh[sh["d"] >= "2021-01-01"].iterrows():
        rows.append(dict(code=str(r["公司代码"]).zfill(6), name=str(r["公司简称"]), delist_date=r["d"].date()))
    for _, r in sz[sz["d"] >= "2021-01-01"].iterrows():
        rows.append(dict(code=str(r["证券代码"]).zfill(6), name=str(r["证券简称"]), delist_date=r["d"].date()))
    meta = pd.DataFrame(rows)
    meta = meta[meta["code"].str[0].isin(list("036")) & ~meta["code"].str.startswith(("900", "200"))]
    meta = meta.drop_duplicates("code").sort_values("code")
    meta.to_csv(ROOT / "data" / "delisted_2021.csv", index=False, encoding="utf-8-sig")
    print(f"退市清单 {len(meta)} 只", flush=True)

    ok = skip = fail = 0
    for i, (_, r) in enumerate(meta.iterrows(), 1):
        code = r["code"]
        path = OUT / f"{code}.parquet"
        if path.exists():
            skip += 1
            continue
        sym = ("sh" if code[0] == "6" else "sz") + code
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                df = ak.stock_zh_a_hist_tx(symbol=sym, start_date="20210101",
                                           end_date=pd.Timestamp(r["delist_date"]).strftime("%Y%m%d"),
                                           adjust="")
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
        if i % 50 == 0:
            print(f"  {i}/{len(meta)} ok={ok} skip={skip} fail={fail}", flush=True)
        time.sleep(0.05)
    print(f"完成 ok={ok} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
