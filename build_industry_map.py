"""构建 code -> 行业 映射（新浪行业板块）"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import akshare as ak
except ImportError as e:  # pragma: no cover
    raise SystemExit("need akshare") from e


def build_industry_map(cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / "industry_map.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    spot = ak.stock_sector_spot(indicator="新浪行业")
    rows = []
    for _, r in spot.iterrows():
        label = r["label"]
        sector = r["板块"]
        try:
            df = ak.stock_sector_detail(sector=label)
        except Exception as e:  # noqa: BLE001
            print(f"  {sector} {label} FAIL {e}")
            continue
        if df is None or df.empty or "code" not in df.columns:
            continue
        for _, s in df.iterrows():
            code = str(s["code"]).zfill(6)
            rows.append({"code": code, "industry": sector, "industry_label": label})
        print(f"  {sector}: {len(df)}")
        time.sleep(0.3)

    out = pd.DataFrame(rows).drop_duplicates("code")
    out.to_parquet(cache)
    print(f"mapped {len(out)} stocks -> {out['industry'].nunique()} industries")
    return out


if __name__ == "__main__":
    build_industry_map(ROOT / "data")
