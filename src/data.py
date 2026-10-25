"""行情数据：新浪源；本地 parquet 缓存

说明：
- 默认拉取【不复权】价，市值 = 收盘价 × 总股本，因子截面可比
- 月度调仓的收益用不复权涨跌，会漏掉分红，对月频选股影响通常很小
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError as e:  # pragma: no cover
    raise SystemExit("请先安装 akshare: pip install -r requirements.txt") from e


def _coerce_numbers(obj):
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if "e" in s.lower():
            try:
                return float(s)
            except Exception:  # noqa: BLE001
                return obj
    return obj


def load_config(path: str | Path = "config.yaml") -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _coerce_numbers(cfg)


def to_sina_symbol(code: str) -> str:
    c = str(code).zfill(6)
    # 上交所主板/科创板
    if c.startswith(("60", "68")):
        return f"sh{c}"
    # 深交所主板/创业板
    if c.startswith(("00", "30", "20")):
        return f"sz{c}"
    # 北交所
    if c.startswith(("43", "83", "87", "88", "92", "93")):
        return f"bj{c}"
    return f"bj{c}"


def fetch_stock_list(cache_dir: str | Path = "data") -> pd.DataFrame:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "stock_list.parquet"
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if len(df) > 100:
                return df
        except Exception:  # noqa: BLE001
            pass
    last = None
    for _ in range(3):
        try:
            df = ak.stock_info_a_code_name()
            df = df.copy()
            df["code"] = df["code"].astype(str).str.zfill(6)
            df = df[["code", "name"]]
            try:
                df.to_parquet(path)
            except Exception:  # noqa: BLE001
                df.to_pickle(path.with_suffix(".pkl"))
            return df
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    if path.with_suffix(".pkl").exists():
        return pd.read_pickle(path.with_suffix(".pkl"))
    raise RuntimeError(f"股票列表下载失败: {last}")


def fetch_history(code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """新浪日线（不复权），含 turnover / outstanding_share / total_mv"""
    end_date = end_date or pd.Timestamp.today().strftime("%Y%m%d")
    sym = to_sina_symbol(code)
    df = ak.stock_zh_a_daily(
        symbol=sym,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "outstanding_share"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "outstanding_share" in df.columns and "close" in df.columns:
        df["total_mv"] = df["close"] * df["outstanding_share"]
    return df


def add_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    """用「送转/拆股」事件重建后复权收盘价（adj_close）。

    只处理价格大跌/大涨 + 股本同向变化的事件；现金分红不处理（影响小）。
    这样既保留 close 的原始价格（用于市值/下单），又给因子和收益提供连续序列。
    """
    if df is None or df.empty or "close" not in df.columns:
        return df
    if "adj_close" in df.columns:
        return df
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    if "outstanding_share" in df.columns:
        shares = pd.to_numeric(df["outstanding_share"], errors="coerce")
        ratio = shares / shares.shift(1)
        pret = close.pct_change(fill_method=None)
        event = (
            ((pret < -0.15) & (ratio > 1.05)) | ((pret > 0.15) & (ratio < 0.95))
        ) & ratio.notna() & (ratio > 0)
        g = pd.Series(0.0, index=df.index)
        g[event] = np.log(1.0 / ratio[event])
        g = g.fillna(0.0)
        rev = g[::-1].cumsum()[::-1]
        adj = np.exp(rev - g)
    else:
        adj = pd.Series(1.0, index=df.index)
    df["adj_close"] = close * adj
    return df


def _with_hfq_or_split_adj(df: pd.DataFrame, code: str, cache_dir: str | Path) -> pd.DataFrame:
    """优先用 data/hist_hfq 的后复权价（含分红/拆分）；没有就退回送转复权。"""
    p = Path(cache_dir) / "hist_hfq" / f"{str(code).zfill(6)}.parquet"
    if p.exists() and df is not None and not df.empty:
        try:
            h = pd.read_parquet(p)
            if "close" in h.columns and len(h):
                out = df.copy()
                out["adj_close"] = h["close"].reindex(out.index).ffill()
                if "open" in h.columns:
                    out["adj_open"] = h["open"].reindex(out.index).ffill()
                out["adj_close"] = out["adj_close"].fillna(out["close"])
                if "adj_open" in out.columns:
                    out["adj_open"] = out["adj_open"].fillna(out["open"])
                return out
        except Exception:  # noqa: BLE001
            pass
    return add_adjusted_close(df)


def load_or_fetch_history(
    code: str,
    start_date: str,
    end_date: str | None,
    cache_dir: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    cache = Path(cache_dir) / "hist"
    cache.mkdir(parents=True, exist_ok=True)
    end_tag = (end_date or "today").replace("-", "")
    path = cache / f"{code}_{start_date.replace('-', '')}_{end_tag}.parquet"
    if path.exists() and not refresh:
        try:
            return _with_hfq_or_split_adj(pd.read_parquet(path), code, cache_dir)
        except Exception:  # noqa: BLE001
            return _with_hfq_or_split_adj(pd.read_pickle(path.with_suffix(".pkl")), code, cache_dir)
    df = fetch_history(code, start_date, end_date)
    if not df.empty:
        df = add_adjusted_close(df)
        try:
            df.to_parquet(path)
        except Exception:  # noqa: BLE001
            df.to_pickle(path.with_suffix(".pkl"))
        df = _with_hfq_or_split_adj(df, code, cache_dir)
    return df


def fetch_index_hist(
    symbol: str = "000300",
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    cache_dir: str | Path = "data",
) -> pd.DataFrame:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    end_tag = (end_date or "today").replace("-", "")
    path = cache / f"index_{symbol}_{start_date.replace('-', '')}_{end_tag}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return pd.read_pickle(path.with_suffix(".pkl"))

    df = None
    last_err = None
    sym = str(symbol)
    # 指数前缀：上证 000xxx / 5xxxxx 用 sh，深证 399xxx 用 sz
    prefixes = ["sh", "sz"] if not sym.startswith("399") else ["sz", "sh"]
    for prefix in prefixes:
        try:
            df = ak.stock_zh_index_daily(symbol=f"{prefix}{sym}")
            if df is not None and len(df):
                break
        except Exception as e:  # noqa: BLE001
            last_err = e
            df = None
    if df is None or not len(df):
        try:
            df = ak.stock_zh_index_daily_em(symbol=sym)
        except Exception as e:  # noqa: BLE001
            last_err = e
            df = None

    if df is None or not len(df):
        raise RuntimeError(f"指数 {symbol} 下载失败: {last_err}")

    df = df.copy()
    if "date" not in df.columns:
        for cand in ("日期", "trade_date", "day"):
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
        else:
            df = df.reset_index().rename(columns={df.columns[0]: "date"})

    if "close" not in df.columns:
        for cand in ("收盘", "close"):
            if cand in df.columns:
                df = df.rename(columns={cand: "close"})
                break

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[pd.to_datetime(start_date):]
    if end_date:
        df = df.loc[: pd.to_datetime(end_date)]
    out = df[["close"]].astype(float)
    try:
        out.to_parquet(path)
    except Exception:  # noqa: BLE001
        out.to_pickle(path.with_suffix(".pkl"))
    return out


def fetch_valuation(code: str) -> pd.DataFrame:
    """东财估值历史：PE(TTM) / PB / 总市值。index=date"""
    df = ak.stock_value_em(symbol=str(code).zfill(6))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    rename = {
        "数据日期": "date",
        "PE(TTM)": "pe_ttm",
        "市净率": "pb",
        "总市值": "total_mv",
        "当日收盘价": "close",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = [c for c in ["pe_ttm", "pb", "total_mv", "close"] if c in df.columns]
    out = df[cols].astype(float)
    # ROE 近似：PB / PE
    if "pb" in out.columns and "pe_ttm" in out.columns:
        out["roe"] = out["pb"] / out["pe_ttm"]
        out.loc[out["pe_ttm"] <= 0, "roe"] = np.nan
        out.loc[out["pb"] <= 0, "roe"] = np.nan
        out["ep"] = 1.0 / out["pe_ttm"]
        out.loc[out["pe_ttm"] <= 0, "ep"] = np.nan
    return out


def load_or_fetch_valuation(
    code: str,
    cache_dir: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    cache = Path(cache_dir) / "valuation"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{code}.parquet"
    if path.exists() and not refresh:
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return pd.read_pickle(path.with_suffix(".pkl"))
    try:
        df = fetch_valuation(code)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if not df.empty:
        try:
            df.to_parquet(path)
        except Exception:  # noqa: BLE001
            df.to_pickle(path.with_suffix(".pkl"))
    return df


def download_valuation_map(
    codes: list[str],
    cache_dir: str | Path,
    sleep: float = 0.05,
    progress_every: int = 25,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    total = len(codes)
    for i, code in enumerate(codes, 1):
        try:
            df = load_or_fetch_valuation(code, cache_dir)
            if not df.empty:
                data[code] = df
        except Exception as e:  # noqa: BLE001
            if i % progress_every == 0:
                print(f"  val skip {code}: {e}")
        if i % progress_every == 0 or i == total:
            print(f"  valuation {i}/{total}  ok={len(data)}")
        time.sleep(sleep)
    return data


def load_delisted_histories(cache_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """加载 2021+ 退市股票行情（腾讯原始价），用于修正幸存者偏差。

    退市股没有 total_mv/outstanding_share；build_universe_at 会跳过市值过滤，
    只用成交额过滤。adj_close 用原始价（退市股分红影响小，且腾讯 hfq 对退市股不可靠）。
    """
    cache = Path(cache_dir)
    meta = cache / "delisted_2021.csv"
    out: dict[str, pd.DataFrame] = {}
    if not meta.exists():
        return out
    try:
        df = pd.read_csv(meta, dtype={"code": str})
    except Exception:  # noqa: BLE001
        return out
    for code in df["code"].dropna().astype(str).str.zfill(6):
        p = cache / "hist_delisted" / f"{code}.parquet"
        if not p.exists():
            continue
        try:
            h = pd.read_parquet(p)
            if h is None or h.empty:
                continue
            h = h.copy()
            h["adj_close"] = pd.to_numeric(h["close"], errors="coerce")
            h["adj_open"] = pd.to_numeric(h["open"], errors="coerce") if "open" in h.columns else h["adj_close"]
            if "total_mv" not in h.columns:
                h["total_mv"] = np.nan
            if "outstanding_share" not in h.columns:
                h["outstanding_share"] = np.nan
            out[code] = h
        except Exception:  # noqa: BLE001
            continue
    return out


def download_universe_history(
    codes: list[str],
    start_date: str,
    end_date: str | None,
    cache_dir: str | Path,
    sleep: float = 0.05,
    progress_every: int = 25,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    total = len(codes)
    for i, code in enumerate(codes, 1):
        try:
            df = load_or_fetch_history(code, start_date, end_date, cache_dir)
            if not df.empty:
                data[code] = df
        except Exception as e:  # noqa: BLE001
            if i % progress_every == 0:
                print(f"  skip {code}: {e}")
        if i % progress_every == 0 or i == total:
            print(f"  download {i}/{total}  ok={len(data)}")
        time.sleep(sleep)
    return data


# ---------------- 数据新鲜度 ----------------
def data_asof(cache_dir: str | Path = "data", benchmark: str = "000300") -> pd.Timestamp | None:
    """本地缓存里最近一个交易日（用指数缓存判断）。找不到返回 None。"""
    cache = Path(cache_dir)
    files = sorted(cache.glob(f"index_{benchmark}_*_today.parquet"))
    if not files:
        return None
    try:
        df = pd.read_parquet(files[-1])
        return pd.Timestamp(df.index.max())
    except Exception:  # noqa: BLE001
        return None


def assert_data_fresh(
    cache_dir: str | Path = "data",
    max_gap_days: int = 10,
    benchmark: str = "000300",
) -> pd.Timestamp:
    """数据太旧就拒绝继续，避免用过期行情出信号。"""
    asof = data_asof(cache_dir, benchmark)
    if asof is None:
        raise SystemExit("找不到本地指数缓存，请先运行 run_backtest.py 或刷新数据。")
    gap = (pd.Timestamp.today().normalize() - asof.normalize()).days
    if gap > max_gap_days:
        raise SystemExit(
            f"本地数据只到 {asof.date()}，已过期 {gap} 天（阈值 {max_gap_days} 天）。\n"
            f"请先刷新数据；确认要用旧数据请加 --force。"
        )
    return asof


def hist_asof(hist_map: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    """个股行情的「中位数」最后日期：少数刚刷新的股票不会掩盖大面积过期。"""
    dates = pd.Series([h.index.max() for h in hist_map.values() if h is not None and len(h)])
    return pd.Timestamp(dates.median()) if len(dates) else None


def assert_hist_fresh(
    hist_map: dict[str, pd.DataFrame],
    max_gap_days: int = 10,
) -> pd.Timestamp:
    """股票行情太旧也拒绝出信号（指数和个股可能不同步）。"""
    asof = hist_asof(hist_map)
    if asof is None:
        raise SystemExit("没有加载到任何个股行情。")
    gap = (pd.Timestamp.today().normalize() - asof.normalize()).days
    if gap > max_gap_days:
        raise SystemExit(
            f"个股行情只到 {asof.date()}，已过期 {gap} 天（阈值 {max_gap_days} 天）。\n"
            f"请先运行 python refresh_data.py stocks；确认要用旧数据请加 --force。"
        )
    return asof
