"""ETF 防守组合：数据 + 月度再平衡回测（熊市/防守用）"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError as e:  # pragma: no cover
    raise SystemExit("请先安装 akshare") from e


def to_sina_etf(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith("5") or s.startswith("6"):
        return f"sh{s}"
    return f"sz{s}"


def fetch_etf_hist(
    symbol: str,
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """ETF 日线。adjust="hfq" 用腾讯后复权（含分红/拆分）；adjust="" 用新浪原始价（下单用）。"""
    sym = to_sina_etf(symbol)
    start = pd.to_datetime(start_date)
    if adjust == "hfq":
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=sym,
                start_date=start.strftime("%Y%m%d"),
                end_date=(end_date or pd.Timestamp.today().strftime("%Y%m%d")).replace("-", ""),
                adjust="hfq",
            )
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.loc[start:]
            if end_date:
                df = df.loc[: pd.to_datetime(end_date)]
            return df[["close"]].astype(float)
        except Exception as e:  # noqa: BLE001
            print(f"  ETF {symbol} hfq 失败（{type(e).__name__}），回退新浪原始价")

    df = ak.fund_etf_hist_sina(symbol=sym)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[start:]
    if end_date:
        df = df.loc[: pd.to_datetime(end_date)]
    return df[["close"]].astype(float)


def load_etf_map(
    symbols: dict[str, str],
    start_date: str,
    end_date: str | None,
    cache_dir: str | Path,
    sleep: float = 0.1,
    adjust: str = "hfq",
) -> dict[str, pd.DataFrame]:
    cache = Path(cache_dir) / ("etf_hfq" if adjust == "hfq" else "etf")
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    end_tag = (end_date or "today").replace("-", "")
    for code, name in symbols.items():
        code = str(code)
        path = cache / f"{code}_{start_date.replace('-', '')}_{end_tag}.parquet"
        if path.exists():
            try:
                out[code] = pd.read_parquet(path)
                continue
            except Exception:  # noqa: BLE001
                pass
        try:
            df = fetch_etf_hist(code, start_date, end_date, adjust=adjust)
            if not df.empty:
                df.to_parquet(path)
                out[code] = df
                print(f"  ETF {code} {name}: {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  ETF {code} FAIL: {e}")
        time.sleep(sleep)
    return out


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).tail(1))


def backtest_etf_portfolio(
    price_map: dict[str, pd.DataFrame],
    weights: dict[str, float],
    cfg: dict,
    regime_series: pd.Series | None = None,
) -> pd.DataFrame:
    """固定权重、月末再平衡。支持市况切换权重（shift_on_regime + weights_bear）。"""
    ccfg = cfg.get("cost", {})
    commission = float(ccfg.get("commission", 0.00025))
    slippage = float(ccfg.get("slippage", 0.001))
    stamp = float(ccfg.get("stamp_duty", 0.0))  # ETF 卖出一般无印花税，这里默认为0
    only_regime = (cfg.get("backtest") or {}).get("only_regime")

    dates = sorted(set().union(*[set(p.index) for p in price_map.values()]))
    start = pd.to_datetime(cfg.get("backtest", {}).get("start_date") or min(dates))
    end = pd.to_datetime(cfg.get("backtest", {}).get("end_date") or max(dates))
    all_dates = pd.DatetimeIndex([d for d in dates if start <= d <= end])
    if len(all_dates) < 30:
        raise ValueError("交易日不足")

    # 统一成字符串键，避免 YAML int / str 混用
    weights = {str(k): float(v) for k, v in weights.items()}
    price_map = {str(k): v for k, v in price_map.items()}
    px = pd.DataFrame({k: p["close"].reindex(all_dates).ffill() for k, p in price_map.items()})
    px = px.dropna(how="all").ffill()
    rets = px.pct_change().fillna(0.0)

    sig_days = month_ends(all_dates)
    exec_of = {}
    for s in sig_days:
        later = all_dates[all_dates > s]
        exec_of[s] = later[0] if len(later) else None

    w_cur = {k: 0.0 for k in weights}  # 目标权重；0=空仓
    nav = 1.0
    peak = 1.0
    rows = []

    def in_regime(dt: pd.Timestamp) -> bool:
        if not only_regime or regime_series is None:
            return True
        return in_regime_named(dt, only_regime)

    def in_regime_named(dt: pd.Timestamp, name: str) -> bool:
        if regime_series is None:
            return False
        if dt in regime_series.index:
            return bool(regime_series.loc[dt] == name)
        prev = regime_series.loc[:dt]
        return bool(prev.iloc[-1] == name) if len(prev) else False

    for dt in all_dates:
        # 市况门控
        if only_regime is not None and regime_series is not None:
            if not in_regime(dt) and any(v > 0 for v in w_cur.values()):
                turn = sum(w_cur.values())
                cost = turn * (commission + slippage + stamp)
                nav *= 1.0 - cost
                w_cur = {k: 0.0 for k in weights}

        # 再平衡日
        if dt in exec_of.values():
            shift_regime = cfg.get("backtest", {}).get("shift_on_regime")
            weights_bear = cfg.get("weights_bear") or {}
            if only_regime:
                target = dict(weights) if in_regime(dt) else {k: 0.0 for k in weights}
            elif shift_regime and regime_series is not None and in_regime_named(dt, shift_regime):
                target = {str(k): float(v) for k, v in weights_bear.items()} if weights_bear else dict(weights)
            else:
                target = dict(weights)
            turn = sum(abs(target.get(k, 0) - w_cur.get(k, 0)) for k in set(weights) | set(w_cur) | set(target))
            cost = turn * (commission + slippage)
            sell_w = sum(w_cur.get(k, 0) - target.get(k, 0) for k in weights if target.get(k, 0) < w_cur.get(k, 0))
            cost += max(sell_w, 0) * stamp
            nav *= 1.0 - cost
            w_cur = dict(target)

        # 当日收益
        day_ret = 0.0
        for k, w in w_cur.items():
            if w > 0 and k in rets.columns:
                day_ret += w * float(rets.at[dt, k])
        nav *= 1.0 + day_ret
        peak = max(peak, nav)
        rows.append(
            {
                "date": dt,
                "nav": nav,
                "drawdown": 1.0 - nav / peak if peak else 0.0,
                "invested": sum(w_cur.values()),
            }
        )

    return pd.DataFrame(rows).set_index("date")
