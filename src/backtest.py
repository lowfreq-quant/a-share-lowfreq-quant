"""月度调仓回测（执行口径 v2，2026-10-18 修正）

修正点（相对旧版）：
1. 持仓判断用 **信号日 s** 的市况（月末收盘后已知），不用执行日收盘——去掉前视；
2. 建仓/调仓在 **执行日开盘**；老仓吃「昨收→今开」跳空，新仓吃「今开→今收」；
3. 市况转出在 t 日收盘确认，**t+1 开盘卖出**（吃 t 日跌幅 + 隔夜跳空），不再用前一日净值卖出；
4. 收益用 ``adj_close``（送转/拆股已复权），避免把 10 送 10 当成 -50%。
"""
from __future__ import annotations

import pandas as pd

from .portfolio import load_industry_map, select_top


def month_end_signals(all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(all_dates, index=all_dates)
    return pd.DatetimeIndex(s.groupby([all_dates.year, all_dates.month]).tail(1))


def run_backtest(
    hist_map: dict[str, pd.DataFrame],
    names: pd.DataFrame | dict[str, str],
    cfg: dict,
    val_map: dict[str, pd.DataFrame] | None = None,
    regime_series: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (日净值表, 调仓记录表)

    regime_series: index=date, values=牛市/熊市/震荡。
    若 cfg['backtest']['only_regime'] 有值，则仅该市况持仓，其余市况空仓。
    """
    start = pd.to_datetime(cfg["backtest"]["start_date"])
    end = pd.to_datetime(cfg["backtest"].get("end_date") or pd.Timestamp.today().strftime("%Y-%m-%d"))

    all_dates = sorted({pd.Timestamp(d) for h in hist_map.values() for d in h.index})
    all_dates = pd.DatetimeIndex([d for d in all_dates if start <= d <= end])
    if len(all_dates) < 40:
        raise ValueError("交易日不足，请扩大日期范围")

    ccfg = cfg["cost"]
    rcfg = cfg.get("risk", {})
    fee = float(ccfg["commission"]) + float(ccfg["slippage"])
    stamp = float(ccfg["stamp_duty"])
    industry_map = load_industry_map(cfg.get("data", {}).get("cache_dir", "data"))
    only_regime = (cfg.get("backtest") or {}).get("only_regime")
    apply_st = bool((cfg.get("backtest") or {}).get("point_in_time_st", True))

    # ---- 价格矩阵：close 原始价，adj_close 复权价，open 原始开盘 ----
    close = pd.DataFrame({c: h["close"] for c, h in hist_map.items()}).reindex(all_dates)
    adj = pd.DataFrame(
        {c: (h["adj_close"] if "adj_close" in h.columns else h["close"]) for c, h in hist_map.items()}
    ).reindex(all_dates)
    openp = pd.DataFrame(
        {c: (h["open"] if "open" in h.columns else h["close"]) for c, h in hist_map.items()}
    ).reindex(all_dates)

    adj_factor = (adj / close.mask(close == 0)).astype(float)
    adj_open = pd.DataFrame(
        {
            c: (h["adj_open"] if "adj_open" in h.columns else openp[c] * adj_factor[c])
            for c, h in hist_map.items()
        }
    ).reindex(all_dates)
    prev_adj = adj.shift(1)
    ret_close = adj / prev_adj - 1.0
    gap = adj_open / prev_adj - 1.0
    intra = adj / adj_open - 1.0

    signal_days = month_end_signals(all_dates)
    exec_of: dict[pd.Timestamp, pd.Timestamp | None] = {}
    for s in signal_days:
        later = all_dates[all_dates > s]
        exec_of[s] = later[0] if len(later) else None
    exec_days = {e: s for s, e in exec_of.items() if e is not None}

    def regime_ok(dt: pd.Timestamp) -> bool:
        if not only_regime or regime_series is None:
            return True
        if dt in regime_series.index:
            return bool(regime_series.loc[dt] == only_regime)
        prev = regime_series.loc[:dt]
        if prev.empty:
            return True
        return bool(prev.iloc[-1] == only_regime)

    def r(df: pd.DataFrame, c: str, dt: pd.Timestamp) -> float:
        if c not in df.columns:
            return 0.0
        v = df.at[dt, c]
        return float(v) if pd.notna(v) else 0.0

    holdings: dict[str, float] = {}
    nav = 1.0
    peak = 1.0
    scale = 1.0
    halted = False
    pending_exit = False

    nav_rows: list[dict] = []
    trade_rows: list[dict] = []

    for dt in all_dates:
        exited_today = False
        traded_today = False

        # 1) 昨收确认转出 → 今开卖出
        if pending_exit and holdings:
            gap_ret = sum(w * r(gap, c, dt) for c, w in holdings.items())
            nav *= 1.0 + gap_ret
            sell_w = sum(holdings.values())
            nav *= 1.0 - sell_w * (fee + stamp)
            trade_rows.append(
                {
                    "date": dt,
                    "signal_day": None,
                    "n_buy": 0,
                    "n_sell": len(holdings),
                    "turnover": sell_w,
                    "cost": sell_w * (fee + stamp),
                    "nav_after_cost": nav,
                    "note": f"CASH_not_{only_regime}",
                }
            )
            holdings = {}
            pending_exit = False
            exited_today = True

        # 2) 执行日：按信号日市况决定是否持有；开盘成交
        s = exec_days.get(dt)
        if s is not None and not halted and regime_ok(s):
            top = select_top(
                s,
                hist_map,
                names,
                cfg,
                val_map=val_map,
                industry_map=industry_map,
                apply_st=apply_st,
            )
            if top:
                w = scale / len(top)
                new = {c: w for c in top}
                old = holdings
                # 老仓隔夜跳空
                gap_ret = sum(wo * r(gap, c, dt) for c, wo in old.items())
                nav *= 1.0 + gap_ret
                # 换手成本
                allc = set(new) | set(old)
                turnover = sum(abs(new.get(c, 0) - old.get(c, 0)) for c in allc)
                sell_w = sum(old.get(c, 0) for c in set(old) - set(new))
                cost = turnover * fee + sell_w * stamp
                nav *= 1.0 - cost
                # 新仓日内
                intra_ret = sum(wn * r(intra, c, dt) for c, wn in new.items())
                nav *= 1.0 + intra_ret
                trade_rows.append(
                    {
                        "date": dt,
                        "signal_day": s,
                        "n_buy": len(set(new) - set(old)),
                        "n_sell": len(set(old) - set(new)),
                        "turnover": turnover,
                        "cost": cost,
                        "nav_after_cost": nav,
                        "note": "",
                    }
                )
                holdings = new
                traded_today = True

        # 3) 普通日：收盘到收盘
        if not traded_today and not exited_today and holdings:
            dr = sum(w * r(ret_close, c, dt) for c, w in holdings.items())
            nav *= 1.0 + dr

        # 4) 收盘确认市况转出 → 次日开盘卖
        if holdings and only_regime and regime_series is not None and not regime_ok(dt):
            pending_exit = True

        # 5) 回撤风控（收盘处理）
        peak = max(peak, nav)
        dd = 1.0 - nav / peak if peak > 0 else 0.0
        if dd >= float(rcfg.get("drawdown_halt", 0.20)) and not halted:
            sell_w = sum(holdings.values())
            nav *= 1.0 - sell_w * (fee + stamp)
            holdings = {}
            halted = True
            trade_rows.append(
                {"date": dt, "signal_day": None, "n_buy": 0, "n_sell": 0, "turnover": sell_w,
                 "cost": sell_w * (fee + stamp), "nav_after_cost": nav, "note": "HALT_DD20"}
            )
            pending_exit = False
        elif dd >= float(rcfg.get("drawdown_reduce", 0.15)) and scale == 1.0 and not halted:
            sell_w = 0.5 * sum(holdings.values())
            nav *= 1.0 - sell_w * (fee + stamp)
            scale = 0.5
            holdings = {c: w * 0.5 for c, w in holdings.items()}
            trade_rows.append(
                {"date": dt, "signal_day": None, "n_buy": 0, "n_sell": 0, "turnover": sell_w,
                 "cost": sell_w * (fee + stamp), "nav_after_cost": nav, "note": "REDUCE_DD15"}
            )

        nav_rows.append(
            {
                "date": dt,
                "nav": nav,
                "drawdown": dd,
                "n_pos": len(holdings),
                "invested": sum(holdings.values()),
                "halted": halted,
            }
        )

    return pd.DataFrame(nav_rows).set_index("date"), pd.DataFrame(trade_rows)
