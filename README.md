# lianghua · A 股低频量化研究

> 面向个人投资者的 A 股低频（月度/季度）量化研究工具：多因子选股、市况判定、ETF 防守配置、波动率目标、回测审计。
>
> English: A low-frequency A-share quantitative research toolkit built on `akshare` — factor stock selection, regime classification, defensive ETF allocation, volatility targeting, and backtest auditing.

**⚠️ 仅供研究与学习，不构成投资建议。历史回测有已知偏差，详见 [DISCLAIMER.md](DISCLAIMER.md)。**

---

## 这个项目做了什么

- **数据层**：`akshare`（新浪/腾讯）拉取全 A 日线、ETF 后复权、指数、行业、估值；本地 parquet 缓存。
- **选股**：全市场多因子（市值、动量、波动、换手、EP/ROE），月度调仓，含成本/涨跌停/停牌过滤。
- **市况**：沪深300 的牛/熊/震荡三态规则（60 日收益 + MA120/MA20/MA60 + 波动率）。
- **配置**：红利 ETF / 十年国债 ETF / 黄金 ETF 的固定权重组合，叠加 5%/60d 波动率目标。
- **审计**：执行口径、复权、幸存者偏差、ST 时点、样本外分段、参数邻域、IF 对冲成本。

## 研究结论（2026-10）

| 策略 | 口径 | 年化 | 夏普 | 最大回撤 |
|---|---|---:|---:|---:|
| **30/50/20 + 5%/60d 波动率目标**（推荐底仓） | 2017-08~2026-10，季度，含 1.5% 现金 | **6.35%** | **1.35** | **4.87%** |
| 30/50/20 基准（无波动率目标） | 同上 | 6.23% | 1.21 | 7.80% |
| 熊市防守 15/15/70 | 2020–2026，后复权 | 4.76% | 1.32 | 3.62% |
| choppy v2 选股（已放弃） | 2021–2026，修正后 | 1.83% | 0.31 | 10.71% |
| 沪深300 买入持有 | 2017-08~2026-10 | 2.51% | 0.15 | 38.45% |

**最重要的发现**：旧选股回测 24.55% 总收益，逐层修正后只剩 **10.61%**（年化 1.83%、2024 年后 -5.83%）：

1. 执行口径（建仓日按昨收、执行日市况前视） → 13.32%；
2. 退市股 + ST 时点（幸存者偏差） → **10.61%**。

所以本仓库**不推荐**把 choppy v2 当实盘策略；推荐的是「30/50/20 + 波动率目标」的防守底仓。完整推导见：
- [量化审计_20261018.md](量化审计_20261018.md)
- [research/ETF配置研究_20261018.md](research/ETF配置研究_20261018.md)
- [research/选股修正_20261018.md](research/选股修正_20261018.md)

## 目录结构

```
lianghua/
  src/                       # 核心库
    data.py                  # akshare 下载 + 缓存 + adj_close
    universe.py              # 股票池过滤
    factors.py               # 因子计算与合成
    portfolio.py             # 选股 + 行业约束
    backtest.py              # 月度调仓回测（执行口径 v2）
    report.py                # 绩效指标与报告
    signals.py               # 最新持仓信号
    regime.py                # 牛/熊/震荡判定
    etf_portfolio.py         # ETF 组合回测
  research/                  # 研究脚本与结果
  config*.yaml               # 策略参数
  run_defensive.py           # 防守组合：backtest/init/mark/sheet/rebalance
  run_backtest.py            # 选股回测
  run_monthly.py             # 月度操作台
  run_signals.py             # 持仓信号
  run_regime_check.py        # 市况检查
  refresh_data.py            # 数据增量刷新
  流程.md / 量化审计_*.md / ...  # 中文研究与交接文档
```

## 安装

Python **3.10+**（代码用了 `X | None` 类型标注）。

```bash
pip install -r requirements.txt
```

## 数据（不随仓库分发）

`data/` 被 `.gitignore` 排除：它约 678 MB、9,000+ 个 parquet 文件，且行情数据受数据源条款约束。首次使用请自行下载：

```bash
python download_all.py              # 全市场日线（断点续传）
python research/fetch_data.py       # ETF 后复权 + 沪深300
python research/fetch_hfq.py        # 50 亿+ 股票后复权（可选，较慢）
python research/fetch_delisted.py   # 2021+ 退市股（可选）
python refresh_data.py              # 日常增量刷新
```

数据来自 `akshare` 封装的新浪/腾讯接口；**不要把 `data/` 重新分发**。

## 用法

### 防守组合（推荐）

```bash
python run_defensive.py backtest        # 基准 + 波动率目标
python run_defensive.py init 1000000    # 建 100 万纸面组合
python run_defensive.py signal          # 当前波动/目标仓位
python run_defensive.py sheet           # 生成月度操作单
python run_defensive.py mark            # 估值 + 写回台账
python run_defensive.py rebalance       # 偏离 ≥5% 才再平衡
```

### 选股（研究用，不建议实盘）

```bash
python run_backtest.py config_choppy_v2.yaml
python run_monthly.py
python run_signals.py config_choppy_v2.yaml
python run_regime_check.py
```

### 研究复现

```bash
python research/run_fixed.py
python research/run_robust.py
python research/vol_target_validate.py
python research/vol_target_ladder.py
python research/corrected_experiment.py --quick --no-hfq
```

## 关键设计

- **执行口径 v2**：信号日收盘定市况 → 次日开盘成交 → 转出次日开盘卖；不再用执行日收盘做前视。
- **复权**：优先 `data/hist_hfq` 的后复权价（含分红/拆分）；没有则退回“送转/拆股”复权。
- **幸存者偏差**：2021+ 退市股 206 只纳入回测；信号日前 15 天无交易则剔除（防止退市后还被买入）。
- **ST 时点**：回测 `point_in_time_st: false`，不用当前 ST 名单过滤历史；实盘仍用当前名单。
- **波动率目标**：月末按 60 日实现波动把总仓位缩到 `min(1, 5%/vol)`，下限 50%。
- **成本**：佣金、印花税、滑点都写进回测。

## 已知局限

- 仓库不含数据；回测需要先下载，首次较慢。
- 后复权全量覆盖未完成（当前 406/2,873 只），其余用送转复权；分红影响约 0.2–0.4%/年。
- ST 历史名称不完整，用“回测不过滤 ST”作保守替代。
- 没有实盘交易接口（无 QMT/券商 API）；信号需手动下单。
- 回测仍有模型风险、数据风险、过拟合风险；**不要直接照抄下单**。

## 许可证

[MIT](LICENSE)。依赖库：`akshare`、`pandas`、`numpy`、`matplotlib`、`PyYAML`、`tqdm`，请遵守各自许可证。

## 致谢

- [akshare](https://github.com/akfamily/akshare)：A 股数据接口。
- 所有把回测偏差讲清楚的开源项目与文章。
