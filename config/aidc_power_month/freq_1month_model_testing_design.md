# AIDC 月总用电量预测 · freq_1month 模型测试设计说明

> **场景**：`aidc_power_month`，A/B 两路数据中心。业务问题 = **月末预测下一自然月的总用电量（kWh）**。
> **路线**：月频建模——以 5min 功率聚合的月电量序列为目标，one-step（`predict_steps: 1`）直接预测下个月总量。样本极少（10 个月度点），定位为**低频对照链**；生产主链是日频 `freq_1day` 逐日预测后求和。
> **本文范围**：`config/aidc_power_month/route_{A,B}/freq_1month/window_length_{7,8,9,10}/` 四组共 72 个配置。A/B 两路配置结构完全相同，仅数据不同，下文不再分开赘述。

---

## 1. 测试总体思路

```text
历史月电量 10 个点（2025-10 ~ 2026-07，1ME 月末标签）
        │
        ├─ 滑窗回测：window_length = W 个月为一窗
        │   W7 → 4 个测试月（2026-04 ~ 07）
        │   W8/W9/W10 → 3 个测试月（2026-05 ~ 07）
        │   每窗：W-1 个月训练 → 预测第 W 个月 → 与真实值评分
        │
        └─ 最终预测：全部 10 个月训练 → 预测 2026-08 月总用电量
```

| 设计要素 | 为什么需要 | 简单示例（数值仅示意） |
|---|---|---|
| 月频 + one-step | 直接回答"下个月总量多少"，避免日频误差的月间累积；代价是样本只有 10 个点 | 特征行 = {上月、上上月、前3月电量, 目标月天气} → y = 目标月总量 |
| 四档窗口 W7~W10 | 样本极少时训练窗长度是一阶超参：太短过拟合、太长损失近端信息 | W8：用 2025-11~2026-06 共 8 个月（7 行训练）预测 2026-07 |
| `history_length: 10`（月数） | 月频下 history/window 语义都是月数；`main.py` 硬校验 window < history | W10 配 history 10 → 恰 1 个起点，W7 → 4 个测试窗 |
| 月均日用电量建模（`target_calendar_normalization: per_calendar_day`） | 2 月 28 天与 7 月 31 天的总量天然差 ~10%，不归一会把"天数"当主要信号 | 建模 y = 月总量 ÷ 当月天数，评分/输出前乘回目标月天数（point 与全分位数同一还原） |

> **核心认识**：月频每个测试窗只有 1 个评分点，全部结论建立在 3~4 个点上——这是探索性实验，不是正式选型依据；正式判定需积累到 A/B 各 31 窗。

---

## 2. 数据处理

### 2.1 目标序列

| 项 | 约定 | 为什么 |
|---|---|---|
| 来源 | 5min 功率（kW）`sum × 1/12` 聚合为月电量（kWh） | 业务计量单位是电量 |
| 范围 | 2025-10 ~ 2026-07，A/B 各 10 行 | 数据上线仅 10 个月，决定了整个实验的小样本性质 |
| 时间标签 | `freq: 1ME`，月末 00:00 | 与聚合产物对齐；`now_time = 2026-07-31` |

### 2.2 天气三段式严格信息集（与日频同一契约）

| 段 | 文件 | 内容 | 为什么需要 |
|---|---|---|---|
| 历史（训练） | `weather_monthly_stats_202510_202607.csv` | 1h 实测聚合的月统计（actual） | 训练期天气已知 |
| 回测 | `weather_monthly_stats_backtest_proxy_202604_202607.csv` | 目标月取**上一年同月**实测，`available_at` = 前一月末 | 回测月真实天气在预测原点不可得 |
| 正式未来 | `weather_monthly_stats_future_202608.csv` | 2025-08 同月纯代理 | 不含 2026-08 任何实测 |

```yaml
strict_weather_information_set: true
weather_history_source: actual   # 训练段
weather_backtest_source: proxy   # 回测段
weather_future_source: proxy     # 正式预测段
forecast_mode: forecast
```

> **核心认识**：历史教训——修复前回测曾直接使用测试月实测气象（oracle 泄露），泄露差分实证（扰动 7 月实测预测不变 10,501,740；扰动 proxy 预测变为 11,512,484）后才确认契约生效。任何"回测分数异常好"的月频结果都应先怀疑天气泄露。

### 2.3 one-step Direct 的目标月对齐（`align_direct_features_to_target: true`）

月频 `predict_steps=1` 下统一语义：

$$\{y_t,\ y_{t-1},\ y_{t-2},\ X_{t+1}\} \rightarrow y_{t+1}$$

| 要素 | 处理 | 为什么 |
|---|---|---|
| lag1 | = 最新已知月 $y_t$（不是 $y_{t-1}$） | one-step 预测原点就是 t，lag1 应为最近月 |
| 天气/日历 | = 目标月 $t+1$ 的值（shift -1 对齐） | 月总量由目标月天气驱动，原点月天气是陈旧信息 |
| 限制 | 仅支持 `predict_steps=1`，多步直接 RAISE | 多步需要逐 horizon 展开（日频方案），月频用不到 |

---

## 3. 特征工程

样本极少决定特征必须极简（每窗仅 2~5 个完整训练行）：

| 类别 | 内容 | 为什么需要 | 示例（仅示意） |
|---|---|---|---|
| 滞后 lag | `[1,2,3]` | 近 3 个月水平；更多 lag 会直接吃掉训练行 | `y_lag_1` = 上月总量（归一化后） |
| 日历 | month / quarter / year（3 个，已剔除全部子日粒度） | 月度季节性主要靠月份本身表达 | `dt_month = 8` |
| 天气 | 4 列：rt_tt2（月均气温）、cal_rh（湿度）、rt_ssr（月总辐射）、rt_ws10（风速），取目标月值 | 夏季空调电量与气温/辐射强相关 | `rt_tt2` 单位 K |
| 高级特征 | 全部关闭（rolling/diff 等） | 10 个样本撑不起更多派生维度 | — |

| 方法 | 特征面 |
|---|---|
| USMD（horizon=1 时即单输出） | lag 3 + 日历 3 + 天气 4 ≈ 10 个 predictor |
| USMDR / USMR | 同 USMD；USMDP 不生成 lag（纯日历+天气模板） |
| ST | 仅消费 lag 列做 NNLS 模板拟合 |

---

## 4. 模型与配置分组

每路 × 每窗口档（W7/W8/W9/W10）各 9 个配置，4 档共 72 个：

| 类别 | 配置 | 目的 |
|---|---|---|
| 概率主链 | lgbm usmd / usmdr / usmr / usmdp（quantile [0.1,0.5,0.9] + conformal） | 树模型四方法全谱系 + 区间 |
| 线性分位数 | qr usmd（quantile + conformal） | 线性分位数对照 |
| 点预测基线 | ridge / enet / lasso usmd（point，`scale_features: true`） | 小样本下正则化线性通常优于树 |
| 季节模板 | st usmd（point，NNLS 学 lag 权重） | naive 的可学习推广，零超参 |

| 训练设置 | 取值 | 为什么 |
|---|---|---|
| `quantile_monotone: true` | 全部 quantile 配置 | 消除 quantile crossing |
| 时间衰减 | 关闭 | 10 个样本再做加权没有意义 |
| Conformal | 开启但 `min_scores: 30` 不可达 | 每配置仅 3~4 个校准分 → **全部跳过**（正确行为，不降阈值） |
| 并行 | 全部 1（`window_parallel_workers: 1`） | 单配置秒级完成，无需并行 |

> **核心认识**：horizon=1 时 USMD ≡ USMDR、USMDP ≡ USMR（实测逐值相同），方法维度存在冗余；保留四名称仅为与日频矩阵对齐。

---

## 5. 评估协议

| 项 | 约定 | 为什么 |
|---|---|---|
| 主指标 | 每测试月 MAPE（eval_mask percentile=5） | 与日频同一评估口径 |
| 汇总 | 3~4 个测试月取 **median** | 单点爆炸不拖垮结论 |
| 对照 | seasonal naive = **上月实际值** | 小样本下 naive 极强，是硬门槛 |
| 还原 | 评分在月总量空间进行（归一化预测先乘回天数） | 业务只认月总量误差 |
| 产物 | `results_test/<scenario>/<setting>/`：test_scores_df.csv、cv_plot_df.csv、test_prediction.png、window_plots/ | setting = `模型-数据-方法-窗口档` |

---

## 6. 已知结论与边界（供阅读结果时参照）

- **ST 季节模板是唯一在 A/B 两路、全部窗口档稳定优于 naive 的模型**（A 路 W10 median 0.18%；B 路 W9 1.72%）；Lasso 在 A 路长窗口有效但 B 路不稳。
- LightGBM 四方法全部明显劣于 naive（约 7.6%~15.9%）——10 个样本不足以支撑树模型。
- QR 误差在多窗爆炸（最高 139%），且 P10=P50=P90 区间完全退化；全部 quantile 配置 P10~P90 回测覆盖率≈0，**概率区间不可用于决策**。
- 详细指标与可视化见同目录 `freq_1month_model_comparison_report.html`。
- 边界：每配置仅 3~4 个测试月，结论为探索性；生产预测以日频主链为准，月频作为独立交叉验证（两条链路月度预测互相印证时可信度最高）。
