# Models README

本文档整理了原始 `README.md` 中关于时间序列预测方法的介绍，并结合业界常用实践进行了补充。

## 1. 当前项目已实现的 7 种预测方法

### 1.1 方法总览

| 方法代码 | 方法名称 | 多步策略 | 目标维度 | 是否常见 | 说明 |
|---|---|---|---|---|---|
| USMDO | 单变量多步直接输出 | MIMO/Direct | 单变量H步 | 中 | 更像“仅外生变量”基线 |
| USMD | 单变量多步直接 | Direct | 单变量H步 | 高 | 经典直接法，避免递归误差累积 |
| USMR | 单变量多步递归 | Recursive | 单变量1步滚动 | 高 | 工程最常见之一，训练简单 |
| USMDR | 单变量多步直接递归 | DirRec/Block-Recursive | 单变量 | 中高 | Direct 与 Recursive 折中 |
| MSMD | 多变量多步直接 | Direct (with multivariate lags) | 单变量H步 | 高 | 零售/负荷常见，效果通常好于 USMD |
| MSMR | 多变量多步递归 | Recursive | 多变量1步滚动 | 中高 | 适合系统联动变量预测 |
| MSMDR | 多变量多步直接递归 | DirRec | 多变量 | 中 | 实现复杂，调参成本高 |

### 1.2 各方法核心思路（整理版）

1. `USMDO`：仅使用外生特征（时间、节假日、天气等）直接输出未来 H 步。适合冷启动或缺少目标历史值场景。
2. `USMD`：使用目标序列滞后特征 + 外生特征，一次输出 H 步。适合短期预测和稳定性优先场景。
3. `USMR`：训练一步模型并递归滚动预测 H 步。训练快、部署简单，但长 horizon 误差会累积。
4. `USMDR`：分块递归（DirRec），在效率和误差累积之间折中。
5. `MSMD`：使用“目标+其他内生变量”的滞后特征，直接预测目标未来 H 步。多变量相关性强时常是首选。
6. `MSMR`：递归预测所有内生变量，保留变量联动关系，适合需要联动输出的系统建模。
7. `MSMDR`：多变量分块递归，思路先进但工程复杂度较高。

## 2. 当前方法是否业界常用

结论：**是，当前 7 种方法在“多步预测策略”层面属于业界主流范式**，尤其是 `Direct / Recursive / DirRec / Multi-output`。

依据：
- 这些策略本身是经典多步预测问题的标准分解方式，学术和工业中长期被采用。
- 现代时间序列库仍显式支持这些范式（如 `sktime` 的 direct/recursive/dirrec/reduction，`skforecast` 的 direct/recursive）。
- 在大规模实战竞赛（例如 M5）中，递归与非递归（direct）策略都被大量使用，尤其与 LightGBM 等模型结合时。

补充判断：
- `USMDO` 在业界更多作为 baseline 或冷启动方案，不是多数高精度生产系统的主模型。
- `MSMD/USMD` 这类“滞后特征 + 树模型 + 直接多步”是目前业务预测（零售、需求、负荷）非常常见的配置。

## 3. 业界常用但当前项目可进一步补充的方法

以下方法是目前业界高频出现、且与你当前框架兼容或可扩展的方向。

### 3.1 统计与可解释基线（建议优先补）

1. `SeasonalNaive / Naive`：低成本强基线，必须保留用于 sanity check。
2. `ETS/Holt-Winters`：季节性强的业务序列常见强基线。
3. `ARIMA/SARIMA/SARIMAX`：经典可解释方案，仍被大量用于产线和监控基线。
4. `AutoARIMA`：自动选阶，适合快速建模和小样本。
5. `Prophet`：节假日/事件驱动场景可解释性较好。

### 3.2 机器学习增强（与你当前路线最匹配）

1. `Global LightGBM/XGBoost`：跨序列联合训练（cross-learning），已被 M5 充分验证。
2. `Quantile LightGBM/CatBoost`：直接输出分位数预测，满足补货/容量场景的风险控制需求。
3. `分层预测 + Reconciliation`：集团-区域-门店-商品等层级业务中很常见。

### 3.3 深度学习（按业务规模逐步引入）

1. `DeepAR`：多序列概率预测工业落地经典（尤其大量相关序列）。
2. `N-BEATS / N-HiTS`：在通用单变量/多序列任务中表现稳定。
3. `TFT`：多输入、多步、需要解释性时常用。
4. `PatchTST / Autoformer / Informer`：长序列场景常见 Transformer 路线。

## 4. 对本项目的建议落地顺序

1. 先补统计强基线：`SeasonalNaive + ETS + SARIMAX`。
2. 在现有树模型基础上补概率预测：`Quantile`（P10/P50/P90）。
3. 补层级预测能力（如按门店/品类聚合后再协调）。
4. 数据规模足够时再引入 `DeepAR/N-BEATS/TFT/PatchTST`，并与现有树模型做集成。

## 5. 参考资料（调研来源）

1. Ben Taieb, S., Hyndman, R.J. (Direct/Recursive/Rectify): https://robjhyndman.com/publications/rectify/index.html
2. sktime Forecasting API（含 Direct/Recursive/DirRec/多类 forecaster）: https://www.sktime.net/en/stable/api_reference/forecasting.html
3. skforecast（Recursive/Direct 文档与实现）: https://skforecast.org/latest/user_guides/autoregresive-forecaster
4. statsmodels ARIMA/SARIMAX: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
5. statsmodels ExponentialSmoothing: https://www.statsmodels.org/v0.10.2/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
6. Prophet（Forecasting at Scale）: https://www.tandfonline.com/doi/abs/10.1080/00031305.2017.1380080
7. M4 Competition 结果总结: https://www.sciencedirect.com/science/article/abs/pii/S0169207018300785
8. M4 冠军方法（ES + RNN 混合）: https://econpapers.repec.org/article/eeeintfor/v_3a36_3ay_3a2020_3ai_3a1_3ap_3a75-85.htm
9. M5 背景论文: https://www.sciencedirect.com/science/article/pii/S0169207021001187
10. M5 Accuracy 结果论文（LightGBM 与递归/非递归策略）: https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
11. DeepAR（AWS 文档）: https://docs.aws.amazon.com/en_us/sagemaker/latest/dg/deepar.html
12. Autoformer (NeurIPS 2021): https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html
13. PatchTST (ICLR 2023): https://iclr.cc/virtual/2023/poster/10876
14. Informer (AAAI 2021): https://aaai.org/papers/11106-informer-beyond-efficient-transformer-for-long-sequence-time-series-forecasting/
