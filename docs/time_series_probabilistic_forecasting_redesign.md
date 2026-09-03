# 时间序列概率预测模块重构设计方案

> 文档状态：**Historical reference（2026-08-31）**
> 本文保留 2026-08-24 专项审计与设计背景，不再描述当前代码路径、配置数量或生产能力。当前事实以 [`multistep_forecasting_redesign.md`](./multistep_forecasting_redesign.md) 为准；canonical 当前只支持 point 与边际 quantile，不执行 CQR。
> 创建日期：2026-08-24  
> 最近修订：2026-08-24——根据方案评审补齐 quantile/interval 分离、目标变换逆序、标签可得性与实施顺序  
> 文档性质：历史专项设计与决策溯源
> 调研依据：[Kaggle 时间序列预测调研](./kaggle_time_series_forecasting_research.md)  
> 参考结构：[时间序列分解模块重构设计方案](./time_series_decomposition_redesign.md)  
> 本文不构成实施授权，也不再作为实施状态机。

## 1. 背景与目标

### 1.1 背景

当前项目已经具备可运行的概率预测主链，但它不是一个独立模块，而是散落在多个层次：

- `config/config_sections.py` 定义 `predict_type`、`quantiles`、`quantile_monotone` 和 CQR 参数；
- `models/ModelTraining.py` 按分位数分别训练模型，并把模型保存为嵌套 `dict`；
- `models/ModelForecasting.py` 在 9 种多步策略中收集分位数输出，并把最接近 0.5 的分位数当点预测；
- `models/ModelTesting.py` 把分位数列写入 `cv_plot_df.csv`，可选记录 CQR nonconformity score；
- `main.py` 在最终预测阶段恢复目标量纲、执行 CQR、执行分位数单调化并保存 `prediction.csv`；
- `utils/quantile.py` 和 `utils/conformal.py` 分别实现逐行排序与对称 CQR 校准。

因此项目当前已经能输出 `predict_q10 / predict_q50 / predict_q90`，也能在最终 forecast 阶段膨胀首尾区间，但还不能把这些结果作为一套可验证、可比较、可部署的概率预测产品。

本次代码与结果复核进一步确认：问题不只是“缺少几个指标”，还包括目标空间、后处理顺序、校准窗口时序、输出一致性和模型能力声明等契约没有统一。

### 1.2 核心判断

项目当前更准确的能力描述是：

> **已实现多步条件边际分位数预测和最终 forecast 的 pooled CQR 后处理，但尚未建立完整的概率预测评估、校准回测和统一数据契约。**

需要明确三层能力边界：

1. `q10/q50/q90` 是每个未来时点的**边际条件分位数**；
2. 当前输出不是联合轨迹分布，不能直接回答月总量分布、连续超限概率或路径场景概率；
3. CQR 改善的是指定区间的边际覆盖率，不会自动把一组分位数升级成联合概率模型。

### 1.3 重构目标

本次重构目标是把现有散列实现升级成独立、可验证的 `probabilistic/` 模块：

1. 建立 `ProbabilisticSpec`，统一配置合法性、量化网格、点预测分位数、区间定义和校准参数；
2. 建立结构化概率模型与预测结果对象，替代裸 `dict` 和 `Forecaster.quantile_outputs` 隐式副作用；
3. 建立可持久化的目标变换栈，严格按训练顺序的逆序恢复到原始目标空间；
4. 修复已确认的时序与结果一致性问题；
5. 新增 pinball、coverage、width、Winkler、crossing、calibration error 和 horizon-wise 诊断；
6. 建立因果的 prequential CQR 回测，使 CQR 能在历史窗口上被真实评价；
7. 把模型 quantile 与 CQR prediction interval 作为两类独立产物，禁止用校准边界覆盖 q10/q90；
8. 保持现有 863 个模型 YAML 可加载，现有 9 种多步预测策略的业务语义不变；
9. 为未来接入路径采样模型保留统一评估入口，但本轮不实现 DeepAR/GRU/Transformer。

### 1.4 成功标准

重构完成必须同时满足：

- 同一份量化结果在训练、回测、forecast、CSV、图和校准器中使用同一数值语义；
- `predict_value` 与 processed q50 严格一致；CQR 不修改 q50；
- `predict_q*` 始终表示模型分位数，CQR 区间使用独立的 `predict_pi*_lower/upper` 列；
- 任何区间 score 都能由保存的对应区间边界逐点复算；
- “最近 N 个校准窗口”按预测原点时间选择，且只有标签已在当前原点可得的记录能够进入校准集；
- 目标还原顺序由训练期记录的变换栈自动逆序执行，不在概率模块中硬编码；
- 点预测配置不生成空概率文件，分位数配置必然生成概率评估文件；
- 固定 horizon 与 `calendar_month` 都有明确的 `horizon_step`；
- 现有配置全部通过新旧配置归一化；
- 全套 unittest、配置 dry-run 和至少一个真实 quantile 配置端到端实跑通过。

### 1.5 非目标

本轮明确不做：

1. 不新增 DeepAR、GRU、TFT、N-BEATS 或 Transformer 训练管线；
2. 不把路径采样、copula、联合轨迹场景或层级 reconciliation 混入当前 GBDT quantile 主链；
3. 不修改 9 种多步预测方法的目标时点定义；
4. 不把概率预测等同于模型融合；
5. 不照搬 M5 的零售层级权重和缩放口径；
6. 不把 CQR 覆盖率提升当作模型精度提升；
7. 不在没有指标证据前直接实现 horizon-binned、asymmetric 或 regime-conditioned conformal；
8. 不把无泄漏的 prequential 流程误报为无需假设的有限样本覆盖定理；
9. 不修改现有实验 YAML；新配置写法先作为兼容入口落地。

---

## 2. 当前实现与使用方式

### 2.1 当前代码触点

| 层次 | 当前位置 | 当前职责 | 主要问题 |
|---|---|---|---|
| 配置 | `config/config_sections.py:261-317,389-400` | quantile、单调化、并行、CQR 参数 | 字段分散，没有统一 spec |
| 配置加载 | `config/config_loader.py:66-118` | 分组 YAML 展平到 dataclass | 分组名称不参与语义校验，CQR 字段可放错组仍生效 |
| 目标构造 | `features/FeatureEngineering.py` | 按 9 种方法生成单/多步目标 | 与概率逻辑共用，边界总体正确 |
| 分位数训练 | `models/ModelTraining.py:451-587,1205-1291` | 每个 q 独立训练、模型参数注入、并行 | if/elif 分散；能力不支持时只 warning |
| 分位数推理 | `models/ModelForecasting.py:537-591` | 识别 quantile bundle、取中位分位数、收集输出 | 裸 dict、形状不统一、侧通道状态 |
| 多步策略推理 | `models/ModelForecasting.py:698-1149` | Direct/Recursive/DirRec/多变量分位数轨迹 | 递归路径统一由中位数回填，语义未显式声明 |
| Blend quantile | `models/ModelForecasting.py:1205-1268` | 每个 q 的 Direct/Recursive 共用一组融合权重 | 权重学自中位数点误差，不优化概率指标 |
| 回测记录 | `models/ModelTesting.py:377-405` | 写 quantile 列、可选写 conformal score | score 在单调化前算，CSV 在单调化后写 |
| 点指标 | `models/ModelTesting.py:635-714` | R2/MSE/RMSE/MAE/MAPE/naive | 无概率指标、无 horizon 指标 |
| 回测后处理 | `models/ModelTesting.py:771-870` | 保存前逐行排序、画首尾区间 | 排序可能改变 q50，但 `Y_preds` 不同步 |
| forecast 编排 | `main.py:620-750` | 目标逆变换、CQR、单调化、CSV 输出 | 处理顺序分散，窗口选择方向错误 |
| CQR 工具 | `utils/conformal.py` | score 与 pooled symmetric CQR | 静默截断长度，参数校验不足，注释与实现冲突 |
| 单调化工具 | `utils/quantile.py` | 对 `predict_q*` 列逐行排序 | 依赖字符串排序，分位数标签语义松动 |
| 持久化 | `models/ModelTraining.py:1284-1300` | `model.pkl` 保存嵌套 quantile dict | 无 schema version、无显式概率元数据 |

### 2.2 当前配置入口

当前基础配置字段：

```yaml
overrides:
  model_strategy:
    model_type: lightgbm
    pred_method: univariate-single-multistep-direct
    predict_type: quantile
    quantiles: [0.1, 0.5, 0.9]
    quantile_monotone: true
  performance:
    quantile_parallel_workers: 1
```

启用当前 CQR：

```yaml
overrides:
  conformal:
    enable_conformal_calibration: true
    conformal_alpha: 0.1
    conformal_calibration_windows: 5
    conformal_min_scores: 30
```

当前仓库的 154 个 CQR YAML 实际都把 `enable_conformal_calibration` 放在 `model_strategy` 组。由于 loader 只把二级字段展平、并不校验字段所属组，配置仍然生效。这不是运行错误，但说明配置来源和职责边界已经漂移。

### 2.3 当前配置规模

2026-08-24 对 `config/**/*.yaml` 的可执行扫描结果：

| 指标 | 数量 |
|---|---:|
| 模型 YAML | 863 |
| `predict_type=point` | 465 |
| `predict_type=quantile` | 398 |
| quantile 网格为 `[0.1,0.5,0.9]` | 398 |
| `quantile_monotone=true` | 390 |
| `enable_conformal_calibration=true` | 154 |
| quantile + LightGBM | 378 |
| quantile + QuantileRegressor | 20 |
| quantile + 其他模型 | 0 |
| quantile + `scale_target=true` | 0 |
| quantile + 目标分解 | 234 |
| quantile + horizon feature | 48 |

398 个 quantile 配置覆盖 USMDP、USMD、USMR、USMDR、MSMD、MSMR、MSMDR、USBR；当前没有 MSBR quantile YAML，但代码分支允许它。

这组事实意味着：

- 新的 quantile grid 校验和模型能力注册表可以在不修改现有 YAML 的情况下落地；
- `scale_target` 的分位数回测错位是潜伏缺陷，不是当前 398 份配置的既有结果污染；
- 单调化语义修复会影响绝大多数 quantile 配置，必须明确旧结果失效范围。

### 2.4 当前训练方式

#### 2.4.1 模型目标注入

`Trainer._inject_quantile_params()` 按模型类型注入目标：

| 模型 | 当前注入 |
|---|---|
| LightGBM | `objective=quantile`, `metric=quantile`, `alpha=q` |
| XGBoost | `objective=reg:quantileerror`, `quantile_alpha=q` |
| CatBoost | `loss_function=Quantile:alpha=q` |
| HistGB | `loss=quantile`, `quantile=q` |
| QuantileRegressor | `quantile=q` |
| 其他模型 | warning 后沿用默认目标 |

“其他模型 warning 后继续”会把不支持分位数损失的普通回归器输出伪装成 q10/q50/q90。当前 YAML 没有踩中，但这是配置契约缺陷，重构后必须 fail-fast。

#### 2.4.2 单输出方法

USMDP、USMR、MSMR 以及 `horizon_feature` 展开后的 Direct 方法，每个 q 训练一个单输出模型：

$$
\hat Q_q(y_{t+h}\mid X_{t,h}) = f_q(X_{t,h})
$$

单输出 quantile 路径可以透传 `sample_weight`。

#### 2.4.3 多输出 Direct/DirRec

USMD、MSMD、USMDR、MSMDR 的常规多输出路径，对每个 q 再为每个 horizon 输出训练子模型：

$$
\hat{\mathbf Q}_q =
\left(f_{q,1}(X_t),\ldots,f_{q,H}(X_t)\right)
$$

默认成本近似为 `Q × H` 个基模型。当前大部分多输出 quantile 走 sklearn `MultiOutputRegressor`，不透传时间衰减权重；项目自定义 `DirectMultiOutputRegressor` 已在点预测路径支持逐输出 `sample_weight`，但 quantile 路径未统一复用。

#### 2.4.4 Recursive/DirRec 的概率路径语义

当前 Recursive 和 DirRec 在每一步同时得到所有分位数，但写回历史的是 `median_quantile` 对应预测：

$$
y^{history}_{t+h} \leftarrow \hat Q_{0.5}(y_{t+h})
$$

下一步的 q10/q50/q90 都条件于同一条中位数回填轨迹。这种做法避免每个 q 建立独立递归状态，但它表达的是“围绕中位数路径的条件边际带”，不是完整路径不确定性传播。

该行为不是简单 bug，但必须成为显式策略契约；当前代码和输出没有记录它。

#### 2.4.5 Blend quantile

USBR/MSBR 为每个 q 分别训练 Direct 与 Recursive 子模型，再用同一组权重融合：

$$
\hat Q_q = w_d\hat Q_q^{direct} + w_r\hat Q_q^{recursive}
$$

`ridge_stacking` 权重来自 median 分位数的 Direct/Recursive 点预测分量，因此它只优化中位数点误差，不保证 pinball、coverage 或区间宽度改善。

### 2.5 当前推理和输出方式

`Forecaster._predict_point_and_quantiles()` 返回：

- point 模型：`(point_pred, None)`；
- quantile bundle：`(median_q_pred, {q: pred_q})`。

如果没有精确 q50，当前会选择最接近 0.5 的分位数并把它当 `predict_value`。这会让 `predict_value` 的统计含义依赖配置网格，且没有 fail-fast。

最终 `prediction.csv` 结构：

```text
time,predict_value,predict_q10,predict_q50,predict_q90
```

当前 setting 自动追加 `-quantile`，因此 point 与 quantile 结果目录不会直接碰撞。

### 2.6 当前后处理顺序

#### 滑窗回测

当前顺序为：

```text
模型预测
  → point 目标逆变换
  → quantile calendar normalization
  → 写 quantile 列
  → 用未单调化首尾 quantile 计算 conformal_score
  → 汇总全部窗口
  → 保存前逐行排序 quantile 列
  → 写 cv_plot_df.csv / 绘图区间
```

#### 最终 forecast

当前顺序为：

```text
模型预测
  → point / quantile 目标逆变换
  → calendar normalization
  → 读取 cv_plot_df conformal_score
  → CQR 校准首尾 quantile
  → 逐行排序 quantile 列
  → 写 prediction.csv
```

两个顺序都没有统一的“原始输出、目标空间输出、单调输出、校准输出”阶段对象。

### 2.7 当前 CQR 使用方式

启用条件：

1. `predict_type=quantile`；
2. 至少两个分位数；
3. `is_testing=true` 先生成 `cv_plot_df.csv`；
4. `enable_conformal_calibration=true`；
5. 有不少于 `conformal_min_scores` 个有限 score；
6. 当前硬性禁止与目标分解组合。

score：

$$
s_i=\max\{\hat q_{low,i}-y_i,\ y_i-\hat q_{high,i}\}
$$

校准量：

$$
E_\alpha=
\operatorname{Quantile}_{\lceil(1-\alpha)(n+1)\rceil/n}(s_1,\ldots,s_n)
$$

输出区间：

$$
[\hat q_{low}-E_\alpha,\ \hat q_{high}+E_\alpha]
$$

注意：score 可以为负，意味着真实值位于原始区间内部。`utils/conformal.py` 的实现和单测允许负值，但函数 docstring 写着“score >= 0”，当前文档与实现冲突。

### 2.8 当前使用边界

- quantile 输出是时点边际，不是轨迹联合分布；
- point 指标全部基于 q50/最近 0.5 的分位数；
- `test_scores_df.csv` 不评价 q10/q90；
- CQR 只改变最终 forecast 的首尾分位数，不改变 q50；
- 当前回测 CSV 保存的是未校准区间，不能直接评价 CQR 后区间；
- 多输出 quantile 不支持时间衰减权重；
- quantile 与 ensemble 主路径不组合；
- quantile 与目标分解可组合，但 conformal 与目标分解当前被拒绝。

### 2.9 当前实际运行与结果解读

本地入口仍是修改 `main.py` 的 `CONFIG_YAML`，然后运行：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python main.py
```

当配置同时设置 `is_testing=true` 和 `is_forecasting=true` 时，执行顺序为：

```text
滑窗回测
  → 生成 cv_plot_df.csv 和 conformal_score
  → 全历史重新训练
  → 读取回测 score 校准最终 forecast
  → 生成 prediction.csv
```

结果目录由 `<scenario_subpath>/<setting>` 隔离；quantile setting 自动含 `-quantile`：

```text
results/pretrained_models/<scenario>/<setting>/model.pkl
results/results_test/<scenario>/<setting>/test_scores_df.csv
results/results_test/<scenario>/<setting>/cv_plot_df.csv
results/results_forecast/<scenario>/<setting>/prediction.csv
```

当前结果解读规则：

1. `test_scores_df.csv` 的 MAPE/MAE 只评价 q50 点预测，不能据此判断 q10/q90；
2. `cv_plot_df.csv` 的 `predict_q10/q90` 是未 CQR 校准的回测边界；
3. `conformal_score` 只为最终 forecast 提供校准样本；当前不能从同一 CSV 直接得到 CQR 后覆盖率；
4. `prediction.csv` 的首尾 quantile 在 CQR 开启时是校准后边界，但没有未来真值，不能当场评价实际覆盖率；
5. 要比较 raw 与 calibrated CQR，必须构造严格时间前推的回测；这正是 §10.8 的重构目标；
6. 现阶段不能用 q50 MAPE “持平”否定 CQR，因为 CQR 按设计不修改 q50；应比较 coverage、width 和 Winkler。

---

## 3. Kaggle 调研对本项目的直接要求

### 3.1 不需要再扩展常规模型数量

[Kaggle 调研 §9](./kaggle_time_series_forecasting_research.md#9-概率预测当前最大能力缺口) 已明确：当前最大缺口不是再接一个树模型，而是建立概率评估闭环。

M5 Uncertainty 的高价值经验是以 pinball/weighted pinball 直接评价分位数，而不是只看 q50 的点误差。Enefit 的高价值经验是按多个时间折验证方向一致，并保持 forecast-origin 信息集严格；两者都不能支持“只看一份 prediction.csv 区间看起来合理”这种判断方式。

### 3.2 必须从窗口总分扩展到 horizon-wise

当前整窗 median MAPE 会掩盖：

- 远 horizon 的区间变窄或漏覆盖；
- 峰段/突变日的 q90 失效；
- q10 与 q90 尾部误差不对称；
- coverage 提升是否只是 width 膨胀；
- crossing 是否集中在某一 horizon。

因此概率指标必须至少支持：

```text
window × horizon_step × quantile / interval × metric × stage
```

### 3.3 CQR 必须先被评价，再扩展算法

调研给出的顺序是正确的：

1. 先实现 pooled CQR 的因果 prequential 评价；
2. coverage 随 horizon 下降时，才进入 horizon-binned CQR；
3. 上下尾显著不对称时，才进入 asymmetric CQR；
4. regime 先作为离线诊断维度，不能直接用未来标签做在线校准条件。

这里的“因果”只表示不消费当前及未来标签。标准 split conformal 的有限样本覆盖结论依赖交换性；滚动时间序列中的序列相关、分布漂移和“最近窗口”选择会削弱该假设，因此项目必须把目标 coverage 当作回测验收目标，而不是无条件定理。

### 3.4 路径级概率属于后续独立能力

当前 quantile-by-horizon 不能回答路径问题。DeepAR/GRU 的价值是通过采样表达联合轨迹，而不是替换现有所有 GBDT。

本轮只要求新评估接口未来能消费路径模型输出，不为不存在的模型预建训练空壳。

---

## 4. 已确认问题与优先级

### 4.1 问题总览

| 优先级 | 问题 | 当前影响 | 处理结论 |
|---|---|---|---|
| **P0** | CQR “最近 N 窗口”实际选到最旧窗口 | 154 个 CQR 配置 | 按 forecast origin 时间选取 |
| **P0** | CQR 边界覆盖写回 q10/q90 | calibrated bound 被误标为 quantile | quantile 与 prediction interval 分列输出 |
| **P0** | score 在单调化前算，CSV 边界在单调化后写 | score 无法由保存区间复算 | 统一阶段顺序并保留 raw/final |
| **P0** | 单调化可能改变 q50，但 `Y_preds/predict_value` 不同步 | 点指标与 q50 输出不一致 | q50 作为锚点，或同步 point |
| **P0** | 无 pinball/coverage/width/Winkler | 无法评价概率质量 | 新建概率指标模块和结果表 |
| **P0** | CQR 没有 prequential 回测 | 只能改 final forecast，不能证明有效 | 历史窗口按时间顺序校准→评价 |
| **P0** | prequential 只比较 origin，未校验标签可得时间 | 长 horizon/延迟数据可能偷看未到达标签 | 以 `label_available_at <= current_origin` 筛选 |
| **P0** | quantile 配置只校验非空 | 非法 q、重复 q、无 q50 可静默运行 | `ProbabilisticSpec` fail-fast |
| **P0** | CQR/quantile 工具静默截断长度 | schema 错位被掩盖 | 精确长度契约，异常 RAISE |
| **P0** | 文档中的目标逆变换顺序与训练顺序不互逆 | calendar/decomposition 组合会还原错位 | 持久化变换栈并严格逆序恢复 |
| P1 | quantile 目标逆变换分散 | `scale_target` 回测分位数列潜伏错位 | point/quantile 共用 TargetTransformPipeline |
| P1 | 多输出 quantile 不透传 sample_weight | Direct/DirRec 丢失时间衰减 | 复用 `DirectMultiOutputRegressor` |
| P1 | 模型分位数能力靠 if/elif + warning | 不支持模型可伪装 quantile | objective capability mapping |
| P1 | 分位数列名不是可逆 codec | 高精度 q 可能列名冲突，q100 字符串排序错误 | 数值网格为权威、列名只做序列化 |
| P1 | quantile bundle 是裸 dict | 无 schema/version/类型约束 | dataclass bundle |
| P1 | Recursive 概率路径语义隐式 | 容易误报为完整概率轨迹 | 显式 `median_path` 策略与元数据 |
| P1 | 窗口并行未强制压低 quantile 并行 | 未来启用 Q 并行时会超额订阅 | 统一并行预算 |
| P2 | Blend 权重只优化 q50 | 概率收益不可解释 | 先输出分量概率指标，再决定优化目标 |
| P2 | 只有 pooled symmetric CQR | horizon/尾部失准无法定向治理 | 指标证据满足后扩展 |
| P2 | 无路径联合概率 | 不能评价月总量/连续超限 | 独立研究支线 |

### 4.2 CQR 校准窗口方向错误

固定 horizon 下：

- `window=1` 是最新测试窗口；
- `window` 越大，时间越早。

代表性结果 `aidc_load_15min_daily/route_A/.../cv_plot_df.csv`：

| window | 测试日期 |
|---:|---|
| 1 | 2026-07-31 |
| 2 | 2026-07-30 |
| 3 | 2026-07-29 |
| 29 | 2026-07-03 |
| 30 | 2026-07-02 |
| 31 | 2026-07-01 |

当前 `main.py:690-692`：

```python
recent_windows = sorted(cv_df["window"].unique())[-n_cal_windows:]
```

当 `n_cal_windows=5` 时实际选中 27–31，即最旧的 5 个窗口。Blend ridge stacking 使用了同样的选择方式。

修复不能简单写成 `[:N]`，因为未来窗口编号规则可能改变；应按 `forecast_origin` 或测试时间最大值排序并执行 as-of 选择。

### 4.3 point 与 q50 不一致

当前单调化对每行全部 `predict_q*` 值排序后重新贴标签。若原始分位数交叉，q50 可能换成原 q10 或 q90 的值；但 `Y_preds/predict_value` 仍保留单调化前的中位数模型输出。

2026-08-24 对存量 `results/results_test` 的扫描：

| 指标 | 结果 |
|---|---:|
| `cv_plot_df.csv` 总数 | 463 |
| 含至少两个 quantile 的 CSV | 236 |
| `Y_preds` 与保存后 q50 不一致 | 169 |
| 保存后仍有 crossing | 33 |

这些存量文件跨多个实验版本，不能当正式精度汇总；但足以证明输出契约不一致是现实问题。

推荐使用**以 q50 为固定锚点的单调投影**：修正上下分位数，不改变 q50。若暂时保留 legacy rearrangement，则必须同步 `predict_value=processed q50`，并明确旧结果失效。

### 4.4 conformal score 与保存区间不一致

score 在窗口内、单调化之前计算；`cv_plot_df.csv` 在全部窗口结束后才单调化。存量扫描中，82 份含 score 的 CSV 无法用保存后的首尾边界复算原 score。

这破坏三个验收条件：

- 校准样本不可审计；
- raw/processed 区间无法区分；
- 后续 coverage 计算不知道应该对应哪个 stage。

### 4.5 概率覆盖率缺口是真实存在的

236 份存量 quantile 回测 CSV 的 q10–q90 经验覆盖率中位数约为 **32.7%**。该扫描混合不同场景、配置和生成时间，只作为诊断基线，不作为正式模型排名。

这一结果与调研中 15min 场景原始区间覆盖率约 30%–43% 的专项分析一致：当前概率预测最大问题是区间严重欠覆盖，而现有 `test_scores_df.csv` 完全看不出来。

### 4.6 raw quantile 与 CQR prediction interval 语义混淆

当前 154 个 CQR 配置全部使用：

- quantiles：`[0.1,0.5,0.9]`，原始中心区间名义覆盖为 80%；
- `conformal_alpha=0.1`，CQR 目标覆盖为 90%。

这在数学上不是无效配置：split conformal 可以把任意基础区间校准到 90%。但校准后的两个边界不再是条件 q10/q90，而是目标 90% coverage 的 prediction interval bounds。当前代码把它们写回 `predict_q10/predict_q90`，会把两类统计对象混为一谈。

新配置必须把以下三个量同时写清：

```text
lower_quantile=0.1
upper_quantile=0.9
calibration_target_coverage=0.9
```

legacy 配置保持 90% 行为，但当 `upper-lower != target_coverage` 时记录显式 WARNING 和元数据。

规范输出必须分开：

```text
predict_q10,predict_q50,predict_q90       # 模型分位数，只做 crossing repair
predict_pi90_lower,predict_pi90_upper     # CQR 后的 90% prediction interval
```

Pinball 只评价 quantile；coverage/width/Winkler 评价 interval。CQR 不覆盖 quantile 列。

### 4.7 潜伏的目标缩放错位

滑窗回测中 point 会经 `target_scaler.restore_predictions()`，quantile 列写入 `cv_plot_df` 前没有统一执行同一逆变换；只有 CQR score 的局部副本尝试恢复量纲。

当前 398 个 quantile YAML 全部 `scale_target=false`，所以不是现役结果污染；但配置系统允许开启，属于启用即踩的潜伏缺陷。

### 4.8 quantile 列名不是可靠数据契约

当前列名：

```python
f"predict_q{int(round(q * 100)):02d}"
```

问题：

- `q=0.101` 与 `q=0.104` 都可能编码为 `predict_q10`；
- `predict_q100` 的字符串排序位于 `predict_q50` 前；
- 代码多处从列名前缀推断顺序，没有共同 parser；
- CSV 没有保存权威 quantile grid 元数据。

现有 q10/q50/q90 没有列名碰撞，但架构不能继续依赖字符串偶然正确。

### 4.9 工具函数 fail-soft 与项目新契约冲突

`compute_nonconformity_scores()` 取三个输入的最短长度，现有单测还断言这种截断。它会把 Direct 输出、目标恢复或 CSV 对齐错误变成静默少算。

多步 Direct 已改为长度不等即 RAISE；概率预测必须采用同一纪律。

### 4.10 目标变换逆序设计有误

当前训练顺序为：

```text
calendar normalization
  → decomposition
  → target scaling（Trainer 内）
```

证据见 `main.py:834-848` 和 `models/ModelTesting.py:173-243`。因此统一组合后的正确逆序必须是：

```text
inverse target scaling
  → decomposition restore
  → inverse calendar normalization
```

本方案原写法把 calendar restore 放在 decomposition 之前，与训练顺序不互逆。修订后不再让概率模块自行排列这些变换，而是由共享 `TargetTransformPipeline` 记录 fit 顺序并自动逆序 restore；point 和 quantile 使用同一实例。

### 4.11 校准样本必须满足标签已可得

仅要求 `record.forecast_origin < current_forecast_origin` 不足以证明标签在当前时刻已经可用。长 horizon、重叠窗口或 T+1/T+2 数据延迟下，较早 origin 对应的部分真实值仍可能尚未到达。

校准记录必须增加 `label_available_at`，筛选条件改为：

```text
record.label_available_at <= current_forecast_origin
```

如业务没有额外发布延迟，默认 `label_available_at = target_time`；配置 `label_availability_delay_steps=N` 时再增加 `N × freq`。存在独立 `available_at` 数据契约时，以数据源声明为准。

### 4.12 P0 指标与实施阶段冲突

§4.1 把概率指标和 prequential CQR 定义为 P0，但原 §13 把它们排到 Phase 4，晚于 spec、类型、训练器和 9 种推理路径迁移。这会导致大重构前仍没有概率验收工具，也与“先增强评估、不改变预测值”的调研结论冲突。

修订后的实施顺序改为：先在最终 `probabilistic/` 包中落 metrics、严格 score、interval 输出和 prequential evaluator，并通过薄适配器接入当前主链；之后再迁移训练和推理对象，避免写临时实现。

---

## 5. 模块放置与边界

### 5.1 推荐位置

新增顶层包：

```text
probabilistic/
  __init__.py
  spec.py
  types.py
  objectives.py
  training.py
  postprocessing.py
  metrics.py
  calibration.py
  evaluation.py
  pipeline.py
```

第一版不为单一 CQR 预建 `calibration/base.py + cqr.py` 子包，也不把 evaluation/diagnostics、types/bundle 过早拆开。出现第二种 calibrator 或第二套持久化 schema 后，再按真实变化轴拆分。

### 5.2 为什么不继续放在 `utils/`

概率预测不是两个无状态数学函数：它连接配置、训练目标、模型持久化、多步推理、目标逆变换、后处理、滑窗评价和最终输出。

继续把逻辑放进 `utils/quantile.py` / `utils/conformal.py` 会保留当前“工具函数存在，但主链顺序由调用方临时拼接”的问题。

### 5.3 为什么不全部放在 `models/`

`models/` 负责回归器和 9 种多步策略；pinball、coverage、校准数据选择、prequential 评价和 artifact schema 不属于单一模型实现。

概率模块应横跨：

- `models/ModelTraining.py`：构造 quantile 模型；
- `models/ModelForecasting.py`：产生各策略的分位数；
- `models/ModelTesting.py`：评价；
- `main.py`：最终 forecast 编排；
- `decomposition/` 与 `features/FeatureScalering.py`：恢复到原始目标空间。

### 5.4 与其他模块的边界

| 模块 | 保留职责 | 概率模块消费方式 |
|---|---|---|
| `features/` | 特征和多步目标列 | 只消费 X/Y，不重定义 shift 语义 |
| `models/ModelFactory.py` | 创建具体回归器 | capability adapter 注入 quantile 目标 |
| `models/ModelTraining.py` | 训练主流程和数据预处理 | 委托 `QuantileTrainer` |
| `models/ModelForecasting.py` | 构造上下文并委托五类多步 executor | 返回统一 `ForecastDistribution` |
| `features/TargetTransformation.py` | 实现共享 `TargetTransformPipeline`，记录 calendar/decomposition/scaler 的训练顺序 | 对 point/quantile 执行同一逆序 restore |
| `decomposition/` | 确定性分量还原 | 作为 TargetTransformPipeline 的一个有序步骤 |
| `utils/eval_mask.py` | 点预测业务掩码 | 不默认改变 conformal 覆盖总体 |
| `main.py` | 流程入口与结果目录 | 只调用概率 pipeline，不手写 CQR 分支 |

---

## 6. 目标架构

### 6.1 分层视图

新架构分 8 层：

1. **Spec**：配置归一化与合法性矩阵；
2. **Capability Mapping**：模型是否支持 quantile、如何注入目标；
3. **Quantile Trainer**：按 q 和输出形态训练，构造强类型 bundle；
4. **Strategy Adapter**：把 9 种多步方法输出转换为统一二维 quantile matrix；
5. **Target Transform Consumer**：消费共享变换栈，按训练顺序的逆序恢复 point/quantile；
6. **Postprocessor**：crossing 诊断和单调化；
7. **Calibrator**：按标签可得时间选择历史 score，新增独立 prediction interval；
8. **Evaluator/Writer**：生成概率预测明细、window/horizon 指标和校准报告。

### 6.2 数据流

```text
X / multi-step Y
  │
  ▼
QuantileTrainer ───────────────► ProbabilisticModelBundle
                                      │
                                      ▼
9 种预测策略 ────────────────► ForecastDistribution(model_space)
                                      │
                                      ▼
TargetTransformPipeline.restore
                       ───────► ForecastDistribution(target_space/raw quantiles)
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                  crossing diagnostics      monotone projection
                                                  │
                                                  ▼
                                      processed quantiles + q50 point
                                                  │
                               ┌──────────────────┴─────────────────┐
                               ▼                                    ▼
                       quantile evaluator                    CQR calibrator
                               │                                    │
                               ▼                                    ▼
                   pinball/crossing             independent prediction interval
                                                                    │
                                                                    ▼
                                                     coverage/width/Winkler + writer
```

### 6.3 核心不变式

1. quantile level 始终以 `float` 数组保存，列名不是权威来源；
2. quantile matrix 统一形状为 `(n_steps, n_quantiles)`；
3. quantile levels 严格递增、唯一且位于 `(0,1)`；
4. tsproj_ml 的 quantile 模式必须含 q50，并且 `point == processed_quantiles[:, q50_index]`；
5. calibrator 只接受 `space=target` 的结果；
6. CQR score 与所引用的 boundary stage 必须一致；
7. 任意 shape 不一致直接 RAISE，不截断、不 edge pad；
8. CQR 只新增 prediction interval，不覆盖 processed quantile；
9. 每条校准样本携带 `forecast_origin/time/label_available_at/horizon_step/window`；
10. 所有窗口选择按时间与标签可得性执行，不按 window ID 猜方向；
11. raw、processed quantile 与 calibrated interval 三类产物不互相覆盖。

### 6.4 构造映射职责

第一版不单独建立通用 registry 文件。以下映射分别放在其真实归属模块中：

- `objectives.py`：model type → quantile objective adapter；
- `postprocessing.py`：crossing method → postprocessor；
- `calibration.py`：calibration method → calibrator；
- `pipeline.py`：根据 `ProbabilisticSpec` 组装上述对象。

映射只做构造，不执行训练、不计算指标、不读 CSV。只有当组件类型扩展到需要插件注册时，再抽取 `registry.py`。

---

## 7. 参数配置架构

### 7.1 当前字段

| 当前字段 | 所在 dataclass | 当前语义 |
|---|---|---|
| `predict_type` | `ModelStrategyConfig` | point / quantile |
| `quantiles` | `ModelStrategyConfig` | 分位数网格 |
| `quantile_monotone` | `ModelStrategyConfig` | 保存前逐行排序 |
| `quantile_parallel_workers` | `PerformanceConfig` | q 级线程并行 |
| `enable_conformal_calibration` | `ConformalConfig` | 开启 CQR |
| `conformal_alpha` | `ConformalConfig` | 目标误覆盖率 |
| `conformal_calibration_windows` | `ConformalConfig` | 校准窗口数 |
| `conformal_min_scores` | `ConformalConfig` | 最小 score 数 |

### 7.2 配置设计原则

1. 现有字段继续作为兼容输入；
2. 新增 `probabilistic: Dict[str, Any]` 原始 mapping；
3. 主链内部只消费 `ProbabilisticSpec`；
4. 新旧字段同义且一致时接受，冲突时 RAISE；
5. `point` 模式不允许配置 calibration；
6. q50 必须显式包含；
7. interval pair 必须引用已配置 quantile；
8. target coverage 与 lower/upper quantile 分开表达；
9. 未实现方法 fail-fast，不做 warning 降级；
10. 原始 YAML dict 在 spec 构建后不进入运行期。

### 7.3 兼容旧写法

```yaml
overrides:
  model_strategy:
    predict_type: quantile
    quantiles: [0.1, 0.5, 0.9]
    quantile_monotone: true
  conformal:
    enable_conformal_calibration: true
    conformal_alpha: 0.1
    conformal_calibration_windows: 5
    conformal_min_scores: 30
```

legacy 归一化：

- `point_quantile=0.5`；
- `crossing_method=rearrangement`（迁移期兼容）；
- interval 为 `(0.1,0.9)`；
- CQR target coverage 为 `1-0.1=0.9`；
- 当 raw nominal coverage `0.9-0.1=0.8` 与 0.9 不同时记录 warning，但不改变旧行为。

### 7.4 新增规范写法

```yaml
overrides:
  probabilistic:
    mode: quantile
    quantiles: [0.1, 0.5, 0.9]
    point_quantile: 0.5
    crossing:
      method: median_preserving_isotonic
      report_raw: true
    intervals:
      - name: p10_p90
        lower_quantile: 0.1
        upper_quantile: 0.9
    calibration:
      method: cqr
      interval: p10_p90
      target_coverage: 0.8
      calibration_windows: 5
      min_windows: 3
      min_scores: 30
      label_availability_delay_steps: 0
      allow_interval_shrink: false
      grouping: pooled
```

该 mapping 由 `BaseModelConfig.probabilistic` 整体保留，内部嵌套由 `resolve_probabilistic_spec()` 自行严格校验。

### 7.5 内部 spec

```python
@dataclass(frozen=True)
class IntervalSpec:
    name: str
    lower_quantile: float
    upper_quantile: float

    @property
    def nominal_coverage(self) -> float:
        return self.upper_quantile - self.lower_quantile


@dataclass(frozen=True)
class CalibrationSpec:
    method: str
    interval_name: str
    target_coverage: float
    calibration_windows: int
    min_windows: int
    min_scores: int
    label_availability_delay_steps: int = 0
    allow_interval_shrink: bool = False
    grouping: str = "pooled"


@dataclass(frozen=True)
class ProbabilisticSpec:
    mode: str
    quantiles: tuple[float, ...]
    point_quantile: float
    recursive_propagation: str
    crossing_method: str
    intervals: tuple[IntervalSpec, ...]
    calibration: CalibrationSpec | None
    schema_version: int = 1
```

### 7.6 合并规则

`probabilistic/spec.py`：

```python
def resolve_probabilistic_spec(cfg) -> ProbabilisticSpec:
    ...
```

规则：

1. 只有旧字段：生成 legacy spec；
2. 只有新 mapping：生成 canonical spec；
3. 两者都没有：返回 `mode=point`；
4. 两者都存在：分别归一化后逐项比较；
5. 完全一致：接受；
6. 任意冲突：`ValueError`；
7. unknown key、unknown method、unknown grouping：`ValueError`；
8. params 深拷贝并冻结。

### 7.7 合法性矩阵

| 模式 | 必填 | 约束 |
|---|---|---|
| point | 无 | 禁止 intervals/calibration |
| quantile | quantiles、point_quantile | q 唯一、严格递增、`0<q<1`；本项目因 `predict_value` 契约要求 point q=0.5 必须存在 |
| interval | lower/upper | lower < upper，二者必须在 quantiles 中 |
| cqr | interval、target coverage | `0<coverage<1`、windows>0、min_windows>0、min_scores>0、delay_steps>=0 |
| median_path | q50 | 当前 Recursive 默认且唯一实现 |
| quantile_path | — | 本轮未实现，配置阶段拒绝 |
| sample_path | — | 本轮未实现，配置阶段拒绝 |

### 7.8 模型能力矩阵

| 模型 | native quantile | 本轮状态 |
|---|---:|---|
| LightGBM | 是 | 支持 |
| XGBoost | 是 | 支持 |
| CatBoost | 是 | 支持 |
| HistGB | 是 | 支持 |
| QuantileRegressor | 是 | 支持 |
| RandomForest | 当前 wrapper 无 quantile 实现 | 拒绝 |
| Ridge / ElasticNet / Lasso | 否 | 拒绝 |
| SeasonalTemplate | 否 | 拒绝 |

---

## 8. 核心数据类型与契约

### 8.1 `QuantileGrid`

```python
@dataclass(frozen=True)
class QuantileGrid:
    levels: tuple[float, ...]
    point_level: float

    def index_of(self, level: float) -> int: ...
    def column_name(self, level: float) -> str: ...
```

职责：

- 数值排序；
- 唯一性与范围校验；
- point level 定位；
- CSV 列名编码；
- 列名碰撞检测。

### 8.2 `ForecastDistribution`

```python
@dataclass
class PredictionIntervalForecast:
    name: str
    lower: np.ndarray
    upper: np.ndarray
    target_coverage: float
    method: str
    base_quantiles: tuple[float, float]


@dataclass
class ForecastDistribution:
    point: np.ndarray
    quantile_grid: QuantileGrid
    quantile_values: np.ndarray
    intervals: dict[str, PredictionIntervalForecast]
    space: str
    quantile_stage: str
    forecast_times: pd.DatetimeIndex
    metadata: dict[str, Any]
```

约束：

- `point.shape == (n_steps,)`；
- `quantile_values.shape == (n_steps,n_quantiles)`；
- `point` 与 q50 严格一致；
- `space ∈ {model,target}`；
- `quantile_stage ∈ {raw,processed}`；CQR 不创建 calibrated quantile stage；
- 每个 interval 的 lower/upper 长度等于 n_steps，且 `lower <= upper`；
- 时间戳严格递增且长度匹配。

### 8.3 `ProbabilisticModelBundle`

```python
@dataclass
class ProbabilisticModelBundle:
    schema_version: int
    spec: ProbabilisticSpec
    model_type: str
    pred_method: str
    models_by_quantile: dict[float, Any]
    recursive_propagation: str
```

Blend 子模型不再靠任意嵌套 dict 猜结构，使用显式对象：

```python
@dataclass
class BlendQuantileModel:
    direct: Any
    recursive: Any
```

部署层再用一个唯一外层 bundle 关联模型和预处理状态：

```python
@dataclass
class ForecastModelBundle:
    schema_version: int
    model: Any  # point model 或 ProbabilisticModelBundle
    feature_scaler: Any
    target_transform: Any  # TargetTransformPipeline
    selected_features: tuple[str, ...]
    input_schema: dict[str, Any]
```

`ForecastModelBundle` 是 `model.pkl` 的唯一内容；不再让多个 pickle 分别持有同一批 per-q models。

### 8.4 `CalibrationRecord`

```python
@dataclass(frozen=True)
class CalibrationRecord:
    forecast_origin: pd.Timestamp
    target_time: pd.Timestamp
    label_available_at: pd.Timestamp
    horizon_step: int
    window: int
    interval_name: str
    lower: float
    upper: float
    y_true: float
    score: float
    stage: str
```

### 8.5 目标空间契约

目标变换属于 point/quantile 共用能力。训练期按实际顺序把变换步骤登记到共享 pipeline；推理期自动逆序执行：

```text
training: calendar normalization → decomposition → target scaling
restore:  inverse target scaling → decomposition restore → inverse calendar normalization
  → target-space raw quantiles
  → crossing/postprocessing
  → CQR interval calibration
```

概率模块只依赖一个最小协议，不拥有 calendar/decomposition/scaler 的实现：

```python
class TargetTransformRestorer(Protocol):
    def restore(self, values: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray: ...
```

point 和 quantile matrix 必须通过同一 restorer；每一步返回新对象，不在原对象上隐式覆盖。

---

## 9. 训练与推理重构

### 9.1 Quantile objective 映射

把 `_inject_quantile_params()` 的 if/elif 迁移为 `objectives.py` 中的显式 capability mapping：

```python
class QuantileObjectiveAdapter(Protocol):
    def supports(self, model_type: str) -> bool: ...
    def inject(self, params: Mapping[str, Any], quantile: float) -> dict[str, Any]: ...
```

不支持时直接 RAISE，不允许默认损失继续训练。

### 9.2 QuantileTrainer

```python
class QuantileTrainer:
    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        categorical_features: list[str],
        sample_weight: np.ndarray | None,
    ) -> ProbabilisticModelBundle: ...
```

职责：

- 遍历 quantile grid；
- 注入 quantile objective；
- 根据单/多输出选择 wrapper；
- 透传类别特征、native data、early stopping、sample weight；
- 控制 Q/H/model 三层并行预算；
- 返回强类型 bundle。

### 9.3 多输出 quantile 权重

默认 `multi_output_strategy=multioutput` 时，改为复用 `DirectMultiOutputRegressor`，把同一窗口的 `sample_weight` 逐输出传给基估计器。

保留限制：

- `regressor_chain` 仍不支持 sample weight；
- 使用时 warning 并记录 bundle metadata；
- 不允许文档再笼统写“多输出 quantile 全部不支持权重”。

### 9.4 并行预算

定义有效并行度：

$$
P_{effective} =
P_{window}\times P_{quantile}\times P_{output}\times P_{model}
$$

当 `window_parallel_workers>1`：

- `quantile_parallel_workers=1`；
- `multi_output_n_jobs=1`；
- `model_thread_count=1`；
- `ensemble_parallel_workers=1`。

最终 forecast 没有窗口并行时，才允许 Q/output/model 内部并行，并继续遵守物理核容量上限。

#### 计算成本预算

标准 q10/q50/q90 下，普通多输出 Direct 每窗口约训练 `Q × H` 个基估计器。当前典型规模：

| 场景 | 每窗口基估计器 | 完整 CV + final 基估计器 |
|---|---:|---:|
| 15min、H=96、31 窗 | 288 | 9,216 |
| 5min、H=288、31 窗 | 864 | 27,648 |
| calendar month、H=31、6 窗 | 93 | 651 |

因此每个 Phase 的行为验收必须区分：合成单测、方法级小窗口 probe、单配置完整窗口实跑和全配置重跑。除最终推广外，不用全矩阵训练代替契约测试。

### 9.5 Strategy Adapter

9 种方法保留现有时序编排，只改变输出接口：

| 方法 | 适配后 quantile matrix 来源 |
|---|---|
| USMDP | 每个未来行 × q |
| USMD/MSMD | Direct `(H,Q)` |
| USMR/MSMR | 逐步追加为 `(H,Q)` |
| USMDR/MSMDR | 逐 block 追加为 `(H,Q)` |
| USBR/MSBR | Direct/Recursive 分位数按 q 融合后 `(H,Q)` |

所有 Direct/DirRec 分位数长度必须精确等于请求步数。

### 9.6 Recursive 传播策略

本轮递归传播内部固定为 `median_path`：所有 q 的下一步预测条件于 q50
回填历史。该机制只有一个实现，不再作为公共 YAML 参数；部署态
`ProbabilisticSpec` 仍显式保存该内部合同及其独立产物版本。

`quantile_path` 不应直接实现为“q10 回填 q10、q90 回填 q90”后宣称联合概率正确；条件分位数的递归组合不等于多步边际分位数。路径级能力应由采样模型解决。

### 9.7 Blend quantile

保留共享权重行为，但新增诊断：

- Direct pinball；
- Recursive pinball；
- Blend pinball；
- 三者 coverage/width；
- crossing 变化。

只有证据显示 q50 权重损害概率质量时，才讨论按 pinball 学权重；不在本轮直接改优化目标。

---

## 10. 后处理、评估、校准与输出

### 10.1 后处理顺序

统一顺序：

```text
target-space raw quantiles
  → 记录 raw crossing 指标
  → median-preserving monotone projection
  → point 绑定 processed q50
  → 记录 processed quantile pinball
  → CQR calibration（若启用，只新增 prediction interval）
  → interval validation
  → 记录 interval coverage / width / Winkler
  → writer
```

### 10.2 单调化策略

支持：

| 方法 | 语义 | 本轮状态 |
|---|---|---|
| `none` | 保留 raw crossing | 支持 |
| `rearrangement` | legacy 逐行排序 | 兼容输入 |
| `median_preserving_isotonic` | q50 固定锚点，其他 q 做单调投影 | 推荐默认 |

推荐一次到位采用 q50 锚定策略，原因：点预测业务指标、q50 和 `predict_value` 必须同义。

当前只有 q10/q50/q90 时使用透明的 q50 锚定裁剪：

```text
q10_processed = min(q10_raw, q50_raw)
q50_processed = q50_raw
q90_processed = max(q90_raw, q50_raw)
```

未来支持五个及以上 quantile 时，再使用 q50 固定锚点的 PAVA/isotonic projection。无论哪种实现，`q50 changed ratio` 必须为 0。

### 10.3 概率指标

#### Pinball loss

$$
L_q(y,\hat q)=
\max\left(q(y-\hat q),(q-1)(y-\hat q)\right)
$$

每个窗口内以 **mean pinball** 作为 proper scoring rule 主指标；跨窗口沿用项目口径取 median。点级 pinball 的 median 只作分布诊断，不作为主排名。可对 q 网格取简单平均，M5 的层级 scale/weight 不直接照搬。

#### Empirical coverage

$$
\widehat{PICP}=\frac{1}{n}\sum_i
\mathbb{1}(l_i\le y_i\le u_i)
$$

#### Average interval width

$$
AIW=\frac{1}{n}\sum_i(u_i-l_i)
$$

不同 route/场景量纲不可直接比较，额外输出：

$$
NormalizedWidth=\frac{AIW}{RobustTargetScale}
$$

`RobustTargetScale` 默认取窗口目标的中位数绝对值（接近 0 时回退 IQR），并把实际口径写入 metadata。

#### Winkler score

对目标误覆盖率 $\alpha$：

$$
W_\alpha(l,u;y)=
\begin{cases}
u-l, & l\le y\le u \\
u-l+\frac{2}{\alpha}(l-y), & y<l \\
u-l+\frac{2}{\alpha}(y-u), & y>u
\end{cases}
$$

#### Crossing

至少输出：

- row crossing rate；
- adjacent pair crossing rate；
- crossing magnitude；
- repair changed ratio；
- q50 changed ratio（推荐策略必须为 0）。

#### Calibration error

$$
CalibrationError=|EmpiricalCoverage-TargetCoverage|
$$

同时保留有方向的：

$$
CoverageGap=EmpiricalCoverage-TargetCoverage
$$

负值表示欠覆盖，正值表示区间过度保守。coverage 输出必须同时带 `n_points`、`n_windows` 和置信区间；horizon-wise 小样本优先使用 Wilson interval，存在明显序列相关时改用按窗口 block bootstrap。

### 10.4 指标掩码

概率覆盖率默认只过滤：

- 非有限 y；
- 非有限预测；
- 明确的数据质量 invalid 标记。

不默认复用 MAPE 的 y-dependent `eval_mask`。原因：按真实 y 分位数过滤会改变 coverage 所对应的总体。

可以额外输出 `eval_mask` 业务切片，但必须标记 `scope=business_masked`，不能与总体 coverage 混写。

### 10.5 horizon-wise 输出

每个滑窗预测点显式增加：

```text
forecast_origin
window
horizon_step
horizon_fraction
```

`calendar_month` 的 H 可为 28/29/30/31；按绝对 `horizon_step` 聚合时同时输出 `n_points/n_windows`，不存在的第 31 步不补值。

### 10.6 新增结果文件

#### `probabilistic_predictions_df.csv`

长表粒度：`window × time × quantile`。

| 列 | 说明 |
|---|---|
| `forecast_origin` | 预测原点 |
| `window` | 兼容窗口号 |
| `time` | 目标时间 |
| `horizon_step` | 1-based 步号 |
| `quantile` | 数值 q |
| `prediction_raw` | target-space raw quantile |
| `prediction_processed` | 单调化后 quantile |
| `y_true` | 回测真值 |
| `point_prediction` | processed q50 |

#### `probabilistic_intervals_df.csv`

长表粒度：`window × time × interval`，不把 CQR bounds 伪装成 quantile：

```text
forecast_origin,window,time,horizon_step,interval_name,
base_lower_quantile,base_upper_quantile,target_coverage,
lower,upper,method,y_true
```

#### `probabilistic_scores_df.csv`

长表粒度：`window × stage × metric × quantile/interval`。

核心列：

```text
window,forecast_origin,scope,record_kind,stage,metric,quantile,
interval_name,lower_quantile,upper_quantile,target_coverage,
value,n_points,n_windows,ci_lower,ci_upper
```

其中 `record_kind=quantile` 时 stage 为 `raw/processed`；`record_kind=interval` 时 stage 为 `base/cqr`。禁止把 CQR interval row 标成某个 quantile stage。

#### `horizon_scores_df.csv`

跨窗口按 `horizon_step` 聚合，包含：

- point MAE/MAPE；
- q pinball；
- interval coverage/width/Winkler；
- crossing；
- naive skill（有合法 baseline 时）。

#### `calibration_report.csv`

记录：

```text
forecast_origin,interval_name,method,target_coverage,
selected_origin_min,selected_origin_max,selected_windows,
selected_label_available_max,n_windows,n_scores,correction,
allow_interval_shrink,status,reason
```

### 10.7 现有输出兼容

继续保留：

- `test_scores_df.csv`；
- `cv_plot_df.csv`；
- `prediction.csv`；
- 现有 q10/q50/q90 列；
- `-quantile` setting 后缀。

兼容原则：

- `cv_plot_df.csv` 中 `predict_qXX` 表示 processed、未 CQR 的回测分位数；
- raw 值进入新长表，不给旧宽表无限加列；
- `prediction.csv` 中 `predict_qXX` 仍表示 processed quantile，CQR 通过 `predict_pi<coverage>_lower/upper` 追加；
- `predict_value == predict_q50` 是硬断言；
- legacy CQR 不再覆盖 q10/q90，这属于纠正统计命名的有意行为变化，需用 setting suffix 隔离结果。

### 10.8 CQR 的 prequential 回测

当前“先汇总所有窗口 score，再只校准 final forecast”不能评价 CQR。

新流程按 forecast origin 从早到晚：

```text
最早窗口：无更早校准样本 → status=insufficient_history
下一窗口：只用更早窗口 score 校准 → 在当前窗口评价
...
最终 forecast：只用 forecast origin 之前最新 N 个已完成窗口
```

对于待评价窗口 $k$，校准集必须满足标签已经可得：

$$
\mathcal C_k=\{s_i:\ label\_available\_at_i \le origin_k\}
$$

严禁把当前窗口尚未到达的标签或更晚窗口真值用于当前窗口校准。若业务数据存在发布延迟，`label_available_at` 必须叠加该延迟。

该流程保证时序因果性，但不自动恢复 exchangeability。项目报告使用“目标 coverage / 经验 coverage”，不得把滚动时间序列结果表述为无需假设的严格有限样本保证。

### 10.9 校准样本选择

新增统一函数：

```python
def select_calibration_records(
    records,
    forecast_origin,
    n_windows,
    grouping,
): ...
```

先把 `max(label_available_at) <= forecast_origin` 的窗口判为完整可用窗口，再按其 forecast origin 降序选最新 N 个；最后对重复 target time 去重，不消费 window ID 顺序。默认不混入只到达部分标签的窗口，避免不同 horizon 获得不对称校准样本。

### 10.10 CQR 边界语义

本轮实现 pooled symmetric CQR：

- `grouping=pooled`；
- correction 对所有 horizon 共用；
- 同时满足 `min_windows` 和 `min_scores` 才校准，避免月频 H=1 时仅靠点数门槛语义失真；
- 默认 `allow_interval_shrink=false`，将负 correction 截为 0，保持“只扩不缩”的保守业务语义；
- 若未来允许 shrink，必须单独验证窄 future interval 不会产生交叉或空区间。

horizon-binned/asymmetric 只在 §3.3 的证据条件满足后进入下一版本。

---

## 11. 与主链的接线和持久化

### 11.1 配置侧

`main.Model.__init__`：

- 调用 `resolve_probabilistic_spec()`；
- 执行模型能力与组合约束校验；
- setting 命名继续由 `spec.mode` 决定 `-quantile`。

### 11.2 训练侧

`models/ModelTraining.py`：

- 删除分位数 if/elif 主体，委托 `QuantileTrainer`；
- point 路径保持原样；
- 返回 `ProbabilisticModelBundle`。

### 11.3 推理侧

`models/ModelForecasting.py`：

- 9 种方法只负责时序输入和递归状态；
- 统一返回 `ForecastDistribution`；
- 不再通过 `self.quantile_outputs` 隐式传结果；
- 迁移期保留只读兼容 property，全部内部调用切换后移除。

### 11.4 测试侧

`models/ModelTesting.py`：

- 每窗口得到 target-space raw distribution；
- 调概率 pipeline 产生 processed 结果和 metrics；
- 汇总后执行 prequential calibration evaluator；
- 旧点指标继续从 processed q50 计算。

### 11.5 forecast 侧

`main.py`：

- 不再直接 import `utils.conformal` / `utils.quantile`；
- 调 `ProbabilisticPipeline.finalize_forecast()`；
- 保存 processed quantile、独立 CQR interval 与校准报告。

### 11.6 持久化

`<checkpoints_dir>/model.pkl` 保存唯一权威的 `ForecastModelBundle`，不再额外保存一份含相同模型的 `probabilistic_bundle.pkl`。其中概率模型字段为 `ProbabilisticModelBundle`，整体至少包含：

- schema version；
- point model 或 `ProbabilisticModelBundle`；
- feature scaler；
- `TargetTransformPipeline`（含 target scaler/decomposition/calendar 状态）；
- selected features 与输入 schema；
- `ProbabilisticSpec`、quantile grid、model type / pred method；
- per-q models 与 recursive propagation。

额外保存不含模型对象、便于人工检查的：

```text
<checkpoints_dir>/probabilistic_schema.json
```

兼容策略：

1. 新模型保存统一 `ForecastModelBundle`；
2. loader 能识别旧 quantile dict 并迁移为 runtime bundle；
3. 未知 schema version 直接失败；
4. 不依赖从文件名猜 quantile grid；
5. 未来如建立统一 deployment bundle，应一次包含 model、feature scaler、TargetTransformPipeline、selected features 和 probabilistic spec，不再增加局部 pickle。

### 11.7 与目标分解的组合

目标分解是确定性平移：

$$
Q_q(y)=Q_q(r)+d
$$

只要所有 quantile 和 score 在 compose 后的 target space 计算，CQR 与目标分解可以合法组合。

重构后解除 `conformal × decomposition` 硬限制的前置条件：

- target-space pipeline 单测通过；
- window score 与 final interval 同空间；
- decomposition + quantile + CQR 端到端实跑；
- 不与 blend residual-space 权重问题混为一谈。

### 11.8 与 target scaler 的组合

同理，inverse target scaling 必须先于 crossing、metrics 和 calibration。

若 scaler 是非线性单调变换（如 log1p），逐 quantile 逆变换合法；calibration correction 必须在逆变换后的业务量纲计算。

---

## 12. 兼容、迁移和结果失效策略

### 12.1 兼容原则

1. 不批量修改 863 个现有模型 YAML；
2. 不改变已有 q10/q50/q90 CSV 列名；
3. point 配置数值与输出不变；
4. quantile 无 crossing、无 CQR、无 scale/decomposition 时，架构迁移前后逐位一致；
5. 已确认错误修复不伪装成“行为兼容”；
6. 语义变化按 setting suffix 或全量重跑隔离。

### 12.2 旧 API 映射

| 旧对象/字段 | 新对象 |
|---|---|
| `predict_type/quantiles/...` | `ProbabilisticSpec` 兼容输入 |
| quantile model nested dict | `ProbabilisticModelBundle` |
| `Forecaster.quantile_outputs` | `ForecastDistribution.quantile_values` |
| `monotonize_quantile_columns()` | `QuantilePostprocessor` |
| `compute_nonconformity_scores()` | strict `CQRCalibrator.score()` |
| `calibrate_quantile_band()` | `CQRCalibrator.calibrate()` |
| `cv_plot_df.conformal_score` | `CalibrationRecord` + 兼容列 |
| 分散的 calendar/decomposition/scaler restore | 共享 `TargetTransformPipeline` |

### 12.3 结果失效范围

| 修复 | 失效范围 |
|---|---|
| CQR 最近窗口方向修复 | 全部旧 CQR final forecast |
| CQR interval 与 q10/q90 分列 | 全部旧 CQR 输出 schema；消费者需改读 `predict_pi*` |
| score 改为 processed 区间后计算 | 全部旧 conformal score |
| q50 锚定单调化 | crossing 被修复过的 quantile 回测/forecast |
| point=q50 强约束 | 当前 `Y_preds != saved q50` 的结果 |
| 多输出 quantile sample weight 真正生效 | 开启 time decay 的多输出 quantile 配置 |
| target-space pipeline | 未来启用 scale_target 的 quantile 结果 |

现有结果不会被代码自动改写。实施后必须：

1. 先跑一个代表配置验证；
2. 明确版本 suffix；
3. 再决定重跑范围；
4. 不把旧/新概率指标混在同一比较表。

### 12.4 Golden baseline 原则

不能先把已知错误输出记录成 golden，再把正确修复当回归。

顺序必须是：

```text
P0 语义修复
  → 合成样本契约通过
  → 记录 corrected baseline
  → 架构迁移等价性对比
```

### 12.5 生产同步边界

`docs/production_sync.md` 规定本仓库是核心算法源。概率模块先在本仓库完成：

- spec；
- metrics；
- postprocessing；
- calibrator；
- bundle。

生产包只做输入/输出 adapter，不在生产侧维护第二份 CQR 算法。

---

## 13. 实施步骤

### 13.1 依赖链

```text
Phase 0 语义契约 + 指标
   ↓
Phase 1 因果 prequential CQR
   ↓
Phase 2 spec/types + TargetTransformPipeline
   ↓
Phase 3 按方法族迁移训练与推理
   ↓
Phase 4 单一 bundle + 全链验收
   ↓
Phase 5 证据驱动扩展讨论
```

Phase 0–1 的实现直接写入最终 `probabilistic/` 包，通过薄适配器接入当前 `ModelTesting/main`，不先写一份临时 utils 实现再迁移。

### 13.2 Phase 0：语义契约与概率指标

目标：

1. 新增严格 quantile grid / interval / CQR 参数校验；
2. score 输入长度不等直接 RAISE；
3. quantile 与 CQR interval 分列输出，CQR 不覆盖 q10/q90；
4. q50 锚定 crossing repair，确保 `predict_value == processed q50`；
5. 实现 mean pinball、coverage、width、normalized width、Winkler、crossing、signed/absolute coverage gap；
6. 显式输出 `forecast_origin/horizon_step/n_points/n_windows`；
7. 修正文档中 score 非负、函数签名和 plus-one 注释漂移。

**HARD 验收：**

- quantile 与 interval 列可同时存在且数值语义不混淆；
- saved score 可由对应 processed interval boundaries 逐点复算；
- q50 repair changed ratio = 0；
- 手算指标全部一致；
- 错长度、非法 alpha、非法 q、重复 q 全部 RAISE。

**SOFT 验收：**单个已有 quantile CSV 可通过离线 evaluator 生成新增指标，不要求本阶段重训全模型。

### 13.3 Phase 1：因果 prequential CQR

目标：

1. `CalibrationRecord` 增加 `label_available_at`；
2. 按标签可得性过滤，再按 origin 选最新 N 个窗口；
3. 同时校验 `min_windows/min_scores`；
4. 对历史窗口执行 prequential calibration；
5. 输出 `probabilistic_intervals_df.csv` 和 `calibration_report.csv`；
6. final forecast 使用相同选择器和 calibrator。

**HARD 验收：**

- 当前窗口不消费自身标签；尚未到达的长 horizon 标签不进入校准集；
- window 1/31 编号方向不影响选择结果；
- raw quantile 与 calibrated interval 的 coverage/width/Winkler 可直接比较；
- 报告明确“经验 coverage”，不宣称无条件有限样本保证。

### 13.4 Phase 2：配置、类型与目标变换栈

目标：

1. 新增 `ProbabilisticSpec`、`QuantileGrid`、`ForecastDistribution`、`PredictionIntervalForecast`；
2. 完成 legacy/new resolver 和 objective capability mapping；
3. 建立 point/quantile 共用 `TargetTransformPipeline`；
4. 训练期登记 calendar→decomposition→scaler，推理期严格逆序 restore；
5. 全部 863 模型 YAML 解析通过。

**HARD 验收：**

- 398 个 quantile YAML 归一化到 `[0.1,0.5,0.9] + point=0.5`；
- 154 个 legacy CQR 配置保留 90% target coverage，但输出改为独立 PI 列；
- 不支持模型 + quantile 在构造阶段失败；
- calendar/decomposition/scaler 合成探针逐步变换后可逆恢复；
- point 与 quantile matrix 使用同一个 restorer。

### 13.5 Phase 3：按方法族迁移训练与推理

避免一次性切换 9 种方法，按输出结构分四个纵向切片：

1. **3-a 单输出**：USMDP、USMR、MSMR；
2. **3-b Direct 多输出**：USMD、MSMD、horizon feature；
3. **3-c DirRec**：USMDR、MSMDR；
4. **3-d Blend**：USBR、MSBR。

每个切片依次完成：RED 契约测试 → bundle/adapter 实现 → 方法级小窗口 probe → 通过后再进入下一切片。

共同目标：

- `QuantileTrainer` 替换对应训练分支；
- 返回统一 distribution，不再依赖 `self.quantile_outputs` 副作用；
- multioutput quantile 复用自定义 wrapper 透传 sample weight；
- 窗口并行时内部 Q/H/model 并行全部压到 1。

**HARD 验收：**

- 各方法 point/quantile shape 契约通过；
- 无 crossing、无 CQR 的兼容样本逐位等价；
- 时间衰减在 multioutput quantile 基模型收到相同 sample weight；
- 每个切片至少有一个真实配置小窗口实跑。

### 13.6 Phase 4：单一持久化与全链验收

目标：

1. `model.pkl` 保存唯一 `ForecastModelBundle`，其中 quantile 模型字段为 `ProbabilisticModelBundle`；
2. 保存 `probabilistic_schema.json`；
3. 旧 dict bundle loader 兼容；
4. README/config/models/utils/production_sync 文档同步；
5. 明确旧结果失效与新 setting suffix；
6. 完成固定 horizon、calendar_month、目标分解+CQR 的代表配置验收。

真实配置分层：

- **方法级 probe**：每个方法族 1–2 窗；
- **完整窗口基准**：15min daily USMD，无天气；
- **CQR 对照**：同配置 raw quantile + independent PI；
- **低频**：calendar_month quantile；
- **组合**：目标分解 + quantile + CQR。

**HARD 验收：**全套单测、863 YAML dry-run、bundle roundtrip、上述代表配置全部通过；输出 schema、指标和失效清单齐全。

### 13.7 Phase 5：只在证据满足后讨论扩展

候选：

1. horizon-binned CQR；
2. asymmetric / block / weighted / adaptive conformal；
3. WIS / quantile-grid 近似 CRPS；
4. probabilistic seasonal baseline；
5. 路径采样模型独立支线；
6. 层级概率 reconciliation。

---

## 14. 测试、风险与设计决策

### 14.1 单元测试

至少覆盖：

1. quantile grid 范围、唯一性、严格排序；
2. q50 缺失失败；
3. quantile column codec 无碰撞；
4. 支持/不支持模型能力矩阵；
5. pinball 手算；
6. coverage/width/Winkler 手算；
7. raw crossing rate 与 repair 指标；
8. q50 锚定单调投影；
9. point=q50；
10. CQR interval 不覆盖 q10/q90；
11. CQR score 精确长度；
12. plus-one quantile；
13. 最近窗口按时间选择；
14. `label_available_at` 阻断未到达标签；
15. min windows / min scores skip；
16. target coverage 与 raw nominal coverage 分离；
17. negative correction 在 no-shrink 模式截为 0；
18. signed/absolute coverage gap；
19. normalized width 和 coverage CI；
20. calendar→decomposition→scaler 的逆序 restore；
21. model/target space 类型守卫；
22. `model.pkl` bundle roundtrip；
23. legacy dict load；
24. 旧/新 spec 等价与冲突。

### 14.2 集成测试

覆盖矩阵：

| 维度 | 最低覆盖 |
|---|---|
| 方法 | 9 种 point/quantile shape；现役 8 种 quantile 全路径 |
| 训练形态 | 单输出、多输出、horizon_feature、blend |
| 后处理 | none/rearrangement/q50-anchored clipping/PAVA |
| 目标空间 | none/scale/calendar/decomposition；组合时验证逆序 restore |
| 校准 | off/pooled CQR/标签未到达/窗口不足/score 不足 |
| horizon | fixed/calendar_month |
| 并行 | window 串行/并行，quantile 串行/内部并行 |
| 输出 | processed quantile 与 independent PI 同时存在、互不覆盖 |

### 14.3 验证命令

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m unittest discover -s tests -p "test_*.py"

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/check_model_configs.py 'config/**/*.yaml'

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m compileall -q probabilistic features decomposition models main.py config tests
```

真实配置验收必须保留命令、配置路径、窗口数、输出文件和指标摘要。

### 14.4 风险：抽象过重

如果只把两个 utils 文件搬进目录，架构问题不会解决；如果一次性引入通用概率分布、采样、密度、copula 等接口，又会制造空壳。

控制：本轮类型只表达 quantile grid；路径样本只写后续边界，不实现虚假接口。

### 14.5 风险：修复与重构混在一起

q50、score stage、CQR 窗口方向都属于数值语义修复，不是纯重构。

控制：Phase 0 先修正 quantile/interval 语义并建立 corrected baseline；Phase 1 只增加因果 CQR 评价；Phase 2–3 再做类型、变换栈和训练推理迁移。结果差异必须归因到明确修复项。

### 14.6 风险：coverage 被区间宽度掩盖

只报告 coverage 会奖励无限宽区间。

控制：coverage、width、Winkler 必须同表输出；任何 CQR 推广同时报告三项。

### 14.7 风险：把边际区间误报成路径风险

控制：文档、CSV metadata 和 API 都使用 `marginal_quantiles` / `marginal_interval` 语义；不使用 `scenario_probability` 等路径级命名。

### 14.8 风险：单调化改善图形但损害统计含义

控制：同时保存 raw/processed；报告 repair ratio、pinball delta、q50 changed ratio。默认方法固定 q50。

### 14.9 风险：校准总体被 eval_mask 改写

控制：总体概率指标与 MAPE 业务掩码分开；任何 masked coverage 必须用独立 scope 命名。

### 14.10 风险：把因果回测误报为理论保证

Prequential 只解决未来信息泄漏，不消除时间序列相关性和分布漂移。控制：报告同时写 target coverage、empirical coverage、样本数和置信区间；标准 split conformal 的理论保证必须附 exchangeability 前提，必要时以 block/weighted/adaptive conformal 作为后续候选。

### 14.11 风险：模型持久化双写

`model.pkl` 与额外概率 bundle 若都包含 per-q models，会产生权威来源冲突。控制：新实现只让 `model.pkl` 持有模型对象；`probabilistic_schema.json` 仅保存可读 metadata，不保存第二份模型。

### 14.12 已定设计决策

| ID | 决策 | 结论 | 代价 |
|---|---|---|---|
| D-001 | 模块位置 | 顶层 `probabilistic/` | 主链四处入口迁移 |
| D-002 | 点预测语义 | quantile 模式必须含 q50，point=q50 | 非 q50 网格将被拒绝 |
| D-003 | 单调化 | 三分位用 q50 锚定裁剪；多分位再用 anchored PAVA | crossing 结果需重跑 |
| D-004 | CQR 窗口 | 按 forecast origin 选最新完成窗口 | 旧 CQR forecast 失效 |
| D-005 | CQR 回测 | prequential + `label_available_at` as-of 筛选 | 最早若干窗口无 calibrated interval |
| D-006 | interval 语义 | raw quantile pair 与 target coverage 分开配置 | 配置字段增加 |
| D-007 | 结果格式 | q 列保留模型分位数；CQR 用独立 PI 列和 interval 长表 | CQR 消费方需迁移字段 |
| D-008 | 路径概率 | 本轮不实现 | 暂不能评价联合轨迹风险 |
| D-009 | 持久化 | `model.pkl` 保存唯一 `ForecastModelBundle`（含模型、scaler、TargetTransform、schema）；JSON 只存 metadata | 需维护旧散装 pkl/dict adapter |
| D-010 | 结果有效性 | 已确认错误修复后旧结果不视为兼容基线 | 需重跑受影响配置 |
| D-011 | 目标还原 | 共享 TargetTransformPipeline 记录训练顺序并逆序 restore | point/quantile 接线均需迁移 |
| D-012 | 覆盖表述 | 时间序列只声明经验 coverage；理论保证附 exchangeability 前提 | 文案更保守但真实 |
| D-013 | 实施顺序 | metrics/prequential 前移，训练推理按方法族纵向迁移 | Phase 数量减少、每阶段可独立验收 |
| D-014 | 模块粒度 | 第一版不预建 calibration 子包/通用 registry | 第二种实现出现时再拆分 |

### 14.13 结论

本项目概率预测的主问题不是“没有 quantile 模型”，而是缺少一个把训练、目标空间、后处理、校准和评价统一起来的概率契约。

正确重构方向是：

1. 先关闭 quantile/interval 混淆、q50、score stage、标签可得性和窗口方向等 P0 语义问题；
2. 先建立 pinball/coverage/width/Winkler/crossing/horizon 评估与因果 prequential CQR；
3. 建立 `ProbabilisticSpec + ForecastDistribution + ProbabilisticModelBundle`；
4. 由共享 TargetTransformPipeline 按训练栈逆序恢复 point/quantile；
5. 按单输出、Direct、DirRec、Blend 四个方法族迁移训练与推理；
6. 最后再依据证据讨论 horizon/asymmetric/block/weighted conformal 和路径模型。

---

## 15. 实施进度记录

> 本节是 §13 的进度跟踪模块，只记录“完成到哪、证据是什么、下一步是什么”，不复述设计。  
> 状态机：`pending → in_progress → completed / blocked / cancelled`。  
> completed 必须附命令、结果摘要和受影响结果说明；无证据不得标 completed。  
> 实施中发现偏差追加到 §15.7，不回改正文中的原始设计结论。

`last_updated: 2026-08-24`

### 15.1 总览

| Phase | 内容 | 状态 | 完成时间 | 证据摘要 |
|---|---|---|---|---|
| 0 | quantile/interval 语义 + 概率指标 | **completed** | 2026-08-24 | 279 tests OK；863 YAML dry-run；QR calendar-month CQR 实跑 |
| 1 | 因果 prequential CQR | **completed** | 2026-08-24 | 286 tests OK；863 YAML dry-run；4 窗口 calendar-month CQR 全链实跑 |
| 2 | spec/types + TargetTransformPipeline | **completed** | 2026-08-24 | canonical/legacy spec、强类型结果、objective mapping、共享逆变换栈；863 模型 YAML 通过 |
| 3 | 按方法族迁移训练与推理 | **completed** | 2026-08-24 | 9 方法 shape；现役 8 种 quantile 真实 probe；multioutput 权重与并行预算完成 |
| 4 | 单一 bundle、文档与全链验收 | **completed** | 2026-08-24 | 唯一 ForecastModelBundle；fixed + calendar-month + decomposition/scaling/CQR 全链通过 |
| 5 | conformal/path 扩展讨论 | **completed（本版不扩展）** | 2026-08-24 | 当前证据不足以启动 horizon/asymmetric/path 新算法；保留独立后续支线 |

### 15.2 Phase 0：语义契约与概率指标（completed 2026-08-24）

- [x] 0-a quantile/CQR 参数 fail-fast
  - `probabilistic/spec.py` 校验非空、唯一、严格递增、`0<q<1`、必须含 q50、interval 引用、alpha/min_scores/windows；main、Trainer、config checker 共用。
- [x] 0-b score 精确长度契约
  - `probabilistic/calibration.py` 对 score/calibration 输入长度不等直接 RAISE；`utils/conformal.py` 只保留兼容入口；负 correction 默认截为 0。
- [x] 0-c quantile 与 CQR PI 分列输出
  - `predict_q*` 保留 processed model quantile；CQR 独立输出 `predict_pi<coverage>_lower/upper`，不再覆盖 q10/q90。
- [x] 0-d q50 锚定 crossing repair + point=q50
  - 三分位路径对 lower/upper 做 q50 锚定修复，q50 不变；测试与 forecast 硬同步点预测。
- [x] 0-e pinball/coverage/width/Winkler/crossing/coverage gap
  - `probabilistic/metrics.py` 实现 mean pinball、coverage、width/normalized width、Winkler、signed/absolute coverage gap、Wilson CI 和 crossing/repair 指标。
- [x] 0-f horizon_step、样本数与 coverage CI
  - `probabilistic/evaluation.py` 写出 `probabilistic_predictions_df.csv`、`probabilistic_scores_df.csv`、`horizon_scores_df.csv`；固定/自然月窗口共用 1-based horizon。
- [x] 0-g 修正文档与实现注释漂移
  - 同步 `AGENTS.md`、`config_sections.py`、Kaggle 调研、compat utils；明确时间序列只报告经验 coverage，prequential/as-of 仍属 Phase 1。

#### Phase 0 验收证据（2026-08-24）

| 验收项 | 命令/方法 | 结果 |
|---|---|---|
| 概率专项 TDD | `python -m unittest tests.test_probabilistic_contracts tests.test_conformal tests.test_probabilistic_postprocessing tests.test_probabilistic_metrics tests.test_probabilistic_evaluation tests.test_probabilistic_pipeline -v` | **24 tests — OK**；每项先观察 RED 再最小实现 GREEN |
| 全量回归 | `python -m unittest discover -s tests -p "test_*.py"` | **Ran 279 tests — OK** |
| 全量模型配置 | `python scripts/check_model_configs.py 'config/**/*.yaml'` | **exit 0**；863 模型 YAML 通过；仅 2 个既有 USMDP rolling 提示 |
| compileall | `python -m compileall -q probabilistic models main.py config scripts tests` | exit 0 |
| 真实配置冒烟 | `qr_usmd_prob_mean_conformal.yaml`，calendar_month，2 个窗口，输出到 `/tmp/tsproj_prob_phase0_smoke2/` | **exit 0，4.003s**；61 个 CV 点、31 个 forecast 点 |
| 输出契约回读 | Python 逐项断言 CSV schema/数值 | point=q50；score 可由保存边界复算；q/PI 分列；183 条 quantile records；279 条 horizon scores |

结果影响：旧 CQR forecast 的首尾 q 列语义失效；新结果必须重跑，且消费方需从 `predict_pi*` 读取 calibrated interval。Phase 0 没有改写仓库内存量 results，真实冒烟产物隔离在 `/tmp`。

### 15.3 Phase 1：因果 prequential CQR（completed 2026-08-24）

- [x] 1-a `CalibrationRecord.label_available_at`
  - 每条记录携带 origin、target、label availability、horizon、window、processed boundaries、y 与 score；额外延迟按 `conformal_label_availability_delay_steps × freq` 计算。
- [x] 1-b as-of 校准样本选择与 target-time 去重
  - 只接收 `window_origin < current_origin` 且窗口内最大 `label_available_at <= current_origin` 的完整历史窗口；按 origin 而非 window ID 取最近 N 窗；重叠 target time 保留较新 origin。
- [x] 1-c min_windows + min_scores
  - 新增 `conformal_min_windows`（默认 3）和既有 `conformal_min_scores` 双门槛；区分 `insufficient_windows`/`insufficient_scores`，并由启动校验与 863 YAML checker fail-fast。
- [x] 1-d 历史窗口 prequential evaluation
  - 按 origin 从早到晚逐窗校准；当前/未来窗口 score 永不进入当前校准集；只对成功校准窗口评价 CQR coverage/width/Winkler。
- [x] 1-e independent interval artifact + calibration report
  - 写出 `probabilistic_intervals_df.csv`；CQR window/horizon metrics 追加到既有概率评分表；`calibration_report.csv` 同时记录 prequential 与 final 审计行。
- [x] 1-f final forecast 共用同一 calibrator
  - `main.py` 移除按 window ID 排序的 legacy 选择，改用与历史评价相同的 `calibrate_with_records`；q 与 PI 继续物理分列。

#### Phase 1 验收证据（2026-08-24）

| 验收项 | 命令/方法 | 结果 |
|---|---|---|
| 概率专项 | Phase 0 套件 + `tests.test_probabilistic_prequential` | **31 tests — OK**；Phase 1 的 7 个新行为均附 RED→GREEN 证据 |
| 全量回归 | `python -m unittest discover -s tests -p "test_*.py"` | **Ran 286 tests in 18.900s — OK** |
| 全量模型配置 | `python scripts/check_model_configs.py 'config/**/*.yaml'` | **863 模型 YAML，exit 0**；仅 2 个既有 USMDP rolling 提示 |
| compileall/diff | `python -m compileall -q probabilistic models main.py config scripts tests` + `git diff --check` | exit 0 |
| 真实配置全链 | QR USMD quantile+CQR，calendar_month，4 个窗口，隔离到 `/tmp/tsproj_prob_phase1_smoke3/` | **exit 0，5.383s**；122 个 CV 点 + 31 个 final forecast 点 |
| prequential 回读 | Python 读取 interval/report/score/horizon artifacts 并逐项断言 | 4 个 origin 的 selected windows = 0/1/2/3；仅 July 窗满足双门槛，生成 31 个 CQR interval 点；6 个 window metric rows + 186 个 horizon metric rows |
| final as-of 回读 | 读取 final audit + `prediction.csv` | origin=2026-08-01；最近完整 origin=2026-07-01；label_available_max=2026-07-31；4 窗/122 scores；point=q50；PI 包含 base q10/q90 |

结果影响：旧 final CQR 按 window ID 选择的校准样本语义失效；新结果必须重跑。Phase 1 未改仓库内存量 results，验收产物仍隔离在 `/tmp`。

### 15.4 Phase 2：配置、类型与目标变换栈（completed 2026-08-24）

- [x] 2-a `ProbabilisticSpec` + legacy/new resolver
  - legacy 字段和 `overrides.probabilistic` 分别归一化；unknown key、非法组合和双入口冲突直接 RAISE；运行期只消费 spec。
- [x] 2-b `QuantileGrid` / `ForecastDistribution` / `PredictionIntervalForecast`
  - quantile level 数值为权威；column codec 检查碰撞；shape/time/space/stage/point=q50 均为强约束。
- [x] 2-c objective capability mapping
  - LightGBM/XGBoost/CatBoost/HistGB/QuantileRegressor 由 `probabilistic/objectives.py` 注入原生 quantile objective；不支持模型构造期失败。
- [x] 2-d `TargetTransformPipeline` 顺序记录与逆序 restore
  - 训练登记 calendar→decomposition→scaler；恢复自动 scaler→decomposition→calendar，长度与时间不一致 RAISE。
- [x] 2-e point/quantile 共用 restorer
  - Forecaster 内一次性恢复 point 和 quantile matrix；main/testing 不再分别手写恢复顺序。
- [x] 2-f 863 模型 YAML 全量解析
  - checker 按 `base_config/overrides` 顶层 schema 识别模型 YAML，避免把 periodicity/peak-valley 等独立工具配置误计为模型。

#### Phase 2 验收证据（2026-08-24）

| 验收项 | 命令/方法 | 结果 |
|---|---|---|
| spec/types/objective/transform 专项 | `tests.test_probabilistic_contracts/types/objectives` + `tests.test_target_transform_pipeline` | legacy/new、codec、shape、能力映射、逆序 restore 全部 GREEN |
| 模型 YAML 实扫 | checker 的 `is_model_yaml` + `resolve_probabilistic_spec + validate_quantile_model_support` | **847 模型 YAML**；465 point；382 quantile；114 CQR；0 error |
| quantile 现状 | 同上 | 382/382 为 q10/q50/q90；370 LightGBM + 12 QuantileRegressor（qr 月频 4×2 配置已删：7-10 行训练窗 × 36 维特征数学上不可用，正则化也无法修复欠定问题）|
| canonical 五分位实跑 | `/tmp/tsproj_prob_phase4_canonical5/` | q05/q10/q50/q90/q95；指定 `central80=q10/q90`；score/metric/report 均未误用 grid extremes |

结果影响：legacy YAML 无需批量修改；新 mapping 可直接使用。过去依赖近似 q50、字符串列排序或分散目标逆变换的结果不作为新契约基线。

### 15.5 Phase 3：按方法族迁移训练与推理（completed 2026-08-24）

- [x] 3-a 单输出：USMDP/USMR/MSMR
- [x] 3-b Direct：USMD/MSMD/horizon feature
- [x] 3-c DirRec：USMDR/MSMDR
- [x] 3-d Blend：USBR/MSBR
- [x] 3-e multioutput quantile sample weight
  - 普通 multioutput quantile 与 Blend Direct 均复用 `DirectMultiOutputRegressor`，同一窗口权重逐输出透传；`regressor_chain` 仍 warning skip。
- [x] 3-f 窗口并行下内部并行预算
  - `utils/parallel_budget.py` 在 `window_parallel_workers>1` 时把 quantile/output/model/ensemble 四层内部并行压到 1。

#### Phase 3 验收证据（2026-08-24）

| 方法族 | 真实配置 probe | 结果 |
|---|---|---|
| 单输出 | USMDP、USMR；月频 H=1，1 窗 | exit 0，typed bundle/distribution 输出 |
| Direct | USMD 月频 H=1；MSMD 5min 配置缩短 H=4；horizon-feature 日频 1 窗 | exit 0，shape 精确匹配 |
| DirRec/递归 | USMDR 月频 H=1；MSMR/MSMDR 5min 配置缩短 H=4 | exit 0，中位路径回填语义不变 |
| Blend | USBR calendar-month H=31；MSBR typed shape 单测 | exit 0；共享权重、point=q50、matrix `(H,Q)` |
| probe 目录 | `/tmp/tsproj_prob_phase3_probes2/`、`/tmp/tsproj_prob_phase3_multi_probes/`、`/tmp/tsproj_prob_phase3_horizon_probe/` | 均与仓库 `results/` 隔离 |

迁移结果：Trainer 返回 `ProbabilisticModelBundle`；Blend 子模型使用 `BlendQuantileModel`；quantile 推理统一返回 `ForecastDistribution`。`Forecaster.quantile_outputs` 只保留只读兼容 property，内部状态不再由外部编排消费。

### 15.6 Phase 4–5（completed 2026-08-24）

- [x] 4-a `model.pkl` 单一 ForecastModelBundle（内含 ProbabilisticModelBundle）
- [x] 4-b `probabilistic_schema.json`
  - JSON 只含 schema/model type/pred method/selected features/input schema/spec metadata，不复制模型对象。
- [x] 4-c legacy dict loader + roundtrip
  - 旧 quantile nested dict 迁移为 runtime bundle；未知 schema version 失败；point/quantile bundle pickle roundtrip 通过。
- [x] 4-d README/config/models/utils/production_sync 同步
- [x] 4-e fixed/calendar_month/decomposition+CQR 代表配置验收
- [x] 4-f 结果失效与重跑清单
- [x] 5-a horizon/asymmetric/block/weighted/adaptive conformal 决策
  - **本版本不启动**：当前只证明 pooled symmetric CQR 全链正确，尚无正式 31 窗 horizon coverage 下降或上下尾不对称证据。
- [x] 5-b path-level 独立支线决策
  - 继续作为独立研究支线；当前边际 quantile/PI 不升级为轨迹联合分布，不预建采样模型空壳。

#### Phase 4 验收证据（2026-08-24）

| 验收项 | 配置/方法 | 结果 |
|---|---|---|
| bundle 专项 | `tests.test_forecast_model_bundle` | 单一 model.pkl、schema JSON、legacy migration、roundtrip 全部通过；不写新的 target_scaler/decomposition sidecar |
| fixed horizon | `lgbm_usmd_prob_mean.yaml`，freq=1ME，1 CV 窗 + final | `/tmp/tsproj_prob_phase4_fixed/`，exit 0；1 行 forecast；point=q50；typed bundle |
| 最强组合 | QR USMD calendar-month + linear decomposition + target scaling + CQR，4 窗 + final | `/tmp/tsproj_prob_phase4_combo/`，exit 0；122 CV 点、31 forecast 点；final 4 窗/122 scores，status=applied |
| 最强组合回读 | Python 严格断言 | `ForecastModelBundle → ProbabilisticModelBundle`；transform steps=`decomposition,target_scaling`；score 复算最大误差 `5.82e-11`；q/PI 分列 |
| canonical 五分位 | 新 mapping，q05/q10/q50/q90/q95，`central80`，allow shrink | `/tmp/tsproj_prob_phase4_canonical5/`，61 CV 点、31 forecast 点；prequential/final report 均为 central80；输出 `predict_pi80_*` |

#### 结果失效与重跑清单

1. 旧 CQR final forecast：window-ID 选择、q 列被校准边界覆盖或缺独立 PI 的结果全部失效；
2. 旧 `conformal_score`：不是由保存后 processed interval 复算的结果全部失效；
3. crossing repair 改变过 q50、导致 `Y_preds != predict_q50` 的 quantile 结果全部失效；
4. 新启用 multioutput time decay 的 quantile 配置：旧结果未真实消费权重，必须重跑；
5. target scaling/decomposition/CQR 组合：只接受共享 `TargetTransformPipeline` 生成的新结果；
6. 现有 results 不自动改写；重跑必须使用新 suffix/目录隔离，不能把旧新概率指标混表。

### 15.7 发现的实施偏差（实施中追加，勿删）

| # | 日期 | 偏差 | 处理 |
|---|---|---|---|
| 1 | 2026-08-24 | 原设计允许 raw/processed 两阶段都计算 interval metrics；真实 QR 冒烟发现 raw quantile crossing 时 lower>upper，raw 不是合法 interval | raw 只保留 pinball/crossing 诊断；coverage/width/Winkler 仅对合法 processed boundaries 计算；补 RED→GREEN 回归测试 |
| 2 | 2026-08-24 | Phase 0 首次真实冒烟在结果保存阶段因上述 raw crossing 失败 | 根因修复后用新隔离目录重跑，同一配置 2 窗口 + final forecast 全链通过 |
| 3 | 2026-08-24 | Phase 1 首次接线只写历史 prequential audit，没有把 final selector 证据追加到 `calibration_report.csv` | 新增 final 审计行；无论 applied/skipped 都记录 status/reason、requested/selected windows 与 label availability 上界 |
| 4 | 2026-08-24 | final 审计追加后 CSV 混用 `YYYY-MM-DD` 与空格时间格式，严格 `parse_dates` 保持 object | 补 RED→GREEN 格式测试；四个审计时间列统一写 ISO `YYYY-MM-DDTHH:MM:SS` |
| 5 | 2026-08-24 | 实施中两次 ad-hoc 计数得到 890/876，与设计基线 863 冲突；根因分别是把独立工具 YAML 当模型、以及用 loader 默认 `data_path` 作启发式导致 13 个假阳性 | 新增 schema 识别测试；checker 只接收顶层含 `base_config/overrides` 且不属独立工具前缀的文件；最终仍为 863（465 point + 398 quantile），设计基线无漂移 |
| 6 | 2026-08-24 | decomposition+scaling+CQR 首次真实组合运行在结果保存阶段触发既有非法 f-string format specifier；此前 residual rows 为空时未执行该分支 | 新增 `ResidualDiagnosticsSaveRegressionTest` 复现 RED；预先构造 `fft_cv_text` 后 GREEN；原组合重跑 exit 0 |
| 7 | 2026-08-24 | canonical 五分位 probe 前复核发现 evaluator/final 仍隐式使用 grid 首尾，若 calibration interval=q10/q90 会误取 q05/q95 | 新增显式 interval RED→GREEN；score、base/horizon metrics、prequential/final CQR 和 report 全部由 `IntervalSpec` 数值 level 驱动 |
| 8 | 2026-08-24 | 月频方法 probe 首次启用天气时，配置的 future 文件从月末开始而严格契约要求预测月初开始 | probe 关闭与策略迁移无关的天气通路后重跑；正式组合验收使用无天气配置，未修改 YAML/数据 |

### 15.8 设计阶段核实证据

| 核实项 | 命令/方法 | 结果 |
|---|---|---|
| 工作树 | `git status --short` | 概率 Phase 0–1 代码、测试和直接相关文档改动；无 YAML/results 改动 |
| 模型 YAML 审计 | Python 遍历 `config/**/*.yaml` + `yaml.safe_load` | 863 模型 YAML；398 quantile；154 CQR |
| quantile 网格 | 同上 | 398/398 均为 `[0.1,0.5,0.9]` |
| 模型能力使用 | 同上 | 378 LightGBM + 20 QR；无不支持模型配置 |
| CQR 参数 | 同上 | 154/154 为 alpha=0.1，字段均写在 model_strategy 组 |
| 存量结果审计 | Python 遍历 `results/results_test/**/cv_plot_df.csv` | 463 CSV；236 含 quantile；169 point/q50 不一致；82 score/保存边界不一致；coverage 中位数约 32.7% |
| 窗口方向 | 读取 15min daily A 路代表性 `cv_plot_df.csv` | window 1=2026-07-31，window 31=2026-07-01 |
| 目标变换顺序 | 读取 `main.py:834-848`、`models/ModelTesting.py:173-243`、`models/ModelForecasting.py:1270-1283` | 训练 calendar→decomposition→scaler；逆序应为 scaler→decomposition→calendar |
| 方案评审修订 | 逐项核对 quantile/CQR 统计语义、持久化与实施依赖 | CQR PI 与 q 列分离；加入 label availability；metrics/prequential 前移；取消双份模型 pickle |
| 文档结构 | 章节/围栏/相对链接脚本检查 | 15 个一级编号章节连续；90 个代码围栏成对；相对链接 0 缺失；H3 编号连续 |
| 全量回归 | `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m unittest discover -s tests -p "test_*.py"` | **Ran 286 tests — OK** |

#### 实施收口验证（2026-08-24）

| 验收项 | 命令/方法 | 最终结果 |
|---|---|---|
| 全量 unittest | `python -m unittest discover -s tests -p "test_*.py"` | **Ran 327 tests in 19.215s — OK** |
| 全模型 YAML dry-run | `python scripts/check_model_configs.py 'config/**/*.yaml'` | **863 configs，exit 0**；仅 2 个既有 USMDP rolling 提示，无硬失败 |
| compileall | `python -m compileall -q probabilistic features decomposition models main.py config scripts tests` | exit 0 |
| diff whitespace | `git diff --check` | exit 0 |
| docstring 行为等价 | `ast_equivalence_check.py ... decomposition/bundle.py` | `IDENTICAL decomposition/bundle.py` |
| 设计文档结构 | Python 检查 H2/代码围栏/相对链接 | 1–15 连续；90 个围栏成对；0 缺失相对链接 |
| 真实全链 | fixed、8 种现役 quantile 方法、horizon-feature、decomposition+scaling+CQR、canonical 五分位 | 全部 exit 0；产物均隔离在 `/tmp` |

#### horizon 聚合补齐 point/naive/crossing 指标（2026-08-24，对照 §10.5）

| 验收项 | 命令/方法 | 结果 |
|---|---|---|
| RED→GREEN | `tests.test_probabilistic_evaluation`（新增 2 用例） | 先确认 `point` 记录缺失（RED），实现后 9 tests OK |
| 全量回归 | `python -m unittest discover -s tests -p "test_*.py"` | **Ran 329 tests — OK** |
| 真实全链 | `qr_usmd_prob_mean_conformal.yaml`，calendar_month，4 窗 + final，隔离到 `/tmp/tsproj_prob_horizon_naive/` | exit 0；`horizon_scores_df.csv` 新增 `mae/mape/naive_mae/naive_skill`（point 记录）与 crossing 三指标（processed 聚合）；31 个 horizon 全覆盖 |
| 手算复核 | Python 重算 h1 MAE/naive MAE/skill | `h1_skill=-3.717275` 与 `1-mae/naive_mae` 完全一致；`cv_plot_df.csv` 新增 `Y_naive` 列 |

说明：naive 对照复用 `ModelTesting` 既有"昨日同时刻"序列（`n_per_day` 步前），注入 `cv_plot_df.Y_naive`；无 naive 对齐的窗口不输出 naive 指标。§10.5 的 naive skill 已就位；多基线套件（上周/同槽位中位数/SeasonalTemplate）仍属 Kaggle 调研 P0 欠账。

### 15.9 环境备忘

- 项目根：`/Users/wangzf/projects/tsproj_ml`；
- 当前分支：`dev`；
- Python 验证统一前缀：`env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python ...`；
- Phase 0–4 已实施并完成本版本全链验收；Phase 5 已裁决为等待指标证据，不在本版扩展 conformal/path 算法。
- 存量 results 未自动改写；旧 CQR/q50-crossing/未消费 multioutput 权重的结果按 §12.3 与 §15.6 清单重跑。
