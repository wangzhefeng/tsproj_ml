# 多步预测模块重构设计方案

> 文档状态：Draft / 未评审
> 创建日期：2026-08-24
> 关联文档：调研依据见 `docs/kaggle_time_series_forecasting_research.md`（§6 多步预测、§9 概率预测）；分解重构先例见 `docs/time_series_decomposition_redesign.md`。
> 本方案不构成实施授权；待决策项保留默认行为。

## 1. 背景与目标

### 1.1 背景

当前项目已经有一条完整可运行的多步预测链路：

- `models/ModelForecasting.py` 的 `Forecaster` 类承载全部 9 种预测方法的推理实现（约 1483 行单类）；
- `models/ModelTraining.py` 的 `Trainer` 承载对应的多输出训练容器（`DirectMultiOutputRegressor`、`HorizonAlignedDirectRegressor`、sklearn `MultiOutputRegressor`/`RegressorChain`）与 bundle 打包（quantile / blend / auxiliary）；
- `features/FeatureEngineering.py` 构造各方法的目标列（Direct 系 `shift_1..H`，Recursive/Pointwise `shift_0`）与 lag 特征；
- `models/ModelTesting.py` 负责滑窗测试与评分；
- `main.py` 负责编排、conformal 校准与 quantile 单调化。

2026-08-24 已完成四项多步语义修复（MSMD/MSMDR 目标对齐、Direct 输出长度契约、USMDP 多步 safe-lag、global panel 实体边界），时间语义层面的缺陷已关闭。

但从架构与能力上看，仍存在以下问题：

1. **概率预测只有产出、没有评估**：全仓无 pinball / PICP / Winkler / width / crossing rate 实现；`test_scores_df.csv` 只有整窗 R2/MSE/RMSE/MAE/MAPE 与季节 naive 对照，无法回答"q10/q90 是否准确、80% 区间是否真的覆盖 80%、覆盖是否靠区间过宽换来、哪些 horizon 失准"。
2. **推理实现平铺且重复**：9 个 `*_forecast` 方法在单一 `Forecaster` 类中各占 80–150 行，"特征工程 → 预处理 → 预测 → 长度契约 → 回填"的模式以三种变体反复出现；方法分发是 50 行 if/elif 长链。
3. **quantile 输出记录三路不统一**：Direct 走 `_record_quantile_direct`（长度精确契约）、Recursive 走 `_record_quantile_recursive_step`（逐步 store）、Pointwise 直接 reshape——下游评估若要统一消费必须感知方法类型。
4. **CQR 评估口径错位**：conformal 校准只发生在 final forecast 阶段；测试段 `cv_plot_df.csv` 的 `predict_q*` 是未校准原值，在测试 CSV 上比较覆盖率会系统性得出"校准无效"的错误结论。
5. **训练侧方法感知分散**：`_should_use_fourmethods_baseline_training`、`_should_use_horizon_feature`、`_is_blend_method`、`_should_use_horizon_aligned_direct` 等布尔判断散布在 Trainer，新增方法需要同时改多处。

### 1.2 目标

本次重构的目标是"补评估闭环 + 收敛推理架构"，不是更换预测算法：

1. 建立概率预测评估闭环：pinball、coverage、interval width、Winkler score、quantile crossing rate，按 window / horizon / quantile / interval 四个粒度输出；
2. 建立 horizon-wise 评估：每个预测步的点指标与概率指标，`fixed_steps` 与 `calendar_month` 双模式；
3. 建立 baseline suite 与 skill score：昨日 / 上周 / 同槽位中位数 / SeasonalTemplate，与模型共用 eval_mask；
4. 把 9 种 forecast 方法收敛为 strategy 注册表模式，消除分发长链与复制粘贴；
5. 把 quantile 输出统一为一个数据对象（`QuantileTrajectory`），评估、单调化、校准消费同一契约；
6. 修正 CQR 评估口径：测试段可同时评估原始区间与校准后区间；
7. 保持行为兼容：同一配置、同一数据下，最终预测数值与现有结果目录语义不变（评估层只新增产物，不改预测值）。

### 1.3 非目标

本次明确不做：

1. 不修改 `config/` 下任何实验配置；
2. 不改变 9 种预测方法的数学语义（目标列、递归回填、blend 权重学习逻辑均不动）；
3. 不引入 DeepAR/GRU 等分布模型、不做层级协调（调研文档已判定为后续研究）；
4. 不改变 CQR 校准算法本身（先评估、后扩展，见调研文档 §9.3）；
5. 不重写训练侧容器（`DirectMultiOutputRegressor` 系列保持原样，仅收敛其选择逻辑的入口）；
6. 不把评估结果反向用于自动选型或自动调参（评估产物只报告、不决策）。

---

## 2. 当前现状与问题

### 2.1 当前事实（代码为准）

| 组件 | 位置 | 现状 |
|---|---|---|
| 方法分发 | `ModelForecasting.py:1293-1343` | if/elif 长链，9 分支 + blend 的 point/quantile 二级分支 |
| USMDP 推理 | `ModelForecasting.py:698-766` | 未来 H 行一次 pointwise predict；safe-lag 模式拼接历史+未来后截取未来行 |
| USMD/MSMD 推理 | `ModelForecasting.py:768-784, 922-948` | `_build_direct_forecast_input` 取 anchor 行 → `_require_direct_prediction_length` 长度契约 → `_record_quantile_direct` |
| USMR 推理 | `ModelForecasting.py:786-848` | 逐步循环：单步特征工程 → 预测 → `_record_quantile_recursive_step` → `_append_history_row` 回填 |
| MSMDR/USMDR 推理 | `ModelForecasting.py:850+, 1032+` | `block_size=min(lags)` 分块，块内一次多输出、块间递归 |
| MSMR 推理 | `ModelForecasting.py:950-1030` | runtime_cache + lag_state 缓存；非目标内生量 aux 轨迹优先、持久性回退 |
| Blend 推理 | `ModelForecasting.py:1153-1255` | `_resolve_blend_weights`（ridge_stacking 读 `blend_weights.csv`）→ 两路预测加权 |
| quantile 训练 | `ModelTraining.py:451-523` | 每 q 独立模型（`_inject_quantile_params` 按 model_type 注入 quantile objective），bundle dict 打包 |
| horizon_feature | `ModelTraining.py:636-751` | `_melt_to_horizon_long` 宽表→N×H 长表，加 `forecast_horizon_idx` 三列 |
| 评分 | `ModelTesting.py:634-714` | 整窗 R2/MSE/RMSE/MAE/MAPE + eval_mask + 季节 naive；calendar_month 追加月总量 MAPE |
| conformal | `main.py:681-726` | 仅 forecast 阶段，首尾 quantile 对称膨胀 |
| 单调化 | `utils/quantile.py` + `main.py:725` | forecast 阶段逐行排序 |

### 2.2 当前问题

1. **评估层缺失（P0）**：
   - 无法验证概率预测质量，CQR 与原始 quantile 的比较只能看 q50 点精度；
   - 无 horizon 维度指标，远 horizon 退化、区间失准被整窗 median 掩盖；
   - 现有季节 naive 只有点口径，无概率口径基线。

2. **推理架构平铺（P1）**：
   - 新增方法 = 在 1483 行单类里再插一个 100 行方法 + 分发分支 + Trainer 布尔判断；
   - Direct 系方法（USMD/MSMD/MSMDR/USMDR）共享的"anchor 输入 → 长度契约 → quantile 记录"模式在 4 处重复。

3. **quantile 输出契约不统一（P1）**：
   - `self.quantile_outputs` 的填充方式随方法变化；评估层（若接入）必须按方法分支；
   - 单调化只覆盖 forecast 阶段，测试段 `predict_q*` 未经单调化（crossing 不可测量）。

4. **CQR 口径错位（P0）**：
   - 测试段评估覆盖率时用的是未校准区间，结论系统性偏差。

---

## 3. 重构范围与优先级

| 批次 | 内容 | 性质 | 优先级 |
|---|---|---|---|
| A | 概率评估闭环：`probabilistic_metrics` + `probabilistic_scores_df.csv` + `horizon_scores_df.csv` + baseline suite + skill score | 纯新增，不改预测值 | **P0** |
| B | CQR 口径修正：测试段落 raw + calibrated 双区间列 | 评估口径修正 | **P0** |
| C | 推理架构收敛：strategy 注册表 + `QuantileTrajectory` 统一契约 | 架构重构，行为等价 | P1 |
| D | 训练侧入口收敛：方法布尔判断归并为方法能力声明 | 架构重构，行为等价 | P2 |

批次 A/B 先行（独立于 C/D 可交付）；C/D 是结构债清偿，须以"行为等价测试"护航。

---

## 4. 模块放置与边界

### 4.1 评估层（新增）

```text
utils/
  probabilistic_metrics.py      # 纯函数指标库，无项目依赖
models/
  ModelTesting.py               # 接入评估产出（不新建文件，扩展现有类）
```

指标库放 `utils/` 的理由：与 `eval_mask`、`quantile`、`multistep_contract` 同级，是纯函数集合；`ModelTesting` 是唯一消费方，不单建 `evaluation/` 包（当前只有一个消费场景，避免过度分层——参照分解重构 §13.1 的教训）。

### 4.2 推理 strategy 层（新增）

```text
models/
  forecast_strategies.py        # 9 种方法的 strategy 注册表
```

不新建顶层包（如 `forecasting/`）的理由：`decomposition/` 独立成包的前提是它有 extractor/forecaster/composer/pipeline 四层结构；多步预测的收敛对象只是"分发 + 公共模式"，一个注册表文件 + `Forecaster` 保留为上下文容器即可。

### 4.3 不动的边界

- `models/ModelTraining.py` 的训练容器与 bundle 打包不动；
- `features/FeatureEngineering.py` 的目标构造不动（2026-08-24 刚验收）；
- `main.py` 的 conformal 校准与 blend 权重学习不动（批次 B 只改测试段口径）。

---

## 5. 目标架构

### 5.1 概率评估闭环（批次 A）

#### 5.1.1 指标库 `utils/probabilistic_metrics.py`

```python
def pinball_loss(y_true, y_pred, quantile) -> np.ndarray:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return np.maximum(quantile * error, (quantile - 1.0) * error)

def mean_pinball(y_true, quantile_pred_map) -> dict[float, float]: ...

def interval_coverage(y_true, lower, upper) -> float:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))

def mean_interval_width(lower, upper) -> float: ...

def winkler_score(y_true, lower, upper, alpha) -> np.ndarray:
    # 标准 Winkler：区间外按距离罚、区间内按宽度罚
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = width.copy()
    score += np.where(below, (2.0 / alpha) * (lower - y_true), 0.0)
    score += np.where(above, (2.0 / alpha) * (y_true - upper), 0.0)
    return score

def quantile_crossing_rate(quantile_pred_map) -> float:
    # 逐点检查 q 低分位预测 > q 高分位预测 的比例
```

设计约束：

- 纯函数、无状态、不依赖 pandas 之外的项目模块；
- 全部可在合成样本上手算复核（测试要求见 §11.1）；
- interval 由 quantile 对（如 (q10,q90)、(q25,q75)）派生，`alpha = 高q - 低q`。

#### 5.1.2 评估产出

| 产物 | 粒度 | 列 |
|---|---|---|
| `probabilistic_scores_df.csv` | window × quantile | `Pinball(q)`、`Pinball vs Naive(q)`、`CRPS(近似，可选)` |
| 同文件追加 | window × interval | `Coverage`、`Nominal Coverage`、`Width`、`Winkler(alpha)`、`Width vs Naive` |
| `horizon_scores_df.csv` | horizon_step | `MAE`、`MAPE`、`Pinball(q50)`、`Coverage(q10-q90)`、`Width`、`Crossing Rate` |
| `test_scores_df.csv` | window（现有文件追加列） | `Quantile Crossing Rate`、`Mean Pinball` |

点预测配置（`predict_type: point`）不生成概率列——指标函数按 `quantile_outputs is None` 短路。

#### 5.1.3 baseline suite

扩展现有 `_build_seasonal_naive` 为 baseline 注册表：

| baseline | 生成方式 | 点口径 | 概率口径 |
|---|---|---|---|
| `naive_last_day` | 现有季节 naive（n_per_day 步前） | MAPE/MAE（已有） | pinball（用 q50=naive，区间 = naive ± 近期残差分位数，可选 P2） |
| `naive_last_week` | 7×n_per_day 步前 | 同上 | 同上 |
| `naive_same_slot_median` | 近 K 个同槽位中位数 | 同上 | — |
| `SeasonalTemplate` | 复用现有 st 模型的模板逻辑 | 同上 | — |

频率自动选择合法基线：5min/15min/1h → last_day/last_week；1D → last_week/same_slot；1ME → 趋势 naive（P2 可选）；历史不足时该 baseline 记 NaN（与现有 naive 的 None 语义一致）。

skill score 统一定义：

```
skill(metric) = 1 - loss_model(metric) / loss_baseline(metric)
```

落在 `test_scores_df.csv` 追加列 `Skill vs naive_last_day(MAPE)` 等。

#### 5.1.4 horizon-wise 评估

数据源：`cv_plot_df.csv` 已含每窗口逐点 `time / actual / predict / predict_q*`——horizon 维度 = 窗口内步索引（`time - window_start` 换算），不需要改 Forecaster。评估在 `test_results_save` 阶段聚合：

- `fixed_steps`：step = 1..H 直接由窗口内行序得到；
- `calendar_month`：各窗口 horizon 长度不同（28/29/30/31），按 step 对齐聚合（短月缺的 step 不计入该月）。

### 5.2 CQR 口径修正（批次 B）

现状：校准只在 forecast。方案（两个层次，第一层必做）：

1. **测试段双区间列（必做）**：`ModelTesting` 在写 `cv_plot_df.csv` 时，若 `enable_conformal_calibration=true`，按与 `main.py` forecast 相同的"最近 N 窗 conformal_score"逻辑，额外写 `predict_q{lo}_cal` / `predict_q{hi}_cal` 两列（原始列保留）。校准逻辑抽为 `utils/conformal.py` 的共享函数，测试与 forecast 调用同一实现，避免口径漂移。
   - 泄漏检查：窗口 w 的校准集只用窗口 < w 的 score（与 ridge blend 权重学习同模式）；
2. **评估层消费（随批次 A）**：coverage/width 指标对 raw 与 cal 两套区间各算一遍，`probabilistic_scores_df.csv` 输出 `Coverage(raw)` / `Coverage(calibrated)` 对照列。

### 5.3 推理 strategy 注册表（批次 C）

```python
# models/forecast_strategies.py
@dataclass(frozen=True)
class ForecastStrategy:
    name: str                                  # pred_method 全名
    fn: Callable[[Forecaster], np.ndarray]     # 执行入口
    is_multi_output: bool                      # 训练容器选择
    supports_quantile: bool
    supports_horizon_feature: bool             # 方法能力声明（收敛 Trainer 布尔判断）

FORECAST_STRATEGIES: dict[str, ForecastStrategy] = {
    "univariate-single-multistep-direct-pointwise": ForecastStrategy(...),
    "univariate-single-multistep-direct": ...,
    # ... 9 项
}

def dispatch(pred_method: str, predict_type: str) -> Callable[[Forecaster], np.ndarray]:
    strategy = FORECAST_STRATEGIES.get(pred_method)
    if strategy is None:
        raise ValueError(f"Unsupported pred_method: {pred_method}")
    return strategy.quantile_fn if predict_type == "quantile" and strategy.quantile_fn else strategy.fn
```

- `Forecaster` 保留全部状态（df_history_for_lags、feature_engineer、scaler、model bundle、quantile_outputs），strategy 函数是方法绑定（第一参数是 Forecaster 实例），不复制状态；
- 分发长链替换为 `dispatch(...)(self)`；
- Direct 系公共模式（anchor 输入构建 → 长度契约 → `_record_quantile_direct`）抽为 `_direct_forecast_common(self, ...)` 供 USMD/MSMD 复用；DirRec 与 Recursive 的循环骨架差异大，不强行合并。

### 5.4 `QuantileTrajectory` 统一契约（批次 C）

```python
@dataclass
class QuantileTrajectory:
    quantiles: list[float]
    values: dict[float, np.ndarray]        # q -> (horizon,) 数组
    crossed: np.ndarray | None = None      # 单调化后重算的 crossing 标记

    def monotonize(self) -> QuantileTrajectory: ...
    def interval(self, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]: ...
    def aligned(self, horizon: int) -> QuantileTrajectory:  # 长度契约
        ...
```

- 三种记录路径（direct/recursive/pointwise）在各自 strategy 末尾统一构造 `QuantileTrajectory`；
- `self.quantile_outputs` 改为持有该对象（保留 dict 兼容视图一个过渡版本）；
- 评估层、单调化、conformal 校准全部消费该对象。

### 5.5 训练侧入口收敛（批次 D）

`Trainer` 的方法布尔判断（`_should_use_fourmethods_baseline_training` / `_should_use_horizon_feature` / `_is_blend_method` / `_should_use_horizon_aligned_direct`）改为读取 `ForecastStrategy` 的能力声明字段，判断逻辑本身不变，只是声明从"散布的方法名字符串比较"收敛为注册表单点维护。

---

## 6. 现有 9 种方法如何映射到新架构

| 方法 | strategy.fn 归属 | 公共模式复用 | quantile 构造 | 行为变化 |
|---|---|---|---|---|
| USMDP | pointwise 专属 fn | —（safe-lag 分支保留在 fn 内） | reshape → trajectory | 无 |
| USMD | direct common | anchor + 长度契约 + record | `_record_quantile_direct` → trajectory | 无 |
| USMR | recursive fn | 循环骨架（单变量版） | recursive store → trajectory | 无 |
| USMDR | dirrec fn | 循环骨架（块版） | 块内 record → trajectory | 无 |
| MSMD | direct common | 同 USMD | 同 USMD | 无 |
| MSMR | recursive fn（多变量版） | runtime_cache 逻辑保留 | recursive store → trajectory | 无 |
| MSMDR | dirrec fn（多变量版） | 同 USMDR + aux 回填 | 同 USMDR | 无 |
| USBR | blend fn | 权重解析保留 | blend 两路加权 → trajectory | 无 |
| MSBR | blend fn（多变量版） | 同 USBR | 同 USBR | 无 |

行为等价的验收方式：每方法一个"重构前后预测值逐元素相等"测试（固定随机种子 + 合成数据）。

---

## 7. 与主链的接线方式

### 7.1 训练侧（不动）

`Trainer.train()` 签名、bundle 打包、容器选择逻辑全部不变。批次 D 只替换布尔判断的来源。

### 7.2 测试侧（批次 A/B 扩展）

```text
_window_test (现有)
  → Forecaster.forecast() (现有)
  → cv_plot_df (现有 + cal 区间列 [批次 B])
  → _evaluate_score (现有)
  → _evaluate_probabilistic (新增 [批次 A]) → probabilistic_scores_df
  → _evaluate_horizon (新增 [批次 A]) → horizon_scores_df
  → _evaluate_baselines (新增 [批次 A]) → test_scores_df 追加列
```

评估函数只读 `cv_plot_df` 与 `quantile_outputs`，不触碰预测路径。

### 7.3 预测侧（批次 C 等价替换）

`main.py:forecast()` 调用 `Forecaster.forecast()` 的入口不变；分发长链在 `Forecaster.forecast()` 内部替换为 `dispatch()`。

### 7.4 产物落盘

新增 CSV 与现有 `test_scores_df.csv` 同目录（`results/results_test/<scenario>/<setting>/`）；forecast 侧不新增产物（概率评估的语义边界是测试段；forecast 无真值）。

---

## 8. 兼容与迁移策略

### 8.1 兼容原则

1. 不修改 `config/`：全部能力通过代码默认行为交付（评估产物默认生成，类似现有 `test_scores_df.csv`）；
2. 预测值零变化：批次 A/B 不进入预测路径；批次 C/D 由行为等价测试护栏；
3. 产物只增不改：现有 CSV 的列与顺序不变，新列只追加；
4. `setting` 与结果目录不受影响（无新配置字段进 setting 公式）。

### 8.2 旧 API 到新 API 的映射

| 旧 | 新 | 过渡策略 |
|---|---|---|
| `self.quantile_outputs: dict` | `QuantileTrajectory` | 保留 `.as_dict()` 视图，评估层完成后切默认 |
| if/elif 分发（50 行） | `dispatch()` | 一次性替换，`grep` 验证无残留 |
| Trainer 方法布尔判断 | strategy 能力声明 | 批次 D 逐个替换，每个有独立等价测试 |

### 8.3 行为兼容要求

- 9 方法预测值逐元素相等（合成数据 + 固定种子）；
- 249 项现有 unittest 全绿；
- 全量 904 YAML 加载 0 错误；
- `check_model_configs.py` 通过。

---

## 9. 本次必须修复的问题

### 9.1 测试段 CQR 口径（批次 B）

问题：校准只在 forecast，测试段评估用的是未校准区间。
修复：共享校准函数 + 测试段双区间列（§5.2）。泄漏控制：窗口 w 只用 < w 的 score。

### 9.2 测试段 quantile 未单调化（随批次 A）

问题：`quantile_monotone` 只在 forecast 生效，测试段 crossing 不可测量。
修复：`ModelTesting` 写 `cv_plot_df` 前对 `predict_q*` 调用同一 `monotonize_quantile_columns`（保留原始列 + 追加 `_mono` 列，两套都可评估），配置开关沿用 `quantile_monotone`。

---

## 10. 预留扩展位

以下方向在本方案中只留接口位，不实现：

1. **CRPS**：pinball 均值已是 CRPS 的离散近似，完整 CRPS 预留 `probabilistic_metrics.crps` 接口位；
2. **horizon-binned conformal**：若批次 A 评估发现 coverage 随 horizon 单调下降，按 horizon 分桶校准是第一个扩展点（调研文档 §9.3 的条件触发路径）；
3. **概率口径 baseline 区间**：naive ± 近期残差分位数（P2，先只做点口径 skill）；
4. **regime 分层诊断**：按负荷水平/节假日/事件切片的 coverage 报告（标签必须预测原点可见，避免泄漏）；
5. **路径级联合概率**：DeepAR 类分布模型（独立研究支线，不进本链路）。

---

## 11. 实施步骤

### Phase A：概率评估闭环（P0，先行）

1. `utils/probabilistic_metrics.py`：pinball / coverage / width / Winkler / crossing 纯函数 + 单测（合成样本手算复核）；
2. `ModelTesting._evaluate_probabilistic`：消费 `cv_plot_df` 的 `predict_q*`，输出 `probabilistic_scores_df.csv`；
3. `ModelTesting._evaluate_horizon`：输出 `horizon_scores_df.csv`（fixed + calendar_month）；
4. baseline suite：`naive_last_day`（复用现有）/ `naive_last_week` / `naive_same_slot_median` + skill 列；
5. 测试段 quantile 单调化双列（§9.2）；
6. 在一个现役 quantile 配置上实跑冒烟：确认预测 CSV 数值不变、只新增评估产物。

### Phase B：CQR 口径修正（P0）

1. `utils/conformal.py` 抽共享校准函数（forecast 逻辑原样迁移）；
2. `ModelTesting` 双区间列（泄漏控制：窗口 w 用 < w 的 score）；
3. 评估层输出 `Coverage(raw)` / `Coverage(calibrated)` 对照。

### Phase C：推理 strategy 收敛（P1）

1. `models/forecast_strategies.py` 注册表 + 9 strategy；
2. Direct 系公共模式 `_direct_forecast_common` 抽取；
3. `QuantileTrajectory` 数据对象，三种记录路径统一；
4. 分发长链替换为 `dispatch()`；
5. 9 方法行为等价测试（逐元素相等）。

### Phase D：训练侧入口收敛（P2）

1. `ForecastStrategy` 能力声明字段（is_multi_output / supports_quantile / supports_horizon_feature）；
2. Trainer 布尔判断逐个改为读声明；
3. 每个替换点独立等价测试。

### 实施顺序与依赖

```text
Phase A ──独立──> 可先行交付
Phase B ──依赖 A 的评估消费──> 随后
Phase C ──独立于 A/B──> 行为等价护栏
Phase D ──依赖 C 的 ForecastStrategy──> 最后
```

---

## 12. 测试与验证

### 12.1 单元测试（新增）

1. pinball 手算复核：`y=10, pred=12, q=0.9 → loss=1.8`；`y=10, pred=8, q=0.9 → 0`；
2. coverage 合成样本：已知覆盖 8/10 点 → 0.8；
3. Winkler：区间内 = 宽度；区间外 = 宽度 + 2/α × 距离（手算两组）；
4. crossing rate：构造 `[q10 > q50]` 的退化输入 → rate = 1.0；
5. 单调化前后 crossing rate 变化断言；
6. baseline：历史不足一天 → NaN；上周同槽位正确对齐；
7. horizon_scores：fixed 3 步 + calendar_month 短月缺步聚合；
8. CQR 双区间：窗口 w 的 cal 列不消费窗口 ≥ w 的 score（spy 断言）；
9. `QuantileTrajectory.aligned`：长度不足/过长 RAISE（与 Direct 长度契约同语义）；
10. strategy 注册表：未知方法 fail-fast；blend 的 point/quantile 二级分发正确。

### 12.2 集成测试

1. 一个 quantile 配置实跑：预测 CSV 数值不变、新增 3 个评估 CSV；
2. 点预测配置：不生成概率列；
3. 9 方法 × 合成数据：重构前后预测值逐元素相等（Phase C 护栏）。

### 12.3 配置与结果验收

- 不修改 `config/`；
- 904 YAML 加载 0 错误；
- 现有 `setting` / `scenario_subpath` 无变化；
- `unittest discover` 全绿；`check_model_configs.py` 通过。

---

## 13. 风险与控制点

### 13.1 风险：评估层侵入预测路径

指标计算若消费不当（如在预测过程中提前算分）会改变行为。
控制点：评估函数只读 `cv_plot_df` 与 `quantile_outputs` 快照，在 `test_results_save` 阶段统一执行。

### 13.2 风险：Phase C 重构引入行为漂移

1483 行单类的等价重构，任何一处隐式状态依赖（如 `self.model` 被 blend 临时改写）都可能在搬移中丢失。
控制点：每方法独立等价测试先行（先写测试锁定当前行为，再搬移）；blend 的 `self.model` 临时改写模式在 strategy fn 内原样保留。

### 13.3 风险：CQR 测试段校准的泄漏

窗口 w 若用了含 w 的 score，校准评估失效。
控制点：spy 测试断言校准输入窗口集合严格 < w；与 ridge blend 权重学习共用同一窗口排除模式。

### 13.4 风险：评估产物膨胀

horizon_scores（H 行）+ probabilistic_scores（window×q 行）在 288 点高频场景会显著增大结果目录。
控制点：CSV 粒度按 (scenario, setting) 一份，不做 per-window 文件；高频场景 horizon 聚合按 4 步分桶（可选，默认逐步）。

### 13.5 风险：能力声明被误当"已实现"

注册表字段（supports_horizon_feature 等）只是声明，不代表全部组合已验证。
控制点：未验证组合保持现有 WARNING/RAISE 行为；声明字段只服务于代码路径选择，不进入文档的"已支持"结论。

---

## 14. 结论

本次重构的核心不是"更换多步预测算法"（9 种方法的数学语义与训练容器已经稳定，且 2026-08-24 刚完成时间语义修复），而是两件事：

1. **补齐概率预测评估闭环**（P0）：pinball/coverage/width/Winkler/crossing + horizon-wise + baseline suite + skill score + CQR 口径修正——这是从"能输出 quantile"到"能验证概率预测"的唯一路径，也是后续一切概率侧决策（CQR 扩展、分布模型对比）的前置；
2. **收敛推理架构**（P1/P2）：strategy 注册表消除分发长链与复制粘贴，`QuantileTrajectory` 统一三种 quantile 记录路径——为未来新增方法（如真正的 global panel 策略）降低边际成本。

全程行为兼容：预测值零变化（A/B）、行为等价护栏（C/D），不改配置、不改结果目录。

---

## 15. 实施进度记录

> 本节是 §11 实施步骤的**进度跟踪模块**：只记录"做到了哪、验证证据是什么、下一步是什么"，不复述设计。每完成一项必须回填证据（命令 + 结果摘要），无证据不得标记 completed。
>
> 状态机：`pending → in_progress → completed / blocked / cancelled`。同一时间只应有一个 in_progress 子项。
>
> Agent 使用约定：接手实施任务时先读本节定位断点；修改状态时同步更新 `last_updated`；实施中发现的问题追加到 §15.6"发现的实施偏差"表，不回改正文设计章节。

`last_updated: 2026-08-24`

### 15.1 总览

| Phase | 内容 | 状态 | 完成时间 | 证据摘要 |
|---|---|---|---|---|
| A | 概率评估闭环（指标库 + 3 评估产物 + baseline suite） | pending | — | — |
| B | CQR 测试段口径修正（双区间列） | pending | — | — |
| C | 推理 strategy 收敛（注册表 + QuantileTrajectory） | pending | — | — |
| D | 训练侧入口收敛（能力声明） | pending | — | — |

### 15.2 Phase A：概率评估闭环

- [ ] A-1 `utils/probabilistic_metrics.py` + 单测（手算复核）
- [ ] A-2 `ModelTesting._evaluate_probabilistic` → `probabilistic_scores_df.csv`
- [ ] A-3 `ModelTesting._evaluate_horizon` → `horizon_scores_df.csv`
- [ ] A-4 baseline suite + skill 列
- [ ] A-5 测试段 quantile 单调化双列
- [ ] A-6 现役 quantile 配置冒烟（预测数值不变 + 新增产物）

### 15.3 Phase B：CQR 口径修正

- [ ] B-1 `utils/conformal.py` 共享校准函数（forecast 逻辑迁移）
- [ ] B-2 测试段双区间列（泄漏 spy 测试）
- [ ] B-3 Coverage(raw)/Coverage(calibrated) 对照列

### 15.4 Phase C：推理 strategy 收敛

- [ ] C-1 `forecast_strategies.py` 注册表
- [ ] C-2 Direct 公共模式抽取
- [ ] C-3 `QuantileTrajectory` 统一契约
- [ ] C-4 分发替换 + 9 方法等价测试

### 15.5 Phase D：训练侧入口收敛

- [ ] D-1 能力声明字段
- [ ] D-2 Trainer 布尔判断替换 + 等价测试

### 15.6 发现的实施偏差

| # | 日期 | 偏差描述 | 影响与处置 |
|---|---|---|---|
| — | — | — | — |
