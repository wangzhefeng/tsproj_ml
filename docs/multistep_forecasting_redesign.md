# 多步预测模块重构设计方案

> 文档状态：Draft / v2 重新定界（2026-08-24——概率预测重构 Phase 0–5 已先于本方案完成并验收（329 tests OK，证据见 `docs/time_series_probabilistic_forecasting_redesign.md` §15），本方案原计划大部分被取代；v2 只保留仍存在的缺口）
> 创建日期：2026-08-24；v2 修订：2026-08-24
> 关联文档：调研依据见 `docs/kaggle_time_series_forecasting_research.md`（§6 多步预测、§9 概率预测）；概率预测已实现状态见 `docs/time_series_probabilistic_forecasting_redesign.md`；分解重构先例见 `docs/time_series_decomposition_redesign.md`。
> 本方案不构成实施授权；待决策项保留默认行为。

## 1. 背景与目标

### 1.1 背景

当前项目多步预测链路的事实状态（2026-08-24 代码核实）：

- `models/ModelForecasting.py`（1623 行）的 `Forecaster` 类承载 9 种预测方法的推理实现；分位数输出已统一收敛为 `probabilistic/types.py::ForecastDistribution`，`Forecaster.quantile_outputs` 退化为只读兼容 property；
- `probabilistic/` 包已落地：metrics（pinball/coverage/width/Winkler/crossing/Wilson CI）、evaluation（`probabilistic_predictions_df.csv` / `probabilistic_scores_df.csv` / `horizon_scores_df.csv` / `probabilistic_intervals_df.csv` / `calibration_report.csv`）、prequential CQR（as-of `label_available_at` 选择）、q50 锚定 crossing repair（raw 列 `predict_q*_raw` 已保留）、`ProbabilisticSpec` / `ForecastModelBundle` / `TargetTransformPipeline`；
- `models/ModelTraining.py` 的 `Trainer` 承载多输出训练容器（`DirectMultiOutputRegressor`、`HorizonAlignedDirectRegressor`）与 bundle 打包；quantile 训练已由 `probabilistic/training.py::QuantileTrainer` 接管；
- 方法分发仍是 50 行 if/elif 长链（`ModelForecasting.py:1400-1450`）；Trainer 的方法布尔判断（`_should_use_fourmethods_baseline_training` / `_should_use_horizon_feature` / `_is_blend_method` / `_should_use_horizon_aligned_direct`）仍散布多处。

2026-08-24 已完成四项多步语义修复（MSMD/MSMDR 目标对齐、Direct 输出长度契约、USMDP 多步 safe-lag、global panel 实体边界），时间语义缺陷已关闭；概率预测评估闭环已由专项重构落地，不属本方案范围。

### 1.2 目标（v2 重新定界）

v1 的四批次目标中，概率评估闭环、CQR 口径修正、quantile 输出统一契约已由 `docs/time_series_probabilistic_forecasting_redesign.md` 以更完整的设计完成。v2 只保留仍存在的缺口：

1. **baseline suite 与 skill score**（唯一 P0）：昨日同时刻 naive 已有，缺上周同时刻、同槽位中位数、SeasonalTemplate 基线与统一 skill 列；
2. **推理 strategy 注册表**（P1）：消除 50 行 if/elif 分发长链与 Direct 系公共模式复制；
3. **训练侧入口收敛**（P2）：Trainer 方法布尔判断归并为注册表能力声明。

非目标保持不变：不改预测算法语义、不改配置、不改结果目录、预测值行为等价。

### 1.3 非目标

本次明确不做：

1. 不修改 `config/` 下任何实验配置；
2. 不改变 9 种预测方法的数学语义（目标列、递归回填、blend 权重学习逻辑均不动）；
3. 不重复概率重构已完成的工作：pinball/coverage/width/Winkler/crossing 指标、horizon-wise 评估、prequential CQR、`ForecastDistribution`、raw quantile 列、q/PI 分列——以 `probabilistic/` 包为唯一实现，本方案只消费；
4. 不引入 DeepAR/GRU 等分布模型、不做层级协调（调研文档已判定为后续研究）；
5. 不重写训练侧容器（`DirectMultiOutputRegressor` 系列保持原样，仅收敛其选择逻辑的入口）；
6. 不把评估结果反向用于自动选型或自动调参（评估产物只报告、不决策）。

---

## 2. 当前现状与问题

### 2.1 当前事实（2026-08-24 代码核实）

| 组件 | 位置 | 现状 |
|---|---|---|
| 方法分发 | `ModelForecasting.py:1400-1450` | if/elif 长链，9 分支 + blend 的 point/quantile 二级分支；**无注册表** |
| quantile 记录 | `ModelForecasting.py:625-646` | `_record_quantile_direct`（长度精确契约）+ `_record_quantile_recursive_step`（逐步 store），最终汇入 `ForecastDistribution` |
| quantile 兼容视图 | `ModelForecasting.py:160-166` | `quantile_outputs` 只读 property，标注"主链应消费 ForecastDistribution" |
| 概率指标 | `probabilistic/metrics.py` | pinball / Wilson CI / interval metrics / crossing metrics 已实现 |
| 概率产物 | `probabilistic/evaluation.py` | 5 个 CSV 长表产物已接入 `ModelTesting.test_results_save`（L844 `write_probabilistic_artifacts`） |
| raw quantile 保留 | `ModelTesting.py:405-408` | 每窗口已写 `{q_col}_raw`（未修复列）+ `q_col`（q50 锚定 repair 后列）并存 |
| CQR | `probabilistic/calibration.py` + `evaluation.py` | prequential as-of 校准、`calibration_report.csv` 审计、forecast 共用同一 calibrator |
| 季节 naive | `ModelTesting.py:350-366` | 仅昨日同时刻一种；无上周/同槽位/模板基线，无 skill 列 |
| Trainer 方法判断 | `ModelTraining.py:243/342/353/617/626` | 四个布尔方法散布，`train()` 多处引用（L910/959/1071/1170） |

### 2.2 仍存在的问题

1. **baseline suite 缺失（P0）**：
   - 只有昨日同时刻 naive，无法回答"比上周同槽位/近邻中位数好多少"；
   - 无 skill score 列，median MAPE 缺少持久性锚点的相对口径；
   - 概率指标的 naive 对照只有 `Pinball vs Naive` 雏形，无系统性概率基线。

2. **推理架构平铺（P1）**：
   - 新增方法 = 在 1623 行单类里再插一个 100 行方法 + 分发分支 + Trainer 布尔判断；
   - Direct 系方法（USMD/MSMD/USMDR/MSMDR）共享的"anchor 输入 → 长度契约 → quantile 记录"模式在多处重复。

3. **训练侧方法感知分散（P2）**：
   - 方法能力（是否多输出、是否支持 quantile、是否支持 horizon_feature）以方法名字符串比较散布，新增方法需改多处。

---

## 3. 重构范围与优先级（v2）

| 批次 | 内容 | 性质 | 优先级 | v1 处置 |
|---|---|---|---|---|
| A′ | baseline suite + skill score | 纯新增，不改预测值 | **P0** | v1 Phase A 的 A-4 残留；其余已被概率重构取代 |
| C′ | 推理 strategy 注册表（dispatch + Direct 公共模式抽取） | 架构重构，行为等价 | P1 | v1 Phase C 的 C-1/C-2/C-4 残留；C-3（QuantileTrajectory）被 `ForecastDistribution` 取代 |
| D′ | 训练侧入口收敛（能力声明） | 架构重构，行为等价 | P2 | 同 v1 Phase D，锚点改为 C′ 的注册表 |

被概率重构取代、从本方案移除的范围（不再重复实施）：

- pinball/coverage/width/Winkler/crossing 指标库 → `probabilistic/metrics.py`（已实现）；
- `probabilistic_scores_df.csv` / `horizon_scores_df.csv` → `probabilistic/evaluation.py`（长表 schema，已实现；v1 的宽表设计作废）；
- raw quantile 列保留 → `ModelTesting.py:405` 已实现（`{q_col}_raw`）；
- CQR 测试段口径 → prequential CQR 已实现（as-of 标签可得性，比 v1 的"窗口 < w"更严格）。

A′ 独立于 C′/D′ 可先行交付；C′/D′ 以行为等价测试护航。

---

## 4. 模块放置与边界

### 4.1 baseline suite（批次 A′）

```text
models/
  ModelTesting.py               # 扩展现有 _build_seasonal_naive 为基线注册表，不新建文件
```

不单建 `utils/baselines.py` 的理由：基线生成本质依赖 `Tester._evaluate_split_index` 的窗口索引语义与 `df_history` 上下文，与 `ModelTesting` 强耦合；现有 naive 即在该文件内。参照 v1 指标库放 `utils/` 的先例被概率重构证伪（实际落在 `probabilistic/metrics.py` 包内）——基线同理跟随评分主体。

### 4.2 推理 strategy 层（批次 C′）

```text
models/
  forecast_strategies.py        # 9 种方法的 strategy 注册表
```

不新建顶层包的理由不变：收敛对象只是"分发 + 公共模式"，一个注册表文件 + `Forecaster` 保留为上下文容器即可。

### 4.3 不动的边界

- `probabilistic/` 包全部不动（本方案只消费其产物）；
- `models/ModelTraining.py` 的训练容器与 bundle 打包不动；
- `features/FeatureEngineering.py` 的目标构造不动；
- `main.py` 的编排不动。

---

## 5. 目标架构

### 5.1 baseline suite（批次 A′）

扩展现有 `_build_seasonal_naive` 为基线注册表：

| baseline | 生成方式 | 点口径 | 概率口径 |
|---|---|---|---|
| `naive_last_day` | 现有季节 naive（n_per_day 步前） | MAPE/MAE（已有） | pinball（q50=naive，区间=naive ± 近期残差分位数，可选 P2） |
| `naive_last_week` | 7×n_per_day 步前 | 同上 | 同上 |
| `naive_same_slot_median` | 近 K 个同槽位中位数 | 同上 | — |
| `SeasonalTemplate` | 复用现有 st 模型的模板逻辑 | 同上 | — |

频率自动选择合法基线：5min/15min/1h → last_day/last_week；1D → last_week/same_slot；1ME → 趋势 naive（P2 可选）；历史不足时该 baseline 记 NaN（与现有 naive 的 None 语义一致）。

skill score 统一定义：

```text
skill(metric) = 1 - loss_model(metric) / loss_baseline(metric)
```

落在 `test_scores_df.csv` 追加列 `Skill vs naive_last_day(MAPE)` 等；概率侧的 naive 对照沿用 `probabilistic_scores_df.csv` 的既有 `Pinball vs Naive` 列结构扩展，不新建文件。

### 5.2 推理 strategy 注册表（批次 C′）

```python
# models/forecast_strategies.py
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # 避免 forecast_strategies <-> ModelForecasting 的 import 环
    from models.ModelForecasting import Forecaster

@dataclass(frozen=True)
class ForecastStrategy:
    name: str                                  # pred_method 全名
    fn: Callable[["Forecaster"], np.ndarray]   # 执行入口（字符串注解，运行时不 import）
    is_multi_output: bool                      # 训练容器选择
    supports_quantile: bool
    supports_horizon_feature: bool             # 方法能力声明（供批次 D′ 消费）

FORECAST_STRATEGIES: dict[str, ForecastStrategy] = {
    "univariate-single-multistep-direct-pointwise": ForecastStrategy(...),
    "univariate-single-multistep-direct": ...,
    # ... 9 项
}

def dispatch(pred_method: str, predict_type: str) -> Callable[["Forecaster"], np.ndarray]:
    strategy = FORECAST_STRATEGIES.get(pred_method)
    if strategy is None:
        raise ValueError(f"Unsupported pred_method: {pred_method}")
    return strategy.quantile_fn if predict_type == "quantile" and strategy.quantile_fn else strategy.fn
```

- `Forecaster` 保留全部状态（df_history_for_lags、feature_engineer、scaler、model bundle），strategy 函数是方法绑定（第一参数是 Forecaster 实例），不复制状态；
- 分发长链替换为 `dispatch(...)(self)`；
- Direct 系公共模式（anchor 输入构建 → 长度契约 → `_record_quantile_direct`）抽为 `_direct_forecast_common(self, ...)` 供 USMD/MSMD 复用；DirRec 与 Recursive 的循环骨架差异大，不强行合并；
- quantile 输出契约**不再另行设计**：沿用概率重构的 `ForecastDistribution`，`_record_quantile_*` 保持现状，strategy fn 内部调用现有记录函数即可。

### 5.3 训练侧入口收敛（批次 D′）

`Trainer` 的方法布尔判断改为读取 `ForecastStrategy` 的能力声明字段，判断逻辑本身不变，只是声明从"散布的方法名字符串比较"收敛为注册表单点维护。

**声明与开关的两层区分（实施纪律）**：注册表字段只承载**方法能力**（USMD/MSMD 支持 horizon_feature、USMDR/MSMDR 不支持——这是方法属性，不随配置变化）；现有判断中的**配置开关**部分（如 `direct_strategy == "horizon_feature"`）仍从 `args` 读取，不硬编码进注册表。替换后的判断形如 `strategy.supports_horizon_feature and args.direct_strategy == "horizon_feature"`，与现行为逐点等价。

---

## 6. 现有 9 种方法如何映射到新架构

| 方法 | strategy.fn 归属 | 公共模式复用 | quantile 构造 | 行为变化 |
|---|---|---|---|---|
| USMDP | pointwise 专属 fn | —（safe-lag 分支保留在 fn 内） | 沿用现有记录路径 → ForecastDistribution | 无 |
| USMD | direct common | anchor + 长度契约 + record | 同上 | 无 |
| USMR | recursive fn | 循环骨架（单变量版） | 同上 | 无 |
| USMDR | dirrec fn | 循环骨架（块版） | 同上 | 无 |
| MSMD | direct common | 同 USMD | 同上 | 无 |
| MSMR | recursive fn（多变量版） | runtime_cache 逻辑保留 | 同上 | 无 |
| MSMDR | dirrec fn（多变量版） | 同 USMDR + aux 回填 | 同上 | 无 |
| USBR | blend fn | 权重解析保留 | blend 两路加权 → ForecastDistribution | 无 |
| MSBR | blend fn（多变量版） | 同 USBR | 同上 | 无 |

行为等价的验收方式：每方法一个"重构前后预测值逐元素相等"测试（固定随机种子 + 合成数据）。

---

## 7. 与主链的接线方式

### 7.1 测试侧（批次 A′）

```text
_window_test (现有)
  → _evaluate_score (现有，追加 baseline skill 列)
  → write_probabilistic_artifacts (现有，扩展 naive 概率对照)
```

baseline 计算复用 `_evaluate_split_index` 的窗口索引，只读 `df_history`，不触碰预测路径。

### 7.2 预测侧（批次 C′ 等价替换）

`main.py:forecast()` 调用 `Forecaster.forecast()` 的入口不变；分发长链在 `Forecaster.forecast()` 内部替换为 `dispatch()`。

### 7.3 产物落盘

批次 A′ 只追加 `test_scores_df.csv` 列与扩展 `probabilistic_scores_df.csv` 既有列结构，不新建文件；批次 C′/D′ 无产物变化。

---

## 8. 兼容与迁移策略

### 8.1 兼容原则

1. 不修改 `config/`：全部能力通过代码默认行为交付；
2. 预测值零变化：A′ 不进入预测路径；C′/D′ 由行为等价测试护栏；
3. 产物只增不改：现有 CSV 的列与顺序不变，新列只追加；
4. `setting` 与结果目录不受影响。

### 8.2 旧 API 到新 API 的映射

| 旧 | 新 | 过渡策略 |
|---|---|---|
| if/elif 分发（50 行） | `dispatch()` | 一次性替换，`grep` 验证无残留 |
| Trainer 方法布尔判断 | strategy 能力声明 | 批次 D′ 逐个替换，每个有独立等价测试 |
| `_build_seasonal_naive` 单基线 | baseline 注册表 | 保留原函数为 `naive_last_day` 实现，注册表包装 |

### 8.3 行为兼容要求

- 9 方法预测值逐元素相等（合成数据 + 固定种子）；
- 现有 unittest 全绿（当前基线 329 tests，2026-08-24）；
- 全量 863 个模型 YAML 加载 0 错误；
- `check_model_configs.py` 通过。

---

## 9. v1 必修复问题的处置状态

v1 §9 的两项已由概率预测重构完成，不再属于本方案：

| v1 项 | 处置 | 证据 |
|---|---|---|
| 9.1 测试段 CQR 口径 | **已由 prequential CQR 取代**（as-of `label_available_at`，比 v1 的"窗口 < w"更严格） | `probabilistic/evaluation.py::evaluate_prequential_cqr`、`calibration_report.csv`；概率重构文档 §15.3 |
| 9.2 测试段原始 quantile 列被单调化覆盖 | **已实现**（`{q_col}_raw` 与 repair 后列并存；crossing 消费 raw 矩阵） | `ModelTesting.py:405-408`、`probabilistic/evaluation.py:168-220` |

---

## 10. 预留扩展位

以下方向在本方案中只留接口位，不实现：

1. **概率口径 baseline 区间**：naive ± 近期残差分位数（P2，先只做点口径 skill）；
2. **regime 分层诊断**：按负荷水平/节假日/事件切片的 coverage 报告（标签必须预测原点可见，避免泄漏）；
3. **global panel 策略方法**：注册表预留新方法挂载点，panel 数据契约与消融另行评估。

---

## 11. 实施步骤

### Phase A′：baseline suite + skill score（P0，先行）

1. `ModelTesting` 基线注册表：`naive_last_day`（复用现有）/ `naive_last_week` / `naive_same_slot_median`；
2. `test_scores_df.csv` 追加 skill 列（`Skill vs naive_last_day(MAPE)` 等）；
3. `probabilistic_scores_df.csv` 扩展 naive 概率对照（沿用既有长表列结构）；
4. 单测：历史不足 → NaN；上周同槽位对齐正确；skill 手算复核；
5. 一个现役 quantile 配置冒烟：预测 CSV 数值不变、只新增评估列。

### Phase C′：推理 strategy 收敛（P1）

1. `models/forecast_strategies.py` 注册表 + 9 strategy；
2. Direct 系公共模式 `_direct_forecast_common` 抽取；
3. 分发长链替换为 `dispatch()`；
4. 9 方法行为等价测试（逐元素相等）。

### Phase D′：训练侧入口收敛（P2）

1. `ForecastStrategy` 能力声明字段（is_multi_output / supports_quantile / supports_horizon_feature）；
2. Trainer 布尔判断逐个改为读声明（声明=方法能力，开关仍读 args，见 §5.3）；
3. 每个替换点独立等价测试。

### 实施顺序与依赖

```text
Phase A′ ──独立──> 可先行交付
Phase C′ ──独立──> 行为等价护栏
Phase D′ ──依赖 C′ 的 ForecastStrategy──> 最后
```

---

## 12. 测试与验证

### 12.1 单元测试（新增）

1. baseline：历史不足一天 → NaN；上周同槽位正确对齐；same_slot_median 手算复核；
2. skill：已知 loss 比值 → 手算复核；
3. strategy 注册表：未知方法 fail-fast；blend 的 point/quantile 二级分发正确；
4. 9 方法行为等价：固定种子合成数据，重构前后预测值逐元素相等；
5. Trainer 声明替换：每个被替换的判断点有等价断言。

### 12.2 集成测试

1. 一个 quantile 配置实跑：预测 CSV 数值不变、`test_scores_df.csv` 只追加 skill 列；
2. 点预测配置：skill 列同样生成（skill 是点口径，不要求 quantile）。

### 12.3 配置与结果验收

- 不修改 `config/`；
- 863 模型 YAML 加载 0 错误；
- 现有 `setting` / `scenario_subpath` 无变化；
- `unittest discover` 全绿；`check_model_configs.py` 通过。

---

## 13. 风险与控制点

### 13.1 风险：评估层侵入预测路径

baseline 计算若消费不当（如在预测过程中提前计算）会改变行为。
控制点：baseline 只读 `df_history` 与窗口索引，在评分阶段统一执行。

### 13.2 风险：Phase C′ 重构引入行为漂移

1623 行单类的等价重构，任何一处隐式状态依赖（如 `self.model` 被 blend 临时改写）都可能在搬移中丢失。
控制点：每方法独立等价测试先行（先写测试锁定当前行为，再搬移）；blend 的 `self.model` 临时改写模式在 strategy fn 内原样保留。

### 13.3 风险：能力声明被误当"已实现"

注册表字段（supports_horizon_feature 等）只是声明，不代表全部组合已验证。
控制点：未验证组合保持现有 WARNING/RAISE 行为；声明字段只服务于代码路径选择，不进入文档的"已支持"结论。

### 13.4 风险：与概率重构的边界漂移

baseline 的概率对照若另起 schema，会与 `probabilistic_scores_df.csv` 长表结构冲突。
控制点：A′ 只扩展既有长表列，不新建文件、不改既有列语义。

---

## 14. 结论

v1 的四个批次中，概率评估闭环（A 的大部分）、CQR 口径修正（B）、quantile 统一契约（C-3）已由概率预测专项重构以更完整的设计完成。v2 的多步预测重构收敛为三项真实欠账：

1. **baseline suite 与 skill score**（P0）：median MAPE 缺持久性锚点的相对口径，这是调研文档遗留的唯一 P0；
2. **推理 strategy 注册表**（P1）：消除分发长链与 Direct 系复制；
3. **训练侧能力声明收敛**（P2）：方法能力单点维护。

全程行为兼容：A′ 不改预测值、C′/D′ 有等价护栏，不改配置、不改结果目录。

---

## 15. 实施进度记录

> 本节是 §11 实施步骤的**进度跟踪模块**：只记录"做到了哪、验证证据是什么、下一步是什么"，不复述设计。每完成一项必须回填证据（命令 + 结果摘要），无证据不得标记 completed。
>
> 状态机：`pending → in_progress → completed / blocked / cancelled / superseded`。同一时间只应有一个 in_progress 子项。
>
> Agent 使用约定：接手实施任务时先读本节定位断点；修改状态时同步更新 `last_updated`；实施中发现的问题追加到 §15.6"发现的实施偏差"表，不回改正文设计章节。

`last_updated: 2026-08-24`

### 15.1 总览（v2）

| Phase | 内容 | 状态 | 完成时间 | 证据摘要 |
|---|---|---|---|---|
| A′ | baseline suite + skill score | pending | — | — |
| C′ | 推理 strategy 注册表 | pending | — | — |
| D′ | 训练侧入口收敛 | pending | — | — |
| v1-A | 概率评估闭环 | **superseded** | 2026-08-24 | 由概率重构 Phase 0–4 完成（`probabilistic/metrics.py` + `evaluation.py` 5 个 CSV）；仅 A-4 baseline suite 残留为 Phase A′ |
| v1-B | CQR 测试段口径 | **superseded** | 2026-08-24 | 由概率重构 Phase 1 prequential CQR 取代（as-of 标签可得性，更严格） |
| v1-C | strategy 注册表 + QuantileTrajectory | **partially superseded** | 2026-08-24 | C-3 被 `ForecastDistribution` 取代；C-1/C-2/C-4 残留为 Phase C′ |

### 15.2 Phase A′：baseline suite + skill score

- [ ] A′-1 基线注册表（naive_last_week / naive_same_slot_median）
- [ ] A′-2 skill 列（`Skill vs naive_last_day(MAPE)` 等）
- [ ] A′-3 naive 概率对照扩展（沿用 `probabilistic_scores_df.csv` 长表）
- [ ] A′-4 单测（NaN 边界、对齐、手算）
- [ ] A′-5 现役 quantile 配置冒烟（预测数值不变 + 新增评估列）

### 15.3 Phase C′：推理 strategy 收敛

- [ ] C′-1 `forecast_strategies.py` 注册表
- [ ] C′-2 Direct 公共模式抽取
- [ ] C′-3 分发替换 + 9 方法等价测试

### 15.4 Phase D′：训练侧入口收敛

- [ ] D′-1 能力声明字段
- [ ] D′-2 Trainer 布尔判断替换 + 等价测试

### 15.5 当前代码核实证据（2026-08-24，v2 定界依据）

| 核实项 | 命令/方法 | 结果 |
|---|---|---|
| 全量 unittest | `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache .venv/bin/python -m unittest discover -s tests -p "test_*.py"` | **Ran 329 tests — OK** |
| 概率包存在性 | `ls probabilistic/` | metrics / evaluation / calibration / spec / types / objectives / training / pipeline / postprocessing 全部存在 |
| strategy 注册表不存在 | `ls models/forecast_strategies.py` | No such file |
| 分发长链仍在 | `grep -n 'elif self.args.pred_method' models/ModelForecasting.py` | 9 分支命中，L1400-1450 |
| Trainer 布尔散布 | `grep -n '_should_use_\|_is_blend_method' models/ModelTraining.py` | 5 处定义 + 4 处引用 |
| baseline 仅 naive_last_day | `grep -n 'naive_last_week\|same_slot_median\|skill' models/ModelTesting.py utils/ probabilistic/` | 无命中 |
| raw quantile 列已实现 | `grep -n '_raw' models/ModelTesting.py` | L405 `{q_col}_raw` 与 repair 后列并存 |
| CQR prequential 已实现 | `grep -n 'evaluate_prequential_cqr' probabilistic/evaluation.py` | L305 |
| quantile 兼容视图 | `ModelForecasting.py:160-166` | 只读 property，标注"主链应消费 ForecastDistribution" |

### 15.6 发现的实施偏差

| # | 日期 | 偏差描述 | 影响与处置 |
|---|---|---|---|
| 1 | 2026-08-24 | 评审发现初稿 §2.2-3/§9.2 的事实陈述与代码相反：测试段 CSV 写盘前已被单调化，缺的不是"补单调化"而是"保留原始列"。§9.2 已订正为"原始 quantile 列被单调化覆盖" | v1 修订完成；后被概率重构以 `{q_col}_raw` 实现（§9 处置表） |
| 2 | 2026-08-24 | 评审发现初稿 §5.2 双区间列未写明实现位置：`_window_test` 在 worker 进程内拿不到其他窗口的 conformal_score。§5.2 已补"实现位置（关键约束）" | v1 修订完成；后被概率重构 prequential CQR 取代（主进程聚合后执行，与订正方向一致） |
| 3 | 2026-08-24 | 评审发现初稿 §12.1-1 pinball 手算示例两值对调，且 §5.1.2 未定义 schema | v1 修订完成；schema 后被概率重构以长表实现（宽表设计作废） |
| 4 | 2026-08-24 | v2 定界：概率预测重构（Phase 0–5，329 tests OK）先于本方案完成，v1 的 Phase A（除 A-4）、B、C-3 被取代。根因：两个方案并行立项，概率重构覆盖了本方案的评估与契约层 | v2 重新定界为 baseline suite + strategy 注册表 + Trainer 收敛；§3 批次表、§5 目标架构、§9 处置状态、§11 实施步骤全部改写；§15.1 总览以 superseded 标注被取代批次 |
