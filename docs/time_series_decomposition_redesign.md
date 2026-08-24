# 时间序列分解模块重构设计方案

## 1. 背景与目标

### 1.1 背景

当前项目已经有一条可运行的目标分解链路：

- `features/TargetDecomposition.py` 中的 `TargetDecomposer` 负责目标分解、趋势外推、季节模板外推和结果还原；
- `main.py`、`models/ModelTesting.py`、`models/ModelForecasting.py`、`models/ModelTraining.py` 分别负责最终预测、滑窗测试、推理还原和持久化；
- `tests/test_target_decomposition.py` 已覆盖部分基础行为。

但从架构上看，这条链路仍然是“单个类 + 多个 if 分支 + 直接侵入主链”的实现，不满足以下目标：

1. 支撑 `linear / stl / mstl` 之外的更多分解方法；
2. 把 Trend / Seasonality / Hybrid 三种语义显式拆开；
3. 允许未来扩展乘性分解、变换域分解、OOF 分量特征、分量级模型和深度 basis 分解；
4. 保持现有业务行为稳定，不改变实验配置。

### 1.2 目标

本次重构的目标不是“换一个算法”，而是把分解从单文件实现升级成一个可扩展的模块：

1. 建立独立顶层模块 `decomposition/`；
2. 将现有 `linear / stl / mstl` 迁移为同一抽象下的注册方法；
3. 保持现有行为兼容：同一配置、同一数据下，最终预测与回测语义不变；
4. 为后续扩展预留方法位置，但不在本次实现；
5. 同步修正两个已确认的语义缺陷：
   - `linear + damped` 未消费 `decomposition_trend_lookback`；
   - `MSTL` 未透传 `decomposition_robust`；
6. 同步修正 `periodicity_analysis.py` 的两周期边界与标准季节强度输出，但单独验收，不与主链重构混为一个验收口径。
7. 在“不修改现有 YAML”的前提下，把分解参数从 `preprocessing.decomposition_*` 的散列字段升级为两层配置契约：现有 preset 简写继续兼容，未来 `method=custom` 支持组件级 extractor/forecaster/composer 编排，内部统一归一化为 `DecompositionSpec` 后再交给 registry/pipeline。

### 1.3 非目标

本次明确不做：

1. 不修改 `config/` 下任何实验配置；
2. 不新增可直接启用的乘性分解、log 分解、OOF 分量特征、分量级模型、深度 basis 分解；
3. 不改变现有 9 种预测方法的接口；
4. 不改变 `scale_target`、blend、conformal 的组合约束；
5. 不把分解逻辑塞进 `features/` 或 `models/` 作为长期主结构。

---

## 2. 当前现状与问题

### 2.1 当前事实

现有实现：

- 只有 `features/TargetDecomposition.py::TargetDecomposer`；
- 支持的 `decomposition_method` 为 `none / linear / stl / mstl`；
- 滑窗测试中先切原始电平，再只在训练段拟合；
- 最终预测在全部已知历史上重新拟合；
- point 和 quantile 都加回同一确定分量；
- 分解器与模型一起保存（v1 为独立 `target_decomposer.pkl`；2026-08-24 起升级为 v2 内嵌式 `decomposition_bundle.pkl`，见 §7.4）。

### 2.2 当前问题

1. `TargetDecomposer` 同时承担了：
   - 方法选择；
   - 成分抽取；
   - 趋势外推；
   - 季节模板构造；
   - 时间轴校验；
   - 未来还原；
   - 兼容 `none`。

   这使任何新方法都必须继续往同一个类里塞分支。

2. Trend、Seasonality、Hybrid 没有形成显式对象边界：
   - Trend 只是 `TargetDecomposer` 内部的一个分支；
   - Seasonality 同时存在于 `datetime_features`、`SeasonalTemplate` 和 `TargetDecomposer`；
   - Hybrid 实际上是“TargetDecomposer + 主模型 + Forecaster”的隐式组合，而不是一个显式 pipeline。

3. 已确认的现役语义缺陷：
   - `linear + damped` 路径没有消费 `decomposition_trend_lookback`；
   - `MSTL` 没有透传 `decomposition_robust`。

4. `periodicity_analysis.py` 的周期诊断与主链边界不一致：
   - 恰好两个完整周期时被拒绝；
   - `stl_seasonal_ratio` 不是标准季节强度；
   - 这些属于诊断层问题，不等于预测链路失效，但会误导分解选型。

5. 持久化虽然存在，但“model + scaler + decomposer + schema”不是一个显式 bundle，未来多成分方法会进一步放大这一点。

---

## 3. 模块放置与边界

### 3.1 推荐位置

新建顶层模块包（实际结构为平铺单文件，与实现一致）：

```text
decomposition/
  __init__.py
  spec.py          # DecompositionSpec / resolve_decomposition_spec / preset 展开
  types.py         # ComponentFrame / ComponentForecast / TARGET_KEY
  base.py          # ComponentExtractor / ComponentForecaster / Composer 抽象
  extractors.py    # NoneExtractor / PolyTrendExtractor / STLExtractor / MSTLExtractor
  forecasters.py   # PolyTrendForecastForecaster / PhaseTemplateForecaster
  composers.py     # AdditiveComposer
  pipeline.py      # DecompositionPipeline
  registry.py      # build_pipeline / 方法注册
  bundle.py        # DecompositionBundle 持久化
  diagnostics.py   # write_diagnostics_report
```

### 3.2 为什么不放在 `features/` 或 `models/`

#### 不放 `features/`

`features/` 的职责是构造输入特征、目标特征、滞后和滚动统计；分解模块服务的是目标空间和结构成分，不属于特征生成层。

#### 不放 `models/`

`models/` 的职责是训练、测试、推理、融合和模型持久化；分解模块不是预测器本身，而是“预测器之外的目标结构层”。

#### 放 `decomposition/`

分解同时连接：

- `data_provider/` / `features/`：历史数据与特征上下文；
- `models/`：主模型训练与推理；
- `main.py`：流程编排；
- `results/`：持久化与报告。

它天然是横向编排层，适合独立顶层包。

---

## 4. 目标架构

### 4.1 分层视图

新架构按 5 层组织：

1. **Extractor**
   - 负责在可见训练窗口内，把 `y` 拆成结构化成分与残差。

2. **Forecaster**
   - 负责把 trend / seasonal 等结构化成分外推到未来时点。

3. **Composer**
   - 负责把 residual 预测与结构分量合回原始目标空间。

4. **Pipeline**
   - 负责把 extractor / forecaster / composer 串起来，并对主链暴露统一接口。

5. **Registry**
   - 负责方法注册、配置分发和统一实例构造。

### 4.2 建议接口

```python
class ComponentExtractor:
    def fit(self, history: pd.DataFrame, time_col: str, target_col: str) -> None: ...
    def transform(self, history: pd.DataFrame) -> ComponentFrame: ...
    def components(self) -> dict[str, pd.Series]: ...


class ComponentForecaster:
    def fit(self, history: ComponentFrame) -> None: ...
    def predict(self, future_times: pd.DatetimeIndex) -> ComponentForecast: ...


class Composer:
    def compose(self, residual_pred: np.ndarray, component_forecast: ComponentForecast) -> np.ndarray: ...
    def restore_quantiles(self, quantile_map: dict[float, np.ndarray], component_forecast: ComponentForecast) -> dict[float, np.ndarray]: ...


class DecompositionPipeline:
    def fit_transform(self, history: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame: ...
    def forecast_components(self, future_times: pd.DatetimeIndex) -> ComponentForecast: ...
    def restore(self, values: np.ndarray, future_times: pd.DatetimeIndex) -> np.ndarray: ...
```

### 4.3 建议数据类型

```python
@dataclass
class ComponentFrame:
    times: pd.DatetimeIndex
    components: dict[str, np.ndarray]
    residual_name: str


@dataclass
class ComponentForecast:
    times: pd.DatetimeIndex
    components: dict[str, np.ndarray]


@dataclass(frozen=True)
class DecompositionSpec:
    # 完整字段契约见 §5.5
    method: str
    composition: str = "additive"
    extractor: ComponentSpec | None = None
    trend_forecaster: ComponentSpec | None = None
    seasonal_forecaster: ComponentSpec | None = None
    composer: ComponentSpec | None = None
```

其中：

- `ComponentFrame` 表示训练窗口内的历史分解结果；
- `ComponentForecast` 表示未来时点的结构化分量预测；
- `DecompositionSpec` 是配置归一化后的唯一运行时输入；组件级字段契约见 §5.5。

---

## 5. 参数配置架构

### 5.1 当前配置事实

对当前 `config/` 下 YAML 的实际扫描结果如下：

| 指标 | 数量 |
|---|---:|
| YAML 总数 | 904 |
| 含分解字段的 YAML | 686 |
| `decomposition_method=linear` | 266 |
| `decomposition_method=none` | 264 |
| `decomposition_method=stl` | 156 |
| `decomposition_method=mstl` | 0 |

当前分解相关字段全部位于 `overrides.preprocessing` 下：

| 字段 | 当前作用 |
|---|---|
| `decomposition_method` | 方法开关与类型 |
| `decomposition_periods` | STL/MSTL 周期 |
| `decomposition_robust` | STL/MSTL robust 开关 |
| `decomposition_trend_degree` | 趋势阶数 |
| `decomposition_trend_forecast` | 趋势外推方式 |
| `decomposition_damping` | damped 系数 |
| `decomposition_trend_lookback` | 近期趋势回看窗口；当前 YAML 未使用 |
| `decomposition_seasonal_cycles` | 季节模板回看周期数 |

这说明现有配置已经形成了事实上的“方法 + 趋势 + 季节 + 稳健性”参数面，但缺少一个统一的分解配置组。

### 5.2 配置设计原则

1. **不突破现有 YAML 约束**：仍然只允许 `overrides` 下一层分组，不能设计 `overrides.preprocessing.decomposition.method` 这种多级嵌套。
2. **旧字段保留为兼容输入**：现有 `preprocessing.decomposition_*` 继续可读，不能要求 686 个 YAML 同步迁移。
3. **新配置分为两层**：第一层是 `method=none/linear/stl/mstl` 的 preset 简写；第二层是 `method=custom` 的组件级配置，面向未来扩展。
4. **新配置组保留原始来源**：`overrides.decomposition` 应整体写入 `cfg.decomposition`，不能继续展平成 `cfg.method`、`cfg.periods` 等全局字段，否则会丢失配置来源并污染扁平命名空间。
5. **内部只认一种对象**：主链不直接消费散落字段，统一先归一化为 `DecompositionSpec`。
6. **冲突必须 fail-fast**：同一语义同时被旧字段和新字段设置且值不同，直接报错，不做静默覆盖。
7. **预留不等于可用**：`custom` 以及未实现的 extractor/forecaster/composer 类型，在对应实现接入前必须直接失败。

### 5.3 外部 YAML 契约

#### 5.3.1 兼容旧写法

现有 YAML 继续有效，不修改：

```yaml
overrides:
  preprocessing:
    decomposition_method: stl
    decomposition_periods: [96]
    decomposition_robust: true
    decomposition_trend_degree: 1
    decomposition_trend_forecast: polynomial
    decomposition_trend_lookback: 28
    decomposition_damping: 0.98
    decomposition_seasonal_cycles: 4
```

#### 5.3.2 新增规范写法

未来新配置推荐使用：

```yaml
overrides:
  decomposition:
    method: stl
    composition: additive
    periods: [96]
    robust: true
    trend_degree: 1
    trend_forecast: polynomial
    trend_lookback: 28
    damping: 0.98
    seasonal_cycles: 4
```

该结构只使用 `overrides` 下一层分组；通过在 `BaseModelConfig` 上声明 `decomposition` 字段，`config_loader` 会把整个 mapping 保留到 `cfg.decomposition`，不会把内部字段展平成全局配置属性。

#### 5.3.3 最小配置

`none`：

```yaml
overrides:
  decomposition:
    method: none
```

`linear`：

```yaml
overrides:
  decomposition:
    method: linear
    trend_degree: 1
    trend_forecast: polynomial
```

`stl`：

```yaml
overrides:
  decomposition:
    method: stl
    periods: [96]
    robust: true
    seasonal_cycles: 4
```

#### 5.3.4 组件级 custom 写法（预留）

未来真正需要组合不同组件时，使用 `method=custom`。示例只表达目标契约，当前实现阶段不得启用：

```yaml
overrides:
  decomposition:
    method: custom
    schema_version: 1
    extractor: mstl
    extractor_params:
      periods: [96, 672]
      robust: true
    trend_forecaster: damped_linear
    trend_params:
      degree: 1
      lookback: 28
      damping: 0.98
    seasonal_forecaster: phase_template
    seasonal_params:
      cycles: 4
    composer: additive
    composer_params: {}
```

其中 `extractor_params / trend_params / seasonal_params / composer_params` 是各组件自己的参数包；未知键、未知组件类型和未实现组件类型都必须由 `resolve_decomposition_spec()` 或 registry 直接拒绝。

### 5.4 字段映射

| 旧字段（`preprocessing`） | 新字段（`decomposition`） | 说明 |
|---|---|---|
| `decomposition_method` | `method` | 方法选择 |
| 无 | `composition` | 组合模式；当前仅允许 `additive` |
| `decomposition_periods` | `periods` | 周期列表 |
| `decomposition_robust` | `robust` | robust 分解 |
| `decomposition_trend_degree` | `trend_degree` | 趋势阶数 |
| `decomposition_trend_forecast` | `trend_forecast` | 趋势外推 |
| `decomposition_damping` | `damping` | damped 系数 |
| `decomposition_trend_lookback` | `trend_lookback` | 近期趋势回看 |
| `decomposition_seasonal_cycles` | `seasonal_cycles` | 季节模板回看周期 |
| 无 | `extractor` | custom 模式的 extractor 类型 |
| 无 | `extractor_params` | extractor 参数包 |
| 无 | `trend_forecaster` | trend 外推器类型 |
| 无 | `trend_params` | trend 外推参数包 |
| 无 | `seasonal_forecaster` | seasonal 外推器类型 |
| 无 | `seasonal_params` | seasonal 外推参数包 |
| 无 | `composer` | 组合器类型 |
| 无 | `composer_params` | 组合器参数包 |

### 5.5 内部归一化对象

新增统一组件配置与归一化对象：

```python
@dataclass(frozen=True)
class ComponentSpec:
    type: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class DecompositionSpec:
    method: str
    composition: str = "additive"
    extractor: ComponentSpec | None = None
    trend_forecaster: ComponentSpec | None = None
    seasonal_forecaster: ComponentSpec | None = None
    composer: ComponentSpec | None = None
```

所有 pipeline / registry / checker 都只消费 `DecompositionSpec`，不再直接读取 `cfg.decomposition_*` 或 `cfg.decomposition` 的原始字段。`DecompositionSpec` 中的各 `params` 必须在构建时完成拷贝与类型归一，后续运行不得再依赖原始 YAML dict。

### 5.6 新旧配置合并规则

`decomposition/spec.py` 中提供：

```python
def resolve_decomposition_spec(cfg) -> DecompositionSpec:
    ...
```

合并顺序：

1. `BaseModelConfig` 增加 `decomposition: Dict[str, Any]` 字段，使 `overrides.decomposition` 被整体保留到 `cfg.decomposition`，而不是展平为全局字段；
2. 只有旧字段：按 `preprocessing.decomposition_*` 生成 preset spec；
3. 只有新字段：按 `cfg.decomposition` 生成 spec；
4. 两者都没有：返回默认 `method=none`；
5. 两者都存在：先把两侧分别归一化为 preset spec，再逐项比较；
   - 语义完全一致：接受；
   - 任何字段不一致：直接 `ValueError`，禁止静默覆盖；
6. 新写法 `method=custom` 时，如果同时存在任何旧分解字段，直接拒绝，避免 preset 与 custom 语义混写；
7. `cfg.decomposition` 中的未知键、`custom` 的未知组件类型、未实现组件类型都在 `resolve_decomposition_spec()` 阶段直接失败；
8. preset 方法由 spec builder 展开为固定组件组合；例如 `stl` 展开为 STL extractor、默认 trend forecaster、phase-template seasonal forecaster 和 additive composer。

### 5.7 配置合法性矩阵

| 配置模式 | 必填字段 | 允许字段 | 不允许字段 |
|---|---|---|---|
| preset `none` | 无 | 无 | 趋势、周期、季节和组件字段均应省略 |
| preset `linear` | 无 | `trend_degree`, `trend_forecast`, `damping`, `trend_lookback` | `periods`, `robust`, `seasonal_cycles` |
| preset `stl` | `periods` 恰好 1 项 | `robust`, `trend_*`, `seasonal_cycles` | 组件级字段 |
| preset `mstl` | `periods` 至少 2 项 | `robust`, `trend_*`, `seasonal_cycles` | 组件级字段 |
| `custom` | `extractor`, `composer` | `*_params`, `trend_forecaster`, `seasonal_forecaster` | preset 简写字段 |
| 预留方法 | 不适用 | 不适用 | 当前一律拒绝 |

当前 `none` 配置已经只设置 `method`，因此 preset 约束与现有 YAML 兼容；`custom` 当前只作为未来契约存在，在组件实现接入前不得用于生产配置。

### 5.8 配置实施步骤

1. 在 `BaseModelConfig` 中新增 `decomposition: Dict[str, Any]` 字段，使 `overrides.decomposition` 保留为原始 mapping，不再展平成全局字段。
2. 保留 `PreprocessingConfig.decomposition_*` 字段作为兼容层，不从 dataclass 中删除。
3. 在 `decomposition/spec.py` 新增 `resolve_decomposition_spec(cfg)`，负责读取 `cfg.decomposition`、旧字段和默认值，并归一化为 `DecompositionSpec`。
4. 在 spec builder 中实现 preset 展开：`none / linear / stl / mstl` 分别映射到固定 extractor/forecaster/composer 组合。
5. `decomposition/registry.py` 只接收 `DecompositionSpec`，由它构造对应 pipeline。
6. `scripts/check_model_configs.py` 改为读取 spec 后校验 preset、组件字段和组合约束。
7. `config/generate_configs.py` 若未来生成新写法，必须显式支持输出 `decomposition` 原始 mapping；不要把组件级字段直接混入现有按字段展平的 `YAML_OVERRIDE_GROUPS`。
8. 本轮不迁移任何现有 YAML；旧写法与新写法的等价性完全由测试证明。

---

## 6. 现有方法如何映射到新架构

### 6.1 `none`

- extractor：空实现；
- forecaster：空实现；
- composer：恒等还原；
- 作为主链默认路径保留。

### 6.2 `linear`

- extractor：一次/二次多项式趋势抽取；
- forecaster：polynomial / damped 趋势外推；
- composer：加性组合；
- 现有行为作为第一批适配方法迁入。

### 6.3 `stl`

- extractor：STL 单周期趋势 + 季节分量；
- forecaster：趋势外推 + 最近若干周期相位模板外推；
- composer：加性组合；
- 与 `linear` 共用趋势外推逻辑，但 extractor 独立。

### 6.4 `mstl`

- extractor：MSTL 多周期趋势 + 多季节分量；
- forecaster：趋势外推 + 各周期模板叠加；
- composer：加性组合；
- 与 `stl` 共享季节模板和外推逻辑。

---

## 7. 与主链的接线方式

### 7.1 输入侧

`main.py`：

- 不再直接实例化 `TargetDecomposer`；
- 通过 registry/pipeline 构造 `DecompositionPipeline`；
- 最终预测仍在全部已知历史上拟合。

### 7.2 滑窗测试侧

`models/ModelTesting.py`：

- 每个窗口仍在训练段内重新拟合 pipeline；
- 测试标签仍保持原始电平；
- 不允许测试段参与 extractor。

### 7.3 推理侧

`models/ModelForecasting.py`：

- 只消费 pipeline 的 `restore()` / `forecast_components()`；
- 不再直接感知具体 `linear/stl/mstl` 分支。

### 7.4 持久化侧

`models/ModelTraining.py`：

- 保存 pipeline 或 pipeline 内部必要状态；
- 保留现有 `target_decomposer.pkl` 语义，但对象类型可切换为 pipeline；
- 如果后续要支持独立加载，再升级成 bundle。

> **实施更新（2026-08-24）**：本阶段先升级为 v2 内嵌式 `DecompositionBundle`。后续概率预测重构已将其收敛进唯一 schema v1 `ForecastModelBundle`：当前只写 `model.pkl`，同时持有 model、feature scaler、`TargetTransformPipeline`、selected features 与 input schema；`probabilistic_schema.json` 仅存 metadata，不再写新的 `decomposition_bundle.pkl`/`target_scaler.pkl`。最新事实见 `time_series_probabilistic_forecasting_redesign.md` §15。

---

## 8. 兼容与迁移策略

### 8.1 兼容原则

本次迁移遵循：

1. 不改 `config/`；
2. 不改 `README` 里对分解功能的公开口径；
3. 不保留 `features/TargetDecomposition.py` 兼容 shim；
4. 所有内部 imports 一次性切到新模块；
5. 现有测试全部改到新接口。

### 8.2 旧 API 到新 API 的映射

| 旧对象 | 新对象 |
|---|---|
| `TargetDecomposer` | `DecompositionPipeline` |
| `resolve_decomposition_method()` | `build_pipeline()` / registry 入口 |
| `history_component` | `ComponentFrame.components` |
| `_forecast_trend()` | trend forecaster |
| `_forecast_seasonal()` | seasonal forecaster |
| `restore()` | composer / pipeline restore |

### 8.3 行为兼容要求

同一输入、同一配置下：

- `linear` 的历史残差应等价于旧实现；
- `stl` 的历史趋势/季节分量应等价于旧实现；
- `mstl` 的历史趋势/季节分量应等价于旧实现；
- 最终 `restore()` 输出应等价于旧实现；
- `none` 应完全不改变现有行为。

---

## 9. 本次必须修复的问题

### 9.1 `linear + damped` 忽略 `decomposition_trend_lookback`

问题：

- STL/MSTL 分支会按 lookback 计算近期趋势斜率；
- `linear` 分支直接对全窗口趋势做 `np.polyfit`；
- 结果是 damped 外推没有按配置消费 lookback。

处理：

- 在 `trend forecaster` 中统一消费 `decomposition_trend_lookback`；
- `linear` 与 `stl/mstl` 使用同一趋势外推实现；
- 补充合成序列测试，验证 lookback 不同会产生不同未来趋势分量。

### 9.2 `MSTL` 未透传 `decomposition_robust`

问题：

- STL 已按配置传递 `robust`；
- MSTL 没有透传；
- 当前没有启用 MSTL 的 YAML，因此这是潜伏缺陷。

处理：

- 在 MSTL extractor 中通过 `stl_kwargs={"robust": decomposition_robust}` 透传；
- 补充 robust true/false 的接线测试；
- 暂不新增任何 MSTL 配置。

### 9.3 `periodicity_analysis.py` 的两周期边界与强度指标

问题：

- 恰好两个完整周期时被拒绝；
- `stl_seasonal_ratio` 不是标准季节强度。

处理：

- 将边界修正为与主链一致的 `2 * period <= n`；
- 保留旧指标名但标注为 legacy/custom；
- 新增标准 `seasonal_strength` / `trend_strength` 输出；
- 这部分单独验收，不与分解主链重构混成一条线。

---

## 10. 预留扩展位

以下方法在本次只做架构预留，不实现。

### 10.1 乘性 / 变换域分解

预留方法键：

- `multiplicative`
- `log_additive`
- `boxcox_additive`

未来扩展点：

- composer 支持非加性还原；
- pipeline 支持“先变换、再分解、再逆变换”。

### 10.2 OOF 分解预测作为二阶段特征

预留方法键：

- `oof_component_feature`

未来扩展点：

- extractor 生成 OOF component forecast；
- 将 component forecast 作为主模型输入特征，而不是直接改写目标。

### 10.3 分量级模型集成

预留方法键：

- `modal_vmd`
- `modal_ceemdan`
- `modal_wavelet`

未来扩展点：

- 每个模态分量独立建模；
- composer 汇总分量输出；
- 分量数量和端点效应需单独治理。

### 10.4 深度 basis / learned decomposition

预留方法键：

- `dlinear`
- `nbeats`
- `learned_basis`

未来扩展点：

- 独立深度训练管线；
- 不与当前 sklearn 风格 `ModelFactory` 混接；
- 作为研究支线，不占主链接口。

### 10.5 概率分量分解

预留方法键：

- `probabilistic_component`
- `quantile_component`

未来扩展点：

- 对 trend/seasonal 分量输出不确定性；
- 分量相关性和路径一致性另行建模；
- 先补边际概率评估，再考虑扩展到分量级。

---

## 11. 实施步骤

### 11.1 Phase 0：修正语义缺陷（completed 2026-08-23）

目标：

1. 修复 `linear + damped` 的 lookback；
2. 修复 MSTL robust 透传；
3. 修复 periodicity_analysis 的边界与强度指标。

验收：

- 合成序列下 `linear+damped` 的 future component 会随 lookback 改变；
- MSTL 构造时 `stl_kwargs` 中可见 `robust`；
- periodicity_analysis 允许恰好两个完整周期；
- 标准季节/趋势强度指标输出稳定。

### 11.2 Phase 1：建立配置契约归一化（completed 2026-08-24）

目标：

1. 在 `BaseModelConfig` 中新增 `decomposition: Dict[str, Any]`，保留新配置组的原始结构；
2. 保留 `PreprocessingConfig.decomposition_*` 兼容字段；
3. 新增 `decomposition/spec.py::resolve_decomposition_spec()`；
4. 在 spec builder 中完成 preset → extractor/forecaster/composer 展开；
5. 让 registry/checker 统一消费 `DecompositionSpec`；
6. 证明旧写法和新写法在同一输入下生成同一个 preset spec。

验收：

- 旧 YAML 写法加载后的 preset spec 与新写法一致；
- 新旧字段同值时不报错，冲突时直接 `ValueError`；
- preset `none / linear / stl / mstl` 的字段合法性矩阵由单元测试覆盖；
- `method=custom` 在组件实现接入前 fail-fast；
- 未实现预留方法在配置阶段即 fail-fast；
- 现有 686 个含分解字段的 YAML 不需要修改。

### 11.3 Phase 2：建立 `decomposition/` 包并迁移现有方法

目标：

1. 在 Phase 1 已有的 `decomposition/spec.py` 基础上补齐组件实现：extractor（none/poly_trend/stl/mstl）、forecaster（poly_trend_forecast/phase_template）、composer（additive）；
2. 实现 registry：只接收 `DecompositionSpec`，按组件类型构造实例并组装 `DecompositionPipeline`；
3. 用 pipeline 统一替换 `TargetDecomposer` 的主链入口（main.py / ModelTesting.py / ModelForecasting.py / ModelTraining.py），并把 main.py 中重复的分解校验段迁移到 spec/registry；
4. 删除旧 `features/TargetDecomposition.py`（无 shim，一次切换），更新所有内部 imports 与测试 imports；
5. 保持 Phase 0 已修复的两处语义（linear+damped 对 y 取 lookback；MSTL robust 透传）不回退。

遗留决策点（实施时已定）：

- periods 未配置时回退 `n_per_day` 的行为保留（当前 156 个 stl YAML 全部显式写 periods，回退路径仅作兜底）；
- pipeline 及其组件必须可 pickle（window_parallel_workers payload 过进程池）。

验收：

- 全套单测通过；
- 现有配置加载无变化；
- 除 Phase 0 两处已修复缺陷外，`none / linear / stl / mstl` 的历史 transform 与未来 restore 输出与旧实现逐位一致（用真实 stl 配置重跑对比 median MAPE）；
- 主链滑窗测试与最终预测行为一致。

### 11.4 Phase 3：补齐 bundle / registry / 诊断输出（completed 2026-08-25）

目标：

1. 将 model + scaler + decomposition pipeline 定义为统一持久化 bundle，加载路径做同构恢复；
2. `check_model_configs.py` 改为通过 `resolve_decomposition_spec()` 校验 preset/组件字段（吸收 Phase 1 顺延的 1.6）；
3. 增加 decomposition 分量诊断报告输出（trend/seasonal/residual 概览，只新增文件不覆盖旧结果）；
4. 未实现方法（custom + RESERVED_METHODS）在 registry 层 fail-fast，并在 checker 输出中可见。

验收：

- 保存/加载后的 pipeline 输出一致；
- `check_model_configs.py` 能识别注册方法；
- 未实现方法不会在主链被误调用；
- 诊断输出只新增，不覆盖旧结果。

### 11.5 Phase 4：再讨论扩展方法接入

仅在配置契约、模块迁移和诊断输出都稳定后，再评估：

- 乘性 / log 分解；
- OOF 分解预测特征；
- 分量级模型；
- 深度 basis 分解。

---

## 12. 测试与验证

### 12.1 单元测试

至少补充：

1. `linear` 重构等价测试；
2. `stl` 重构等价测试；
3. `mstl` 重构等价测试；
4. `none` 透传测试；
5. damped lookback 生效测试；
6. MSTL robust 透传测试；
7. periodicity 两周期边界测试；
8. point/quantile restore 一致性测试；
9. pipeline 持久化测试；
10. 旧 `preprocessing.decomposition_*` 写法与新 `decomposition` 写法的 spec 等价测试；
11. 新旧字段冲突时的 `ValueError` 测试；
12. preset `none / linear / stl / mstl` 字段合法性矩阵测试；
13. `cfg.decomposition` 保留原始 mapping、不展平为全局字段的 loader 测试；
14. `custom` 组件字段、未知键和未知组件类型的 fail-fast 测试；
15. 未实现预留方法的 fail-fast 测试。

### 12.2 集成测试

至少覆盖：

1. `is_testing=true` 时每个窗口仅用训练段拟合；
2. `is_forecasting=true` 时最终预测使用完整已知历史；
3. `test_scores_df.csv` / `prediction.csv` 与迁移前结构兼容；
4. 现有 9 种预测方法不回归。

### 12.3 配置与结果验收

必须满足：

- 不修改 `config/`；
- 现有 686 个含分解字段的 YAML 无需迁移且全部可加载；
- 不改变现有实验结果目录；
- 不因为重构导致新增 `setting` 或 `scenario_subpath`；
- 现有 `check_model_configs.py` 通过；
- `unittest discover` 全部通过。

---

## 13. 风险与控制点

### 13.1 风险：抽象过重

如果只做一个 `DecompositionPipeline` 的包装，而不真正拆 extractor / forecaster / composer，那么只是把分支搬到另一个文件，架构没有实质变化。

控制点：

- extractor、forecaster、composer 必须分层；
- registry 只做构造，不做预测逻辑。

### 13.2 风险：改动主链接口过深

`main.py`、`ModelTesting.py`、`ModelForecasting.py` 都是主链关键触点，如果一次性重写过多，会把重构变成高风险迁移。

控制点：

- 本次只做“入口替换 + 语义修复”；
- 不扩展算法；
- 不改配置；
- 不改结果目录。

### 13.3 风险：诊断层与预测层混淆

`periodicity_analysis.py` 属于诊断工具，不是预测主链；若把它的修复和分解主链混在一起，会扩大验收面。

控制点：

- 诊断修正单独验收；
- 不影响主链行为时不做额外回归重写。

### 13.4 风险：预留位被误当“已实现”

如果 registry 中预留了 `vmd / nbeats / oof_component_feature`，但没有明确 `not implemented` 边界，后续容易误用。

控制点：

- 预留方法只允许出现在文档和 registry 说明里；
- 不允许主链默认调用；
- 调用未实现方法必须 fail-fast。

### 13.5 风险：新配置组来源与可变 mapping 泄漏

新方案通过在 `BaseModelConfig` 上保留 `decomposition: Dict[str, Any]`，避免把 `method`、`composition`、`periods` 等字段展平成全局配置属性；这解决了命名空间污染，但引入两个边界：

- loader 不再替我们检查 `decomposition` 内部未知键；
- 原始 dict 可变，不能直接贯穿训练流程。

控制点：

- `resolve_decomposition_spec()` 必须显式检查允许键、组件类型和组件参数；
- 构建 `DecompositionSpec` 时必须深拷贝/归一化 params，禁止后续模块持有原始 YAML dict；
- `cfg.decomposition` 只作为配置输入，不作为预测期状态。

---

## 14. 结论

本次重构的核心不是“新增更多分解算法”，而是把分解从单个实现升级为一个清晰、可扩展、可验收的架构：

- 顶层模块放在 `decomposition/`；
- 参数配置升级为“旧 `preprocessing.decomposition_*` 兼容输入 + 新 `decomposition` preset/组件级两层输入 + 内部 `DecompositionSpec` 唯一消费”；
- 现有 `linear / stl / mstl` 迁入统一 pipeline；
- 现有行为保持兼容；
- `linear+damped` 与 `MSTL robust` 两个语义缺陷在本次一并修复；
- `periodicity_analysis` 的诊断问题单独修正、单独验收；
- 乘性分解、OOF 分量特征、分量级模型和深度 basis 分解都预留位置，但本次不实现。

该方案满足“马上重构”的目标，同时不会把实验配置、主链模型和结果目录卷进不必要的变化。

---

## 15. 实施进度记录

> 本节是 §11 实施步骤的**进度跟踪模块**：只记录"做到了哪、验证证据是什么、下一步是什么"，不复述设计。每完成一项必须回填证据（命令 + 结果摘要），无证据不得标记 completed。
>
> 状态机：`pending → in_progress → completed / blocked / cancelled`。同一时间一个 Phase 内只应有一个 in_progress 子项。
>
> Agent 使用约定：接手实施任务时先读本节定位断点；修改状态时同步更新 `last_updated`；实施中发现的问题追加到 §15.7"发现的实施偏差"表，不回改正文设计章节。

`last_updated: 2026-08-25`

### 15.1 总览

| Phase | 内容 | 状态 | 完成时间 | 证据摘要 |
|---|---|---|---|---|
| 0 | 修复 3 个语义缺陷（单独验收） | **completed** | 2026-08-23 | 见 §15.2；199 tests OK |
| 1 | 配置契约归一化（DecompositionSpec） | **completed** | 2026-08-24 | 见 §15.3；222 tests OK；904 YAML 加载 |
| 2 | 建立 decomposition/ 包并迁移现有方法 | **completed** | 2026-08-25 | 见 §15.4；229 tests OK；904 YAML；e2e STL 实跑 |
| 3 | bundle / registry / 诊断输出补齐 | **completed** | 2026-08-25 | 见 §15.5；235 tests OK；checker+e2e 通过 |
| 4 | 扩展方法接入讨论（乘性/OOF/分量级/深度） | pending | — | 依赖解除，待用户决策 |

### 15.2 Phase 0：修复语义缺陷（单独验收）—— completed

**验收口径**：仅修复行为缺陷，不做架构变更；诊断层（periodicity）与主链（TargetDecomposition）分开验收。

#### 0-a linear+damped 消费 decomposition_trend_lookback

- 状态：**completed**（2026-08-23）
- 改动：`features/TargetDecomposition.py`
  - 抽出 `_resolve_trend_lookback()`；
  - linear 分支改走 `_fit_trend_forecast()` 统一拟合，damped 近期斜率改为从**观测 y** 的 lookback 段估计。
- 实施偏差：§9.1 原方案"linear 与 stl/mstl 共用趋势外推即可"存在数学问题——linear 的趋势分量是全局多项式拟合线，对它取任何 lookback 窗口重拟合斜率都相同，lookback 永远无效。实际修复对 y 而非 trend 分量取 lookback，已在代码注释中说明理由（详见 §15.7 偏差 #1）。
- 测试：`tests/test_target_decomposition.py::test_linear_damped_forecast_consumes_trend_lookback`（RED→GREEN，lookback=7 斜率 10.0 / lookback=40 斜率 1.79，未来分量明显分化）
- 行为影响：仅 `decomposition_trend_forecast: damped` 且 lookback≠全窗时数值变化；当前仓库仅 2 个 damped 配置（load_month A/B 消融），均为待跑状态，无既有结果作废。

#### 0-b MSTL 透传 decomposition_robust

- 状态：**completed**（2026-08-23）
- 改动：`features/TargetDecomposition.py` MSTL 分支构造改为 `MSTL(y, periods=..., stl_kwargs={"robust": ...})`
- 测试：`tests/test_target_decomposition.py::test_mstl_passes_robust_to_statsmodels`（spy 断言 stl_kwargs == {"robust": True/False}）
- 行为影响：当前 0 个 MSTL 配置，纯潜伏缺陷修复，无结果影响。

#### 0-c periodicity_analysis 两周期边界 + 标准强度指标

- 状态：**completed**（2026-08-23）
- 改动：`data_process/periodicity_analysis.py`
  - `_stl_seasonal_component` 边界 `period >= len(y)//2` → `2*period > len(y)`（与主链一致，恰好两个完整周期可用）；
  - 新增标准 `stl_seasonal_strength` / `stl_trend_strength`（FPP 公式 `max(0, 1-Var(R)/Var(S+R))`）；`stl_seasonal_ratio` 保留并标注 legacy。
- 测试：`tests/test_periodicity_analysis.py`（新增文件，3 个用例：两周期可分解、强季节信号强度>0.5、白噪声强度下界 0）
- 行为影响：仅诊断输出，新增列不覆盖旧列；已有 periodicity 报告 CSV 不含新列，需重跑工具才会出现。

#### Phase 0 验收证据（2026-08-23）

| 验收项 | 命令 | 结果 |
|---|---|---|
| 全量单测 | `uv run python -m unittest discover -s tests` | **Ran 199 tests — OK** |
| 全量模型 YAML 加载 | 遍历 `config/**/*.yaml` + `load_yaml_config` | **863 loaded, 0 errors** |
| compileall | `uv run python -m compileall -q <改动目录>` | exit 0 |

§11.1 四条验收逐项对照：
1. linear+damped future component 随 lookback 改变 → ✅ 测试断言 recent[-1]-recent[0]>30 vs long<10
2. MSTL 构造时 stl_kwargs 可见 robust → ✅ spy 测试两值覆盖
3. periodicity 允许恰好两个完整周期 → ✅ 2*period=24=序列长测试
4. 标准强度指标输出稳定 → ✅ 合成正弦>0.5 / 白噪声∈[0,0.5) 两端验证

### 15.3 Phase 1：配置契约归一化（completed 2026-08-24）

按 §5.8 的 8 步执行，实际落地范围与偏差说明：

- [x] 1.1 `BaseModelConfig` 加 `decomposition: Dict[str, Any]`（completed 2026-08-24）
  - 改动：`config/config_sections.py:249-253` `PreprocessingConfig` 增加 mapping 字段；
  - 验证：loader 将 `overrides.decomposition` 整体保留到 `cfg.decomposition`，`method/periods` 等不出现在全局命名空间（loader 冒烟脚本断言 `not hasattr(cfg, 'method')`）。
- [x] 1.2 `PreprocessingConfig.decomposition_*` 兼容字段保留不删（completed；本次零改动，仅确认存在）
- [x] 1.3 `decomposition/spec.py::resolve_decomposition_spec()`（completed 2026-08-24）
  - 新文件 `decomposition/spec.py`（373 行）+ `decomposition/__init__.py`；
  - 实现 `ComponentSpec / PresetParams / DecompositionSpec / RESERVED_METHODS`；
  - 合并规则实现 §5.6 全部 8 条：单侧归一化、双侧等价接受、冲突 ValueError、custom×旧字段拒绝、未知键/预留方法 fail-fast、params 冻结为 `MappingProxyType`。
- [x] 1.4 preset 展开（completed 2026-08-24）
  - `DecompositionSpec.expand_preset()`：none→空组件；linear→poly_trend extractor + poly_trend_forecast；stl/mstl→对应 extractor + poly_trend_forecast + phase_template seasonal + additive composer。
  - 注意（与 §4.2 的接口分工差异）：Phase 1 阶段 `expand_preset()` 产出的是**组件配置描述**（type + params），不是组件实例；实例构造属 Phase 2 registry 职责，见偏差 #3。
- [x] 1.5 registry 只消费 DecompositionSpec — **顺延至 Phase 2**（Phase 1 无 registry 实体，spec 已按该契约设计；标记为 Phase 2 的 2.1 首项）
- [x] 1.6 `check_model_configs.py` 改读 spec — **顺延至 Phase 2/3**（与 registry 一并对齐；当前 checker 仍读旧字段，行为兼容不受影响，见偏差 #3）
- [x] 1.7 `generate_configs.py` — **不实施**（当前无生成新写法配置的需求，§5.8 第 7 步本身为条件项）
- [x] 1.8 新旧写法 spec 等价测试（completed 2026-08-24）
  - 新文件 `tests/test_decomposition_spec.py`：23 个用例，覆盖 §12.1 第 10-15 条全部（等价、冲突 ValueError、合法性矩阵、loader mapping 保留、custom/未知键/预留方法 fail-fast、params 冻结拷贝）。

#### Phase 1 验收证据（2026-08-24）

| 验收项 | 命令 | 结果 |
|---|---|---|
| spec 单测 | `uv run python -m unittest tests.test_decomposition_spec -v` | **Ran 23 tests — OK** |
| 全量单测 | `uv run python -m unittest discover -s tests` | **Ran 222 tests — OK**（199→222） |
| 全量 YAML 加载+spec 解析 | 遍历 904 个 YAML + `resolve_decomposition_spec` | **904 loaded, 0 failed**；methods: none=482 / linear=266 / stl=156（与 §5.1 基线一致） |
| compileall | `compileall -q decomposition config tests/test_decomposition_spec.py` | exit 0 |

§11.2 验收逐项对照：
1. 旧 YAML preset spec == 新写法 → ✅ `test_legacy_and_new_same_semantics_produce_equal_specs`
2. 新旧字段同值不报错 / 冲突 ValueError → ✅ 等价用例 + `test_conflicting_legacy_and_new_raises`
3. preset 合法性矩阵单测覆盖 → ✅ none/linear/stl/mstl 各有专项用例
4. custom fail-fast → ✅ `test_custom_not_implemented` + custom×旧字段拒绝
5. 未实现预留方法配置阶段失败 → ✅ `test_reserved_methods_rejected`（5 个代表键）
6. 686 个含分解 YAML 零修改 → ✅ 全量 904 个 YAML 全部通过（含分解的 686 在其中，未改任何 config 文件）

### 15.4 Phase 2：decomposition/ 包迁移（completed 2026-08-25）

> 状态：completed（2026-08-25）。

- [x] 2.1 新建包骨架（completed 2026-08-25）
  - `decomposition/` 包：`__init__.py / spec.py / types.py / base.py / extractors.py / forecasters.py / composers.py / pipeline.py / registry.py`
- [x] 2.2 迁移 none/linear/stl/mstl 组件实现（completed 2026-08-25）
  - `NoneExtractor / PolyTrendExtractor / STLExtractor / MSTLExtractor`；
  - `PolyTrendForecastForecaster / PhaseTemplateForecaster`；
  - `AdditiveComposer`；
  - `DecompositionPipeline.from_args(cfg)` 替代 `TargetDecomposer(args)`。
- [x] 2.3 主链入口替换（completed 2026-08-25）
  - `main.py`：import 切换 + `resolve_decomposition_spec().method` 替代 `resolve_decomposition_method()`；
  - `models/ModelTesting.py`：`DecompositionPipeline.from_args(args)`；
  - `models/ModelForecasting.py` / `models/ModelTraining.py`：通过 `target_decomposer` 属性接口（`is_fitted/restore`）已天然兼容，无需改动。
- [x] 2.4 删除 features/TargetDecomposition.py（completed 2026-08-25）
  - 全仓引用已清零后 `rm`；`compileall` exit 0。
- [x] 2.5 等价性验证（completed 2026-08-25）
  - `tests/test_decomposition_equivalence.py`：7 用例，none/linear(1次,2次)/stl(robust T,F)/pickle roundtrip，`atol=1e-6~1e-10` 全过；
  - `tests/test_target_decomposition.py`：7 用例全部切到新 pipeline（含 Phase 0 的 lookback/robust 测试），语义不回退；
  - 端到端：`lgbm_usmd_prob_mean_decomp_stl7.yaml`（power_month A 路）2 窗口实跑，median MAPE 0.0179（与历史水平一致）。

#### Phase 2 验收证据（2026-08-25）

| 验收项 | 命令 | 结果 |
|---|---|---|
| 全量单测 | `uv run python -m unittest discover -s tests` | **Ran 229 tests — OK**（222→229） |
| 全量 YAML 加载+pipeline 构造 | 904 个 YAML + `build_pipeline(spec)` | **904 loaded, 0 failed** |
| compileall | `compileall -q decomposition features models main.py config tests` | exit 0 |
| 端到端 STL 实跑 | power_month A 路 `decomp_stl7` 配置 2 窗口 | **median MAPE 0.0179**，主链无报错 |

遗留决策点执行结果：
- main.py 重复校验段：`resolve_decomposition_method()` 已替换为 `resolve_decomposition_spec().method`，校验逻辑保持原位（Phase 3 checker 对齐时统一迁移）；
- periods 回退 n_per_day：保留（`_SeasonalExtractorBase._resolve_periods` 支持回退，当前 156 个 stl YAML 全部显式写 periods 不受影响）；
- pickle 兼容：`test_pipeline_pickle_roundtrip` 通过。
- 遗留决策点（正文未写明，实施时需定）：
  - main.py:236-252 重复校验段迁移到 spec/registry
  - periods 未配置时回退 n_per_day 的行为是否保留（当前 156 个 stl 配置全部显式写 periods）
  - pipeline 组件保持可 pickle（window_parallel_workers payload 过进程池）

### 15.5 Phase 3：bundle / registry / 诊断（completed 2026-08-25）

- [x] 3.1 持久化统一 bundle（completed 2026-08-25）
  - 新增 `decomposition/bundle.py`：`DecompositionBundle`（pipeline + spec 摘要 + schema_version），save/load 带 schema 校验和类型校验；
  - `ModelTraining.model_save()` 在保存 `target_decomposer.pkl` 的同时写入 `decomposition_bundle.pkl`（向后兼容：旧 pkl 保留，bundle 为新增）。
  - **后续更新（2026-08-24）**：v2 局部 bundle 随后被唯一 `ForecastModelBundle` 吸收；当前 `model.pkl` 同构保存 model、feature scaler、目标变换栈、selected features 和 input schema，局部 decomposition bundle 仅保留为历史兼容类型。
- [x] 3.2 registry 与 check_model_configs 对齐（completed 2026-08-25）
  - `scripts/check_model_configs.py` 改为通过 `resolve_decomposition_spec(cfg)` 校验，异常写入 problems 而非崩溃；checker dry-run 全配置通过。
- [x] 3.3 decomposition 诊断报告输出（completed 2026-08-25）
  - 新增 `decomposition/diagnostics.py`：`write_diagnostics_report()` 输出 `decomposition_diagnostics.csv`（y/deterministic/residual/trend/seasonal 各分量的 mean/std/min/max/n）；
  - `ModelTesting._window_test` 在 fit 后写入；无 test_results_dir 的测试环境自动跳过；端到端实跑已验证文件产出。
- [x] 3.4 未实现方法 fail-fast 边界（completed 2026-08-24，Phase 1 已覆盖）
  - `resolve_decomposition_spec()` 拦截 custom + RESERVED_METHODS；
  - `decomposition/registry.py::build_pipeline()` 兜底拦截 custom；
  - checker 在 spec 解析异常时写入 problems。

#### Phase 3 验收证据（2026-08-25）

| 验收项 | 命令 | 结果 |
|---|---|---|
| 全量单测 | `uv run python -m unittest discover -s tests` | **Ran 235 tests — OK**（229→235） |
| checker dry-run | `check_model_configs.py 'config/aidc_power_month/.../add_decomposition/*.yaml'` | **全部配置通过校验** |
| 端到端 STL 实跑 | power_month A 路 `decomp_stl7` 配置 2 窗口 | **median MAPE 0.0179**，`decomposition_diagnostics.csv` 产出 |
| bundle roundtrip | `test_bundle_roundtrip_preserves_predictions` | pickle 保存→加载→预测逐位一致 |
| 全量 YAML + pipeline 构造 | 904 个 YAML + `build_pipeline(spec)` | **904 loaded, 0 failed** |
| compileall | `compileall -q decomposition scripts features models main.py config tests` | exit 0 |

§11.4 验收逐项对照：
1. 保存/加载后 pipeline 输出一致 → ✅ bundle roundtrip 测试
2. check_model_configs.py 能识别注册方法 → ✅ spec-based 校验 + dry-run 全过
3. 未实现方法不在主链被误调用 → ✅ spec/registry 双层 fail-fast
4. 诊断输出只新增不覆盖 → ✅ `decomposition_diagnostics.csv` 新文件，旧结果目录无修改

### 15.6 Phase 4：扩展方法讨论（B1 残差频谱诊断 completed 2026-08-24）

依赖已解除（Phase 1-3 全部 completed）。既有结论：VMD/Wavelet 需先有残差频谱证据；EMD 系列最后；Prophet/TBATS 不进主链。

B1 残差频谱诊断已完成：
- 新增 `decomposition/residual_diagnostics.py`：`diagnose_window_residual`（单窗 FFT/ACF）、`summarize_window_residuals`（跨窗 CV 评估）、`write_residual_diagnostics`（CSV 输出）；
- `ModelTesting._window_test` 每窗口收集 residual 诊断行随结果返回；
- `test_results_save` 汇总全部窗口并写 `residual_diagnostics.csv`；
- 判定逻辑：FFT 主导周期 CV < 0.5 且最强 ACF 峰 > 0.3 → `stable_band_detected=True`（提示存在未被分解吸收的频带结构，可评估 VMD/Wavelet）；
- 测试：`tests/test_residual_diagnostics.py` 7 用例（周期检测、白噪声拒绝、漂移拒绝、CSV 结构）。

VMD/Wavelet 的启动与否取决于实跑 `residual_diagnostics.csv` 的 stable_band 结果；乘性/log 分解（B2）和 OOF 分量特征（B3）待用户决策。

### 15.7 发现的实施偏差（实施中追加，勿删）

| # | 日期 | 偏差 | 处理 |
|---|---|---|---|
| 1 | 2026-08-23 | 0-a 修复方案偏离 §9.1（对 y 而非 trend 分量取 lookback，原因：全局多项式重拟合任何窗口斜率相同） | 已在代码注释说明；Phase 2 迁移时保持该语义 |
| 2 | 2026-08-23 | 全量 YAML 加载实测 863 个（§5.1 的 686 为"含分解字段"口径、904 为含非模型 yaml 总数口径），两个口径均与文档一致，无矛盾 | 记录口径差异，无需处理 |
| 3 | 2026-08-24 | Phase 1 将原计划的 1.5（registry 消费 spec）和 1.6（checker 改读 spec）顺延至 Phase 2/3：Phase 1 无 registry 实体，强行提前只会造空壳；checker 在 registry 落地前读旧字段行为不变 | 已在 §15.3 标注顺延；Phase 2 的 2.1/2.3 吸收 |
| 4 | 2026-08-24 | `_preset_params_from_legacy` 中 `none` 分支存在冗余条件（method=none 时提前 return None 的判定与外层 `_legacy_explicitly_set` 部分重叠），行为正确但逻辑可简化 | 行为已由 23 个测试锁定；Phase 2 重构 spec.py 时一并简化 |
| 5 | 2026-08-25 | Phase 2 等价性验证发现 STL 历史分量偏差 0.73：`STLExtractor.components()` 初版用 polyfit 重构值替代 STL 原始输出，导致历史 transform 与旧实现不等价 | 修复：fit 阶段缓存 `exact_deterministic`（STL/MSTL 原始 trend+seasonal），components() 返回缓存值 |
| 6 | 2026-08-25 | Phase 2 发现 linear+damped lookback 失效：`expand_preset()` 初版只给 trend_forecaster 传 mode/lookback，未传给 extractor（extractor 需在 fit 阶段决定近期斜率来源） | 修复：extractor params 也带 mode/lookback；等价性测试 lookback=7/40 分化恢复 |
| 7 | 2026-08-25 | 诊断输出写入 `_window_test` 后，测试 SimpleNamespace 缺 `test_results_dir` 导致 1 个测试报 AttributeError | 修复：`getattr(args, "test_results_dir", None)` 防御，无目录跳过诊断 |
| 8 | 2026-08-25 | main.py:243-259 周期校验段仍直读旧字段 `cfg.decomposition_periods`：新写法 `overrides.decomposition` 的 STL/MSTL 配置会因 legacy 属性为默认 `[]` 被误拒 | **已修复**（待办 #1）：main.py 改读 `spec.preset.periods` |
| 9 | 2026-08-25 | §11.4 目标 1 部分完成：`DecompositionBundle` 只包 pipeline + spec 摘要，model/scaler 仍是独立 pkl，加载侧无代码消费 bundle | **已修复**（待办 #2）：v2 内嵌式 bundle，ModelDeployPkl 自动识别，一次切换完成 |
| 10 | 2026-08-25 | §12.1 第 3 条 mstl 等价测试缺失；且旧文件删除后等价测试变成新 pipeline 自比，对新旧逐位一致性不再约束 | **已修复**（待办 #3）：旧实现冻结为 tests/fixtures/，等价测试恢复真实新旧对比并补 mstl 用例 |
| 11 | 2026-08-25 | `ModelTesting._window_test` 每窗口写同一路径 `decomposition_diagnostics.csv`，逐窗口覆盖且重跑覆盖上次结果，偏离"只新增不覆盖"口径 | **已修复**（待办 #4，方案 A）：按窗口序号命名 `_win{N}` |
| 12 | 2026-08-25 | 实现与文档接口漂移：`ComponentFrame.residual_name` 未实现；`base.py` 用未文档化的 `"__y__"` 键；`pipeline.py` 有死代码 `hasattr(..., "attach_trend"): pass`；`trend_coefficients_context_ns` 为动态贴附属性未在 `__init__` 声明；§3.1 目录结构（子包）与实际（平铺单文件）不符 | **已修复**（待办 #5）：全部清理完毕；文档 §3.1/§4.3 已同步 |

### 15.8 环境备忘

- 2026-08-23 起点工作区：`.gitignore`、`docs/kaggle_time_series_forecasting_research.md`（用户侧修改）+ 本设计文档；AGENTS.md 两条旧文案已于 2026-08-23 修正为 `decomposition_method` 口径。
- 验证命令统一前缀：`env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python ...`

### 15.9 遗留待办清单（2026-08-25 review 发现）

> 本节跟踪 Phase 0-3 完成后 review 出的未收口事项。每项关闭时回填证据并标 completed；对应实施偏差见 §15.7 #8-#12。

| # | 优先级 | 待办 | 状态 | 证据 |
|---|---|---|---|---|
| 1 | **P0** | main.py:243-259 校验段改读 `spec.preset.periods`，删除 legacy 直读；否则新写法 stl/mstl 配置被误拒 | **completed**（2026-08-24） | main.py 改读 `spec.preset.periods`；新写法 stl/mstl spec 探针通过；252 tests OK |
| 2 | P1 | bundle 扩展为 model + scaler + pipeline 统一持久化，定义加载入口 | **completed，并被后续统一 bundle 吸收**（2026-08-24） | 当前由 schema v1 `ForecastModelBundle` 统一持有 model、feature scaler、`TargetTransformPipeline`、selected features/input schema；`ModelDeployPkl.load_model` 兼容历史对象并拒绝未知 schema |
| 3 | P1 | 补 mstl 新旧等价测试：从 `git show e4502fd:features/TargetDecomposition.py` 恢复旧实现做一次性 transform/restore 数值对比，结果固化进测试 | **completed**（2026-08-24） | 旧实现冻结为 `tests/fixtures/legacy_target_decomposition.py`；`test_decomposition_equivalence.py` 改用真实旧实现（非自比），新增 mstl robust T/F 两例，9 用例全过 |
| 4 | P2 | 诊断报告避免覆盖：按窗口/run 加后缀，或全窗口聚合后写一次 | **completed**（2026-08-24，方案 A） | `write_diagnostics_report` 加 `suffix` 参数；`_window_test` 传 `f"_win{window}"`；新测试 `test_diagnostics_suffix_avoids_overwrite` |
| 5 | P2 | 低成本收尾：清理 `spec.py` none 冗余分支（偏差 #4）、`pipeline.py` 死代码、`trend_coefficients_context_ns` 动态属性入 `__init__`；文档 §3.1/§4.3 与实现对齐 | **completed**（2026-08-24） | spec 冗余分支已删；`attach_trend` 死代码已删；`trend_coefficients_context_ns` 入 `__init__` 并去掉 type: ignore；`__y__` 提为 `TARGET_KEY` 常量（types.py，三处引用统一）；`ComponentFrame.residual_name` 文档同步为 TARGET_KEY 语义；§3.1 目录结构更新为平铺单文件结构 |
| 6 | — | Phase 4 扩展方法接入讨论（乘性/OOF/分量级/深度） | B1 残差诊断 completed；B2/B3 pending | 实跑 stable_band 结果裁决 VMD/Wavelet；B2/B3 待用户决策 |

建议处理顺序：#1 → #4 → #5 → #3 → #2 → #6。
