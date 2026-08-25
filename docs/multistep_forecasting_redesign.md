# 多步预测策略重构设计方案

> 文档状态：Implemented / v3 代码收口；存量结果重跑另行跟踪
> 创建日期：2026-08-24；最近更新：2026-08-25
> 适用范围：九种多步预测策略的配置解析、目标构造、训练计划、推理执行、组合能力、运行时状态与兼容迁移。
> 关联文档：多步预测调研依据见 `docs/kaggle_time_series_forecasting_research.md` §6；实施先例见 `docs/time_series_decomposition_redesign.md` 的状态跟踪方式。
> 代码实施已按 §14 完成；受数值语义修复影响的存量结果见 `docs/multistep_forecasting_invalidation.md`，不因代码收口自动视为已重跑。

## 1. 结论先行

当前项目不是“缺少一个分发注册表”，而是**同一预测策略在六个层次被重复解释**：

1. 配置层判断方法是否合法；
2. 数据层判断保留单变量还是多变量；
3. 特征层决定 lag 与目标列如何构造；
4. 训练层根据目标列形状选择训练容器；
5. 推理层通过九个方法和长分支执行；
6. 缩放、模型封装和增强逻辑再次按方法名猜测目标结构。

只把 `if/elif` 换成函数映射不会消除这些问题，反而会把分散语义藏到注册表后面。v3 采用一次到位的方案：

- 对外继续接受现有九个 `pred_method` 全名和短码，配置入口不改；
- 对内将九种方法分解为两个正交维度：
  - **输入范围**：仅目标序列 / 全部内生序列；
  - **推进族**：逐点、直接、递归、分块直接递归、直接递归融合；
- 由一个 `ResolvedStrategy` 同时生成 `TargetPlan`、`TrainingPlan`、`FeaturePlan` 和 `RuntimePlan`；
- 五类执行器只消费已解析计划，不再读取方法名字符串；
- 所有短输出、缺列、非法回填、缺权重和不完整面板输入改为就地失败，不再静默截断、补零或降级；
- 复合能力通过组合对象实现，不再依赖临时改写 `Forecaster.model`、读取结果目录文件或共享可变历史。

该方案会保留外部配置契约和结果目录契约，但**不会承诺所有数值逐元素不变**：修复 DirRec 训练/推理计划不一致、静默降级等缺陷时，新训练模型的结果发生变化是预期行为，必须重跑相应配置并重新验收。

---

## 2. 重构目标与成功标准

### 2.1 目标

1. 九种方法的数学语义能够由单一、可检查的策略规格完整表达；
2. 方法合法性、目标步集合、训练布局、推理推进方式和增强能力来自同一解析结果；
3. 新增一种推进族时，不需要修改 `FeatureEngineering`、`Trainer`、`Forecaster`、`DataLoader` 和缩放器中的多组字符串分支；
4. 每类执行器具有明确的输入、状态、调用次数和输出长度契约；
5. 多变量未来内生值、融合权重、特征 schema 和全局面板均成为一等接口；
6. 旧配置和旧模型可迁移读取，新增模型产物自描述且可独立推理；
7. 已确认的多步语义修复持续受测试保护。

### 2.2 成功标准

| 维度 | 验收标准 |
|---|---|
| 单一事实源 | 活动主链中九个方法全名只允许出现在策略目录、兼容适配器和针对性测试中 |
| 目标语义 | 九种方法的 `TargetPlan.label_steps`、目标列顺序和输出宽度均有表驱动测试 |
| 训练/推理一致 | `TrainingPlan.output_width == RuntimePlan.model_output_width`；DirRec 的有效块长只解析一次 |
| 输出契约 | 每条单序列预测严格返回 `horizon` 个有限数值；子执行器长度不符在本层报错 |
| 状态隔离 | 一次预测调用不修改输入模型句柄、原始历史帧或其他执行器状态 |
| 组合可复现 | 融合权重与辅助轨迹模型随模型产物保存，推理不依赖测试结果目录 |
| 特征一致 | 训练 schema 缺列直接失败；额外列按显式策略处理；不再用 `0.0` 伪造缺失列 |
| 配置兼容 | 当前全部模型 YAML 可加载；现有 `pred_method`、短码、setting 和场景目录规则不变 |
| 方法覆盖 | 九种方法均有最小端到端点预测测试，覆盖 H=1、H>1、尾块不足和多变量回填 |
| 面板能力 | 启用全局训练时，训练、滑窗、未来输入和输出均按 `(series_id, time)` 分组且不串序列 |

### 2.3 非目标

本方案不处理：

1. 新增深度学习或统计预测模型；
2. 更换 LightGBM、XGBoost 等底层估计器工厂；
3. 修改业务评估指标、绘图或结果目录命名；
4. 自动选择最优多步策略或自动推广实验配置；
5. 与多步推进无关的特征扩展、数据清洗和结果美化；
6. 为兼容错误行为而保留静默截断、补零或隐式降级。

---

## 3. 重构前主链基线（2026-08-24）

### 3.1 代码规模与职责分布

2026-08-24 代码核实：

| 文件 | 行数 | 与多步策略相关的职责 |
|---|---:|---|
| `models/ModelForecasting.py` | 1623 | 九种推理方法、递归状态、分块、融合、回填、特征重建、输出整形 |
| `models/ModelTraining.py` | 1293 | 单/多输出训练、horizon 长表、融合子模型、训练容器选择 |
| `models/ModelTesting.py` | 1020 | 滑窗构造、模型训练与预测调用、窗口结果收集 |
| `features/FeatureEngineering.py` | 1668 | 九方法的 lag/目标列分支、horizon 外生展开、分组顺序特征 |
| `main.py` | 1049 | horizon、组合校验、训练/测试/预测编排、融合权重学习 |
| `features/FeatureScalering.py` | 667 | 按方法名解析目标列；推理缺列补训练期默认值 |
| `data_provider/data_loader.py` | 751 | 单/多变量方法集合、历史列裁剪、未来时间模板 |

### 3.2 六层重复解释

| 层 | 当前做法 | 后果 |
|---|---|---|
| 配置 | `config_sections.py` 保存方法集合，`main.py` 再做局部组合校验 | 合法方法与合法组合不是一个契约 |
| 数据 | `data_loader.py` 维护单变量方法集合 | 新增方法时容易漏掉历史列裁剪规则 |
| 目标 | `FeatureEngineering.py` 用长 `if/elif` 生成 `shift_0` 或 `shift_1..H` | 目标步语义与训练容器隐式耦合 |
| 训练 | `Trainer` 用目标列数量和多个方法判断决定容器 | 形状被当成策略语义，错误目标也可能被“合理训练” |
| 推理 | `Forecaster` 内九个方法 + 最终分发长链 | 状态、特征、模型和执行算法无法独立测试 |
| 缩放 | `FeatureScalering.py` 再按方法名推导目标列 | 与特征层目标定义可能漂移 |

### 3.3 当前配置使用面

对 `config/` 下 863 个带 `base_config` 的模型 YAML 解析结果：

| 方法 | 配置数 |
|---|---:|
| USMDP | 144 |
| USMD | 368 |
| USMR | 172 |
| USMDR | 108 |
| MSMD | 17 |
| MSMR | 17 |
| MSMDR | 17 |
| USBR | 18 |
| MSBR | 2 |

增强配置分布：

| 能力 | 非默认配置数 | 当前事实 |
|---|---:|---|
| `direct_strategy=horizon_feature` | 58 | 只应作用于 USMD/MSMD |
| `block_size>0` | 12 | 均为 DirRec，值为 96 |
| `endogenous_backfill_strategy=auxiliary` | 4 | 只用于 MSMR/MSMDR |
| `blend_weight_strategy=ridge_stacking` | 20 | 覆盖全部 USBR/MSBR 配置 |
| `align_direct_features_to_target=true` | 72 | 同一字段承担两种时间对齐语义 |
| `enable_global_training=true` | 0 | 代码仅有局部接线，无现役配置 |

这些数字说明重构不能只保护四种常用单变量方法，也不能把增强能力视为无人使用的实验代码。

---

## 4. 九种策略的真实语义

### 4.1 统一符号

- $H$：外层预测步数，即 `predict_steps`；
- $B$：DirRec 有效块长；
- $t$：训练特征行对应的时间；
- `shift_0`：标签为当前行目标 $y_t$；
- `shift_h`：标签为未来第 $h$ 步目标 $y_{t+h}$；
- 仅目标输入：内生 lag 只来自目标列；
- 多内生输入：内生 lag 来自目标列及其他内生列。

`shift_0` 不等于“预测零步”。在当前行已经由 lag 特征表达过去状态时，`shift_0` 是递归模型的一步训练标签；预测阶段把未来时间行作为“当前行”，因此仍产生下一待预测时点的值。

### 4.2 总表

| 短码 | 输入范围 | 推进族 | 训练标签 | 模型输出 | 推理调用方式 | 主要状态 |
|---|---|---|---|---|---|---|
| USMDP | 仅目标 | 逐点 | `shift_0` | 标量/每行一值 | 对 H 个未来特征行批量预测 | 无递归状态 |
| USMD | 仅目标 | 直接 | `shift_1..H` | H 维 | 原点特征调用一次 | 无递归状态 |
| USMR | 仅目标 | 递归 | `shift_0` | 标量 | 每步调用一次，共 H 次 | 目标 lag 状态 |
| USMDR | 仅目标 | 分块直接递归 | `shift_1..B` | B 维 | 每块调用一次，共 `ceil(H/B)` 次 | 目标 lag 状态 |
| MSMD | 多内生 | 直接 | `shift_1..H` | H 维 | 原点特征调用一次 | 无递归状态 |
| MSMR | 多内生 | 递归 | `shift_0` | 标量 | 每步调用一次，共 H 次 | 全部内生 lag 状态 |
| MSMDR | 多内生 | 分块直接递归 | `shift_1..B` | B 维 | 每块调用一次 | 全部内生 lag 状态 |
| USBR | 仅目标 | 直接+递归融合 | `shift_1..H` + `shift_0` | H 维 + 标量 | 两个子执行器后加权 | 两套子模型、融合权重 |
| MSBR | 多内生 | 直接+递归融合 | `shift_1..H` + `shift_0` | H 维 + 标量 | 两个子执行器后加权 | 两套子模型、全内生状态、融合权重 |

### 4.3 USMDP：逐点直接

**当前实现**：

- 默认不生成目标 lag，以未来日期、气象、自定义外生等逐行预测；
- 开启目标时点对齐后，历史与未来时间轴拼接，允许 `min(lags) >= H` 的 safe-lag；
- 模型一次接收 H 行，每行输出一个值；
- 若 `predictor_features` 为空，当前代码退回原始未来帧。

**合理语义**：逐点方法的核心不是“直接多输出”，而是“每个未来时点都有一行独立特征，标签步为 0”。safe-lag 是该方法的特征计划，不应由推理函数临时解释。

**缺陷**：

1. `align_direct_features_to_target` 同时承担 safe-lag 开关和其他 Direct 方法的单步时间对齐，字段语义过载；
2. 空特征退回原始未来帧会让训练 schema 对齐器静默补列，掩盖特征构造失败；
3. 是否允许 lag 由方法名和布尔字段共同决定，没有结构化 `FeaturePlan`。

### 4.4 USMD / MSMD：直接多输出

**当前实现**：

- 构造 `shift_1..H`；
- 默认每个输出各训练一个基估计器；
- 可选 horizon 长表，把 N×H 目标展开为 N×H 行并加入 horizon 特征；
- 可选按目标步取未来外生特征；
- 推理端已具备 Direct 输出长度精确校验。

**合理语义**：USMD 与 MSMD 共享同一 Direct 执行器，差别仅是 `input_scope`。训练布局是 `MULTI_OUTPUT` 或 `HORIZON_LONG`，不应再由 `Trainer` 自行猜测。

**缺陷**：

1. USMD/MSMD 的公共逻辑仍以两个方法保存；
2. horizon 长表、目标步外生和普通多输出分别在特征层与训练层判断，缺少统一兼容矩阵；
3. 方法支持能力与配置开关混在多个布尔函数中；
4. 训练列 schema 和推理列 schema 发生漂移时可被默认值补齐，而不是在策略边界失败。

### 4.5 USMR / MSMR：一步递归

**当前实现**：

- 训练使用 `shift_0`；
- USMR 每步重新构造特征并把目标预测写回历史；
- MSMR 使用缓存的静态外生与 lag deque；
- MSMR 对其他内生列按“辅助轨迹 → 最后已知值 → 0.0”回填；
- 未来行不足时循环提前停止并返回短数组。

**合理语义**：两者应共享一个递归执行器和显式 `RecursiveState`。多变量差异只体现在内生状态的提供者集合，不应复制推进循环。

**缺陷**：

1. 未来行不足返回短结果，错误直到主链末端才暴露；
2. 标量转换函数可从多维输出取第一个元素，模型形状错误可能被吞掉；
3. 缺少其他内生历史时补 `0.0`，制造无业务依据的数据；
4. 注释称 MSMR“递归预测所有内生变量”，实际只预测目标，其他列采用给定轨迹、辅助模型或持久性；
5. USMR 与 MSMR 使用不同特征构建路径，行为等价性没有系统测试。

### 4.6 USMDR / MSMDR：分块直接递归

**当前实现**：

- 推理块长由显式 `block_size`，否则取 `min(lags)`，无 lag 时取 1；
- 特征层只有显式 `block_size>0` 时才把训练目标宽度缩为块长，否则仍构造 H 个目标；
- 每块只调用一次模型，取 `min(block_size, remain, len(pred_vec))` 个输出；
- 逐点把块预测写回历史，再预测下一块；
- MSMDR 对其他内生列使用已知未来、辅助轨迹或持久性。

当前 125 个 DirRec 配置中：

- 12 个显式 `block_size=96`，训练与推理按 96 对齐；
- 其余使用默认解析；其中 28 个配置满足 `min(lags) < H`，训练构造 H 个输出但推理每块只消费前 `min(lags)` 个；
- 该问题在独立多输出训练下主要造成无效训练、模型体积膨胀和计划不自描述，但在其他多输出容器下还会引入更强耦合风险。

**合理语义**：`B` 必须在配置解析阶段唯一确定，并同时写入目标计划、训练计划、模型元数据和推理计划：

$$
B = \min\left(H,\ \begin{cases}
\text{block\_size}, & \text{block\_size}>0 \\
\min(\text{lags}), & \text{lags 非空} \\
1, & \text{其他}
\end{cases}\right)
$$

显式 `block_size>H` 不应静默截断，应在配置阶段报错。

**缺陷**：

1. 默认块长只在推理层解析，训练与推理计划分裂；
2. `take` 允许模型短输出，若 `len(pred_vec)==0`，`while` 不前进，存在死循环风险；
3. 末块可截取，但完整块必须严格输出 B 维；当前未区分“合法尾块截取”和“非法模型短输出”；
4. USMDR 文档称“只训练一个模型”，实际默认容器按每个输出训练基估计器；
5. 两个 DirRec 实现的循环、块校验和状态更新大量重复。

### 4.7 USBR / MSBR：直接递归融合

**当前实现**：

- 一个训练表同时包含 `shift_1..H` 和 `shift_0`；
- `Trainer` 按列名拆成 Direct 与 Recursive 两组并返回 dict bundle；
- 推理时临时把 `self.model` 改成 Direct 子模型，执行后重置历史，再改成 Recursive 子模型；
- 两路长度按 `min(len(direct), len(recursive))` 静默截短；
- 拟合权重从 `test_results_dir/blend_weights.csv` 读取，文件缺失或无效时退回固定权重。

**合理语义**：Blend 不是第六种底层推进算法，而是 `DirectExecutor + RecursiveExecutor + WeightResolver` 的组合。训练产物必须携带两个子模型和最终权重，推理只消费产物，不读取外部结果文件。

**缺陷**：

1. `self.model` 临时改写和历史重置依赖调用顺序，异常发生时可能留下污染状态；
2. 两路长度不一致被静默截断；
3. 权重不随模型保存，同一模型在不同目录状态下可能产生不同结果；
4. 显式要求拟合权重却缺文件时静默改成固定权重，实际执行策略与配置不一致；
5. 固定权重缺少“恰好两个、有限、非负、总和大于 0”的统一校验；
6. USBR/MSBR 的差异仍通过字符串前缀判断，而不是输入范围规格。

---

## 5. 横切能力现状

### 5.1 horizon 长表

该能力是 Direct 的训练布局，不是独立预测方法。当前只应适用于 USMD/MSMD。新架构把它表示为：

```text
TrainingLayout.HORIZON_LONG
```

由 `TrainingPlan` 生成长表所需的 horizon 列、展开顺序和目标步映射；推理端仍由 Direct 执行器一次接收 H 行。

### 5.2 目标步外生对齐

当前有两条路径：

1. 横向 `feature_h1..feature_hH`，每个输出基估计器只消费对应步；
2. 长表中把第 h 步外生折叠回基础列。

两条路径共享同一个“未来第 h 步可见外生”语义，应由 `FeaturePlan.exogenous_timing` 描述，不应分别靠 Trainer 和 FeatureEngineering 的布尔判断维护。

### 5.3 多变量未来内生回填

新架构定义统一提供者链：

```text
KnownFutureProvider
  → ConfiguredBackfillProvider
      → PersistenceProvider 或 AuxiliaryProvider
```

规则：

- 未来帧显式给值时优先使用；
- 配置 persistence 时必须有最后已知有限值；
- 配置 auxiliary 时，辅助模型训练失败或轨迹缺步直接失败；
- 不允许最后回退到 `0.0`；
- 每个提供者返回值与来源标签，供运行日志审计。

### 5.4 特征缓存

MSMR 已固定使用 lag 状态缓存，MSMDR 可选 `enable_feature_cache`。缓存是执行优化，不应改变特征值。新架构要求：

- 同一 `FeaturePlan` 同时提供 reference builder 与 cached builder；
- 对相同上下文逐步比较两者生成的列、顺序和值；
- 一致后才能启用缓存；
- 执行器不直接拼字符串构造 lag 列。

### 5.5 全局面板

Global panel 工程链路现已闭合：

- 历史和未来均以 `(series_id, time)` 为复合主键，重复键直接失败；
- lag、shift、rolling、expanding、diff、pct_change 和 time-since 全部按 series 分组；
- 滑窗边界、递归状态和未来 H 行按 series 独立构造；
- `global_incomplete_series_policy` 显式选择 `raise` 或整组 `drop`；
- 未知 series 当前只允许 `raise`，不静默编码；
- 最终输出保留 `series_id`，每条序列严格 H 行。

`tests/test_global_panel_contract.py` 覆盖复合键、不同长度、逐 series 窗口、五推进族、九方法真实 Forecaster 及完整 YAML→训练→pkl→预测链路。当前仍无现役 YAML 启用该能力，因此“工程支持”不等于已有业务精度结论；组合限制见 §12。

---

## 6. 缺陷清单与优先级

| ID | 缺陷 | 影响 | 优先级 | 目标处置 |
|---|---|---|---|---|
| M-001 | 方法语义在六层重复声明 | 新方法/新能力易漏改、漂移 | P0 | `ResolvedStrategy` 单一事实源 |
| M-002 | DirRec 默认块长未进入训练目标计划 | 无效训练、产物不自描述 | P0 | 统一 `effective_block_size` |
| M-003 | DirRec 接受短/空块输出 | 短结果或死循环 | P0 | 完整块严格 B 维、进度断言 |
| M-004 | Recursive 未来不足时提前返回 | 延迟报错、诊断差 | P0 | 上下文建立时要求未来恰好 H 行 |
| M-005 | 标量/向量转换静默取首项或展平 | 模型 shape 错误被吞掉 | P0 | family-specific shape validator |
| M-006 | Blend 两路按最短长度截断 | 组合结果不完整 | P0 | 两路均严格 H 维 |
| M-007 | Blend 推理改写共享模型与历史 | 异常路径污染状态 | P0 | 子上下文隔离、无共享句柄改写 |
| M-008 | 拟合权重依赖结果目录且可静默退回 | 不可复现、配置与执行不一致 | P0 | 权重嵌入模型产物，缺失即失败 |
| M-009 | 辅助回填失败后静默持久性降级 | 显式策略未真正执行 | P1 | 严格 provider 契约 |
| M-010 | 多变量历史缺列时补 0.0 | 伪造内生状态 | P0 | 缺列/无有限尾值直接失败 |
| M-011 | 推理 schema 缺列按默认值补齐 | 掩盖训练/推理漂移 | P0 | 缺列失败，NaN 与缺列分开处理 |
| M-012 | USMDP 空 predictor 回退原始帧 | 特征构造错误被掩盖 | P0 | 空 predictor 直接失败 |
| M-013 | 增强字段在不适用方法上可静默 no-op | 配置误写难发现 | P1 | 非默认无效字段一律拒绝 |
| M-014 | 全局面板只有局部特征支持 | 容易误宣称可用 | P1 | 端到端补齐后再开放 |
| M-015 | 注释与实现不一致 | 误导维护者与后续设计 | P1 | 以规格表生成/校验文档描述 |
| M-016 | 九方法缺少统一行为测试矩阵 | 重构无法证明边界 | P0 | 表驱动契约测试 + 端到端测试 |

---

## 7. 根因分析

### 7.1 方法名承担了过多职责

`pred_method` 同时编码：

- 输入变量范围；
- 标签布局；
- 训练输出宽度；
- 推理推进方式；
- 是否需要递归状态；
- 是否允许某些增强能力。

各模块只能重新解析字符串来获取自己需要的部分，最终形成多份不一致的隐式规格。

### 7.2 训练目标表被当成策略协议

当前 Trainer 常通过“目标列有一列还是多列”“是否有 `_shift_0`”来判断训练方式。该做法把数据形状当成协议：一旦特征层生成了错误列，Trainer 不一定报错，反而可能训练出结构合法但语义错误的模型。

### 7.3 推理上下文是大型可变对象

`Forecaster` 同时持有模型、历史、未来、特征工程器、缩放器、递归缓存、辅助轨迹、融合分量和输出收集器。方法之间通过改写这些字段复用功能，导致执行顺序成为隐式依赖。

### 7.4 兼容策略偏向“继续跑”而不是“语义真实”

当前多处容错选择补零、截短、取首元素或降级。它们减少了即时报错，却让配置声明、实际执行和最终模型产物不一致。多步预测中这类错误会沿 horizon 扩散，必须优先可诊断性而不是无条件继续。

---

## 8. 目标架构

### 8.1 包结构

```text
models/
  multistep/
    __init__.py
    spec.py                 # 枚举、StrategySpec、九方法 catalog
    resolve.py              # 旧配置 → ResolvedStrategy，集中校验
    plans.py                # TargetPlan / FeaturePlan / TrainingPlan / RuntimePlan
    contracts.py            # history/future/schema/model/output 契约
    artifacts.py            # 自描述模型产物与旧产物适配
    state.py                # RecursiveState / panel state
    backfill.py             # EndogenousFutureProvider 实现
    weights.py              # 固定/拟合融合权重对象
    executors/
      pointwise.py
      direct.py
      recursive.py
      dirrec.py
      blend.py
```

不建立继承层次很深的 `BaseStrategy`。五个执行器共享的是计划对象和小型纯函数，不是通过基类继承大型 Forecaster 状态。

### 8.2 九个外部 ID，五个内部执行器

```text
USMDP ───────────────> PointwiseExecutor
USMD  ─┐
MSMD  ─┴─────────────> DirectExecutor
USMR  ─┐
MSMR  ─┴─────────────> RecursiveExecutor
USMDR ─┐
MSMDR ─┴─────────────> DirRecExecutor
USBR  ─┐
MSBR  ─┴─────────────> BlendExecutor
```

单变量/多变量差异由 `InputScope` 和 `FeaturePlan` 表达，推进循环不复制。

### 8.3 解析流程

```text
legacy args + horizon
        │
        ▼
resolve_strategy(...)
        │
        ├── StrategySpec       稳定方法身份与推进族
        ├── TargetPlan         标签步与目标列顺序
        ├── FeaturePlan        lag、时间对齐、输入范围、未来值来源
        ├── TrainingPlan       单输出/多输出/长表/组合训练
        └── RuntimePlan        调用粒度、块长、状态与输出 shape
```

各下游模块只接收自己需要的 plan，不再读取 `args.pred_method`。

---

## 9. 核心规格与契约

### 9.1 StrategySpec

```python
@dataclass(frozen=True)
class StrategySpec:
    method: str
    code: str
    input_scope: InputScope
    rollout: RolloutFamily
```

只保留稳定语义，不放 `supports_xxx` 布尔集合。能力是否适用由 `rollout`、`input_scope` 和已解析选项共同推导，避免注册表演化成另一组散乱开关。

### 9.2 ResolvedStrategy

```python
@dataclass(frozen=True)
class ResolvedStrategy:
    spec: StrategySpec
    horizon: int
    target_plan: TargetPlan
    feature_plan: FeaturePlan
    training_plan: TrainingPlan
    runtime_plan: RuntimePlan
```

关键不变量：

```text
target_plan.output_width == training_plan.output_width
training_plan.output_width == runtime_plan.model_output_width
runtime_plan.forecast_length == horizon
```

Blend 例外不是破坏不变量，而是它的 `TrainingPlan` 和 `RuntimePlan` 都显式包含两个子计划。

### 9.3 TargetPlan

```python
@dataclass(frozen=True)
class TargetPlan:
    label_steps: tuple[int, ...]
    column_names: tuple[str, ...]
    layout: TargetLayout
```

| 推进族 | `label_steps` |
|---|---|
| Pointwise | `(0,)` |
| Direct | `(1, 2, ..., H)` |
| Recursive | `(0,)` |
| DirRec | `(1, 2, ..., B)` |
| Blend | Direct 子计划 `(1..H)` + Recursive 子计划 `(0,)` |

`FeatureEngineering` 根据计划调用通用 lag/shift 原语，不再包含九方法分支。

### 9.4 FeaturePlan

至少包含：

```python
@dataclass(frozen=True)
class FeaturePlan:
    input_scope: InputScope
    lag_columns: tuple[str, ...]
    lag_policy: LagPolicy
    row_alignment: RowAlignment
    exogenous_timing: ExogenousTiming
    backfill_policy: BackfillPolicy | None
    panel_key: str | None
```

其中：

- USMDP safe-lag 解析成 `LagPolicy.SAFE_TARGET_ROW`；
- 普通 Direct 解析成 `RowAlignment.FORECAST_ORIGIN`；
- 目标步外生解析成 `ExogenousTiming.BY_HORIZON`；
- 多变量递归方法解析出明确 backfill provider。

### 9.5 TrainingPlan

```python
class TrainingLayout(Enum):
    SINGLE_OUTPUT = "single_output"
    MULTI_OUTPUT = "multi_output"
    HORIZON_LONG = "horizon_long"
    COMPOSITE = "composite"
```

Trainer 改为按 `TrainingPlan.layout` 分发。目标列数量仅用于验证计划执行结果，不再反向决定策略。

### 9.6 ForecastContext

创建执行器前统一验证：

1. `horizon > 0`；
2. 单序列未来帧恰好 H 行；
3. 时间戳唯一、严格递增、频率一致；
4. 历史含目标列和计划要求的全部内生列；
5. lag 为正且历史支持计划要求的上下文；
6. 模型产物的方法 ID、输入 scope、目标步和输出宽度与计划一致；
7. 训练特征 schema 的列在推理构造后全部存在；
8. 面板模式下每个 series 都满足独立的 1–7。

### 9.7 输出 shape 契约

| 执行器 | 模型单次输出要求 | 最终输出 |
|---|---|---|
| Pointwise | H 行各一值，允许 `(H,)` 或 `(H,1)` | `(H,)` |
| Direct | 严格 H 值 | `(H,)` |
| Recursive | 每次严格一个值 | `(H,)` |
| DirRec | 每个完整块严格 B 值；尾块仍要求模型输出 B 值，再取前 `remain` | `(H,)` |
| Blend | 两个子执行器均严格 H 值 | `(H,)` |

禁止通用 `flatten()` 后“看起来长度够”就接受；shape 由推进族专属 validator 解释。

### 9.8 特征 schema 契约

将“列缺失”和“列值缺失”分开：

- 训练 schema 中的列在推理帧不存在：**RAISE**；
- 推理帧有额外列：默认丢弃并记录，或由严格模式拒绝；
- 列存在但值为 NaN：按训练期已声明的 impute 策略处理；
- 目标 lag、horizon 索引、series ID 等策略关键列禁止默认值填补；
- schema 随模型产物保存，不从底层估计器属性猜测。

---

## 10. 五类执行器设计

### 10.1 PointwiseExecutor

流程：

```text
FeaturePlan 构造 H 行未来特征
  → strict schema check
  → 单次 batch predict
  → pointwise shape check
  → H 维轨迹
```

特殊要求：

- safe-lag 仅由 FeaturePlan 生成；
- predictor 列为空直接失败；
- 不允许读取或回填未来目标；
- 训练与推理使用同一行对齐定义。

### 10.2 DirectExecutor

流程：

```text
原点上下文
  → 构造一次输入（1 行或 H 行，取决于 TrainingLayout）
  → predict
  → exact-H validation
```

`MULTI_OUTPUT` 与 `HORIZON_LONG` 共享执行器，仅输入批形状不同。目标步外生由 FeaturePlan 提供，不在执行器内判断列名后缀。

### 10.3 RecursiveExecutor

```python
state = RecursiveState.from_context(context)
for step in range(H):
    x_step = feature_builder.build_step(step, state)
    y_step = require_scalar(model.predict(x_step))
    state.append_target(y_step)
    state.append_other_endogenous(backfill.values_at(step))
    values[step] = y_step
return values
```

要求：

- `RecursiveState` 是本次调用私有对象；
- 输入历史帧不可原地修改；
- 每步输出必须是一个有限值；
- backfill 在循环开始前完成可用性校验，不能运行到中途才发现轨迹短缺；
- USMR 与 MSMR 使用同一循环。

### 10.4 DirRecExecutor

```python
state = RecursiveState.from_context(context)
for block_start in range(0, H, B):
    pred = require_block(model.predict(build_block_input(state)), width=B)
    take = min(B, H - block_start)
    state.append_block(pred[:take], backfill.slice(block_start, take))
    values[block_start:block_start + take] = pred[:take]
return values
```

要求：

- `B` 只能来自 `ResolvedStrategy`；
- 循环索引用 `range`，不使用依赖输出长度前进的 `while`；
- 完整模型输出宽度固定为 B；
- 尾块只截最终轨迹，不降低模型输出契约；
- USMDR 与 MSMDR 共享循环。

### 10.5 BlendExecutor

```text
DirectExecutor(child_context_direct)
RecursiveExecutor(child_context_recursive)
        │                    │
        └──── exact-H ───────┘
                  │
          WeightResolver
                  │
             H 维结果
```

要求：

- 两个 child context 从同一不可变输入构造；
- 不修改顶层模型句柄；
- 权重对象包含来源、样本数、两个归一化权重；
- 固定权重必须有限、非负、长度为 2、总和大于 0；
- 拟合权重在测试/训练编排阶段产生并封装进模型产物；
- 请求拟合权重但没有合格权重时失败，不退回固定值。

---

## 11. 配置解析与兼容策略

### 11.1 外部配置保持不变

以下字段继续接受：

- 九个现有 `pred_method` 全名；
- `direct_strategy`；
- `block_size`；
- `align_direct_features_to_target`；
- `use_horizon_exogenous_for_direct`；
- `endogenous_backfill_strategy`；
- `blend_weight_strategy` / `blend_weights`；
- `enable_feature_cache`；
- `enable_global_training` / `series_id_feature`。

加载后立即规范化为 `ResolvedStrategy`，下游不再直接消费这些原始字段。

### 11.2 非默认 no-op 一律拒绝

| 字段 | 合法范围 |
|---|---|
| `direct_strategy=horizon_feature` | USMD / MSMD |
| `block_size>0` | USMDR / MSMDR |
| `endogenous_backfill_strategy!=persistence` | MSMR / MSMDR |
| `blend_weight_strategy!=fixed` | USBR / MSBR |
| `blend_weights` 非默认 | USBR / MSBR |
| `align_direct_features_to_target=true` | USMDP safe-lag，或现有允许的单步 Direct 对齐 |
| `use_horizon_exogenous_for_direct=true` | 目标计划含未来步的 Direct/DirRec 路径 |
| `enable_feature_cache=true` | 已有 reference/cache 一致性测试的递归路径 |

不适用方法上保留默认值是兼容配置；写入非默认值则视为配置错误。

### 11.3 旧模型产物

增加 `LegacyArtifactAdapter`：

1. 识别现有裸模型和 dict bundle；
2. 从旧模型可用属性推导实际输出宽度；
3. 按旧方法 ID 构造只读兼容元数据；
4. 旧模型推理仍执行严格最终 H 长度检查；
5. 新训练产物一律保存显式 `ResolvedStrategy` 摘要和 schema；
6. 不原地改写旧模型文件。

DirRec 旧模型可能保存 H 个输出而运行时块长为 B。兼容适配器允许其作为 `legacy_output_width=H` 读取并显式取前 B；新训练模型必须只保存 B 个输出。这样既能读取旧模型，又不会延续新训练的计划分裂。

### 11.4 结果失效规则

以下变更会改变新训练结果，相应旧结果必须标记失效并重跑：

- 默认 DirRec 训练目标由 H 收敛为 B；
- 显式 auxiliary 不再静默降级；
- 拟合融合权重改为模型产物内权重；
- schema 缺列从补值改为失败后修正了真实特征链；
- 全局面板从单 series 补值升级为逐 series 推理。

仅做内部职责迁移且契约测试证明等价的阶段，不要求重跑历史实验。

---

## 12. 全局面板完整设计

### 12.1 数据契约

```text
历史主键：(series_id, time)，全局唯一
未来主键：(series_id, time)，每个 series 恰好 H 行
目标：每个 series 一列目标，可共享模型
输出：保留 series_id、time、预测值
```

禁止用“历史最后一个 series_id”代表全部未来。

### 12.2 训练

- 滑窗按 forecast origin 生成，但每个窗口内按 series 分组取训练/测试切片；
- lag、shift、rolling 等顺序算子始终 group-aware；
- 不完整 series 可按显式策略整组拒绝或排除，不能与其他 series 拼接补足；
- series ID 作为类别特征时，训练 schema 保存类别集合与未知 series 策略。

### 12.3 推理

- 为每个 series 创建独立 `ForecastContext` 与 `RecursiveState`；
- 可共享只读模型和静态 schema；
- 不共享 lag deque、回填轨迹或历史帧；
- 输出按输入 series 顺序和时间排序；
- 单个 series 失败时默认整批失败，避免部分结果被误当完整结果。

### 12.4 开放条件

以下开放条件已由 `tests/test_global_panel_contract.py` 全部覆盖：

1. 两序列相同时间戳不串 lag/shift；
2. 两序列不同长度时按策略处理；
3. 九种方法中声明支持的每个推进族至少一个面板端到端测试；
4. 递归与 DirRec 状态互不污染；
5. 输出每个 series 恰好 H 行；
6. 滑窗边界按 series 独立成立。

---

## 13. 已确定的设计决策

| ID | 结论 | 拒绝的方案 | 理由 |
|---|---|---|---|
| D-001 | 建立 `models/multistep/` 独立包 | 单个 `forecast_strategies.py` 薄注册表 | 薄注册表不能收敛目标、训练、状态与产物协议 |
| D-002 | 九方法分解为 InputScope × RolloutFamily | 九个独立 Strategy 类 | 九类会继续复制 US/MS 推进循环 |
| D-003 | plan 驱动 Trainer | Trainer 继续按目标列数量猜测 | shape 不是策略协议 |
| D-004 | DirRec 统一解析 B，训练和推理都使用 B | 保留训练 H、推理取前 B | 无效训练且产物不自描述 |
| D-005 | 严格缺列/短输出/回填失败 | WARNING 后补零、截短或降级 | 配置与实际执行必须一致 |
| D-006 | 融合权重随模型产物保存 | 推理时读取结果目录 | 模型必须可独立复现 |
| D-007 | 递归状态按调用隔离 | 继续改写 Forecaster 共享历史 | 组合执行和异常恢复不可控 |
| D-008 | 全局面板端到端补齐 | 只保留分组特征能力 | 局部接线不能对外宣称完整功能 |
| D-009 | 旧配置原样接受，新产物采用显式元数据 | 一次性改 863 个 YAML | 无必要扩大迁移面 |
| D-010 | 缺陷修复允许数值变化并要求重跑 | 以逐元素相等为最高目标 | 不能用兼容名义保留错误语义 |

---

## 14. 实施路线

### Phase 0：行为锁定与红灯测试（P0）

目标：先证明当前缺陷和当前正确语义，再改生产代码。

任务：

1. 建立九方法 `TargetPlan` 期望表测试；
2. 为 DirRec 默认 B/H 分裂写失败测试；
3. 为 DirRec 空输出死循环风险写可终止失败测试；
4. 为 Recursive 未来短帧写就地失败测试；
5. 为 Blend 两路不等长、缺拟合权重写失败测试；
6. 为缺内生历史补 0、schema 缺列补值、USMDP 空 predictor 写失败测试；
7. 锁定已正确行为：Direct exact-H、safe-lag 因果性、目标 `shift_1..H`、分组顺序特征边界；
8. 建立九方法最小端到端测试夹具。

退出条件：每个待修缺陷都有可复现红灯；每个保留语义都有绿灯基线。

### Phase 1：策略规格与集中校验（P0）

目标：先建立单一事实源，不迁移执行算法。

任务：

1. 新建 `spec.py`、`plans.py`、`resolve.py`、`contracts.py`；
2. 注册九个稳定方法 ID 与短码；
3. 实现配置到 `ResolvedStrategy` 的规范化；
4. 把 `main.py` 组合校验和 `utils/multistep_contract.py` 规则迁入解析器；
5. `DataLoader`、`FeatureScalering` 先改为消费解析结果；
6. 非默认 no-op 配置 fail-fast；
7. 全量 YAML 加载回归。

退出条件：目标计划、训练计划、运行时计划三者不变量通过；活动主链不再各自维护方法集合。

### Phase 2：目标与训练计划收敛（P0）

目标：消除 FeatureEngineering 和 Trainer 对方法名的解释。

任务：

1. `FeatureEngineering` 改为通用 lag 原语 + `TargetPlan`；
2. Direct/DirRec/Blend 的目标列完全由计划生成；
3. Trainer 只按 `TrainingLayout` 选择路径；
4. 修复默认 DirRec 只训练 B 个输出；
5. 新训练模型写入方法 ID、目标步、输出宽度和 feature schema；
6. 保留旧产物适配器。

退出条件：FeatureEngineering 和 Trainer 中无九方法字符串分支；28 个受影响配置可明确列出重跑范围。

### Phase 3：五执行器迁移（P0/P1）

按最小风险顺序逐族迁移，每次只迁一个 vertical slice：

1. Direct（已有 exact-H 契约，最简单）；
2. Pointwise；
3. Recursive；
4. DirRec；
5. Blend。

每个 slice 的步骤：

1. 先对当前实现建立相同输入的行为夹具；
2. 新执行器接入一对 US/MS 方法；
3. 修复该族已列缺陷；
4. 跑定向测试与全量测试；
5. 删除该族在旧 Forecaster 中的分支；
6. 再迁下一族。

退出条件：`Forecaster` 只负责构造上下文、调用 catalog executor 和向上层返回结果；不再包含九方法推进实现。

### Phase 4：复合能力自描述化（P1）

任务：

1. `EndogenousFutureProvider` 三种实现；
2. auxiliary 训练完整性与轨迹长度校验；
3. `BlendArtifact` 保存子模型与权重；
4. 拟合权重在编排阶段产生并写入产物；
5. 移除推理对 `test_results_dir` 的依赖；
6. reference/cached 特征构建逐值一致性测试；
7. 统一组件日志：块长、调用次数、回填来源和权重来源。

退出条件：同一模型文件在无测试目录环境下可复现相同点预测；显式能力不能静默降级。

### Phase 5：全局面板端到端完成（P1/P2）

任务：

1. DataLoader 支持 `(series_id,time)` 主键；
2. Tester 生成逐 series 滑窗；
3. ForecastContext 按 series 拆分；
4. 五类执行器支持独立 series state；
5. 输出保留 series ID；
6. 增加不等长、缺 series、未知 series 和部分失败测试；
7. 至少一个双序列完整配置冒烟。

退出条件：满足 §12.4 全部开放条件后，才能把全局训练从实验状态改为支持状态。

### Phase 6：旧链清理与文档同步（P2）

任务：

1. 删除旧分发长链、重复方法集合和兼容期临时代理；
2. 更新方法 docstring，修正 USMDR/MSMR 失真描述；
3. 更新 AGENTS.md 的多步策略约定；
4. 更新配置文档的适用矩阵；
5. 对受语义修复影响的结果执行失效清单；
6. 保留旧模型只读适配，不继续生成旧格式。

退出条件：全仓活动代码方法名声明单点化，文档、配置校验和运行日志与新规格一致。

### 14.7 依赖图

```text
Phase 0
   │
   ▼
Phase 1 ──> Phase 2 ──> Phase 3 ──> Phase 4
                                  │
                                  ▼
                               Phase 5
                                  │
                                  ▼
                               Phase 6
```

---

## 15. 测试矩阵

### 15.1 表驱动单元测试

对九个方法至少断言：

- `InputScope`；
- `RolloutFamily`；
- `label_steps`；
- `TrainingLayout`；
- 模型单次输出宽度；
- 预期调用次数；
- 是否需要递归状态；
- 非默认增强字段合法性。

### 15.2 边界值

| 维度 | 用例 |
|---|---|
| horizon | 1、2、B−1、B、B+1、非 B 整数倍 |
| block | 默认、1、H、0、负数、显式大于 H |
| lag | 空、含 0、负数、`min(lag)=H`、`min(lag)<H` |
| future | 0 行、H−1 行、H 行、H+1 行、重复时间戳、乱序 |
| model output | 标量、正确向量、短向量、长向量、二维歧义、NaN、inf |
| endogenous | 已知未来、持久性、辅助完整、辅助缺步、历史缺列 |
| blend | 合法固定权重、负权重、零和、缺拟合权重、子轨迹不等长 |
| panel | 1/2 个 series、不同长度、交错行、未知 series、单 series 失败 |

### 15.3 性质测试

1. Direct 调用模型一次，最终长度恒为 H；
2. Recursive 调用模型 H 次，每次只追加一个目标值；
3. DirRec 调用次数为 `ceil(H/B)`，状态总追加 H 次；
4. Blend 输出是两个 H 维子轨迹的逐点凸组合；
5. 任何执行器都不修改输入历史帧；
6. cache 开关不改变特征矩阵和预测值；
7. panel 中任一 series 的输入变化不影响其他 series 的 lag 状态。

### 15.4 集成验证

实施各阶段必须运行：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m unittest discover -s tests -p "test_*.py"
```

并执行：

1. 863 个模型 YAML 加载和策略解析；
2. `scripts/check_model_configs.py` 全量 dry-run；
3. 九方法最小点预测配置冒烟；
4. 至少一个显式 block、一个 horizon 长表、一个 auxiliary、一个拟合 blend 权重配置；
5. Phase 5 增加一个双序列全局训练冒烟。

### 15.5 数值兼容分级

| 变更类型 | 要求 |
|---|---|
| 纯职责迁移 | 固定模型与固定输入逐元素相等 |
| 严格 shape/schema 校验 | 合法输入逐元素相等；非法输入由静默行为改为明确失败 |
| DirRec 目标宽度修复 | 首输出模型语义一致；训练成本、产物宽度和重训结果重新验收 |
| 融合权重入产物 | 权重相同时逐元素相等；缺权重时由降级改为失败 |
| auxiliary 严格化 | 辅助完整时逐元素相等；失败时不再持久性降级 |
| 全局面板完成 | 旧单序列模式等价；新增多序列模式按独立契约验收 |

---

## 16. 风险与控制

### 16.1 旧模型序列化兼容

风险：现有模型可能是裸估计器、dict bundle 或不同输出宽度的多输出容器。

控制：先实现只读 `LegacyArtifactAdapter`，收集真实产物形态测试；新产物格式只新增，不原地迁移历史文件。

### 16.2 DirRec 数值变化

风险：统一 B 后，28 个默认块长小于 H 的现役配置会减少无效输出训练；底层训练随机性和多输出容器差异可能改变结果。

控制：列出受影响配置，先单配置冒烟，再按项目 A/B 两路完整窗口口径重新判定；旧结果标记失效，不与新结果混用。

### 16.3 严格 schema 暴露存量隐患

风险：现有配置可能依赖缺列补值才能运行。

控制：Phase 0 先记录真实缺列；区分“列缺失”和“值缺失”；修正特征构造根因，不给策略关键列开白名单绕过。

### 16.4 五执行器迁移面较大

风险：一次性移动全部逻辑难以定位回归。

控制：按 Direct → Pointwise → Recursive → DirRec → Blend 垂直切片；一个推进族通过全量测试后才迁下一个。

### 16.5 面板能力扩大主链复杂度

风险：多序列滑窗和输出会影响 DataLoader、Tester、Forecaster 与结果保存。

控制：Phase 5 与单序列核心重构解耦；先完成稳定的单序列五执行器，再增加 panel context；没有端到端证据前不宣传支持。

### 16.6 过度抽象

风险：为九种方法建立复杂继承树，代码量反而增加。

控制：只保留四个计划值对象、五个执行器和少量 Protocol；共享纯函数优先于基类；每个抽象必须消除至少两个现有字符串分支或重复循环。

---

## 17. 实施进度记录

> 状态机：`pending → in_progress → completed / blocked / cancelled`。
> `completed` 必须附命令、结果与关键产物；无验证证据不得标记完成。
> 实施中发现偏差追加到 §17.4，不回改已批准的设计结论；如需改变结论，新增决策记录。

`last_updated: 2026-08-25`

### 17.1 已完成的前置语义修复

| 项目 | 状态 | 当前证据 |
|---|---|---|
| MSMD/MSMDR 直接目标从 `shift_1` 开始 | completed | `FeatureEngineering.py` 显式 `start_step=1`；`tests/test_multistep_target_alignment.py` |
| Direct 输出不足不再边缘填充 | completed | `_require_direct_prediction_length`；目标对齐测试覆盖 |
| USMDP 多步 safe-lag | completed | `utils/multistep_contract.py`；`tests/test_usmdp_safe_lag.py` |
| 面板顺序特征不跨 series | completed | `tests/test_global_panel_feature_boundaries.py`；`tests/test_global_panel_contract.py` |
| 全局面板端到端 | completed（工程能力） | 双序列 YAML→DataLoader→Ridge→bundle/pkl→A/B 预测冒烟；现役启用配置仍为 0 |

### 17.2 重构阶段总览

| Phase | 内容 | 状态 | 完成时间 | 证据 |
|---|---|---|---|---|
| 0 | 行为锁定与红灯测试 | completed | 2026-08-25 | 九方法 plan/shape/状态与缺陷回归测试 |
| 1 | 策略规格与集中校验 | completed | 2026-08-25 | `models/multistep/spec.py`、`plans.py`、`resolve.py`；863 YAML 全通过 |
| 2 | 目标与训练计划收敛 | completed | 2026-08-25 | FeatureEngineer/Trainer/Scaler 消费 resolved plan；DirRec 统一 B |
| 3 | 五执行器迁移 | completed | 2026-08-25 | `models/multistep/executors/`；Forecaster 删除九个旧推进代理 |
| 4 | 复合能力自描述化 | completed | 2026-08-25 | typed artifacts、backfill/weights、统一执行摘要、legacy adapter |
| 5 | 全局面板端到端完成 | completed | 2026-08-25 | 双序列完整配置冒烟与九方法逐 series Forecaster 测试 |
| 6 | 旧链清理与文档同步 | completed | 2026-08-25 | 代码陈旧引用审计；AGENTS、README、config/models 与关联概率设计文档同步 |

### 17.3 当前代码审计证据

| 核实项 | 结果 |
|---|---|
| 模型 YAML | 863 个模型 schema，加载/策略解析错误 0；另有 13 个独立聚合/检测 YAML |
| 九方法使用面 | 144 / 368 / 172 / 108 / 17 / 17 / 17 / 18 / 2 |
| DirRec 配置 | 125 个；12 个显式 block；28 个默认路径存在训练 H、运行块长小于 H |
| horizon 长表 | 58 个非默认配置 |
| auxiliary 回填 | 4 个非默认配置 |
| 拟合 blend 权重 | 20 个非默认配置 |
| 全局训练 | 0 个启用配置；双序列完整配置与九方法工程冒烟已通过 |
| 推理分发 | `Forecaster` 只构造上下文并按 rollout family 调用五类 executor |
| 训练判断 | `Trainer` 消费 `TrainingPlan`，不再维护九方法目录 |
| 特征目标构造 | `FeatureEngineering` 消费 `TargetPlan` / `FeaturePlan` |
| schema/产物 | 严格 feature schema；新产物 typed，旧裸模型/dict 由单一 adapter 只读兼容 |
| 全量验证 | 371 tests OK；增强路径 42 tests OK；compileall 与 `git diff --check` exit 0 |
| 结果失效 | 94 个唯一配置，当前命中 43 组 stale 测试结果；详见失效清单，不在本轮全量重跑 |

### 17.4 已发现的设计偏差

| # | 日期 | 偏差 | 处置 |
|---|---|---|---|
| 1 | 2026-08-24 | 旧方案把问题缩小为“分发注册表”，未覆盖目标、训练、状态、产物和配置的重复语义 | v3 改为完整 `ResolvedStrategy` + 五执行器架构 |
| 2 | 2026-08-24 | 旧方案要求所有阶段预测值零变化，与“消除缺陷”冲突 | v3 改为兼容分级；缺陷修复允许可解释数值变化并要求重跑 |
| 3 | 2026-08-24 | DirRec 初步审计把所有 `block_size=0` 视为错配，范围过大 | 通过 863 配置实际加载复算；准确影响面为 28 个默认配置 |
| 4 | 2026-08-24 | global panel 曾被描述为实体边界已关闭，实际只有特征层分组 | v3 明确为端到端未完成能力，纳入 Phase 5 |
| 5 | 2026-08-24 | Blend 被当成独立大方法迁移，未解决共享状态与目录权重依赖 | v3 改为两个子执行器 + 权重对象的组合 |
| 6 | 2026-08-25 | 全量 glob 得到 876 个 YAML，曾被误写成模型配置数 | 按顶层 schema 复核：863 个模型 YAML + 13 个聚合/检测 YAML；验收只统计前者 |
| 7 | 2026-08-25 | 九方法 panel 冒烟暴露 MSMR/MSBR 递归缓存丢失 `series_id` | 缓存显式携带 panel key；A/B 不串号回归通过 |
| 8 | 2026-08-25 | 结果重跑与代码重构收口不是同一完成条件 | 生成逐配置失效清单；遵守“不全量重跑”，旧结果标 stale 但不删除/覆盖 |

---

## 18. 最终验收清单

- [x] 九种外部方法 ID 和短码保持兼容；
- [x] 九方法目标计划表全部通过；
- [x] FeatureEngineering、Trainer、Forecaster、DataLoader、Scaler 不再各自解释方法字符串；
- [x] DirRec 训练与推理共享唯一 B；
- [x] 所有执行器严格返回 H 个有限值；
- [x] 无短输出截断、空输出死循环、缺内生补零、缺权重降级；
- [x] Blend 与 Recursive 不修改共享模型和历史；
- [x] 新模型产物包含方法、目标步、输出宽度、feature schema 和组合元数据；
- [x] 旧模型通过只读适配器加载；
- [x] 863 个模型 YAML 加载与 dry-run 通过；
- [x] 全量 unittest 通过（371 tests）；
- [x] 九方法最小端到端冒烟通过；
- [x] 受语义修复影响的结果已生成逐配置失效清单；
- [ ] 失效清单中的现役配置完成重跑与测试/模型/预测三类产物验收（本轮明确不全量重跑）；
- [x] 全局面板满足 §12.4 并标记为工程支持（现役配置 0，尚无精度结论）；
- [x] `AGENTS.md`、配置、models、根 README 与关联概率设计文档同步新契约。

代码重构与文档同步已收口；只有存量结果重跑项完成后，才能把“存量结果迁移”也标记为 Completed。
