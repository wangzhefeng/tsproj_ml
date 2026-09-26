# model_training

`model_training/` 只负责模型训练行为，不拥有结果 IO 或 bundle 类型。包根 `__init__.py` 只做包结构说明，不 re-export 符号（消费方全路径导入，与 `models` 门面约定一致）。

- `trainer.py`：`CanonicalTrainer`，训练 strategy artifact。模型组准备载荷以 `PreparedModelGroup` NamedTuple 按字段名消费（2026-09-26 收口，替代裸五元组下标）；independent 与 generic 适配器两条 fit 分支收敛为「构造+checkpoint+组装」统一形状，仅调度入口不同，均走同一 `fit_model_group` 调度（串行或线程池）。`_target_block` 对完整 time-major 网格走 reshape 快路径，非完整块保留逐坐标列索引。
- `weights/`：训练样本权重子包（2026-09-26 由顶层 `sample_weight.py` 收口并激活）。三段链路：**配置合同**在 `forecasting_core/specs/validation.py`（`validation.training.sample_weight`，method/halflife_days/anchor/normalization）→ **算法本体**在本包 `temporal.py`（指数衰减 `2^(-(样本年龄-最老样本年龄)/半衰期)`，锚点 `cutoff`（默认，历史截止）或 `latest_origin`（最新监督原点），归一化 `mean`（默认）/`sum`/`none`；拒绝未来监督原点、未知参数和权重下溢）→ **接线**在 `model_pipeline/fold_fit.py`（`_fit_point/_fit_quantile` 按训练 origins + 标签末端 cutoff 计算权重透传 trainer）。`resolve.py` 为前两段之间的翻译层：读 spec、对 catalog `sample_weight=False` 的模型（如 seasonal_template）前置 RAISE、产出与 origins 对齐的权重数组。`fit_final()` 经 `final_bundle_inputs` 走同一接线函数，跳过则 RAISE 防静默丢失。原唯一生产入口（红太阳年度脚本）已随 2026-09-25 场景收敛退役；其他入口不能仅凭schema接受该字段就声称已经使用权重。
- `quantile.py`：`CanonicalMarginalQuantileTrainer`/`CanonicalMarginalQuantileArtifact`，按 quantile grid 逐 level 训练编排。level 并行默认上限收口为模块常量 `DEFAULT_LEVEL_WORKERS_CAP`（约束规则——level 并行时嵌套 output workers 压 1——不变）。
- `strategies/`：七种标准多步 executor。
- `estimators/`：能力探测和 independent/chain/native adapter。

本包依赖 `forecasting_core` 与 `models`，不依赖 `probabilistic` 或 `model_forecasting`。Quantile estimator factory 由上层注入。

**预测侧消费事实**（2026-09-26 核实，避免误判为分层违规）：`model_forecasting` 六处 import 本包（`predictor.py`/`deployment.py` 消费 strategies executor 与 quantile artifact，`persistence.py` 消费 `CanonicalStrategyArtifact`/`CanonicalTrainer`），`model_performance/resource_planner.py` 消费 `supports_native_multi_quantile` 与 `target_plan_for_config`——「加载 bundle 做预测」路径依赖训练包。该边在分层 DAG 中合法（`model_forecasting → model_training` 在 ALLOWED_PACKAGES），语义纯度问题已登记 OPT-025（与 OPT-023 绑定同一触发条件），触发时上收 executor 与 artifact 合同，届时本条随之修订。

ETS/naive/theta 的 catalog 标记为 native_history，不进入本包监督回归 trainer/adapter；由 pipeline 对每折有界真实历史直接调用 fit_history（runner 按 `models/adapters/native_registry.py::NATIVE_HISTORY_MODELS` 注册表分发）。资源合同允许 native_history 工作负载零特征，记录原始历史行数与实际候选拟合数，不把 horizon 个输出误报为 horizon 个回归模型。

## 输入输出与策略组织

`CanonicalTrainer` 接收 `ForecastConfigSpec`、显式 `estimator_factory`、`EstimatorCapabilities`、固定 `feature_schema` 和可选 `FitCheckpoint`。`train()` 消费按策略调用组织的二维设计序列 `X_by_call` 与标签 `Y`，返回 `CanonicalStrategyArtifact`，而不是最终部署 bundle。

`strategies/base.py` 维护 target plan、坐标和模型组 artifact，七个策略模块复用这组合同。Direct 的 horizon-feature 属于 layout，不是新策略；Local/Global 属于 training scope。MO 分块合法性由 core spec 校验。

`estimators/capabilities.py` 从模型 catalog 与原生能力探测建立支持矩阵（ndarray 合同适配器与工厂已下沉 `models/adapters/canonical.py`，本模块保留能力注册表、`resolve_model_capabilities` 与依赖 checkpoint 的 `SharedMultiQuantilePool`）；`estimators/multi_target.py` 提供 independent、regressor-chain、native adapter。能力不足直接报错，不以自动降级掩盖不支持的目标维度或线程策略。行为探测（`probe_native=True`）按 `(model_type, params)` 正向缓存：探测自带合成两列设计、结论与运行时 feature 宽度无关，逐折重复解析不再重复 clone+fit+predict。共享池 `fit_position` 记录每 position 首次载荷的字节级 blake2b 摘要，跨 level 摘要不一致直接 RAISE——位置对齐不变量从串行约束注释升级为运行时防御。independent 拟合态回填收口在 `assemble_from_task_results` 类方法，调度器不再从类外改写 adapter 私有字段。

## 恢复与边界

checkpoint 通过 `forecasting_core.checkpoints.FitCheckpoint` 注入，训练层不导入运行时文件实现。特征准备、目标变换和 final bundle 落盘由上层负责；quantile 层注入不同分位点的 estimator factory，不在训练包复制目标函数参数规则。
