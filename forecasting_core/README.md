# forecasting_core

`forecasting_core/` 是不依赖其他项目包的稳定预测合同层，不执行训练、回测或预测；bundle 另提供显式 schema JSON 写出方法。

## 内容

- `specs/`：problem/data/feature/strategy/estimator/config 严格不可变规格。
- `tensors.py`：point `(N,H,K)`、marginal quantile `(N,H,K,Q)`、sample 类型边界。
- `probabilistic_spec.py`：point/quantile、quantile grid、interval/calibration 配置合同。
- `artifacts.py`：`ForecastModelBundle`、`MarginalForecastDistribution`、`QuantileGrid`。

旧一维 `ForecastDistribution` 与无消费者的 `calibration_runtime_kwargs` / `validate_probabilistic_args` 已退出；使用现役张量分布及概率 spec 解析接口。不提供旧类型 import/pickle 兼容。`PredictionIntervalForecast`、`QuantileGrid` 和明确 unsupported 的 joint-sample 边界保留。

## 依赖规则

本包不得 import 任何其他项目包。`tests/test_package_layering.py` 对所有 Python 文件做 AST 门禁，包括函数内 import。

## 资源与恢复合同

- `runtime_resources.py`：`RuntimeWorkload`、`RuntimeResourceBudget`、`RuntimeExecutionPlan` 及其序列化 payload；本包不执行资源探测或启动并行任务，实际规划在运行时层。
- `checkpoints.py`：`FitCheckpoint` Protocol 与结构化 `FitCheckpointError`；训练侧只依赖该接口，本地文件实现位于 `model_forecasting/checkpoints.py`。
- `ForecastModelBundle.schema_payload()` 提供不含模型对象的审计元数据；`write_schema_json()` 可显式写出 JSON。模型 pickle 与生命周期持久化由上层服务负责，因此不能把本包描述为绝对无 IO。

## 张量与配置边界

`N/H/K/Q` 分别表示序列、预测步、目标、分位点。point 与 quantile 使用明确轴和时间网格，展平固定 time-major；`(N,S,H,K)` 仅为 joint-sample 类型边界，`generate_joint_samples()` 不提供生成能力。

单模型配置由 `ForecastConfigSpec` 表达；引用式融合规格在 `model_ensemble/specs.py`。core 不为旧配置、旧一维分布或旧模型导入路径建立兼容层。新增字段须同时明确 canonical payload、校验和 fingerprint 语义，不能只增加序列化字段。
