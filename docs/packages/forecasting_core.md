# forecasting_core

`forecasting_core/` 是不依赖其他项目包的稳定预测合同层，不执行训练、回测、预测或文件写入；bundle 只生成 schema payload。

## 内容与子包边界

- `specs/`：problem/data/feature/strategy/estimator/config 严格不可变规格。
- `tensors/`：axes/storage 管理轴与不可变存储；point/quantile/sample 分别表达 `(N,H,K)`、`(N,H,K,Q)`、`(N,S,H,K)`；layout 管理 time-major 展平与轴匹配。pickle 重建函数跟随具体类型。
- `probability/`：grid/spec/calibration/intervals/distribution 管理分位网格、概率语义、校准状态、区间与预测分布；`_validation.py` 仅供本子包共用标量检查。真正的校准算法仍在顶层 `probabilistic/`。
- YAML 与部署概率语义统一由 `probabilistic_spec_from_mapping()` 校验；不把默认值写回原始配置，合法存量配置 fingerprint 不变。计数只接受整数、开关只接受 bool；point 拒绝 quantile-only 字段；单分位不自动生成区间。旧 args 接口不保留。
- `bundle.py`：只拥有部署模型 `ForecastModelBundle`，不混放预测分布。
- `temporal/origin.py`：预测原点解析 `resolve_origin()` 与数据源能力 Protocol `SupportsLatestTargetTime`。
- `temporal/windows.py`：时间合同校验、预测网格及训练历史边界（含 next_day 完整日校验与 DST 防护）；`temporal/sampling.py`：监督原点抽样。回测与生产共用，不读取数据。
- `execution/design.py`：`IndexedDesign` 以不可变 bytes backing 隔离输入别名；compiler 先冻结共享列，再跨调用/切片零复制复用，估计器边界才展开。`retained_bytes` 按 backing storage 去重核算。
- `execution/strategy.py`：共用 `TargetCoordinate`、完整签名 `FeatureProvider`；执行器和模型组 artifact 仍在训练包。
- `specs/yaml.py`：从文本严格解析 YAML、拒绝重复键，不打开文件。
- `specs/training.py`：sample_weight 的唯一字段与数值合同，配置解析和训练算法共用；未消费训练选项直接拒绝。

旧一维 `ForecastDistribution` 与无消费者的 `calibration_runtime_kwargs` / `validate_probabilistic_args` 已退出；使用现役张量分布及概率 spec 解析接口。不提供旧类型 import/pickle 兼容。`PredictionIntervalForecast`、`QuantileGrid` 和明确 unsupported 的 joint-sample 边界保留。

## 依赖规则

本包不得 import 其他项目包；包间门禁见 `tests/test_package_layering.py`。包内方向为 probability→tensors，specs→probability/temporal，bundle→probability；tensors/execution/temporal 不反向依赖配置聚合器或 bundle，模块图无环。根入口与新增子包入口只说明职责，消费者直引所属模块，specs 保留既有规格公共导出。

## 资源与恢复合同

- `execution/resources.py`：`RuntimeWorkload`、`RuntimeResourceBudget`、`RuntimeExecutionPlan` 及序列化 payload；不探测资源或启动并行任务。
- `execution/checkpoints.py`：`FitCheckpoint` Protocol 与结构化错误；本地文件实现位于 `model_performance/checkpoints.py`。
- `ForecastModelBundle.schema_payload()` 提供审计元数据；JSON 写入统一为 `model_predicting.artifacts.persistence.write_bundle_schema_json()`，旧实例写盘方法已退出。
- bundle 仅接受 schema-2；反序列化重新校验。CQR 保存状态必须含有限 correction（applied 时）、校准原点、区间/覆盖率及样本计数；保存与部署再次校验。缺少这些事实的旧 CQR bundle 需显式重训，不伪造状态、不自动迁移或删除旧结果。

## 张量与配置边界

`N/H/K/Q` 分别表示序列、预测步、目标、分位点。point 与 quantile 使用明确轴和时间网格，展平固定 time-major；`(N,S,H,K)` 仅为 joint-sample 类型边界，`generate_joint_samples()` 不提供生成能力。

单模型配置由 `ForecastConfigSpec` 表达；引用式融合规格在 `model_ensemble/configuration/specs.py`。子包化改变类及 pickle 重建函数路径，不保留旧源码转发或 pickle 兼容层；受影响旧产物显式重训，不自动删除或重跑。新增字段须明确 canonical payload、校验和 fingerprint 语义，不能只增加序列化字段。
