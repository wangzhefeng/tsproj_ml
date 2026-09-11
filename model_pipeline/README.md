# model_pipeline

联通严格原始历史通路：`runner.py` 在逐折 fit 边界接入原生 ETS（消费 `builder.target_history(origin)` 的完整时序，不使用监督标签拟合）及 seasonal residual（每个监督原点独立减基线、预测原位恢复）；`supervised_design.py` 负责基线张量、原始单位递归 provider 与预热长度。残差限 Local/raw-history/point/无目标变换，仍沿用 backtest-only 与禁止 final bundle 的合同。块天气与同槽/近期状态在公共 compiler 中实现，不在联通脚本旁路实现。

`model_pipeline/` 负责单模型生命周期、监督设计与批量运行编排；根 `run.py` / `batch_run.py` 调用本包。融合仍由独立 `model_ensemble/` 负责，通过入口注入 runner 与执行服务复用单模型链。

- `runner.py`：`CanonicalBaseModelRunner` 与 `run_canonical_config()`；提供训练、预测、历史准备和只读证据能力，run 入口设置线程限制后委托生命周期。
- `lifecycle.py`：`run_lifecycle()` 管理完成状态与异常传播，`execute_lifecycle()` 组织回测、CQR、final fit、预测和持久化；包含 `CanonicalRuntimeResult` 与产物元数据助手。结果类型仍由 runner 公开导出。
- `supervised_design.py`：`SupervisedDesignBuilder`、information set、训练/预测设计、监督标签窗口、`probe_training_design()`、`minimum_history_rows()`；批编译不支持的设计保留 single 路径，不隐式替换 provider。
- `fold_fit.py`：fold/final 特征选择、变换与训练服务；回测和 final fit 共用配置训练窗口。
- `run_state.py`：running/completed/failed 状态写入和 completed 校验；状态不是跨目录事务，外部直接加载 pickle 不自动消费状态。
- `batch_runtime.py`：`run_canonical_batch()`、`verify_batch_results()` 与报告，负责跨配置 preflight、调度、恢复和验收。
- `batch_artifacts.py`：产物路径、摘要、时间网格、维度与 bundle 身份验收；只看文件存在不能判 completed。

## 编排边界

天气阶段通过 `forecast_designs(..., data_phase="historical"|"future")` 传至不可变请求。默认 historical，供滑窗测试、OOF 及当前生命周期末次历史留出预测使用；训练始终 historical/实测列，测试预测为 historical/预报列。真正未来调用方必须显式传 future，递归 provider 捕获同阶段信息集。训练设计缓存仅哈希映射天气源的 history，不依赖未来文件；其他 source 原合同不变。

显式 `validation.train_history_steps` 时，每个 runner 固定 `history_start = origin - (W-1) * freq`，只编译有界数据。调度依据 registry 的时间覆盖事实，不复用跨折 expanding；`for_backtest_window()` 构造独立 runner，拟合、预测和递归 provider 共用下界。主 runner 最后窗口的编译仅用于资源规划，不供各折训练。此模式暂限 backtest-only，拒绝 final fit/bundle，不能作为部署完成。

`SupervisedDesignBuilder` 经 `SourceRegistry.target_history_coverage()` 获取目标源序列/时间覆盖，再在本包决定 `series_order`、unknown/incomplete policy、训练窗口和监督张量。数据读取、验证及公共 identity 选择由数据层提供；runner、batch runtime、lifecycle 通过 registry 的公开 `base_dir`/`generators` 取得上下文，不穿透私有状态。递归预测目标 provider 与 oracle 标签策略仍属于本包，不迁入通用数据层。

fixed-step/calendar-month 循环分别位于 `model_testing/fixed_step.py` 与 `model_testing/calendar_month.py`；通过显式回测协议消费 runner 能力，逐折评分都委托 `model_testing/scoring.py`。目标变换识别与并行拟合所需的标签历史由 runner 的 `backtest_target_histories()` 提供，测试包不导入具体变换实现。预测/部署和 bundle 构造持久化位于 `model_forecasting/`，final bundle 构造由 `build_strategy_model_bundle()` 统一处理。

`CanonicalBaseModelRunner.execution_evidence(artifact, target_transform)` 提供公开只读证据能力，供回测与融合成员通过协议调用；不为收集证据再次拟合或预测。CQR 收集在 final fit 前完成，部署只应用已保存校准状态。

天气源的 `weather_evidence` 随 source lineage 持久化到生命周期产物与 bundle；去重身份包含证据，不能把不同原点/快照折叠为同一来源路径。非天气源保持既有字段格式。

资源规划、checkpoint、性能档和内存缓存属于 `model_performance/`；目标/特征变换属于 `feature_engineering/transforms/`。raw-design 缓存同时绑定源内容、生成器、依赖清单与编译链实现，修改设计或 compiler 后不得误用旧设计；缓存身份不等于配置语义 fingerprint，不自动删除正式结果。
