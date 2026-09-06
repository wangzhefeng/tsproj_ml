# model_pipeline

`model_pipeline/` 负责单模型生命周期、监督设计与批量运行编排；根 `run.py` / `batch_run.py` 调用本包。融合仍由独立 `model_ensemble/` 负责，通过入口注入 runner 与执行服务复用单模型链。

- `runner.py`：`CanonicalBaseModelRunner` 与 `run_canonical_config()`；提供训练、预测、历史准备和只读证据能力，run 入口设置线程限制后委托生命周期。
- `lifecycle.py`：`run_lifecycle()` 管理完成状态与异常传播，`execute_lifecycle()` 组织回测、CQR、final fit、预测和持久化；包含 `CanonicalRuntimeResult` 与产物元数据助手。结果类型仍由 runner 公开导出。
- `supervised_design.py`：`SupervisedDesignBuilder`、information set、训练/预测设计、监督标签窗口、`probe_training_design()`、`minimum_history_rows()`；批编译不支持的设计保留 single 路径，不隐式替换 provider。
- `fold_fit.py`：fold/final 特征选择、变换与训练服务；回测和 final fit 共用配置训练窗口。
- `run_state.py`：running/completed/failed 状态写入和 completed 校验；状态不是跨目录事务，外部直接加载 pickle 不自动消费状态。
- `batch_runtime.py`：`run_canonical_batch()`、`verify_batch_results()` 与报告，负责跨配置 preflight、调度、恢复和验收。
- `batch_artifacts.py`：产物路径、摘要、时间网格、维度与 bundle 身份验收；只看文件存在不能判 completed。

## 编排边界

fixed-step/calendar-month 循环分别位于 `model_testing/fixed_step.py` 与 `model_testing/calendar_month.py`；通过显式回测协议消费 runner 能力，逐折评分都委托 `model_testing/scoring.py`。目标变换识别与并行拟合所需的标签历史由 runner 的 `backtest_target_histories()` 提供，测试包不导入具体变换实现。预测/部署和 bundle 构造持久化位于 `model_forecasting/`，final bundle 构造由 `build_strategy_model_bundle()` 统一处理。

`CanonicalBaseModelRunner.execution_evidence(artifact, target_transform)` 提供公开只读证据能力，供回测与融合成员通过协议调用；不为收集证据再次拟合或预测。CQR 收集在 final fit 前完成，部署只应用已保存校准状态。

资源规划、checkpoint、性能档和内存缓存属于 `model_performance/`；目标/特征变换属于 `feature_engineering/transforms/`。raw-design 缓存同时绑定源内容、生成器、依赖清单与编译链实现，修改设计或 compiler 后不得误用旧设计；缓存身份不等于配置语义 fingerprint，不自动删除正式结果。
