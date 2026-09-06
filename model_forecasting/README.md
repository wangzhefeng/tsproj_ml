# model_forecasting

`model_forecasting/` 是单模型生命周期编排与推理层。

- `design.py`：information set、特征设计、监督样本和 fixed-step 回测几何。
- `fit_service.py`：fold/final transform、训练与预测服务。
- calendar-month 回测生命周期已于 2026-09-06 R5 迁入 `model_testing/calendar_month.py`（逐折评分体共用 `model_testing/scoring.py`，方案 v3）。
- `persistence.py`：schema-2 bundle 构造与持久化。
- `runtime.py`：`CanonicalBaseModelRunner` 与顶层单模型生命周期编排。
- `forecaster.py`：point 与 marginal quantile 推理，recursive quantile 使用 median path。
- 目标/特征变换状态与严格逆变换、scaler 窗口选择已于 2026-09-06 R3 迁入 `feature_engineering/transforms/`（行为不变重构，方案 v3）。
- `evidence.py`：现有模型状态的只读参数快照及 JSON 安全转换，不执行拟合/预测。
- `lifecycle.py`：单模型 running/completed/failed 状态；批产物验收检查完成状态，不是目录事务。
- `results.py`：canonical long result 读写和绘图。

稳定类型全部来自 `forecasting_core/`。本包可以向下编排各流水线阶段，但不得 import `model_ensemble`。

`CanonicalBaseModelRunner.execution_evidence(artifact, target_transform)` 为公开只读能力，供 fixed-step、calendar-month 和通过 Protocol 边界调用的 Ensemble 成员复用。单模型逐折证据位于 holdout metadata，final 证据位于 `runtime.run_evidence` 与 `result_metadata.json`。适用语义进入内部 fingerprint；缓存命中保留历史证据，缺证据显式 unavailable，不以当前环境伪造历史证据、不为补证据重跑。自然月回测/CQR 收集在 final fit 之前完成，不再于 run 返回后改写 final 产物。

外部直接加载 pickle 不会自动消费完成状态。

## 批调度、资源与部署

资源规划、性能档、checkpoint、fold 变换缓存与有界内存缓存已于 2026-09-06 R2 迁入 `model_performance/`（行为不变重构，方案 v3）。

- `batch_runtime.py`：`run_canonical_batch()` 与 `verify_batch_results()`，负责跨配置 preflight、调度、恢复和验收；CLI 为根目录 `batch_run.py`。
- `batch_artifacts.py`：产物摘要、时间网格、维度与 bundle 身份验收；只看文件存在不能判 completed。
- raw-design 缓存同时绑定源内容、生成器、依赖清单与普通编译链实现内容；修改 compiler/registry/design/策略几何后不得命中旧原始设计。配置 fingerprint 与缓存身份分离，缓存变更不删除存量正式结果。
- `deployment.py`：`predict_strategy_bundle()`，消费已加载 bundle 和显式部署输入，不重新训练。

## 执行链与结果

`run_canonical_config()` 编排设计准备、滚动回测、可选 CQR 收集、final fit、正式预测与持久化。fixed-step 和 calendar-month 各自使用配置训练窗口；自然月折按真实月份长度构造。底层 trainer/forecaster 不拥有这个生命周期。

活动配置通常写入 `results/{pretrained_models,results_test,results_forecast}/<scenario>/<identity>/`，未声明目录时才采用 runtime 回退布局。`prediction.csv` 与 `cv_plot_df.csv` 分别使用 `(series_id,time,target)` 和追加 `window` 的 long 唯一键；quantile 评估另写概率评分文件。

测试临时产物不替代正式模型验收。
