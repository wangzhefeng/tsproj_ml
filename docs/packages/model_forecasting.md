# model_forecasting

`model_forecasting/` 是预测、部署与预测产物层；单模型生命周期和批调度属于 `model_pipeline/`。

- `persistence.py`：`build_strategy_model_bundle()` 统一 schema-2 final bundle 构造，`persist_model_bundle()` 持久化。
- `predictor.py`：point 与 marginal quantile 推理，recursive quantile 使用 median path；包含张量 crossing 修复。
- `deployment.py`：`predict_strategy_bundle()`，消费已加载 bundle 和显式部署输入，不重新训练。
- `evidence.py`：现有模型状态的只读参数快照及 JSON 安全转换，不执行拟合/预测。
- `results.py`：预测 canonical long result 写盘、绘图与 `CanonicalResultReader`；回测产物写盘属于 `model_testing/reporting.py`。

`CanonicalResultReader.read_prediction(path)` / `read_backtest(path)` 读取完整 long 表并解析时间，拒绝非 canonical 文件；不提供隐式格式转换或筛选。原未生效的 `target` / `series_id` 参数已移除，调用方需在返回的 DataFrame 上显式筛选。未使用的 `_plot_timeseries` 再导出已退出；现役 long 转换再导出保留。

稳定类型全部来自 `forecasting_core/`。本包不得反向 import `model_pipeline` 或 `model_ensemble`；目标/特征变换属于 `feature_engineering/transforms/`。

证据采集由上层 runner 的公开只读能力调用。单模型逐折证据位于 holdout metadata，final 证据位于 `runtime.run_evidence` 与 `result_metadata.json`。适用语义进入内部 fingerprint；缓存命中保留历史证据，缺证据显式 unavailable，不以当前环境伪造历史证据、不为补证据重跑。

外部直接加载 pickle 不会自动消费完成状态。

目标分解对象使用版本化状态；旧分解对象布局/已移除组件路径明确拒绝加载，不进行兼容转换。启用分解的配置带 `component_fit_v2` 语义身份；存量结果保留，需用户显式重新训练后才能用于新部署。

## 编排与部署边界

批调度与产物完成验收见 [`model_pipeline.md`](model_pipeline.md)；资源规划、性能档、checkpoint、fold 变换缓存与有界内存缓存见 [`model_performance.md`](model_performance.md)。

部署只使用已保存的模型、变换和校准状态；不会重新选择训练窗口或读取训练期缓存来重建模型。

## 执行链与结果

`model_pipeline.runner.run_canonical_config()` 组织完整生命周期。fixed-step 和 calendar-month 各自使用配置训练窗口；自然月折按真实月份长度构造。底层 trainer/forecaster 不拥有这个生命周期。

活动配置通常写入 `results/{pretrained_models,results_test,results_forecast}/<scenario>/<identity>/`，未声明目录时才采用 runtime 回退布局。`prediction.csv` 与 `cv_plot_df.csv` 分别使用 `(series_id,time,target)` 和追加 `window` 的 long 唯一键；quantile 评估另写概率评分文件。

测试临时产物不替代正式模型验收。

### 结果身份与 Direct 方法

单模型 identity 为 `<method_label>-<model_type>-<training_scope>-k<target_count>-<fingerprint前12位>`。
`ForecastConfigSpec.result_method()` 是路径及结果元数据的方法描述唯一入口；标签来自实际配置，不来自 YAML 文件名。

| Direct 有效配置 | method_label |
|---|---|
| 未声明 direct 变换或 independent_models | direct |
| single_model_horizon，horizon_feature.enabled=false | direct-pointwise |
| single_model_horizon，horizon 启用（无论是否周期编码） | direct-pointwise-horizon |

共享模型的 horizon_feature.enabled 默认 true，cyclical 默认 false；禁用 horizon 时周期编码不生效。
Direct 仅分上述三类。cyclical 是第三类内部的特征变体，不进入可读前缀；有效值保留在 result_method.horizon_feature_cyclical 中，不同周期编码配置由 fingerprint 区分。
非 Direct 策略仍使用原策略名。方法描述额外记录 layout、有效 horizon 开关及 align_to_target，不改变 canonical payload/fingerprint。
不同场景的 pointwise 文件名可能表示不同特征，必须按配置解释。旧目录不会自动迁移、回退查找或触发重训；需先审核迁移清单，再单独授权迁移及路径引用修正。

当前活动配置的生命周期末次预测仍是历史留出评估，天气从 history 的预报列取值。真正未来部署上游构造设计时须显式 `forecast_designs(..., data_phase="future")`，且配置 future_path；默认不会自动借用 future 文件。部署层只消费传入设计，不负责判断文件阶段。
