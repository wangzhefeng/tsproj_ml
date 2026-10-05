# model_predicting

`model_predicting/` 是预测、部署与预测产物层；单模型生命周期和批调度属于 `pipeline/`。子包按消费方划分（2026-09-28）：`contracts/` 注入协议、`loops/` 执行、`artifacts/` 落盘与证据；无子包级转发门面，消费方走完整点路径。

- `loops/predictor.py`：point 与 marginal quantile 推理，recursive quantile 使用 median path；包含张量 crossing 修复与 `assemble_marginal_quantile_distribution()`（median path 逐分位组装唯一实现，训练期与部署期共用）。`crossing.report_raw`（默认 true）在组装段产出修复前后 crossing 诊断（`build_crossing_report()`，消费 `model_evaluation.metrics.crossing_metrics`），写入分布 metadata 的 `crossing_report`；回测侧由 `model_testing/loops/scoring.py` 合入逐窗 execution_evidence 落盘。
- `loops/deployment.py`：`predict_strategy_bundle()`，消费已加载 bundle 和显式部署输入，不重新训练；分位数组装经 `loops/predictor.py` 共享函数，仅单 level 预测回调与 crossing 配置来源（bundle spec）不同。
- `contracts/protocols.py`：`FeatureProvider` 特征注入协议唯一来源；训练期由 `pipeline/fold_fit` 注入，部署期由部署调用方/ensemble 注入。
- `artifacts/persistence.py`：`build_strategy_model_bundle()` 统一 schema-2 final bundle 构造，`persist_model_bundle()` 持久化。
- `artifacts/evidence_collect.py`：运行时只读采集——现有模型状态的参数快照、环境版本与 JSON 安全转换，不执行拟合/预测（2026-09-28 自 `evidence.py` 改名，与 `evidence_assembly.py` 的「组装」对偶）。
- `artifacts/evidence_assembly.py`：生命周期产物证据组装（visibility proof / holdout proof 摘要 / source lineage / feature lineage 四个纯函数）；自 `pipeline/lifecycle.py` 迁入（2026-09-27 证据域聚集），实现逐字保真，由 lifecycle 调用。
- `artifacts/results.py`：预测 canonical long result 写盘、绘图与 `CanonicalResultReader`；回测产物写盘属于 `model_testing/artifacts/reporting.py`。

`CanonicalResultReader.read_prediction(path)` / `read_backtest(path)` 读取完整 long 表并解析时间，拒绝非 canonical 文件；不提供隐式格式转换或筛选。原未生效的 `target` / `series_id` 参数已移除，调用方需在返回的 DataFrame 上显式筛选。未使用的 `_plot_timeseries` 再导出已退出。无 schema 校验的 `read_scores` 已删除（2026-09-28，全仓零消费）。long 转换唯一实现在 `model_testing/artifacts/tensor_frames.py`，本模块只 import 自用件、不再转发再导出（2026-09-28 门面收口；model_ensemble 经门禁白名单直引 tensor_frames）。

`dependency_versions()` 的运行时依赖包名清单与实现指纹共用 `utils/runtime_env.RUNTIME_DEPENDENCY_PACKAGES` 唯一来源（2026-09-28 单源化）；缺失哨兵串两侧各自维护（证据 `"not_installed"` / 指纹 `"absent"`，后者进指纹身份不扰动）。

包目录取名 `model_predicting`（2026-09-28 自 `model_forecasting` 改名，行为零变化；历史引用见 git）。

稳定类型全部来自 `forecasting_core/`。本包不得反向 import `pipeline` 或 `model_ensemble`；目标/特征变换属于 `feature_engineering/transforms/`。

证据采集由上层 runner 的公开只读能力调用。单模型逐折证据位于 holdout metadata，final 证据位于 `runtime.run_evidence` 与 `result_metadata.json`。适用语义进入内部 fingerprint；缓存命中保留历史证据，缺证据显式 unavailable，不以当前环境伪造历史证据、不为补证据重跑。

外部直接加载 pickle 不会自动消费完成状态。

目标分解对象使用版本化状态；旧分解对象布局/已移除组件路径明确拒绝加载，不进行兼容转换。启用分解的配置带 `component_fit_v2` 语义身份；存量结果保留，需用户显式重新训练后才能用于新部署。

## 编排与部署边界

批调度与产物完成验收见 [`pipeline.md`](pipeline.md)；资源规划、性能档、checkpoint、fold 变换缓存与有界内存缓存见 [`model_performance.md`](model_performance.md)。

部署只使用已保存的模型、变换和校准状态；不会重新选择训练窗口或读取训练期缓存来重建模型。
启用 point `absolute_residual` 时，在目标逆变换后返回 `PointIntervalForecast`；未启用仍返回原点张量。按保存的序列/目标/horizon 轴匹配，拒绝使用校准原点以后的残差去预测历史；样本不足组以 `pi_available=false` 明示，不伪造 quantile。预测/回测图支持独立 `predict_pi*` 区间带。

## 执行链与结果

`pipeline.runner.run_canonical_config()` 组织完整生命周期。fixed-step 和 calendar-month 各自使用配置训练窗口；自然月折按真实月份长度构造。底层 trainer/forecaster 不拥有这个生命周期。

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
