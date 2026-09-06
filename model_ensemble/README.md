# model_ensemble

`model_ensemble/` 是引用式模型融合能力包。

- `loader/specs`：成员引用、顶层共享 problem/data/probabilistic/origin 合同。
- `oof/cache`：严格时间 OOF 与 `(member,source,path_role,file_sha256)` 内容寻址缓存。文件 SHA 实现复用 `data_loading.sources.provenance.file_sha256`，本包保留原导出；成员身份组合、OOF 缓存读写与校验仍由本包负责。
- `methods/`：averaging、weighted、linear blending、stacking。
- `predictor.combine_members` 校验成员顺序后委托各 `combine_*`；weighted 与 linear blending 共用预测期加权运算，但学习权重的算法独立。旧 NNLS 函数只保留在 `tests/fixtures/legacy_nnls.py` 作独立黄金参照，生产包不再导出。
- `trainer/predictor/runtime`：融合器学习、成员 final fit、自包含 Ensemble bundle 和 canonical result。
- `deployment.py`：从已加载 bundle 预测；调用方显式提供按 bundle input schema 编译的部署特征及需要时的 feature provider，不读取成员 YAML 或 OOF cache。

Quantile linear blending 按 target 最小化 simplex 约束 pooled pinball；同一 target 的全部 quantile level 共享权重。产物同时保存 quantile grid、有效样本数、optimizer 状态/消息与 fallback 原因，并写入 `resolved_model.json` 供人工审计。

成员 OOF/final 证据通过 `contracts.member_execution_evidence` 调用 runner 的公开只读能力，不导入 L2 执行实现。自定义 runner 未提供该能力时显式记录 unavailable；已提供能力但执行失败直接报错。

`OOFPredictionArtifact.execution_evidence` 与 folds/预测身份分离，写入 `oof_metadata.json` 并在缓存命中时原样读回；旧缓存缺此字段返回空证据，不重跑、不回填当前环境。`resolved_config.json` 和 `result_metadata.json` 的 `run_evidence` 包含 `member_oof`、`member_final` 及 source hashes；成员证据不完整时汇总不能称 recorded。

## 生命周期与依赖注入

1. `run_ensemble_config_file()` 解析引用式 YAML 并校验成员共享合同，不接受 ensemble-of-ensemble。
2. `generate_oof_for_config()` 生成或读取成员 OOF；验证标签与训练标签用 `is_label_safe` 隔离（`ensemble.oof.gap_steps` 隔离合同），不能用成员 final fit 的训练内预测学习融合权重。
3. `fit_ensemble()` 学习融合参数，`evaluation.py::evaluate_fused_oof()` 给出融合 OOF 评分（挂在 `run_ensemble_config` 返回的 `fused_oof_scores` 与 audit）；该评分与 final 预测结果分开记录。
4. 成员按与单模型一致的显式窗口 final fit，持久化自包含 bundle。`predict_ensemble_bundle()` 使用已保存成员与融合器，不重新读取 OOF cache 或成员 YAML。

`contracts.py` 提供 `BaseModelRunner`、`BaseModelRunnerFactory`、`EnsembleRuntimeServices` 与 `FusionMethod`；入口注入执行服务，融合包不直接构造 L2 runtime 私有对象。成员顺序、目标顺序、时间轴和 quantile grid 必须一致。
