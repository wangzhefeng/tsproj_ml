# model_ensemble

`model_ensemble/` 是引用式模型融合能力包。

- `loader/specs`：成员引用、顶层共享 problem/data/probabilistic/origin 合同。
- `oof/cache`：严格时间 OOF 与 `(member,source,path_role,file_sha256)` 内容寻址缓存。
- `methods/`：averaging、weighted、linear blending、stacking。
- `predictor.combine_members` 校验成员顺序后委托各 `combine_*`；weighted 与 linear blending 共用预测期加权运算，但学习权重的算法独立。旧 NNLS 函数只保留在 `tests/fixtures/legacy_nnls.py` 作独立黄金参照，生产包不再导出。
- `trainer/predictor/runtime`：融合器学习、成员 final fit、自包含 Ensemble bundle 和 canonical result。
- `deployment.py`：从已加载 bundle 预测；调用方显式提供按 bundle input schema 编译的部署特征及需要时的 feature provider，不读取成员 YAML 或 OOF cache。

Quantile linear blending 按 target 最小化 simplex 约束 pooled pinball；同一 target 的全部 quantile level 共享权重。产物同时保存 quantile grid、有效样本数、optimizer 状态/消息与 fallback 原因，并写入 `resolved_model.json` 供人工审计。

成员 OOF/final 证据通过 `contracts.member_execution_evidence` 调用 runner 的公开只读能力，不导入 L2 执行实现。自定义 runner 未提供该能力时显式记录 unavailable；已提供能力但执行失败直接报错。

`OOFPredictionArtifact.execution_evidence` 与 folds/预测身份分离，写入 `oof_metadata.json` 并在缓存命中时原样读回；旧缓存缺此字段返回空证据，不重跑、不回填当前环境。`resolved_config.json` 和 `result_metadata.json` 的 `run_evidence` 包含 `member_oof`、`member_final` 及 source hashes；成员证据不完整时汇总不能称 recorded。数值/部署验收状态见 [权威设计 §6.3](../docs/multistep_forecasting_redesign.md#63-可靠性收口实现与受限验证)。
