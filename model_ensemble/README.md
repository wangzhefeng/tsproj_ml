# model_ensemble

`model_ensemble/` 是引用式模型融合能力包。

- `loader/specs`：成员引用、顶层共享 problem/data/probabilistic/origin 合同。
- `oof/cache`：严格时间 OOF 与 `(member,source,path_role,file_sha256)` 内容寻址缓存。
- `methods/`：averaging、weighted、linear blending、stacking。
- `trainer/predictor/runtime`：融合器学习、成员 final fit、自包含 Ensemble bundle 和 canonical result。
- `deployment.py`：从已加载 bundle 预测；调用方显式提供按 bundle input schema 编译的部署特征及需要时的 feature provider，不读取成员 YAML 或 OOF cache。

Quantile linear blending 按 target 最小化 simplex 约束 pooled pinball；同一 target 的全部 quantile level 共享权重。产物同时保存 quantile grid、有效样本数、optimizer 状态/消息与 fallback 原因，并写入 `resolved_model.json` 供人工审计。
