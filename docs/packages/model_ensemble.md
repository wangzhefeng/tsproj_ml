# model_ensemble

`model_ensemble/` 是引用式模型融合能力包。

- `configuration/{specs,loader,preflight}.py`：成员引用、共享 problem/data/probabilistic/origin、方法参数与成员能力预检；严格 YAML reader 与单模型共用 `forecasting_core.specs.yaml`，嵌套及 merge 后重复键同样拒绝。
- `training/oof.py` 与 `outputs/cache.py`：严格时间 OOF 与 `(member,source,path_role,file_sha256)` 内容寻址缓存。文件 SHA 实现复用 `data_loading.sources.provenance.file_sha256`；成员身份组合、OOF 缓存读写与校验仍由本包负责。
- generated weather 的 OOF 身份同时绑定 manifest/raw/normalized 和天气实现哈希；其他尚无传递资产合同的生成器继续 RAISE。OOF 保存 Global series 坐标，当前内部 schema 常量为 4；本次子包迁移沿用原值，不改变缓存身份、不删除存量目录。天气融合真实业务场景仍需单独验收。
- `methods/`：averaging、weighted、linear blending、stacking、adaptive_weighted；参数与支持面见 [配置合同](../config/ensemble.md)。
- `inference/predictor.py::combine_members` 校验成员顺序后委托各 `combine_*`；weighted 与 linear blending 共用预测期加权运算，但学习权重的算法独立。旧 NNLS 函数只保留在 `tests/fixtures/legacy_nnls.py` 作独立黄金参照，生产包不再导出。
- `training/{trainer,backtesting,diagnostics}.py`：融合器学习、成员 final fit、独立外层回测和 meta-train 诊断；`inference/forecast.py` 统一顶层 point_quantile/crossing；`outputs/{reporting,persistence}.py` 拥有报告、bundle 组装与完成清单。根 `runtime.py` 只编排阶段。
- `inference/deployment.py`：从已加载 bundle 预测；调用方显式提供按 bundle input schema 编译的部署特征及需要时的 feature provider，不读取成员 YAML 或 OOF cache。
- 运行身份绑定有序成员语义 fingerprint 和实际原点；未解析 YAML 的 document fingerprint 仅作配置溯源，不作为运行产物身份。OOF key 同时绑定实际成员原点；CSV 读回采用正确舍入，保证 float64 字节校验精确往返。
- 顶层部署概率规格经 `forecasting_core.probability.spec` 唯一解析器构造，不复制首成员的 point_quantile；weighted 的 quantile 误差按样本/步长/分位共同池化，各分位共享逐目标权重；成员 pinball 复用通用指标。

Quantile linear blending 按 target 最小化 simplex 约束 pooled pinball；同一 target 的全部 quantile level 共享权重。产物同时保存 quantile grid、有效样本数、optimizer 状态/消息与 fallback 原因，并写入 `resolved_model.json` 供人工审计。

成员 OOF/final 证据通过 `contracts.member_execution_evidence` 调用 runner 的公开只读能力，不导入 L2 执行实现。自定义 runner 未提供该能力时显式记录 unavailable；已提供能力但执行失败直接报错。

`OOFPredictionArtifact.execution_evidence` 与 folds/预测身份分离，写入 `oof_metadata.json` 并在缓存命中时原样读回；旧缓存缺此字段返回空证据，不重跑、不回填当前环境。`resolved_config.json` 和 `result_metadata.json` 的 `run_evidence` 包含 `member_oof`、`member_final` 及 source hashes；成员证据不完整时汇总不能称 recorded。

## 生命周期与依赖注入

显式 `validation.train_history_steps` 当前仅为单模型 backtest-only 合同；Ensemble 顶层及引用此字段的成员均拒绝，避免 OOF/final 绕过原始历史边界。

1. `run_ensemble_config_file()` 解析引用式 YAML 并校验成员共享合同，不接受 ensemble-of-ensemble。
2. `generate_oof_for_config()` 生成或读取成员 OOF；验证标签与训练标签用 `is_label_safe` 隔离（`ensemble.oof.gap_steps` 隔离合同），不能用成员 final fit 的训练内预测学习融合权重。
3. `training.backtesting.run_outer_backtest()` 在每个外层原点重新生成 label-safe 内层 OOF、学习融合器，再评分外层留出；标准 `cv_plot_df.csv`、`test_scores_df.csv`、概率评分和逐窗图只来自外层。`eval_mask` 只影响评分，不修改绘图真值。
4. `fit_ensemble()` 为最终原点学习融合参数；`training.diagnostics.evaluate_fused_oof()` 的 `fused_oof_scores` 是 meta-train 诊断，单列到 `meta_train_predictions.csv` / `meta_train_scores*.csv`，不得当作独立泛化成绩。
5. 成员按与单模型一致的配置窗口 final fit；融合层 CQR 仅收集外层留出残差，随 bundle 保存原点与校准状态。部署不读取成员 YAML 或 OOF cache，不重估动态权重，不用新真值更新状态。

`contracts.py` 提供 `BaseModelRunner`、`BaseModelRunnerFactory` 与 `EnsembleRuntimeServices`；入口注入执行服务和 calibration factory，融合包不直接构造 L2 runtime。成员可有不同 lag/策略，但 holdout 时间、series/target 顺序、quantile grid 必须一致；训练索引按成员自己的监督原点映射，OOF 数组按 fold-major / series-minor 展平。

OOF 缓存采用同父目录 staging + 一次 rename 发布，锁位于 `_locks/`；中断 staging 留作诊断、读者不扫描。发布后的缺文件/校验错误直接拒绝；重试只恢复未发布条目，不静默覆盖损坏缓存。

`pretrained_models/run_state.json` 记录 running → completed/failed；关键文件、身份、预测/回测网格通过读回检查后才标 completed，并保存 SHA256 清单。状态不是跨目录事务，不支持同身份并发写结果；直接加载 pickle 不自动检查状态。失败重跑可复用完整 OOF，仍会重做外层模型与 final fit。

身份使用解析后的成员语义与实际原点，加 `nested_evaluation_v1` 内部语义盐；原点/成员变化及独立评估产生新身份，旧结果保留。旧 point NNLS 后归一化和 Ridge stacking alpha=1 数值语义不变；新旧标准测试成绩因评估口径不同不能直接比较。

## 子包边界与迁移

根只保留 `runtime.py`、`contracts.py`、`artifacts.py` 和公开配置入口 `__init__.py`。methods 只依赖共享 artifact；inference 可调用 methods，不能反向调用 training/outputs；training 可调用 inference/methods，outputs 可调用 inference 做报告组装，两者不互相调度；全流程调度仅在根 runtime。

子包 `__init__.py` 不新增转发门面，消费方使用完整模块路径。旧平铺实现路径移除（evaluation → training.diagnostics），仓库外直接import调用方须更新；标准bundle内的artifact类路径保持不变，迁移本身不改变配置/OOF身份、预测值或结果路径。不承诺自行pickle配置spec/整个runtime返回对象的旧模块路径兼容。
