# feature_engineering

`feature_engineering/` 是 canonical 唯一特征编译层。

- `compiler.py`：lag、known-future、static、datetime、advanced transformation、visibility proof 与 lineage。
- `selection.py`：每个训练窗独立拟合的监督特征选择。
- `transform_specs.py`：feature/target transformation 严格配置归一化。

特征只能使用预测原点可见信息。实验性 FFT/wavelet 等脚本位于 `docs/feature_engineering/`，未接入生产；现役中国节假日 source 由 `data_loading/holiday_generator.py` 提供，两者不要混淆。

## 缓存与训练态

- `cache.py`：raw-design 内容寻址缓存、进程锁、元数据与载荷校验。缓存身份绑定 source 内容、生成器、依赖及编译链实现，不等同于配置语义 fingerprint。
- `scaling.py`：`CanonicalFeatureScaler`，在训练设计上拟合并将同一状态用于预测设计；不负责目标逆变换。
- `CanonicalFeatureSelector` 只在本训练窗拟合，保留的特征索引随训练 artifact 使用；不在全量数据上先选列再做回测。

## 输入输出与对齐

`FeatureCompiler.compile()` 消费物化信息集，返回 `CompiledFeatures`，包含设计值、`FeatureSchema`、lineage 和 `VisibilityProof`。`batch_eligibility()` 检查批编译能力，`compile_batch()` 提供受支持设计的批量编译；批量路径与普通路径必须保持等价。

known-future 按目标时刻取值；Direct 历史 lag/rolling/diff 的锚点由 `features.transformations.direct.align_to_target` 决定。目标日对齐且 lag 足够深时消费原点前真实历史；越过原点的 observed-past 访问必须显式 provider，不能隐式填补。

目标变换规格归一化位于本包，目标变换实际拟合与恢复由 `model_forecasting/transforms.py` 编排。raw-design 缓存也不是 `.uv_cache/`：删除依赖下载缓存不应清理模型设计缓存。

```bash
# 项目根目录；按修改范围选用
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_feature_visibility_compiler
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_compiler_batch_equivalence
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_compiled_feature_cache
```
