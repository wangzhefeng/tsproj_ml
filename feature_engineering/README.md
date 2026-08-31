# feature_engineering

`feature_engineering/` 是 canonical 唯一特征编译层。

- `compiler.py`：lag、known-future、static、datetime、advanced transformation、visibility proof 与 lineage。
- `selection.py`：每个训练窗独立拟合的监督特征选择。
- `transform_specs.py`：feature/target transformation 严格配置归一化。

特征只能使用预测原点可见信息。实验性 FFT/wavelet/holiday 脚本位于 `docs/feature_engineering/`，未接入生产。
