# decomposition 模块说明

`decomposition/` 是时间序列目标分解组件库：在可见训练窗口内把目标拆成结构化分量（趋势/季节/残差），各分量独立外推后合回原始目标空间。设计文档见 [`docs/time_series_decomposition_redesign.md`](../docs/time_series_decomposition_redesign.md)。

## 架构

```text
Extractor（拆分）→ Forecaster（分量外推）→ Composer（合成）→ Pipeline（编排）→ Registry（构造）
```

## 文件职责

| 文件 | 职责 |
|---|---|
| `spec.py` | `DecompositionSpec` 配置契约：旧 `decomposition_*` 散列字段与新 `decomposition` mapping 两种写法归一化；方法 none/linear/stl/mstl 校验 |
| `types.py` | 共享数据类型（分量帧、窗口元数据） |
| `base.py` | 组件抽象基类与共享时间轴工具 |
| `extractors.py` | 训练窗口内拆分 y：线性/二次/damped 趋势、STL/MSTL 季节分量 |
| `forecasters.py` | 分量外推：多项式/damped 趋势、相位均值季节模板 |
| `composers.py` | 残差预测 + 结构分量合成回目标空间（当前仅 additive，其余 fail-fast） |
| `pipeline.py` | `DecompositionPipeline` 编排器，对外暴露 fit/transform/component/restore |
| `registry.py` | 从 spec 构造 pipeline；未实现方法在此层 fail-fast |
| `diagnostics.py` | 分解诊断报告（`decomposition_diagnostics.csv`） |
| `residual_diagnostics.py` | 残差 FFT/ACF 频谱诊断，为扩展方法提供量化证据 |
| `periods.py` | 周期检测纯数值核心（FFT/ACF/STL 季节强度），跨包共用：`residual_diagnostics` 与 `data_process/periodicity_analysis` 的唯一事实源 |
| `bundle.py` | 历史分解 bundle 的只读加载兼容；新产物只写 `ForecastModelBundle`，不再新建 sidecar |

## 边界

- 分解严格遵守时间因果：每个训练窗口内独立拟合，测试标签保留原始电平；canonical 主链中由 `model_forecasting/transforms.py` 按 `(series_id, target)` 隔离调用，恢复严格逆序（scaling ← decomposition ← calendar normalization）。
- 与 canonical 配置的关系：YAML 的 `features.transformations.decomposition` 声明方法与参数。
- `bundle.py` 只读适配历史产物，不原地改写。

## 验证

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python -m unittest \
  tests.test_decomposition_spec tests.test_decomposition_equivalence \
  tests.test_decomposition_bundle -v
```
