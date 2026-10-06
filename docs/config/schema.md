# 唯一 schema 与加载严格性

模型 YAML 顶层分为：

```text
problem / data / features / strategy|ensemble /
estimator / probabilistic / validation / output
```

- 单模型使用 `strategy`；Ensemble 使用成员 `config_ref`，两者互斥。
- 引用式融合的权重、内外层窗口、动态衰减及拒绝组合见 [ensemble.md](ensemble.md)；YAML 身份与解析后运行身份分开记录。

Ensemble 顶层：

```text
problem/data/probabilistic/ensemble/validation/output
```

未知顶层和嵌套字段在 loader 阶段 RAISE。Transformation 只使用 `direct/advanced/feature_scaling/target/datetime_categorical/interactions/seasonal_baseline` 新结构。

全部 transformation 使用唯一嵌套 schema：

```yaml
features:
  transformations:
    direct:
      layout: independent_models
    advanced:
      rolling:
        columns: [load]
        windows: [96, 192]
        stats: [mean, std]
    target:
      calendar_normalization: {method: none}
      decomposition: {method: none}
      scaling: {method: none}
```

未知顶层或嵌套字段在 `load_yaml_config()` 阶段 RAISE。

## 加载严格性与 fingerprint

- `load_yaml_config()` 按互斥字段集合分派返回 `ForecastConfigSpec | EnsembleConfigSpec`；未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE；只接受 canonical schema，legacy 形态一律 RAISE。
- 已删除的公共配置字段（顶层 `schema_version`、`problem.information_mode`、`output.setting_suffix`、`probabilistic.recursive_propagation`、`probabilistic.schema_version`）重新声明一律 RAISE。配置无显式版本字段：代码即版本，格式变更通过严格字段合同自然 RAISE。
- 预测输入始终执行严格 as-of；监督训练仅通过内部 `target_access=supervised_labels` 放开预测期 target 标签，不放开 known-future 或历史 target revision 的时间边界。递归 quantile 内部固定走 `median_path`。
- canonical fingerprint 只取语义 payload；并行度、日志和输出目录相关字段不进入 fingerprint。结果 identity = 可读前缀 + 12 位 fingerprint；语义相同的配置别名共享 identity，这不是 hash 碰撞。
- Direct 结果前缀沿用 direct、direct-pointwise、direct-pointwise-horizon 三类；cyclical 属于第三类内部特征变体，仅在元数据记录并由 fingerprint 区分，不从 pointwise 文件名推断语义。统一规则见 [`../packages/model_predicting.md`](../packages/model_predicting.md)，修改展示前缀不改变 fingerprint、不自动迁移旧目录。
- 启用目标分解时语义 payload 带 `decomposition_semantics: component_fit_v2`，区分已修复的 STL/MSTL 外推参数语义。分解别名与严格参数校验唯一入口是 `decomposition/configuration/spec.py`；不修改 YAML、不自动重跑或清除旧结果。

## 训练与概率边界

- `validation.training` 只接受 `sample_weight`、`origin_sampling`，不能为 null。未接线的 early_stopping_patience/tuning/augmentation/feature_selection/learning_rate/huber_delta/blend_weight_windows/estimator_ensemble 及 `validation.train_outlier` 一律拒绝；模型参数仍放 `estimator.params`，特征选择放 `features.selection`。
- sample_weight：`method: exponential`（可省略），必填正数 `halflife_days`；可选 `anchor: cutoff|latest_origin`、`normalization: mean|sum|none`，默认 cutoff/mean。anchor 是可得性上界，合法锚点平移不改变相对权重；none 下最新样本权重为 1。
- point 只使用点预测及可选 absolute_residual 校准，不接受 quantiles/point_quantile/crossing。quantile 支持单分位网格（仍须包含 point_quantile），此时不自动生成区间。
- crossing.report_raw、calibration.allow_interval_shrink 只接受 YAML 布尔值；窗口、样本数、标签延迟只接受整数且拒绝 bool。字符串 `"false"`、小数计数和负小数延迟不会被自动转换。
- YAML 与部署概率解析使用同一校验；合法配置不自动补默认字段，因此不改变原有 payload/fingerprint。新声明的有效权重选项属于显式语义输入。
