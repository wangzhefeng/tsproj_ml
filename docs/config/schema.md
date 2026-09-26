# 唯一 schema 与加载严格性

模型 YAML 顶层分为：

```text
problem / data / features / strategy|ensemble /
estimator / probabilistic / validation / output
```

- 单模型使用 `strategy`；Ensemble 使用成员 `config_ref`，两者互斥。

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
      scaling: {method: none, inverse: false}
```

未知顶层或嵌套字段在 `load_yaml_config()` 阶段 RAISE。

## 加载严格性与 fingerprint

- `load_yaml_config()` 按互斥字段集合分派返回 `ForecastConfigSpec | EnsembleConfigSpec`；未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE；只接受 canonical schema，legacy 形态一律 RAISE。
- 已删除的公共配置字段（顶层 `schema_version`、`problem.information_mode`、`output.setting_suffix`、`probabilistic.recursive_propagation`、`probabilistic.schema_version`）重新声明一律 RAISE。配置无显式版本字段：代码即版本，格式变更通过严格字段合同自然 RAISE。
- 预测输入始终执行严格 as-of；监督训练仅通过内部 `target_access=supervised_labels` 放开预测期 target 标签，不放开 known-future 或历史 target revision 的时间边界。递归 quantile 内部固定走 `median_path`。
- canonical fingerprint 只取语义 payload；并行度、日志和输出目录相关字段不进入 fingerprint。结果 identity = 可读前缀 + 12 位 fingerprint；语义相同的配置别名共享 identity，这不是 hash 碰撞。
- Direct 结果前缀沿用 direct、direct-pointwise、direct-pointwise-horizon 三类；cyclical 属于第三类内部特征变体，仅在元数据记录并由 fingerprint 区分，不从 pointwise 文件名推断语义。统一规则见 [`../packages/model_forecasting.md`](../packages/model_forecasting.md)，修改展示前缀不改变 fingerprint、不自动迁移旧目录。
- 启用目标分解时语义 payload 带 `decomposition_semantics: component_fit_v2`，区分已修复的 STL/MSTL 外推参数语义。分解别名与严格参数校验唯一入口是 `decomposition/configuration/spec.py`；不修改 YAML、不自动重跑或清除旧结果。
