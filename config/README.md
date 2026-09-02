# Config

`config/` 当前包含 707 个活动 schema-2 模型 YAML：671 个普通单模型、6 个独立 Ensemble 成员、30 个 Ensemble；按 typed spec 计为 677 个 `ForecastConfigSpec` + 30 个 `EnsembleConfigSpec`。另有 41 个数据工具 YAML。三个 AIDC 15min 负荷场景的 `add_ensemble/` 直接引用 `add_exogenous/` 现役模型，不维护重复 member；其他场景既有 6 个独立 member 保持不变。3 个声明列与物理资产列族错配的配置已于 2026-08-31 经批准原样迁至 `docs/archived_configs/`，不计入活动集。

## 唯一 schema

单模型顶层：

```text
schema_version/problem/data/features/strategy/estimator/probabilistic/validation/output
```

Ensemble 顶层：

```text
schema_version/problem/data/probabilistic/ensemble/validation/output
```

未知顶层和嵌套字段在 loader 阶段 RAISE。Transformation 只使用 `direct/advanced/feature_scaling/target/datetime_categorical/interactions` 新结构。

Fixed-step validation 使用 `history_steps/train_window_steps/fold_count/stride_steps`，均以监督 origin steps 保存；calendar-month 使用 `train_window_days/fold_count/stride_months`，训练窗按原始日数计并动态解析每月 H。

经用户裁决，无权威日期类型资产的 396 个配置已移除 `date_type` source，不生成伪标签。

算力天气 `cal_rh` 在离线数据准备阶段由 `rt_tt2`/`rt_dt` 按 legacy Magnus–Tetens 公式派生；迁移入口为 `config/aidc_electricity_computility/derive_cal_rh.py`。canonical runtime 不恢复旧 `FeatureEngineering` 的现场计算或插值。

## 审计

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_runtime_assets.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_ensemble_configs.py
```
