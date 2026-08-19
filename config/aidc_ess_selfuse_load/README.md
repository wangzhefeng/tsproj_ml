# ESS 站用电策略特征

本目录维护储能站用电预测的策略驱动特征工程、生成配置和消融实验配置。

## 入口与目录

| 路径 | 作用 |
|---|---|
| `build_strategy_features.py` | 薄 CLI 入口，只负责参数解析和调用流水线 |
| `strategy_features.yaml` | v2 数据边界、运行阈值、相似日参数和 A/B 路数据源 |
| `strategy_features/` | v2 可测试实现：时间窗口、状态编码、周期画像、相似日与主流水线 |
| `route_{A,B}/add_strategy_features/` | v2 C0–C3 与 P6 C5 联合聚类消融配置 |

生成数据落在 `dataset/aidc_ess_selfuse_load/forecasting_data/strategy_features/`，由仓库级 `dataset/` 忽略规则排除，不提交 Git。

## 构建与验证

在仓库根目录执行：

```bash
# 完整计算和契约校验，不写文件
UV_CACHE_DIR=.uv_cache uv run --no-sync python \
  config/aidc_ess_selfuse_load/build_strategy_features.py \
  --config config/aidc_ess_selfuse_load/strategy_features.yaml \
  --validate-only

# 写入输出；已有文件时必须显式 --force
UV_CACHE_DIR=.uv_cache uv run --no-sync python \
  config/aidc_ess_selfuse_load/build_strategy_features.py \
  --config config/aidc_ess_selfuse_load/strategy_features.yaml \
  --force
```

可用 `--data-root` 覆盖默认数据根目录，便于测试临时数据或其他数据版本。

## v2 特征结构

v2 同时维护两个时间轴：

- **自然日** `[00:00, 23:55]`：用于相似日检索和全天计划曲线。
- **调度周期** `[22:00, 次日 21:55]`：用于充电、待机、放电周期画像。

模型特征分为四层：

| 层 | 主要特征 | 未来可用性 |
|---|---|---|
| 日滞后与已完成周期 | `ess_lag_288`、`pcs_actual_lag_288`、前一已完成周期时长及一致率 | 来源时间必须不晚于 `as_of_time` |
| 已知计划与周期画像 | `pcs_plan`、计划方向 one-hot、充放电时长/能量/切换次数/首时刻周期编码 | 由未来计划直接构造 |
| 相似日模板 | 计划距离、新颖度、有效样本数、相似日/稳健近期模板及 gate 融合模板 | 只检索历史完整自然日 |
| 联合聚类 lag1 | D−1 的 ESS/实际 PCS/计划 PCS 三视图 one-hot、中心距离、rare、ready | artifact 只在固定 reference 期拟合，D 日只读取 D−1 |

联合聚类使用 `2025-11-01～2026-06-27` 固定 reference：每个 view 独立拟合 StandardScaler+PCA（解释率 ≥90%），block 归一后在 K=2～5 中按 silhouette 选 KMeans；小类不合并，只标记 rare。回测 artifact 与未来最终 forecast artifact 不得混用。

## C0–C5 消融

A/B 路使用相同的消融层级：

| 配置 | 特征集合 |
|---|---|
| C0 | datetime 基线，不启用项目原生目标滞后 |
| C1 | C0 + 日滞后与已完成周期特征 |
| C2 | C1 + 已知计划及计划周期画像 |
| C3 | C2 + 相似日与条件 Gate 模板 |
| C5 | C3 + D−1 联合聚类 one-hot/中心距离/rare/readiness |

所有配置均使用 USMDP point、`predict_steps: 288`、`detrend_target: true`，通过不同 `setting_suffix` 隔离结果目录。C5 只做 31 窗 testing、关闭 final forecasting，确保 `fit_end=2026-06-27` 的回测 artifact 不被当成最终预测 artifact。

## 因果与数据契约

1. `as_of_time` 是历史事实边界；任何未来特征不得读取该时刻之后的 ESS 或实际 PCS。
2. 未来必须是完整、唯一、递增且严格对齐 5 分钟网格的自然日；历史缺口只记录审计，不直接阻断。
3. `contracts.py` 维护唯一的模型列清单、未来关键列和禁止列/命名模式。
4. 未来关键列必须有限且无缺失；禁止出现目标、实际 PCS、同日事后聚类等泄漏列。
5. `lag_feature_ready` 与 `template_feature_ready` 是显式 readiness 标志，不用隐式 NaN 表示是否可用。
6. 默认不覆盖已有输出；写入必须显式 `--force`，避免配置与历史产物不一致。

## 输出

每路生成：

- `model_features_history_{A,B}.csv`
- `model_features_future_{A,B}.csv`
- `audit/calendar_day_quality_{A,B}.csv`
- `audit/dispatch_cycle_summary_{A,B}.csv`
- `audit/similar_day_matches_{A,B}.csv`
- `audit/feature_build_audit_{A,B}.json`
- `audit/joint_cluster_assignments_{A,B}_fit-20260627.csv`
- `artifacts/joint_cluster_{A,B}_fit-20260627.joblib`

## 测试

```bash
UV_CACHE_DIR=.uv_cache uv run --no-sync python -m unittest discover \
  -s tests -p "test_*.py"
```

策略特征专项测试为 `tests/test_ess_strategy_*.py`，覆盖状态编码、双时间轴、周期画像、相似日、联合聚类以及流水线泄漏/落盘契约。
