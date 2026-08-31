# 多步预测 Canonical 架构与实施记录

> 文档状态：**当前唯一权威设计／C0–C7 全部完成并关闭（2026-08-31）**
> 当前实现以代码、活动 schema-2 YAML 和本次 fresh 验证为准；历史方案见 Git 及文末溯源，不再作为现状事实源。
> 七类运行、全量门禁、source header 合同和活动文档均有 fresh 证据；Global N2K2 fixture 已获批作为能力验收，`AGENTS.md` 当前合同已同步。

## 1. 当前边界

`tsproj_ml` 只保留 canonical 运行体系：

- 845 个活动模型 YAML 均为 `schema_version: 2`：797 个普通单模型、24 个引用式 Ensemble、24 个 `ensemble_members/` 独立基模型；另有 41 个数据工具 YAML，不计入模型数。3 个无有效资产的配置原样保存在 `docs/archived_configs/`，不进入 loader/audit/runtime。
- `load_yaml_config()` 严格分派 `ForecastConfigSpec | EnsembleConfigSpec`。未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE。
- `main.py`、`run.py` 不接受 legacy 配置或旧产物。旧 `base_config + overrides`、九方法运行类、旧宽表、旧 pkl 和迁移兼容层均已删除。
- `docs/model_ensemble_redesign.md` 与 `docs/time_series_probabilistic_forecasting_redesign.md` 仅作历史决策参考，本文和 `AGENTS.md` 才是当前事实源。

## 2. 正交预测合同

### 2.1 张量

| 类型 | Shape | 语义 |
|---|---|---|
| point | `(N,H,K)` | N 个 series、H 个未来步、K 个 target |
| marginal quantile | `(N,H,K,Q)` | 各 target/时点的边际条件分位数 |
| joint samples | `(N,S,H,K)` | 仅保留类型边界；生成器明确 unsupported |

内部二维展开固定为 time-major：`(h1,target1)...(hH,targetK)`。边际 quantile 不得表述为联合轨迹分布。

### 2.2 七种策略

`recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo` 只定义未来 H 步的训练与推进方式。Pointwise/horizon-feature 是 Direct layout，Local/Global 是 training scope，模型融合不是第八种 strategy。

MO 策略严格要求 `1 < B < H` 且 `H % B == 0`；不满足时在配置解析期 RAISE，不由运行时截断。

### 2.3 数据角色

`DataSourceSpec.columns` 是入模投影视图；每个声明列必须显式且只声明一个角色：

`target / observed_past / known_future / static / key / ignored`

动态 source 通过 `SourceRegistry + InformationSetRequest` 执行 as-of：

- `target` 的未来真实值不可见；
- `observed_past` 只能使用 forecast origin 及以前的记录，未来轨迹必须绑定显式 provider；
- `known_future` 可以对应未来 target time，但 `available_at <= forecast_origin`；
- static 按 series key 唯一匹配；
- 非 ignored 声明列缺失、重复、越界、不可见或 schema 不匹配均 RAISE；物理文件其他列在 registry 边界丢弃，不会隐式入模。

数据填补与异常清洗只允许发生在离线 `data_process/`，或未来以 config 驱动且满足 as-of 的 compiler 前置步骤中。训练窗口内清洗与评估掩码是两套独立语义。

## 3. 时间几何

### 3.1 Fixed-step

- `problem.horizon` 以 `freq` 为步数单位。
- fixed-step 使用 `validation.history_steps/train_window_steps/fold_count/stride_steps`，四者统一以监督 origin steps 保存。
- 训练标签结束必须严格早于 holdout 标签开始。
- final fit 使用与回测相同的最近训练窗口，不再隐式切为全部历史。

### 3.2 Calendar-month

- `horizon_mode=calendar_month` 时使用 `train_window_days/fold_count/stride_months`；训练窗口按原始日数计，折间距按自然月计。
- 每个历史折和最终目标月按真实月份动态解析 28/29/30/31 个日步。
- 生产折由 `model_testing.validation.calendar_month_folds()` 生成。
- final forecast 必须覆盖完整目标自然月。

### 3.3 月频

`1ME/1MS` 使用月步 offset；seasonal-naive 按月步定位，禁止近似转换为固定天数 Timedelta。

## 4. 当前模块架构

```text
main.py / run.py / config/config_loader.py
        ├── model_forecasting/      单模型生命周期
        └── model_ensemble/         引用式融合生命周期
                    │
                    ▼
probabilistic/                      quantile 训练与独立后处理能力
                    │
                    ▼
data_loading/ → feature_engineering/ → model_training/
                           ├── model_testing/
                           └── model_evaluation/
                    │
                    ▼
forecasting_core/                   稳定 specs/tensors/artifacts
models/ decomposition/ data_process/
                    │
                    ▼
utils/
```

### 4.1 `model_forecasting/` 职责

| 模块 | 当前职责 |
|---|---|
| `design.py` | information-set materialization、特征设计、监督样本和 fixed-step backtest geometry |
| `fit_service.py` | fold/final transform、point/quantile fit 与 predict 服务 |
| `backtest_runtime.py` | calendar-month backtest 生命周期 |
| `persistence.py` | schema-2 bundle 构造与持久化 |
| `forecaster.py` | point 与边际 quantile 推理 |
| `transforms.py` | feature/target transform 状态及严格逆变换 |
| `results.py` | canonical long result 读写 |
| `runtime.py` | `CanonicalBaseModelRunner` 与顶层单模型编排；不再内嵌上述实现 |

### 4.2 包依赖约束

- `forecasting_core/` 不 import 任何流水线、能力或编排包。
- `model_training/` 不 import `probabilistic/`。
- `probabilistic/` 不 import `model_forecasting/`。
- `model_ensemble/` 通过 `EnsembleRuntimeServices` 注入 runner factory 和 bundle persistence，不 import `model_forecasting.runtime`。
- 项目包内禁止函数级延迟 import 绕环，禁止跨包导入下划线私有实现。
- `tests/test_package_layering.py` 使用 AST 扫描顶层和函数内 import，并验证包依赖图无环。

## 5. 训练、变换与模型

`CanonicalTrainer/CanonicalForecaster` 支持 Local/Global、K1/K2、七策略、point 和 independent quantile。

目标变换固定顺序：

```text
calendar normalization → decomposition → target scaling
```

推理恢复严格逆序，状态按 `(series_id,target)` 隔离并随 bundle 保存。Feature scaling 与 feature selection 均在每个训练折内拟合，不能读取 holdout。

模型能力由 `model_training/estimators/capabilities.py` 显式探测；不支持的 target adapter、quantile 或并行组合直接 RAISE，不静默降级。

## 6. 引用式 Ensemble

顶层 `model_ensemble/` 是唯一融合实现，四种方法为：

- `averaging`
- `weighted`
- `linear_blending`
- `stacking`

成员必须引用现役单模型 YAML，并共享 problem/data/probabilistic/origin 合同。OOF cache 对 `(member, source, path_role, file_sha256)` 内容寻址；损坏或输入内容变化必须失效。

Quantile linear blending 按 target 最小化 simplex pooled pinball，一组 target 权重跨 Q 共享。最终 Ensemble bundle 内嵌全部成员 bundle 和 method artifact；部署预测不读取成员 YAML 或 OOF cache。

## 7. 概率边界

当前生产能力只覆盖 point 与边际 quantile：

- quantile 模式写逐 level pinball、central interval coverage/width/Winkler/coverage gap；
- crossing 使用 median-preserving isotonic 后处理；
- `probabilistic/calibration.py` 只保留 CQR 数学内核；canonical runtime 不执行 CQR，也不输出 `predict_pi*`；
- 不支持 joint sample generator、联合路径概率或 ensemble-of-ensemble。

## 8. 结果与产物合同

```text
results/<scenario>/<result_identity>/
├── pretrained_models/
│   ├── model.pkl
│   └── resolved_model.json
├── results_test/
│   ├── cv_plot_df.csv
│   ├── test_scores_df.csv
│   ├── test_scores_probabilistic_df.csv   # quantile only
│   └── result_metadata.json
└── results_forecast/
    ├── prediction.csv
    └── resolved_config.json
```

- `prediction.csv` 唯一键：`(series_id,time,target)`。
- `cv_plot_df.csv` 唯一键：`(series_id,time,target,window)`。
- 单模型和 Ensemble 均保存 schema-2 `ForecastModelBundle`。
- fingerprint 只取语义 payload；日志、并行度和输出目录不进入 fingerprint。
- resolved config/model 必须足以复算 training policy、轴顺序、source/feature lineage 和产物身份。

## 9. 架构整改实施进度

| Slice | 状态 | 当前证据 |
|---|---|---|
| C0/C1 | completed | strict typed schema、生产 parser/constructor 门禁、845 个活动配置；3 个错配配置已批准归档 |
| C2 | completed | fixed-step/calendar-month typed geometry；845 YAML + 迁移 manifest；final fit 同训练窗；615 个子日单模型 sample-count 一致 |
| C3 | completed | proper quantile blending、自包含 bundle、bundle-only point/quantile 部署、优化审计与 canonical long result 已落地；用户批准的 H16/H31 两个代表均 exit 0 且完整产物核验通过 |
| C4 | completed | 稳定概率合同唯一位于 `forecasting_core/`；Protocol 注入、包 DAG、runtime 窄模块和重复合同清理完成 |
| C5 | completed | typed 845-row catalog、CLI/checker 单模型边界、legacy/test-only/零消费者链和 requirements 门禁已收口 |
| C6 | completed | README、权威设计、§17 与 AGENTS typed geometry 已同步；活动 Markdown 断链 0 |
| C7 | completed | fresh 615 tests、三项审计、compile/diff/Markdown 和 H16/H31 代表均已通过；完整证据见架构计划 §17.5 |

C7 的命令、计数和七类产物核验结果只在 `docs/architecture_convergence_plan.md` 的最终执行记录维护；其他文档不复制测试数。

## 10. 历史决策与偏差

- 18 个旧 MDR 配置因 `H % B != 0` 于 2026-08-29 获批映射为 `recursive`：14 个 `aidc_power_month`、4 个 `aidc_load_month`。`validation.legacy_origin` 已删除，本条是批准记录。
- 旧 Direct/Recursive Blend 已迁入 Ensemble，不再作为 strategy。
- decomposition `forecasters.py/composers.py` 曾被误判为死链；包内消费者复核证明它们属于 `DecompositionPipeline` 活路径，因此保留。
- 缺失日期类型未来资产时不伪造数据；受影响配置移除该 source。
- legacy runtime 曾在 `FeatureEngineering.extend_weather_feature()` 内由 `rt_tt2`/`rt_dt` 现场计算 `cal_rh`，随后线性插值、双向填充；canonical 删除隐式数据改写后，2026-08-31 经批准将同一公式迁到离线入口 `config/aidc_electricity_computility/derive_cal_rh.py`，为 23 个天气 CSV/27,180 行追加派生列，原有单元格 SHA 全部保持不变。
- 3 个 `add_training_inference_pod` 配置声明 `*_all_jobs_*` 列，但物理资产使用另一列族；经批准按字节原样迁至 `docs/archived_configs/`，活动模型数由 848 调整为 845。
- 活动 YAML 中没有 Global N2K2；2026-08-31 经批准接受完整生产 runtime fixture 作为该能力验收，不把它表述为现役生产配置。
- 配置语义改变后旧结果失效；旧 pkl/旧宽表在本仓库不可读。

## 11. 验证入口

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m unittest discover -s tests -p "test_*.py"

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_runtime_assets.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_ensemble_configs.py

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m compileall -q \
  forecasting_core data_loading feature_engineering model_training model_testing \
  model_evaluation model_forecasting probabilistic model_ensemble models \
  decomposition data_process utils config scripts tests main.py run.py

git diff --check
```

本轮 fresh 命令、精确计数、七类产物和阻断项只在 `docs/architecture_convergence_plan.md` §17 维护；其他文档中的旧测试数只代表历史环境。

## 12. 历史溯源

- 2026-08-24：概率预测专项方案，现保留为历史参考。
- 2026-08-27 至 29：预测问题正交化、schema-2 迁移、legacy 删除。
- 2026-08-29：引用式 Ensemble v4 实施。
- 2026-08-30 至 31：包级 DAG、时间几何、配置资产、产物和文档收敛；完整执行记录见 `docs/architecture_convergence_plan.md`。

未经 wangzf 明确指示，不 commit、不 push；模型效果消融不属于本架构验收。
