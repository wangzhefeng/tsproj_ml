# 模型融合模块重设计 Implementation Plan

> **文档状态：Historical reference（2026-08-31）**。本文保留 Ensemble v4 的设计决策与实施溯源，不再描述当前代码路径、配置计数或验收状态。当前事实以 [`multistep_forecasting_redesign.md`](./multistep_forecasting_redesign.md) 和 [`architecture_convergence_plan.md`](./architecture_convergence_plan.md) 为准。

> **v4（2026-08-29）**：基于 v3 的程序化评审修订，整体替换 v3。基线事实全部复核属实（824 YAML / 24 ensemble / 485 tests / 848 计数、Direct 可引用、Recursive 需新增，均经独立脚本验证）。v4 主要修正：
> 1. **parity 分级**（B1）：核实当前 stacking 的 NNLS 权重按 outer 滑窗逐窗重训（`forecasting/runtime.py:1583-1595`，final 用末窗 indices 再学一次 `:1731-1737`），不存在可供复现的"当前固定权重"。stacking→linear_blending 的端到端"逐值一致"声明不可实现，降级为函数级 fixture；weighted→averaging 保持端到端逐值一致。
> 2. **calendar_month OOF 边界**（B2）：power_month 2 个 ensemble 为 `freq: 1D + horizon_mode: calendar_month`，变长 horizon 下固定 fold 几何语义未定义。首版 OOF 对 calendar_month 只允许 `fold_count=1`，`fold_count>1` 显式 RAISE。
> 3. **fixture 时序**（B3）：E0 fixture 从终局一次性消费改为每个切片（E1–E6）的回归门禁。
> 4. 补齐 6 项缺失决策：stacking H 维语义/截距/标准化/alpha、weighted 默认 metric、bundle 校验冲突改动点、禁止字段 RAISE 清单、outer backtest vs OOF 几何归属、OOF 孤儿缓存报告。
>
> **Provenance**：Agent MC（Hermes coder profile）/ Model GLM / 2026-08-29。依据：通读 v3 全文，交叉核对 `forecasting/runtime.py`（ensemble 段 1115–1881）、`forecasting/ensemble.py`（NNLS 组合）、`probabilistic/types.py`（ForecastModelBundle 校验）、24 个 ensemble YAML 逐文件语义审计、824 YAML 全量语义比对（Direct/Recursive 匹配计数）、485 unittest 本机复现 OK、`results/` 无 canonical ensemble 产物核实。
>
> **V3→V4 变更链**：V3 修正了 v2 的 phase-gate/legacy 残留与 824→848 计数；V4 不改架构主体与切片划分，只修正 parity 语义、补 calendar_month 边界、固化 6 项实现决策到对应切片。
>
> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 在不恢复任何 legacy 路径、不重写已验证单模型主链的前提下，将模型融合集中到顶层 `ensemble/` 包，支持独立基模型引用、可复用 rolling-origin OOF，以及 averaging、weighted、linear blending、stacking 四种明确方法。

**Architecture:** `forecasting` 继续负责单个 canonical 基模型的特征、变换、策略、估计器、训练和预测；`ensemble` 只引用完整基模型 YAML、生成 OOF、拟合融合器并组合已恢复到原始目标空间的标准预测。正式 Ensemble bundle 内嵌每个成员的 BaseModel bundle，不依赖 OOF 缓存部署。

**Tech Stack:** Python 3.10、dataclass、NumPy/SciPy、scikit-learn、pandas、现有 canonical tensor/bundle、unittest、uv。

---

## 1. 当前实现基线

### 1.1 已核实事实

| 项目 | 当前事实 |
|---|---|
| 配置 | 824 个 schema-2 canonical YAML；无 legacy YAML、`legacy_origin`、`testing/forecasting/window_count` |
| 配置 shape | 800 个单模型；24 个现役 ensemble |
| Ensemble 配置 | 全部 Local、K=1、LightGBM、Direct+Recursive、权重 `[0.5,0.5]` |
| 方法分布 | `weighted=12`、`stacking=12`；当前 stacking 实际算法是 NNLS 非负归一化权重，且**权重按 outer 滑窗逐窗重训**（`forecasting/runtime.py:1583-1595`），final 阶段用末窗 train indices 再学一次（`:1731-1737`） |
| 概率分布 | point 14、quantile 10 |
| horizon 模式 | 22 个 `fixed_steps`（H=288/96/16）；power_month route_A/B 2 个为 `calendar_month`（H 随目标月 28–31 变长） |
| Runtime | `run_canonical_config()` 每次完整执行 rolling backtest、final fit、模型保存和 forecast |
| Legacy | `forecasting/legacy/`、`models/multistep/`、legacy config templates、DataLoader、旧 FeatureEngineering、legacy reader/adapter、迁移器和 invalidation manifest 均已删除 |
| 旧模型融合 | `models/ModelEnsemble.py` 仍有 averaging/weighted/blending/RidgeCV stacking 实现，但 `TimeSeriesEnsembleRegressor` 实际消费为 0；`ModelTraining.py` 仅残留死 import |
| 当前 canonical ensemble | `forecasting/ensemble.py` + `forecasting/runtime.py:1115-1881`；成员共享同一 features/transform/estimator，只能变化 strategy |
| 产物 | 当前 `results/` 中 canonical `resolved_config.json/resolved_model.json` 均为 0，无正式 canonical ensemble 产物需要兼容 |

Fresh 验证（2026-08-29 复核）：

```text
ensemble/结构定向：11 tests OK
全量 unittest：485 tests OK
配置 checker：checked=824 passed=824 hard_failures=0 warnings=0
24 Direct 成员语义匹配：24/24 恰好命中 1 个现有单模型 YAML
24 Recursive 成员语义匹配：24/24 无匹配，且 24 个成员语义两两唯一
```

### 1.2 对 v2 计划的修正

1. **删除 G0 phase gate**：phase 字段已从 824 YAML、spec、checker、测试和文档整链删除；不得恢复。
2. **删除所有 legacy 迁移/manifest 任务**：当前仓库明确无 legacy 读取能力；不得重建 `LegacyArtifactAdapter`、`LegacyResultReader`、迁移器或 invalidation manifest。
3. **不再要求配置数量保持 824**：24 个 Direct 成员均可复用现有基模型；24 个 Recursive 成员均无等价 YAML，且 24 个语义互不重复，因此需要新增 24 个独立基模型，最终应为 848 个模型 YAML。
4. **迁移方法按当前有效行为，不按历史名称**：当前等权 `weighted` → `averaging`；当前 NNLS `stacking` → `linear_blending`。新 Ridge stacking 作为新增能力和后续实验，不冒充等价迁移。
5. **不创建 `forecasting/model_runtime.py` 搬迁主链**：在现有 `forecasting/runtime.py` 内提取公开窄接口，避免大范围移动已验证代码。
6. **不依赖已删除 DataLoader/FeatureEngineering**：全部成员继续走 `SourceRegistry + FeatureCompiler + CanonicalTrainer/Forecaster`。
7. **不做兼容旧 artifact**：只保持现有 schema-2 单模型 bundle 行为；ensemble branch 可直接升级，因为磁盘无 canonical ensemble 产物。
8. **清理而非实现 CLI phase**：`run.py` 仍残留 `--is-testing/--is-forecasting` 及日志字段，随入口适配删除，不能重新接入 runtime。

### 1.3 对 v3 的 v4 修正

1. **parity 分级（B1）**：当前 stacking 权重逐滑窗重训，`stacking→linear_blending` 迁移不存在可复现的固定权重，端到端"逐值一致"声明撤销；改为 NNLS 函数级黄金 fixture + 回测指标允许变化。当前权重学习用单个 validation origin（`primary_train_indices[-1]`）拟合 NNLS，方差极大，本就是待改进行为——不得将其端到端输出录为 golden 规范。
2. **calendar_month OOF 边界（B2）**：OOFSpec 固定 fold 几何不适用于变长 horizon；首版对 `calendar_month` 只允许 `fold_count=1`，多 fold 显式 RAISE，列入"明确不做"。
3. **fixture 生命周期（B3）**：E0 录制的 fixture 成为 E1–E6 每个切片的回归门禁，防止跨切片失效无人发现。
4. **固化实现决策**：stacking meta 学习语义（§3）、weighted metric 防病态（§3）、EnsembleConfigSpec 禁止字段（§5.2）、几何归属（§5.2）、bundle 校验冲突点（§8.2）、OOF 孤儿缓存审计（§7.2/E6）。

---

## 2. 最终术语与边界

```text
完整基模型
  problem/data + features/preprocessing/target transform
  + multi-step strategy + estimator
                    ↓ 成员独立恢复到原始目标空间
OOF 数据
  shared rolling-origin folds + gap
  → OOFPredictionArtifact
                    ↓
融合方法
  averaging | weighted | linear_blending | stacking
```

- **多步策略**：只定义未来 H 步如何训练和推进。
- **估计器**：只定义每次拟合使用的学习器。
- **基模型**：一个现役、可独立运行的 canonical 单模型 YAML。
- **融合方法**：只消费相同 `(series_id,time,target[,quantile])` 坐标上的成员预测。

所有成员必须共享预测问题和信息边界；成员可拥有不同 source 子集、features、feature scaler、target transform、strategy 和 estimator。任何成员预测必须先恢复到原始目标空间，再进入 OOF 和融合。

---

## 3. 四种融合方法

简单平均是 forecast combination 的强基线，复杂方法必须与 averaging 比较。[1]

| 方法 | 参数学习 | 项目定义 |
|---|---:|---|
| `averaging` | 否 | 成员等权平均 |
| `weighted` | 是 | 每 target 按 OOF 误差倒数归一化；默认 metric=`rmse`，可选 `mae`；`mape` 需显式开启并执行 nonzero 下界保护（低负荷时段 MAPE 病态放大，禁止无保护使用） |
| `linear_blending` | 是 | 每 target 在 OOF 上求非负、和为 1、无截距的最小 MSE 线性组合 |
| `stacking` | 是 | 每 target 用成员 OOF 预测训练 meta estimator；默认 Ridge（见下） |

统一边界：

- 参数固定 per-target 学习，跨 series/horizon 汇总。
- 不做 per-series/per-horizon 参数。
- averaging/weighted/linear_blending 支持 point 和边际 quantile；同一 target 的全部 quantile level 使用同一组非负权重。
- stacking 首版只支持 point；quantile stacking 直接 RAISE。
- stacking 必须使用 OOF 预测训练 meta estimator，成员最终再在完整可见历史上重训；禁止 in-sample/prefit meta 训练。[3]
- 业内对 blending 命名不完全统一；本项目使用 `linear_blending` 明确表示"共享 OOF 上的受约束线性组合"。

**stacking meta 学习语义（v4 固化）**：

- **H 维约简**：成员 OOF 预测为 `(N,H,K)`；对每个 target k，将全部 fold 的 `(样本, h)` 展平为一维训练列（shape `(n_folds × N × H,)`），每个 target 训练**一个** Ridge——跨 horizon 汇总，与"不做 per-horizon 参数"一致。
- **截距**：`fit_intercept=False`。成员预测已在原始目标空间、量纲一致，截距会引入无成员对应的系统性偏移；与 blending 的无截距边界保持同族。
- **特征标准化**：meta 输入按 target 用 OOF 均值/标准差标准化（成员误差量级差异大时 Ridge 惩罚不公平）；标准化参数随 method artifact 保存，预测期复用。
- **正则**：`Ridge(alpha=1.0)` 首版固定，不进配置；后续调参走 E8 实验，不提前参数化。

---

## 4. 目标模块结构

```text
ensemble/
├── __init__.py
├── specs.py          # EnsembleConfigSpec、MemberRef、OOFSpec、MethodSpec
├── loader.py         # config_ref 解析和共享合同校验
├── contracts.py      # BaseModelRunner protocol、method protocol
├── oof.py            # OOF 生成和 outer-cutoff 过滤
├── cache.py          # 文件 SHA-256、原子读写、失效校验
├── artifacts.py      # OOFPredictionArtifact、MethodArtifact、EnsembleArtifact
├── trainer.py        # OOF、融合器拟合、最终成员重训
├── predictor.py      # 成员预测和组合
├── runtime.py        # 完整 backtest/final/persist 编排
└── methods/
    ├── averaging.py
    ├── weighted.py
    ├── linear_blending.py
    └── stacking.py
```

收口规则：

- `models/ModelEnsemble.py` 保留文件，但只重导出顶层包公共 API。
- `forecasting/ensemble.py` 保留文件，但只重导出顶层包 typed specs/artifacts/组合函数。
- `ModelTraining.py` 删除旧 ModelEnsemble 死 import。
- 新实现只能位于 `ensemble/`；两个兼容入口不得含算法逻辑。
- 不创建任何 legacy 兼容模块。

---

## 5. 配置 schema

### 5.1 单模型

当前 `ForecastConfigSpec` 改为**仅表示单基模型**：

- 顶层必须有 `problem/data/features/strategy/estimator/probabilistic/validation/output`。
- `strategy` 必须非空。
- 删除 `ForecastConfigSpec.ensemble`。
- 800 个现有单模型 YAML 内容不变。

### 5.2 EnsembleConfigSpec

Ensemble YAML 只允许：

- `schema_version`
- `problem / data / probabilistic`
- `ensemble`
  - `members[] = {name, config_ref}`
  - `oof = {train_window_length, fold_count, stride, gap_steps}`
  - `method = {name, ...params}`
- `validation / output`

**禁止字段（v4 固化）**：ensemble YAML 顶层出现 `features`、`strategy`、`estimator` 任意一个即 RAISE（含 `strategy: null`——现有 24 个 YAML 恰是被清除对象）；`ensemble.oof` 出现未知字段即 RAISE。E2 的互斥 fixture 按禁止字段逐项建用例。

**几何归属（v4 固化）**：ensemble 的 outer backtest / final forecast 边界仍由顶层 `validation`（`history_length/window_length/max_test_windows/test_window_stride/forecast_origin`）决定，与单模型同一合同；`ensemble.oof` 只决定融合器学习的内部 fold 几何。两套几何独立配置、互不推导。

`load_yaml_config()` 根据互斥字段集合返回：

```text
ForecastConfigSpec | EnsembleConfigSpec
```

两者仍使用顶层 `schema_version: 2`；不为 800 个单模型增加无意义类型字段。

### 5.3 成员引用合同

`ensemble/loader.py` 必须：

1. 只允许引用单模型 `ForecastConfigSpec`；禁止 ensemble-of-ensemble、循环引用、重复 name/ref。
2. EnsembleConfig 与成员的 targets、freq、horizon、training scope、series keys、information mode、quantile grid 必须一致。
3. EnsembleConfig.data 是允许的数据源全集；成员 target/key source 必须完全一致，其他 source 必须是该全集中定义完全相同的子集。
4. 成员各自的 features、变换、strategy、estimator 独立生效。
5. 成员 validation/output 只服务独立运行；ensemble OOF、outer backtest、forecast origin 和输出由 EnsembleConfig 决定。
6. 任一成员能力不支持当前 probabilistic mode 即 RAISE，不静默丢成员。

---

## 6. 共享基模型运行接口与时间切分

### 6.1 CanonicalBaseModelRunner

在现有 `forecasting/runtime.py` 中提取公开窄接口，封装当前已验证私有能力：

- 构建设计和监督样本；
- 按指定 train indices 拟合成员自己的 feature/target transform；
- 训练 point/quantile artifact；
- 在指定 origin 预测并恢复到原始目标空间；
- 返回成员 bundle、capabilities、lineage 和标准 tensor。

单模型 `run_canonical_config()` 必须先改为消费该接口并保持输出逐值不变；`ensemble` 随后只依赖此公开接口，不 import runtime 私有函数。

### 6.2 共享 RollingOriginSplitter

新增 `forecasting/validation.py`，把当前 `_label_start/_label_end/_holdout_training_indices/_rolling_backtest_windows` 的通用时间合同抽成 typed folds：

- 单模型 outer backtest 和 ensemble OOF 共用同一实现；
- 支持固定 train window、fold count、stride、gap；
- Global 按 origin 切分，同一 origin 的全部 series 不可拆分；
- 严格要求 `training_label_end_max < validation_label_start`。

时间序列验证必须保持时间顺序；随机 KFold 不允许。[2][4]

---

## 7. OOFArtifact 与缓存

### 7.1 OOF 规则

- learned 方法使用 EnsembleConfig 定义的同一组 rolling-origin folds。
- 每个 fold 对每个成员独立 fit/predict/restore。
- outer backtest 在拟合融合器时，只读取 `OOF.label_end < outer_holdout.label_start` 的前缀。
- final forecast 使用 forecast origin 前全部合法 OOF。
- averaging 不依赖 OOF 拟合，但可复用 OOF 生成成员审计指标。

**calendar_month 边界（v4）**：`horizon_mode=calendar_month` 时 horizon 随目标月在 28–31 变长，固定 `stride/gap_steps` 的 fold 几何未定义。首版 OOF 对 calendar_month 场景**只允许 `fold_count=1`**（单内层 holdout，`label_end = origin + horizon × offset` 已有定义）；`fold_count>1` 且 calendar_month 直接 RAISE，多 fold 变长几何列入"明确不做"。E3 必须包含该 RAISE 用例。当前 power_month 2 个配置迁移为 `linear_blending, fold_count=1`，与该边界自洽。

### 7.2 缓存

```text
<results_root>/_ensemble_oof/<oof_fingerprint>/
├── oof_predictions.csv
├── oof_metadata.json
└── member_manifest.json
```

当前 824 配置共 2118 个 source，全部为 file source、generator=0。因此首版只支持 file source 内容 SHA-256；未来出现 generator member 时先 RAISE，不预建版本接口。

OOF fingerprint 包含：

- member name + 单模型 semantic fingerprint；
- Ensemble problem/data/probabilistic；
- train window/folds/stride/gap；
- 所有实际读取 source 文件的内容 SHA-256；
- OOF schema/logic version。

缓存损坏、列不全、hash 不符直接重建或 RAISE；不得部分复用。

**孤儿缓存治理（v4）**：`_ensemble_oof/<fingerprint>/` 按 fingerprint 键控、跨配置共享，会随配置废弃积累无主目录。`scripts/audit_ensemble_configs.py` 必须报告"不被任何现役 ensemble 配置 fingerprint 引用的孤儿 OOF 目录"清单；删除仍需人工确认（auto-accept 下删除类操作必须先问 wangzf）。

---

## 8. Artifact 与 bundle

### 8.1 EnsembleArtifact

正式 artifact 必须包含：

- resolved EnsembleConfig；
- 每个成员的完整 schema-2 BaseModel bundle；
- method artifact：等权、per-target weights、blend coefficients 或 per-target meta estimators（stacking 需额外保存 per-target 的 meta 特征标准化参数）；
- OOF fingerprint、fold 边界摘要、member order、N/H/K/Q；
- config fingerprint 和 source lineage。

部署预测不读取 OOF cache、backtest CSV 或基模型 YAML。

### 8.2 ForecastModelBundle 调整

- 单模型 bundle 合同保持不变。
- ensemble bundle：`model_type="ensemble"`；top-level `feature_scaler/target_transform` 为 None；成员 bundle 各自保存自己的 transform/schema。
- 单模型 bundle 必须是 `strategy_spec + estimator_spec`。
- ensemble bundle 必须是 `ensemble_spec`，top-level `strategy_spec/estimator_spec` 均为 None；成员 estimator 信息只在 member bundles/spec 中保存。
- **现有校验冲突点（v4 明确）**：`probabilistic/types.py::ForecastModelBundle.__post_init__` 当前要求 schema-2 bundle 的 `estimator_spec` 必为 dict（`RAISE "canonical ForecastModelBundle requires estimator_spec"`），与 ensemble bundle 的 `estimator_spec=None` 直接冲突。E5 必须将该行校验改为"单模型 bundle 要求 estimator_spec 为 dict；ensemble bundle（`ensemble_spec` 非空）要求 estimator_spec/strategy_spec 均为 None"，并同步更新 `tests/test_forecast_model_bundle.py` 的既有断言。
- 当前磁盘无 canonical ensemble bundle，因此不保留旧 canonical ensemble artifact 兼容分支；相应结果统一失效。

新增审计输出：

- `ensemble_member_scores.csv`
- `ensemble_error_correlation.csv`
- `ensemble_weights.csv`（前三种方法）
- `ensemble_meta_report.json`（stacking）
- `ensemble_oof_reference.json`

learned 方法必须同时报告 best single、averaging 和当前方法；只超过单体但不超过 averaging，不能判定复杂方法有效。[1]

---

## 9. 现役 24 配置的精确迁移

程序化 semantic audit 已确认（v4 复核通过）：

- 24 个 Direct 成员均恰好匹配 1 个现有单模型 YAML，可直接引用。
- 24 个 Recursive 成员均无匹配，且 24 个语义全部唯一；必须新增 24 个 `ensemble_members/*_recursive.yaml`。
- 迁移后模型 YAML 总数：`800 + 24 ensemble + 24 recursive member = 848`。

方法迁移：

| 当前配置 | 当前有效算法 | 新方法 |
|---|---|---|
| `weighted` + `[0.5,0.5]`（12） | 固定等权 | `averaging` |
| `stacking`（12） | 单内层 holdout + NNLS 非负归一，权重按 outer 滑窗逐窗重训 | `linear_blending`，首轮 `fold_count=1`（calendar_month 2 个配置亦为 `fold_count=1`，见 §7.1） |

**parity 分级（v4 修正）**：

1. weighted 12 个：等权迁移语义严格等价，保留**端到端逐值一致**承诺——E0 录制滑窗预测/指标黄金 fixture，E6 迁移后逐值比对。
2. stacking 12 个：当前权重逐滑窗重训，OOF 前缀学习的权重与逐窗权重必然不同，端到端逐值一致**不可实现也不追求**。E0 只对 `fit_nonnegative_stacking_weights` 录制**函数级黄金用例**（给定成员预测+actual 输入 → 权重输出），E6 迁移后该函数行为逐值一致；回测指标允许变化。
3. stacking 迁移后 12 个配置的 `cv_plot_df/test_scores_df` 与旧结果不可比，按项目既定约定：**旧 ensemble 结果按配置变更作废**，`results/` 中不对应当前 setting 的旧目录在迁移完成后清理（删除前请示 wangzf）。
4. 不新增 `legacy_origin`、迁移器或 invalidation manifest。
5. 新增永久 `scripts/audit_ensemble_configs.py`，检查引用、共享合同、方法分布、OOF fingerprint、精确计数 848 和孤儿 OOF 缓存目录报告。
6. 当前无 canonical ensemble 结果可保留；新 identity/fingerprint 直接生效，不移动 results。
7. 真正 `weighted` 和 Ridge `stacking` 只在后续实验中新建，不通过迁移伪造已有结果。

---

## 10. 实施切片

### E0：基线、parity fixture 与文档状态

**Files:**
- Modify: `docs/redesign/multistep.md`
- Modify: `AGENTS.md`
- Create/Modify: `tests/test_ensemble_parity.py`

**Work:** 记录 824/24/485 基线；录制两层 fixture：(a) 单模型主链 golden（现有）；(b) weighted 12 配置端到端滑窗黄金 fixture + `fit_nonnegative_stacking_weights` 函数级黄金用例（v4 分级）；锁定 point/quantile 输出和 full-lifecycle 产物；权威文档标记 ensemble redesign in_progress。

**Gate:** 不修改生产行为；两层 fixture 在现实现上通过。

---

### E1：共享 validation 与 CanonicalBaseModelRunner

**Files:**
- Create: `forecasting/validation.py`
- Modify: `forecasting/runtime.py`
- Modify: `tests/test_canonical_runtime_smoke.py`
- Modify: `tests/test_canonical_global_runtime.py`
- Create: `tests/test_canonical_base_model_runner.py`

**TDD:** 先锁定现有单模型 backtest/final 输出；抽取 splitter 和 runner；同配置预测、指标、bundle schema、fingerprint 和文件布局逐值不变。

**Gate:** 全量 485 基线测试仍通过；**E0 两层 fixture 回归通过（v4）**；尚不改 ensemble schema。

---

### E2：顶层包与新 schema 旁路实现

**Files:**
- Create: `ensemble/__init__.py`
- Create: `ensemble/specs.py`
- Create: `ensemble/loader.py`
- Create: `ensemble/contracts.py`
- Create: `tests/test_ensemble_specs.py`
- Create: `tests/test_ensemble_loader.py`

**TDD:** 直接调用新 parser/loader 验证两种字段集合互斥、引用循环/嵌套/问题或数据边界不一致 RAISE、**禁止字段（`features/strategy/estimator`/`strategy: null`）逐项 RAISE（v4）**；现有 `config_loader.py`、`ForecastConfigSpec` 和 24 个旧 ensemble YAML 本切片保持不变。

**Gate:** 新 schema 可独立解析合成 fixture，现有 824 配置 checker 和全量测试仍绿；生产 loader 的原子切换留到 E6，不建立长期双 schema。

---

### E3：OOF 与缓存

**Files:**
- Create: `ensemble/oof.py`
- Create: `ensemble/cache.py`
- Create: `ensemble/artifacts.py`
- Create: `tests/test_ensemble_oof.py`
- Create: `tests/test_ensemble_cache.py`

**TDD:** fold 时间边界、H-step、Global origin 原子性、成员独立 transform/restore、outer cutoff、文件内容变更使 hash 失效、缓存损坏 RAISE、同 fingerprint 复用、**calendar_month + fold_count=1 可解析且 fold_count>1 RAISE（v4）**。

**Gate:** 不实现融合方法；OOFArtifact 可独立审计。

---

### E4：四种方法

**Files:**
- Create: `ensemble/methods/averaging.py`
- Create: `ensemble/methods/weighted.py`
- Create: `ensemble/methods/linear_blending.py`
- Create: `ensemble/methods/stacking.py`
- Create: `tests/test_ensemble_methods.py`

**TDD:** 等权；inverse-error（默认 RMSE；MAPE 仅显式开启且带 nonzero 下界保护，v4）；非负和为1的无截距最小 MSE blending；OOF per-target Ridge stacking（跨 (样本,h) 展平、fit_intercept=False、per-target 标准化、alpha=1.0，v4 固化）；stacking quantile RAISE；前三种 quantile 使用同权重并保持 axes/levels。

**Gate:** method 类不切 fold、不训练成员、不读文件。

---

### E5：Ensemble runtime、bundle 与持久化

**Files:**
- Create: `ensemble/trainer.py`
- Create: `ensemble/predictor.py`
- Create: `ensemble/runtime.py`
- Modify: `probabilistic/types.py`（**具体点：`ForecastModelBundle.__post_init__` 的 `estimator_spec` 必为 dict 校验，改为单模型/ensemble 分支校验，见 §8.2，v4**）
- Modify: `models/ModelSaveLoad.py`
- Modify: `main.py`
- Create: `tests/test_ensemble_runtime.py`
- Modify: `tests/test_forecast_model_bundle.py`

**TDD matrix:** 相同策略不同估计器、不同策略相同估计器、二者都不同；Local K1、Global N2K2；point 四方法；quantile 前三方法；outer holdout 不影响此前融合器；pickle 后预测一致；full lifecycle 六类产物齐全。

**Gate:** Ensemble bundle 自包含，部署不依赖 YAML/OOF/CSV；**E0 fixture 回归通过（v4）**。

---

### E6：原子 schema 切换、24 配置迁移与代码收口

**Files:**
- Modify: `forecasting/specs/config.py`（`ForecastConfigSpec` 改为 base-only）
- Modify: `config/config_loader.py`（按互斥 shape 分派）
- Modify: `main.py`（统一分派单模型/ensemble runtime）
- Modify: `run.py`（适配 `EnsembleConfigSpec`，删除失效 phase CLI 参数和日志）
- Modify: 24 个现役 ensemble YAML
- Create: 24 个专用 Recursive BaseModel YAML
- Create: `scripts/audit_ensemble_configs.py`
- Modify: `scripts/check_model_configs.py`
- Modify: `models/ModelEnsemble.py`（thin re-export）
- Modify: `forecasting/ensemble.py`（thin re-export）
- Modify: `models/ModelTraining.py`（删死 import）
- Modify: current ensemble tests to new package/schema

**Execution:** 在同一切片中先生成并审计 24 个 Recursive 配置，再改 24 个 Ensemble YAML，最后切换 loader/entrypoint；任一中间步骤不作为可交付状态。

**Gate:** checker 精确 `848/848`；方法分布 `averaging=12, linear_blending=12`；24 Direct ref 和 24 Recursive ref 均唯一有效；`run.py` 无 `is-testing/is-forecasting`；两个 shim 无实现代码；**parity 分级通过：weighted 12 端到端逐值一致 + NNLS 函数级 fixture 一致（v4）**；audit 报告孤儿 OOF 缓存目录（v4）。

---

### E7：文档与全量验收

**Files:**
- Modify: `README.md`
- Modify: `forecasting/README.md`
- Modify: `models/README.md`
- Modify: `config/README.md`
- Modify: `tests/README.md`
- Modify: 场景模型测试说明
- Modify: `tests/test_canonical_only_runtime.py`

**Commands:**

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/check_model_configs.py

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/audit_ensemble_configs.py

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run \
  python -m unittest discover -s tests -p "test_*.py"

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m compileall ensemble forecasting config models probabilistic tests main.py run.py

git diff --check
```

**Hard acceptance:** 848 configs、0 hard failure、全量 unittest exit 0、无随机 KFold、无 in-sample stacking、无 legacy 模块/字段/文档复活、两个兼容 shim 无算法逻辑、**weighted parity 逐值一致 + NNLS 函数级 parity 一致（v4 分级口径）**、权威文档记录 fresh 精确证据。

---

### E8：真实效果消融（独立运行授权）

同一成员池和 OOFArtifact 比较：best single、averaging、weighted、linear_blending、stacking。

顺序：单配置 smoke → route_A 31 同窗 → 达门槛后 route_B 31 同窗。

判定：

- A/B median MAPE 同向且绝对改善至少 0.005；
- learned 方法必须同时优于 best single 和 averaging；
- 报告 MAE、naive skill、horizon-wise error、OOF error correlation、权重稳定性、成员胜率和训练成本；
- quantile 额外报告 pinball、coverage、width/Winkler；
- 不把同模型多 seed 作为首批成员；优先误差结构不同的基模型（见 `docs/redesign/kaggle_research.md` §5、§13）。

未达门槛不推广，但模块保留。

---

## 11. 明确不做

- 不恢复 `testing/forecasting` phase 字段或分支。
- 不恢复 legacy YAML、legacy reader/adapter、`forecasting/legacy`、`models/multistep`、迁移器或 invalidation manifest。
- 不把 BR 当第八种策略。
- 不支持 ensemble-of-ensemble。
- 不做 per-series/per-horizon 参数。
- 不允许成员突破共享 as-of 信息边界。
- 不允许 stacking 使用 in-sample/prefit/outer holdout 训练 meta model。
- 不支持 quantile stacking 或联合轨迹分布。
- **不支持 calendar_month 场景的 OOF `fold_count>1`（v4）**。
- **不为 stacking meta 引入截距、per-horizon 模型或可配置正则（v4）**。
- 不自动运行真实批次、删除 results、commit 或 push。

---

## 12. 完成定义

1. ForecastConfigSpec 只表示单模型；EnsembleConfigSpec 只表示引用式融合配置。
2. 所有融合实现位于顶层 `ensemble/`；旧两个模块仅重导出。
3. 单模型主链在 runner/splitter 抽取后行为逐值不变。
4. OOFArtifact 基于文件内容 SHA-256 安全复用，outer backtest 无泄漏；calendar_month 仅 `fold_count=1`。
5. 四种方法语义、概率边界和错误路径有独立测试；averaging 是硬基线。
6. Ensemble bundle 自包含成员模型和融合器；bundle 校验单模型/ensemble 分支清晰。
7. 24 配置完成精确引用式迁移，模型 YAML 总数为 848；parity 按 v4 分级口径验收。
8. checker/audit/tests/compile/diff 全绿，文档与当前纯 canonical 架构一致。
9. 效果结论只由 E8 的 A/B 同窗实验给出。

---

## Sources

[1] https://otexts.com/fpp3/combinations.html — Forecasting: Principles and Practice — Forecast combinations
[2] https://otexts.com/fpp3/tscv.html — Forecasting: Principles and Practice — Time series cross-validation
[3] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html — scikit-learn StackingRegressor
[4] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html — scikit-learn TimeSeriesSplit
