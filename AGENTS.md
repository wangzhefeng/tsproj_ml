# AGENTS.md — tsproj_ml

本文件是项目级约定的单一事实来源，供所有 AI 编码工具（Codex、Claude Code、Hermes Agent、ZCode、Kimi Coding 等）共同遵守，与具体工具或模型无关。`CLAUDE.md` 只含 `@AGENTS.md` 引用以导入本文件；项目约定一律只改这里，不写入任何工具专属配置文件。

通用编码准则（先想后写、简单优先、最小改动、目标驱动）放在各工具的用户全局配置中，本文件有意不重复。

本文件只保留每个任务都需要的运行方式、核心不变量与参考文档索引；子系统细节按 §参考文档索引 按需查阅，不在本文件复制。

## 运行与验证

- 运行与测试统一使用根目录 `.venv/`（`.venv/bin/python`）；依赖唯一来源为 `pyproject.toml` + `uv.lock`，依赖变更用 `uv add`；测试不调用 uv、不设置 `UV_CACHE_DIR`。
- 统一入口：`env -u PYTHONPATH .venv/bin/python run.py --config-yaml <path>`（必须显式指定配置，无默认值）；单模型与引用式 Ensemble 按配置类型自动分派；批量执行用 `batch_run.py`。
- 常规修改运行 `env -u PYTHONPATH .venv/bin/python tests/run_suite.py fast` 并补相关定向测试；收口用 `all`。新测试默认纳入 integration，不得通过遗漏发现或 skip 提速；执行及覆盖映射见 `tests/README.md`。
- 日志目录 `logs/main/`；`LOG_NAME` 由 `os.environ.get('LOG_NAME', 'main')` 提供默认值。

## 核心不变量

- 只接受 `schema_version: 2` canonical YAML；无 legacy 层，非 canonical 输入（YAML/pkl/宽表）一律 RAISE。
- 预测输入严格 as-of；observed-past 列必须显式 provider，禁止隐式 persistence；进入模型的信息集「缺失/异常 = RAISE」，填补清洗只发生在离线数据准备或 config 驱动的 compiler 前置步骤。
- 包依赖只允许从上往下（L4 入口 → L3 能力扩展 → L2 编排 → `forecasting_core` 合同层 → 阶段顶层包 → L1 → L0），同层禁止互依，禁止函数内延迟 import 绕行与下划线私有跨包导入；门禁 `tests/test_package_layering.py`。
- canonical fingerprint 只取语义 payload；并行度、日志、输出目录不进 fingerprint。修改语义后适用新身份，不自动重跑、不删除存量结果。
- 结果目录统一 `results/{pretrained_models,results_test,results_forecast}/<scenario_subpath>/<result_identity>/`；identity 规则与 long 结果 schema 见 `model_forecasting/README.md`。
- target transform 顺序固定 `calendar normalization → decomposition → scaling`，point/quantile 严格逆序恢复，状态按 `(series_id,target)` 隔离。
- 回测与 final fit 使用同一配置训练窗口；融合成员 final fit 与对应单模型同一显式窗口。
- 模型静态描述唯一入口 `models/catalog.py`；构造参数严格校验，未知参数 RAISE，不静默丢弃、不静默降级。
- 改动被测模块必须同步修复测试导入与接口断言。

## 参考文档索引

按任务查阅，非常驻加载：

| 任务 | 先读 |
|---|---|
| 写/改配置 YAML、时间边界、低频与外生约定 | `config/README.md` |
| 动生命周期编排/监督设计/批调度 | `model_pipeline/README.md` |
| 动资源规划/性能档/checkpoint/运行缓存 | `model_performance/README.md` |
| 动特征编译/目标与特征变换 | `feature_engineering/README.md`、`feature_engineering/transforms/README.md` |
| 动目标分解/分量外推/通用周期诊断 | `decomposition/README.md`、`timeseries_analysis/README.md` |
| 动回测几何/逐折评分/回测产物 | `model_testing/README.md` |
| 动预测/部署/证据与 bundle 持久化 | `model_forecasting/README.md` |
| 动训练器/策略 executor/estimator 能力 | `model_training/README.md` |
| 动模型工厂/参数校验/别名 | `models/README.md` |
| 动 ensemble/OOF/融合方法 | `model_ensemble/README.md` |
| 动离线数据准备/事件标签工具链 | `data_process/README.md` |
| 动算力场景数据/预处理脚本 | `config/aidc_electricity_computility/electricity/2026-06-11/scripts/README.md` |
| 改测试结构/执行分层 | `tests/README.md` |

## 仓库维护注意

- **Agent 工作文档分流**：Hermes Agent 生成的实施计划只放 `.hermes/plans/`；Codex/Superpowers 生成的设计与计划只放 `.agents/superpowers/{specs,plans}/`；`docs/` 只保留面向项目使用者的长期有效文档。
- `tests/` 已纳入版本控制；AIDC 专项脚本目录路径含日期段、不是合法 Python 包，相关测试用 `sys.path.insert` 引导后按模块名导入。
- `docs/feature_engineering/` 实验脚本已删除（2026-09-06）：FFT/小波能力经 trailing 窗重写后接入 `advanced.fourier`/`advanced.wavelet`，周期诊断 EDA（含 FFT top-k 与 Engle-Granger 协整检验）并入 `data_process/periodicity_analysis.py`，历史内容从 Git 溯源。
- `utils/`（L0）只保留 frequency/log_util/runtime_env；指标在 `model_evaluation/`，评估掩码在 `model_evaluation/mask.py::build_eval_mask`。
