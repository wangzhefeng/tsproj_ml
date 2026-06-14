# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Project Conventions

### 配置系统
- YAML 配置：`base_config` 指向 Python 模块（默认 `config.univariate_config`），`overrides` 为扁平化键值覆盖
- 修改 YAML 后务必运行验证：配置值可能被回退（已知问题：`true` 会变成 `false`）

### 时间边界约定
- `now_time` 配置值 = 最后一个已知数据点（如 `2026-06-11T23:55:00`）
- 内部分界点 = `floor("1D") + 1day` = 次日 00:00:00
- `start_time = now_time - history_days`，`future_time = now_time + predict_days`
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——所以终日 23:55 是最后一个被包含的点

### 预测方法限制
- **USMDP + advanced_features 不兼容**：`add_rolling_statistics` 和 `add_diff_features` 依赖目标列 `y`，训练时存在但预测时（未来数据）不存在 → LightGBM Fatal: feature count mismatch
- 可通过将操作列改为外生特征列来启用，但默认配置下必须禁用

### 性能并行约定（PerformanceConfig）
- **窗口并行与内部并行不叠加**：`window_parallel_workers > 1` 时，滑窗测试会把每个窗口 payload 的 `multi_output_n_jobs`、`model_thread_count` 强制改写为 1（见 `main.py` 的 `Model.test`）；故测试阶段总并行度≈`window_parallel_workers`。内部参数只在最终 forecast 的单模型上生效（forecast 用原始 `args`）——所以 `is_testing=true` 的场景可同时设高窗口并行与内部并行而不超额订阅。
- **方法→有效杠杆**：多输出方法（USMD/USMDR/MSMD/MSMDR，走 `MultiOutputRegressor`/`DirectMultiOutputRegressor`）调 `multi_output_n_jobs`；单输出方法（USMDP/USMR/MSMR）只 `model_thread_count` 有效。
- 容量上限：`multi_output_n_jobs × model_thread_count ≤ 物理核数`，优先「多并行单线程」。本机 M2 共 8 核（4 性能 + 4 能效）。

### 关键约定
- 输出目录：`results/`（不是 `saved_results/`）；三个子目录（pretrained_models/results_test/results_forecast）按 `<scenario>/<setting>` 嵌套，`<scenario>` 由 `data_dir` 解析得到（与 config 路径对齐），不同场景互不覆盖
- 测试汇总指标：使用 median（中位数），不是 mean——单窗口 MAPE 爆炸会拖垮均值
- `MAPE Accuracy` 业务口径：按 `eval_mask` 掩码过滤后计算（默认 `mode: percentile` 即窗口正样本 `P5`；`absolute`/`combined` 模式启用 `min_value` 绝对下限，`max_value` 上限与 mode 正交、三模式均生效）；阈值落 `test_scores_df.csv` 的 `MAPE Threshold`/`MAPE Upper Threshold` 列，无有效点写 `NaN`
- `prediction.png` 按 `eval_mask` 掩码历史上下文异常点（断线）；未来预测原值保留在 `prediction.csv` 和 `prediction_plot_concat.csv`
- `EvalMaskConfig`（评估/绘图掩码，不改数据）≠ `TrainOutlierConfig`（滑窗训练窗口清洗，插值改数据）
- 本地默认入口：修改 `main.py` 中的 `CONFIG_YAML`，再直接运行 `uv run python main.py`
- `run.py` 保留为兼容入口，但当前文档和脚本不再把它作为推荐运行方式
- 日志目录：`logs/main/`（直接运行 `main.py`）
- `log_util.py` 在模块导入时执行，`LOG_NAME` 用 `os.environ.get('LOG_NAME', 'main')` 提供默认值
- 在这台机器上如果 `uv` 触发 `~/.cache/uv` 权限问题，优先使用 `UV_CACHE_DIR=.uv_cache`
