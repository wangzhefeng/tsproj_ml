# AGENTS.md

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
- `lgbm_usmdo.yaml` 是 A1_01a 电负荷场景的当前配置
- 修改 YAML 后务必运行验证：配置值可能被回退（已知问题：`true` 会变成 `false`）

### 时间边界约定
- `now_time` 配置值 = 最后一个已知数据点（如 `2026-06-11T23:55:00`）
- 内部分界点 = `floor("1D") + 1day` = 次日 00:00:00
- `start_time = now_time - history_days`，`future_time = now_time + predict_days`
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——所以终日 23:55 是最后一个被包含的点

### 预测方法限制
- **USMDO + advanced_features 不兼容**：`add_rolling_statistics` 和 `add_diff_features` 依赖目标列 `y`，训练时存在但预测时（未来数据）不存在 → LightGBM Fatal: feature count mismatch
- 可通过将操作列改为外生特征列来启用，但默认配置下必须禁用

### 关键约定
- 输出目录：`results/`（不是 `saved_results/`）
- 测试汇总指标：使用 median（中位数），不是 mean——单窗口 MAPE 爆炸会拖垮均值
- 日志目录：`logs/main/`（直接运行 main.py）或 `logs/run/`（通过 run.py）
- `log_util.py` 在模块导入时执行，`LOG_NAME` 用 `os.environ.get('LOG_NAME', 'main')` 提供默认值
