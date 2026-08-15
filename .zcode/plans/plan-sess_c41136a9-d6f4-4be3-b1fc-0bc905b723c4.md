# CLAUDE.md 改为 @引用存根 + 汇总分析报告

## 已确认的决策
用户选择：CLAUDE.md 使用「@引用存根」形态（非软链接）。内容只保留在 AGENTS.md。

## 实施步骤

1. **替换 CLAUDE.md**：删除当前软链接（`rm CLAUDE.md`），新建存根文件：简短说明「项目约定以 AGENTS.md 为单一事实来源，勿在本文件添加约定」+ `@AGENTS.md` 导入语法（Claude Code 原生支持，自动加载全文；Codex CLI 本就读 AGENTS.md）。
2. **同步 AGENTS.md 头部说明**：当前写的是「CLAUDE.md is a symlink to this file」，改为「CLAUDE.md contains only an @AGENTS.md reference」。
3. **验证**：`git status` 确认（AGENTS.md: typechange symlink→普通文件；CLAUDE.md: 内容 modified），`cat` 确认存根内容与 AGENTS.md 正文完整。不主动 commit。

## 分析结论（第 1、4 点，沿用已完成的核实，仓库自上轮无变化）

- **内容核实**：AGENTS.md 全部事实性声明逐条对照代码均一致（CONFIG_YAML 指向、freq `1D` 映射、conformal/eval_mask 默认值、df.csv 215 列、2026-08-10 空目录与 YAML data_dir 指向、blend Ridge 参数、main.py 硬校验规则）；「tests/ 已纳管」表述与 HEAD 提交 dff3eb5 一致；测试时长声明已实测验证（63 tests，12.2s，全绿，符合「12~16s」）。v1 硬限制条目如实反映 main.py 校验，属有效约束清单，保留。正文不做内容修改。
- **tests 覆盖**：63 个测试全绿；配置系统/AIDC 脚本/异常双路径/评估掩码覆盖较好；缺口集中在 models/ 层——ModelTraining 各分支（ensemble/blend/quantile/horizon_feature、DirectMultiOutputRegressor 权重转发）、Forecaster 九方法真实推理、ModelFactory 十类模型、FeatureScalering 逆变换、quantile/frequency 纯函数、端到端冒烟均零覆盖。补测优先级：① ModelFactory 冒烟 ② Forecaster 九方法冒烟 ③ Trainer bundle 结构断言 ④ 纯函数 ⑤ 端到端小样本。