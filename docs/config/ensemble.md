# 引用式 Ensemble 配置合同

完整生命周期入口为 `run.py --config-yaml <ensemble.yaml>`；成员引用相对该 YAML 解析，数据路径相对运行目录解析。批量单模型入口不自动支持 Ensemble。

## 方法与参数

| method.name | params | 含义 |
|---|---|---|
| `averaging` | 空 | 成员等权；支持 point/quantile |
| `weighted` | `metric: rmse\|mae\|mape`，`weight_scope: target\|target_horizon` | 按 OOF 逆误差归一化；默认 rmse、target |
| `linear_blending` | `weight_scope: target\|target_horizon` | point 保留 NNLS 后归一化；quantile 最小化 simplex pooled pinball |
| `stacking` | 空 | 仅 point；逐目标标准化、中心化后 Ridge(alpha=1)，不开放任意 estimator 参数 |
| `adaptive_weighted` | `halflife_days` 必填正有限数；`metric`、`weight_scope` 同 weighted | 按 OOF 标签结束时间衰减后的逆误差；在每个拟合原点重估并冻结 |

各分位共享相应 target 或 target×horizon 权重，不分别学习分位权重。`mape` 使用非零分母下限，近零目标场景需谨慎选择。

动态权重示例（仅方法片段，数值仅为示意）：

```yaml
ensemble:
  method:
    name: adaptive_weighted
    params:
      metric: mae
      weight_scope: target_horizon
      halflife_days: 7.0
```

时间权重与 `2 ** (-(origin - label_end).days / halflife_days)` 成比例（实现用精确秒数换算天，并平移共同年龄避免下溢）。仅使用当前原点已完整可得的 OOF 标签；Global 各序列同折使用相同时间权重。bundle 保存原点、标签结束时间、样本权重和最终融合权重；部署不接收新真值、不在线更新，拒绝预测到拟合原点或之前。

## 窗口与共享合同

- `ensemble.oof.{train_window_steps,fold_count,stride_steps,gap_steps}` 控制内层 OOF；每个外层原点仅使用其之前的完整 OOF 标签，折数不足直接报错。
- 顶层 `validation.{history_steps,train_window_steps,fold_count,stride_steps}` 控制外层评估几何；每个成员实际外层拟合和 final fit 使用该成员自己的 `train_window_steps`，内层则统一使用显式 OOF 窗口。
- 顶层 `forecast_origin/schedule_mode` 拥有调度语义；显式 `training_scope` panel 策略向成员传递。不同 lag 允许不同预热和训练候选集，但共享 holdout 时间、series/target 顺序及 quantile grid。
- 顶层 `eval_mask/aggregate_weighting/seasonal_naive_lag` 用于外层评分；图保留未掩码真值。meta-train 诊断不是外层测试成绩。
- 成员 source 必须是顶层共享 source 的 canonical 同语义子集（包含生成器选项、可用时间等），不能仅同名同路径。
- 顶层 `point_quantile/crossing` 决定融合后的分布；不能借首成员规格。融合层 CQR 用外层留出残差，成员不能再声明独立 calibration。

## 显式拒绝的组合

未知键/参数、嵌套或 merge 后重复 YAML 键、ensemble-of-ensemble、quantile stacking、point absolute_residual、自然月等非固定步长 horizon、非默认 refit_every、train_history_steps、training_window/forecast_window、output.overlay 均拒绝。

顶层不接受 `validation.training` 或 `train_outlier`；成员 training 仅接受已有消费链的 sample_weight/origin_sampling，其他休眠选项拒绝。成员 seasonal_baseline 以及原生历史模型不支持本融合 OOF/部署链，不隐式降级。

## 产物与兼容性

- `resolved_config.json` 分别记录 YAML document fingerprint 与解析后运行 fingerprint；后者绑定成员语义、实际原点和融合内部语义盐。
- 标准测试表/逐窗图来自独立 outer holdout；`meta_train_*` 仅诊断融合器拟合样本，不与旧版本测试成绩直接比较。
- 语义变化使用新身份，不迁移、不删除旧结果；OOF 内部版本由 `model_ensemble/outputs/cache.py` 维护（当前 4，子包迁移不升级）。部署只依赖已保存 bundle 与显式特征输入。Direct 的特征若逐步变化，也必须提供逐调用 feature provider，不能将第一步设计当成所有步的设计。
- `run_state.json` 是完成证据而非跨目录事务；同身份并发写结果不受支持。缓存写入中断可重试，已发布损坏条目明确拒绝。
- 独立外层评估会增加训练量；合成测试和 CLI 小样本验收不代表业务预测精度或正式场景已重跑。
