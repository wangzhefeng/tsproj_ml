# model_tuning

自动调参是 L3 实验编排能力：`tune.py` → `model_tuning/` → 现有 `pipeline/`，不另建训练、回测、评估或部署实现。当前支持单模型 point、fixed-step、可执行完整生命周期的严格预测配置；不支持 Ensemble、研究回放、raw-history/backtest-only 模型。

## 结构约定

- `specs.py`：独立搜索配方的严格解析、Optuna 分布、候选 canonical 配置生成。
- `runtime.py`：串行 trial 生命周期、固定验证窗口、独立 holdout、最优 YAML 导出、失败记录与产物验收。
- `__init__.py`：包说明，不 re-export。
- 根 `tune.py`：CLI；模型配置仍由公共 canonical loader 解析。

## 配方合同

配方与模型 YAML 分开；必须显式给出 `trials`、`seed`、`metric`（MAE/RMSE/MAPE，按折 aggregate 等权平均，最小化）、`holdout_origin`、`holdout_fold_count`、`parameters`。未知字段、重复 YAML 键、非有限范围及非法参数路径直接 RAISE。

参数路径仅允许 `estimator.params.<name>` 或既有 `features` 叶子路径；不允许更改数据、时间几何、评估 mask、模型类型或输出路径。分布为 `int`/`float`（low/high，可选 log）或 `categorical`（choices，可含列表型特征组合）。每个候选经 canonical parser、特征与模型运行时校验，非法候选记录失败，不伪造大损失。

示意配方（日期和特征名必须匹配实际数据；不是活动场景的推荐参数）：

```yaml
trials: 10
seed: 0
metric: RMSE
holdout_origin: '2026-01-03T23:00:00'
holdout_fold_count: 1
parameters:
  estimator.params.num_leaves:
    type: int
    low: 7
    high: 31
  features.target_lags.load:
    type: categorical
    choices: [[2, 3], [2, 3, 4]]
```

## 执行与产物

入口：`env -u PYTHONPATH .venv/bin/python tune.py --config-yaml <base.yaml> --search-yaml <search.yaml> --study-dir <new-directory>`。

study 目录必须不存在；候选、holdout/final 和导出配置使用独立输出子空间，禁止覆盖已有 study。产物结构保留 `pretrained_models/results_test/results_forecast/<scenario>/<identity>`。搜索失败也保留已完成的 trial 记录。

搜索只使用基准配置的显式 forecast_origin 及其以前的信息。holdout 的首个预测原点必须不早于搜索截止时间；holdout 评分不参与 Optuna 选优。候选实际回测窗口必须与基准几何一致，且训练样本窗完整，不允许通过改变特征预热长度缩短评分窗口获利。

`best.yaml` 为完整 canonical 配置，不依赖搜索配方，可单独经 `run.py` 执行；其输出命名空间与本次已验证的 final 产物分离。完成必须验证最终模型、回测、预测、配置 fingerprint、完成状态和文件摘要，不能仅以最优 trial 存在宣称完成。

Optuna 使用带 seed 的 TPE；首版串行、无 pruning/恢复旧 study。依赖现有 raw-design 内容寻址缓存复用合法设计，不新建平行缓存；模型自身随机性由 estimator 参数控制，搜索 seed 不冒充模型随机种子。

## 验证边界

真实执行采用合成 Ridge/LightGBM 样例，证明搜索、YAML 重载、独立 holdout、final bundle 与预测链闭环；不代表业务精度收益。活动配置、存量结果和依赖锁文件不因调参能力接入而改写。
