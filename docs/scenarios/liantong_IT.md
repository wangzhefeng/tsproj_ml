# 联通 IT 四组回测结果（历史溯源）

> 联通场景已随 2026-09-25 收敛退役，本文为其存量回测结果的溯源记录，自 `results/results_test/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/README.md` 迁入。

本目录与 `config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/` 的四组配置对应。

## 本次整理后的存量结果

| 分组 | 配置数 | 已有完整回测结果 |
| --- | --- | --- |
| baseline | 10 | 0，尚未运行 |
| baseline_opt | 9 | 0，尚未运行 |
| add_weather | 9 | 3：Direct、RecMO、MIMO |
| add_weather_opt | 9 | 0，尚未运行 |

`add_weather/` 下的结果目录：

- `direct-lightgbm-local-k1-faa2eee1f750/backtest_only/`
- `recmo-lightgbm-local-k1-9396699425c3/backtest_only/`
- `mimo-lightgbm-local-k1-4a6d6b618279/backtest_only/`

三套结果均为原有六项天气、未优化配置的历史回测。每套包含 17 个窗口、4,896 条逐点预测，`run_state.json` 为 completed；不代表 final fit 或部署模型已完成。

## 迁移与输出合同

本次按用户授权，将原来直接位于本目录的三个 result_identity 目录整体迁入 `add_weather/`。完整 config fingerprint 与当前对应配置一致，identity 不变；迁移前后逐文件 SHA-256 一致（每套24个文件，共72个文件），预测、评分、图片及原始元数据均未改写。

其余三个分组只建立空目录，不复制旧结果、不伪造已完成模型。没有重跑配置、修改天气或清理结果。

37份配置的 `output.scenario_subpath` 已包含组名，常规运行会按以下结构写入：

`results/results_test/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/<group>/<result_identity>/backtest_only/`

本场景没有 pretrained_models 或 results_forecast 存量产物，因此本次未创建虚假的训练模型/未来预测目录；共享 `_compiled_features/` 缓存及其他场景未移动。
