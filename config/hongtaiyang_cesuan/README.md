# hongtaiyang_cesuan — 2025 年滚动回测

## 目录约定与验收口径

- `prepare.py`：严格检查三个原始 CSV，按完整自然日 mean 生成两个日均负荷 CSV，不填补。
- `generate_configs.py`：生成完整物理模型 YAML。
- `<site>/<target>/freq_<frequency>/lgbm_<method>.yaml`：五个任务 × direct/direct-pointwise/direct-pointwise-horizon/mimo，主运行矩阵20份canonical配置。recursive配置与对应结果已按授权删除；通用recursive实现仍供其他场景使用。实验配置单独放 `experiments/`，不混入主矩阵。
- 三个direct布局均保留 `align_to_target: false` 和原有lag，历史特征冻结在原点；pointwise共享一个模型、无horizon特征，pointwise-horizon增加 `forecast_horizon_idx`。不照搬其他场景的目标时刻对齐，否则本场景浅lag会请求未来目标值。
- `annual_backtest.py`：本场景年度回测入口，复用 canonical 数据注册表、特征编译、模型工厂、训练器和策略 executor；不改全局生命周期。单独 `run.py` 只执行 YAML 的参考窗口，不代表全年回测。
- 原始 CSV 不修改；日频数据写入 `dataset/hongtaiyang_cesuan/<site>/freq_1day/demand_load.csv`。

## 原始数据可视化

`plot_raw_data.py` 提供绘图函数和独立 CLI。直接读取两个站点的三个原始15min CSV，每个序列生成全年总图和12个月分面图；全部使用原始点，不聚合、不平滑、不填补、不修改CSV。纵轴保留源数据数值，不在缺少单位元数据时擅自标注kW。

输出目录为 `dataset/hongtaiyang_cesuan/<site>/visualization/`，文件名为 `<target>_2025_raw.png` 和 `<target>_2025_raw_monthly.png`，共6张图。

```bash
env -u PYTHONPATH .venv/bin/python config/hongtaiyang_cesuan/plot_raw_data.py
```

此脚本只画原始数据，不训练模型、不改变年度结果。预测效果图由年度回测入口通过 `annual_reporting.py` 接入通用 `model_testing.reporting.write_backtest_results()` 生成，不在原始数据目录混放。

## 时间与训练

- 15min 数据按原生频率，每天预测次日 96 点，训练和全部历史特征只允许此前 30 天原始数据。
- 日频目标是日均负荷，不是最大需量。每个自然月月初一次预测整月；2 月使用 1 月最后 30 天，3 月使用 1—2 月，以后扩展到全部已知历史。
- 2月不再执行递归。`annual_recipe.json` 明确选用日历条件基线：普通日参考水平由原点前非命名节日、且负荷不低于该历史中位数0.5倍的子集估计；按星期取中位数，样本不足时取该子集总体中位数。春节参考水平取原点前已观测春节日的中位数。仅基线估计筛选参考样本，不删除或填补原始数据。
- 官方春节结束后3天线性过渡至普通日参考水平，是经过本年对照选择的预测假设，不是企业真实复工计划。配方和代码身份写入结果审计；绝不从未来真值推断复工时间。3月起恢复各配置声明的LGBM策略与布局。
- `active_1day_profile.json` 保存逐站点、逐方法的采用设置。guangdianchang的两个pointwise采用L2、命名节日数值特征和30天半衰期权重；xinnengyuan的plain pointwise保留L1，采用命名节日特征和60天权重。其他方法的常规月份参数恢复基线，只使用改进后的2月冷启动。
- 时间权重按监督原点年龄计算并归一化到均值1，真实传给 `CanonicalTrainer.train(sample_weight=...)`。保持原始历史逐月扩展，不在2月基线阶段套用时间衰减。该配置由本场景年度入口消费，不宣称其他生命周期入口已经接入权重。
- 日频使用小树、较小 min_child_samples，允许短样本学习；不表示已经证明准确度。特征：target lag、trailing rolling/expanding、datetime、中国节假日生成器。
- 1 月预测值等于真实值，不增加来源标记列，纳入全年评分。因此全年分数包含人为零误差区间，不代表纯模型样本外表现。

## 产物与运行

从仓库根使用 `.venv/bin/python`。年度程序必须显式选择 `--freq 1D` 或 `--config-yaml`，避免误启动15min长任务；`--rerun` 强制重新拟合各窗口，不复用旧窗口CSV。`--max-windows` 仅用于冒烟验证。

```bash
env -u PYTHONPATH .venv/bin/python config/hongtaiyang_cesuan/prepare.py
env -u PYTHONPATH .venv/bin/python config/hongtaiyang_cesuan/generate_configs.py --freq 1D --profile config/hongtaiyang_cesuan/active_1day_profile.json
env -u PYTHONPATH MPLBACKEND=Agg .venv/bin/python config/hongtaiyang_cesuan/annual_backtest.py --freq 1D --rerun
```

结果在 `results/results_test/hongtaiyang_cesuan/<site>/<target>/freq_<frequency>/<result_identity>/annual_<recipe_identity>/`。

- `windows/*.csv`：每个预测日/月的断点产物；只复用与配置、数据内容、实现身份一致的结果，损坏直接拒绝。
- `cv_plot_df.csv`：通用canonical long回测结果，是结果图的数据来源。
- `test_prediction.png`：全年真值/预测对比总图；`windows_results/window_00.png` 为1月真值接入图，`window_01.png` 至 `window_11.png` 为2—12月预测对比图。
- `test_scores_df.csv`、`test_scores_horizon_df.csv`、`result_metadata.json`：通用逐窗/逐步评分及结果元数据。
- `annual_scores_df.csv`：通用评分器对全年逐点汇总的指标，不能以不同自然月RMSE的平均代替全年RMSE。MAPE按通用规则排除真值为零的点。
- `diagnostic_scores_df.csv`：全年、2—12月、逐月、日历类型及事后高低负荷分组的Bias/MAE/RMSE/MAPE；事后分组不进入模型。
- `prediction.csv`：额外保留三列年度交付表 `time,y_true,y_pred`，不替代canonical结果；1月也包含在内。
- `scores.json`：额外保留全年MAE/RMSE摘要，数值来自通用评分器。
- `audit.json`：配置/数据/实现身份及每窗训练边界、策略与样本量；不把真实值回填段伪称模型预测。
- `status.json`：区分 running/partial/failed/completed。中断后以原命令重启，复用已完整写入且校验通过的窗口；中断窗口重算。

完整验收入口（未完成时退出码 1，不把已有结果交集当作全量完成）：

```bash
env -u PYTHONPATH .venv/bin/python config/hongtaiyang_cesuan/verify_results.py --freq 1D
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p test_hongtaiyang_cesuan.py
```

每次拟合采用单线程，配置串行执行；不要同时启动重复的年度任务。除已列明的损失、日历和权重消融外，未进行大规模超参数搜索；不使用未来真值校准或未授权裁剪。

当前执行范围：四方法×两站点，共8份1D全年结果，各365行、13张图；1月等于真值并计分，其余不以未来真值兜底。后续清理已按授权删除31个旧基线、未采用候选和诊断结果目录，仅保留当前8份正式结果、104张图及最新结果包。此入口是历史回测，不导出部署bundle，也不宣称完成实盘部署。

## 优化与条件推广

`optimize_1day.py --stage cold|screen|combined` 运行受控对照，设计见 `experiments/README.md`。2月分别比较非递归短样本pointwise和日历条件基线；L2、命名节日、权重先独立测试再组合。`compare_optimization.py` 读取保留基线与新结果做全年复核，不训练模型。

上述实验工具依赖本地历史基线清单和逐点结果；旧结果清理后不能直接重新计算历史比较，需先恢复对应实验输入。保留的汇总报告仅供查阅，不冒充可重算的原始证据。日常运行与验收使用annual_backtest.py和verify_results.py，不依赖已删除的旧基线。

采用门槛：3—12月MAE至少改善2%，RMSE不恶化超过2%，且全年MAE改善。不能仅靠2月收益推广常规设置。全年复核未过门槛的5个配置已恢复常规参数并重跑，只有3个pointwise配置通过。

`active_15min_profile.json` 仅同步这3个对应的15min负荷配置；其他9份非递归15min配置内容不变，包括全部光伏配置。日频证据不等于高频效果证据；本轮没有运行15min模型，也不把月频春节冷启动套到高频任务。

```bash
# 仅生成已经过日频门槛的配置，不启动15min训练
env -u PYTHONPATH .venv/bin/python config/hongtaiyang_cesuan/generate_configs.py --freq 15min --profile config/hongtaiyang_cesuan/active_15min_profile.json
```

诊断月份/春节参数的选择与全年复核都使用本年数据，改善属于当前历史回放证据，不能宣称独立跨年泛化。

## 15min耗时评估（不运行模型）

原始15min计时结果已按授权删除。`estimate_15min.py --daily-report <当前8份日频验收JSON>` 仅展示冻结历史报告中仍保留方法的旧估算，明确标记historical_only，不再读取已删除的recursive配置，也不重新估算当前优化配置的耗时。

历史估算保存在 `.hermes/plans/hongtaiyang-15min-estimate.json`，不能代替当前配置实测。本轮消融证据为 `hongtaiyang-opt-{cold,screen,combined}.json`，最终验收与对照为 `hongtaiyang-opt-final-{verification,comparison}.json`（均在 `.hermes/plans/`）。未启动新的15min业务模型测试。

`dataset/`、`results/`及`.hermes/plans/`遵循仓库忽略规则，不随代码提交发布；新克隆需另行取得数据与需要的本地报告，不能将Git推送视为数据或结果交付。
