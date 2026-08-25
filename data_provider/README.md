# data_provider 模块说明

`data_provider/` 负责把原始 CSV 转成主流程可用的历史/未来时间序列数据，并提供滑窗测试训练段异常处理。

## 文件职责

| 文件 | 职责 |
|---|---|
| `data_loader.py` | 读取目标、日期、天气 CSV；构造单序列或 global panel 历史/未来时间轴；统一目标列为 `y` |
| `outlier_handling.py` | 滑窗测试训练段异常处理，不修改测试真实值 |

离线/批处理脚本（频率聚合 `data_aggregate.py`、填充回测 `fill_method_backtest.py`、离线异常清洗 `outlier_process.py`、算力预处理 `computility_process.py` 等）统一放在 `data_process/` 和 `config/aidc_electricity_computility/electricity/2026-06-11/scripts/`。

## DataLoader 输入契约

核心字段来自配置实例：

- `data_dir`：数据目录，`main.Model.__init__` 会转成 `Path`。
- `data_path`：目标序列 CSV。
- `target_ts_feat`：目标序列时间列。
- `target`：目标值列，历史处理后映射为 `y`。
- `freq`：固定频率，例如 `5min`、`15min`、`1h`。

可选外生文件：

- 日期：`date_history_path`、`date_future_path`、`date_ts_feat`
- 天气：`weather_history_path`、`weather_future_path`、`weather_ts_feat`
- 自定义外生注册表：`custom_features`（多文件来源，每项 `{name, history_path, future_path, ts_col, columns, categorical_columns}`，历史/未来列名一致）

日期和天气外生文件同时存在历史/未来版本时，会先按时间列纵向拼接、排序、去重形成 canonical 全量表，再按 `forecast_start_time` 显式切成历史片段和未来片段供两个阶段分别使用。自定义外生特征走同一切分机制，特征合并（按精确时间戳 left merge）在 `features/FeatureEngineering.py` 的 `extend_custom_feature` 完成。

## 历史与未来边界

`main.Model` 计算时间窗口后传给 `DataLoader`：

- 历史区间：`[now_time - history_length, now_time)`
- 预测区间：`[now_time, now_time + predict_steps × freq)`

`process_history_data()` 会按历史区间构造模板并映射真实目标；`process_future_data()`
只构造未来模板和外生特征，不读取未来真实目标。

## Global panel 契约

启用 `enable_global_training=true` 时：

- `series_id_feature` 指定实体列，历史和未来主键变为 `(series_id, time)`；同一时刻允许不同序列并存，但复合键重复直接失败。
- 每条序列独立对齐完整历史时间轴；缺失策略由 `global_incomplete_series_policy` 控制，`raise` 硬失败，`drop` 整条排除，不跨序列填值。
- 未来模板按训练期已知 series ID 展开为每条序列各 H 行；外生时间索引先去重，再由预测层逐序列消费。
- 未见过的 series ID 由模型产物中的 panel metadata 校验，当前 `global_unknown_series_policy` 只支持 `raise`。

滑窗分组、递归状态隔离和逐序列执行位于 `models/multistep/panel.py`；DataLoader 只负责复合键数据物化，不解释具体推进策略。

## 异常处理边界

`outlier_handling.py` 用于滑窗测试训练段：

- 默认关闭。
- 只作用于每个测试窗口的训练片段。
- 输出 `train_outlier_report.csv` 所需的标准列。
- 不改变测试段真实值，避免评估泄漏。

离线原始负荷异常清洗（`data_process/outlier_process.py`）的输入输出说明见根 [README.md](/Users/wangzf/projects/tsproj_ml/README.md)「输出结构」一节。
