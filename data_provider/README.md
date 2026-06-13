# data_provider 模块说明

`data_provider/` 负责把原始 CSV 转成主流程可用的历史/未来时间序列数据，并提供两类异常处理工具。

## 文件职责

| 文件 | 职责 |
|---|---|
| `data_loader.py` | 读取目标、日期、天气 CSV；构造历史/未来时间轴；统一目标列为 `y` |
| `outlier_handling.py` | 滑窗测试训练段异常处理，不修改测试真实值 |
| `outlier_process.py` | 离线负荷数据异常标记、清洗和图片输出 |

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

日期和天气外生文件同时存在历史/未来版本时，会拼接为统一表后供历史和未来阶段使用。

## 历史与未来边界

`main.Model` 计算时间窗口后传给 `DataLoader`：

- 历史区间：`[now_time - history_days, now_time)`
- 预测区间：`[now_time, now_time + predict_days)`

`process_history_data()` 会按历史区间构造模板并映射真实目标；`process_future_data()`
只构造未来模板和外生特征，不读取未来真实目标。

## 异常处理边界

`outlier_handling.py` 用于滑窗测试训练段：

- 默认关闭。
- 只作用于每个测试窗口的训练片段。
- 输出 `train_outlier_report.csv` 所需的标准列。
- 不改变测试段真实值，避免评估泄漏。

`outlier_process.py` 用于离线处理原始负荷文件：

- 输入通常是一个 `df_power.csv`。
- 输出保存在源数据目录。
- 标记结果：`df_power_outlier_detection.csv`
- 清洗结果：`df_power_remove_outlier.csv`
- 图片结果：`df_power_anomalies.png`

离线清洗输出的文件名可变更，但 CSV 内容契约不应随意改变。
