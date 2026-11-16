# 节假日数据工具与场景使用说明

## 入口与边界

[export_chinese_holiday_csv.py](<export_chinese_holiday_csv.py>) 从共享日历实现导出 CSV；`--start`、`--end` 均包含，`--output` 必填，默认日频，也支持日内频率。从项目根运行：

```bash
env -u PYTHONPATH .venv/bin/python scripts/holiday_data/export_chinese_holiday_csv.py --start 2025-01-01 --end 2026-12-31 --output dataset/shared/holidays/chinese_holiday_20250101_20261231.csv
```

输出路径相对调用者当前目录；可以用绝对脚本路径从其他目录调用。命令会覆盖指定输出文件，本次迁移不重建任何数据。

只归集独立节假日数据工具：[配置矩阵生成器](<../generate_load_15min_matrix.py>) 仍留在 scripts 根目录；在线生成器仍属于 data_loading，场景专用冷启动和评估脚本仍属于对应场景。

## 数据如何进入模型

1. YAML 显式声明 `source_type: generated`、`generator: chinese_holiday`、`availability: generator_defined`，所需列声明为 `known_future`。
2. [在线生成器](<../../data_loading/calendar_generator/chinese_holiday.py>) 调用 chinese-calendar，经[基础日历计算](<../../data_loading/calendar_generator/calendar_features.py>)生成每日属性，再映射到请求的预测时刻；同日所有 5min/15min 点使用相同属性。
3. [特征编译器](<../../feature_engineering/compiler.py#L1753-L1809>) 按目标时刻取已声明列，形成训练/预测特征并记录 VisibilityProof；不是目标值清洗或节日负荷倍率修正。
4. `available_at=forecast_origin` 表示 calendar-known 假设，未核验历史公告版本，不能当作真实历史发布时间证据。库不支持的日期报错，不退化为普通工作日。

[共享 CSV](<../../dataset/shared/holidays/chinese_holiday_20250101_20261231.csv>) 当前包含 2025–2026 年共730行，供审计或显式 file source 使用。2026-11-16 检索配置未发现引用该目录的 YAML；在线生成不依赖此 CSV。导出包含六项基础属性，但不包含在线按需扩展的四项春节属性，也不附带 available_at。

## 场景清点（2026-11-16）

以下数字是直接声明生成器的物理 YAML 数量，不是去重实验数、运行完成数或效果证据。基础两列指 `is_holiday` 和 `next_holiday_days`。

| dataset 场景 | 使用方式 | 声明数 |
|---|---|---:|
| aidc_load_15min_daily | A/B 两路 add_exogenous 的 holiday、holiday-weather 变体使用基础两列；不是所有实验组默认启用 | 36 |
| aidc_load_15min_rolling | 同上 | 36 |
| aidc_load_15min_short | 同上，按15min预测网格映射 | 36 |
| aidc_load_month | baseline_point、decomposition、experiments 及路线根最优副本使用基础两列；实际为日频，不是月度节假日计数 | 80 |
| aidc_hvac_load_5min | 原路线配置的 baseline/add_weather/add_endogenous_it/add_endogenous_route 共576份；A1_all 的 add_datetime_holiday 另18份；均为基础两列 | 594 |
| aidc_electricity_computility | 联通2026-08-31的 baseline、天气、优化、算力及消融组使用基础两列；不能泛化为所有算力子场景 | 71 |
| hongtaiyang_cesuan | 22份均用基础两列，其中8份需求负荷配置另加春节/命名节日/调休特征 | 22 |
| aidc_ess_selfuse_load | 当前配置未发现节假日 source；普通 datetime 特征不等同于法定节假日 | 0 |
| aidc_power_month | 当前配置未发现节假日 source，也未配置月度节假日聚合 | 0 |
| aidc_load_5min | 当前配置未发现节假日 source | 0 |
| shared | 共享资产目录，不是预测场景 | — |

宏太阳8份扩展配置额外声明 `is_spring_festival`、`is_named_holiday`、`days_to_spring_festival`、`days_after_spring_festival`、`is_adjusted_workday`。[春节特征实现](<../../data_loading/calendar_generator/named_holidays.py>)的距离针对当年春节，非负并截断到31天。[冷启动](<../../config/hongtaiyang_cesuan/scripts/cold_start.py>)还按过去普通日/春节负荷估计水平及节后恢复；[年度诊断](<../../config/hongtaiyang_cesuan/scripts/annual_reporting.py>)按春节、其他命名节日、普通休息日、工作日分组评分。两者不等同于普通模型特征开关，也不代表企业实际停复工计划。

## 容易误读的口径

- `is_holiday` 是休息日标记，包含普通周末；调休上班的周末为0。
- `next_holiday_days`/`prev_holiday_days` 是距休息日的距离，不是仅距春节/国庆；休息日为0，搜索不到时用删失哨兵400。
- `holiday_name` 仅命名节日非空；`solar_term` 为当前节气，非节气日继承上一节气。基础生成器输出这些列，不代表所有配置都使用。
- 月末/稀疏网格取对应日期状态，不会自动生成整月休息日数、工作日数或节日占比。
- 日历列明确启用才入模；新增列属于实验语义变化，本次只迁移脚本和修正文档，不修改模型 YAML、不重跑业务预测。
