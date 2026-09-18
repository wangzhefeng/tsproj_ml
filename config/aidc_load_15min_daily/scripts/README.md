# aidc_load_15min_daily 场景天气适配

本目录保存场景级离线入口，公共天气算法和源缺口治理不在本目录重复实现。
`prepare_weather.py` 只读取共享处理后小时天气，按本场景的15min频率与固定历史窗口（2025-01-01至2026-08-31）生成资产，不扩窗、不训练模型、不修改模型YAML。

```bash
# 共享源新增或修补规则变化时先构建一次；原始 extracted/ 不回写
env -u PYTHONPATH .venv/bin/python scripts/build_scenario_weather.py --output dataset/shared/weather/processed/weather_hourly.csv
# 只更新本场景的派生天气
env -u PYTHONPATH .venv/bin/python config/aidc_load_15min_daily/scripts/prepare_weather.py
```

输入默认 `dataset/shared/weather/processed/weather_hourly.csv`，可用 `--processed <CSV>` 指定；要求同名 `.meta.json` 并核对CSV哈希。输出：`dataset/aidc_load_15min_daily/weather_history_15min_20250101_20260831.csv`，同名metadata记录场景入口及共享资产的CSV/metadata哈希。重复运行重写派生产物，不删除其他文件；从任意工作目录用绝对脚本路径运行均可。

处理方式：5min/15min为小时内hold；日/月为完整覆盖聚合，实测不完整拒绝，预报不完整保留NaN供请求级RAISE。窗口与聚合字段由本场景入口定义。训练实测/历史预测预报映射不变，离线插值/再分析不等于真实发布时间证据。

构建某个场景不会执行共享源修补，也不会写其他场景目录。公共合同和验证见仓库 `scripts/README.md`。
