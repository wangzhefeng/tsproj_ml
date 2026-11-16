# aidc_aidc_ess_selfuse_load 场景天气绑定

本目录保存场景级离线入口，公共天气算法和源缺口治理不在本目录重复实现。
模型配置经 `scripts/build_scenario_weather.py::weather_source` 直连共享资产 `dataset/shared/weather/processed/weather_5min.csv`（小时气泡内 hold，六项列合同）；场景内不再生成天气拷贝。绑定核验入口为仓库根 `scripts/bind_weather.py --scenario aidc_ess_selfuse_load`，逐目标核验 5min 网格覆盖与六项有限性，发布 `dataset/aidc_ess_selfuse_load/analysis/weather_binding/manifest.csv`。

```bash
# 共享源新增或修补规则变化时先构建一次；原始 extracted/ 不回写
env -u PYTHONPATH .venv/bin/python scripts/build_scenario_weather.py --output dataset/shared/weather/processed/weather_hourly.csv --resample 15min 5min
# 核验绑定并发布清单（重建需 --replace）
env -u PYTHONPATH .venv/bin/python scripts/bind_weather.py --scenario aidc_ess_selfuse_load
```

要求共享资产同名 `.meta.json` 并核对CSV哈希。训练实测/历史预测预报映射不变，离线插值/再分析不等于真实发布时间证据。
