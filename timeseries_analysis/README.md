# timeseries_analysis

低层时间序列统计分析算法库，无项目包依赖；不读写文件、不整理 DataFrame、不决定 as-of 窗口。

- `periods.py`：FFT 主导周期/top-k、ACF 正相关局部峰、STL 趋势与季节强度。
- 周期单位为样本步；FFT 幅度保留原始谱幅值，不能与 feature_engineering 的归一化幅值混用。
- STL 周期无效或数值分解失败时返回未检出报告，这是离线诊断合同，不是模型输入清洗或运行时降级。
- 调用方：data_process 的离线周期报告与 decomposition 的残差频谱诊断；排序、缺失样本选择、时间单位换算由离线调用方负责。
