# extrapolation

分量拟合与未来外推。`trend.py` 拟合多项式趋势及 damped 近期斜率，`seasonal.py` 从真实历史季节分量拟合相位模板。所有参数由 preset 显式传入，不接收 extractor 对象或动态注入的拟合状态。
