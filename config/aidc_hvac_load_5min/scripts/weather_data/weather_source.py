"""HVAC 场景天气源合同：直连共享 processed 5min 资产，不落场景内拷贝。"""
from pathlib import Path

PROCESSED_WEATHER_5MIN = 'dataset/shared/weather/processed/weather_5min.csv'

WEATHER_INFERENCE_COLUMNS = {'rt_tt2': 'pred_tt2', 'cal_rh': 'pred_rh', 'rt_ssr': 'pred_ssrd',
                             'rt_ws10': 'pred_ws10', 'rt_ps': 'pred_ps', 'rt_rain': 'pred_rain'}


def weather_source():
    """file 两段制天气源：训练读 rt_ 实测列，推理按 inference_columns 换 pred_ 预报列。

    availability=forecast_origin 是可得性假设，不是供应商发布时间证据；窗口对齐由
    装载层按目标请求网格切片并要求精确覆盖，绑定核验见 bind_weather.py。
    """
    return {
        'name': 'weather', 'source_type': 'file',
        'history_path': PROCESSED_WEATHER_5MIN,
        'time_col': 'ts', 'series_id_cols': [], 'availability': 'forecast_origin',
        'columns': [{'name': column, 'role': role, 'categorical': False}
                    for role, names in (('known_future', WEATHER_INFERENCE_COLUMNS),
                                        ('ignored', WEATHER_INFERENCE_COLUMNS.values()))
                    for column in names],
        'inference_columns': dict(WEATHER_INFERENCE_COLUMNS),
    }
