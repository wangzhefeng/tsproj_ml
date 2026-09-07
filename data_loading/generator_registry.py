"""跨能力内建生成器目录；具体请求上下文由 SourceRegistry 绑定。"""
from data_loading.calendar_generator import BUILTIN_GENERATORS as CALENDAR_GENERATORS
from data_loading.weather_generator.generator import weather_generator

BUILTIN_GENERATORS = {**CALENDAR_GENERATORS, 'weather': weather_generator}

__all__ = ['BUILTIN_GENERATORS']
