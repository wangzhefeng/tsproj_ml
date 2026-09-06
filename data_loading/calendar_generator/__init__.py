"""内置生成器统一注册表；不通过旧兼容模块反向导入。"""
from data_loading.calendar_generator.calendar_features import chinese_holiday_frame
from data_loading.calendar_generator.chinese_holiday import chinese_holiday_generator, GENERATOR_NAME
from data_loading.sources.source_io import SourceGenerator

BUILTIN_GENERATORS: dict[str, SourceGenerator] = {
    GENERATOR_NAME: chinese_holiday_generator,
}

__all__ = ["BUILTIN_GENERATORS", "chinese_holiday_frame", "chinese_holiday_generator", "GENERATOR_NAME"]
