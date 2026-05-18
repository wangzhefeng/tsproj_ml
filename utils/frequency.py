# -*- coding: utf-8 -*-

"""
时间频率工具。
"""

from pathlib import Path

from pandas.tseries.frequencies import to_offset


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def resolve_freq_step_minutes(freq: str) -> float:
    """
    解析 pandas 频率字符串对应的步长（分钟）。

    要求频率必须是固定时长，并且不超过 1 天。
    """
    try:
        offset = to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"Invalid pandas frequency: {freq}") from exc

    try:
        step_nanos = offset.nanos
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise ValueError(
            f"Frequency '{freq}' is not a fixed-duration offset. "
            "Please use fixed frequencies such as '5min', '15min', '30min', '1h', or '1D'."
        ) from exc

    step_minutes = step_nanos / (60 * 1_000_000_000)
    if step_minutes <= 0:
        raise ValueError(f"Frequency '{freq}' must be positive.")
    if step_minutes > 24 * 60:
        raise ValueError(f"Frequency '{freq}' must not exceed 1 day for this project.")

    return int(step_minutes)


def resolve_samples_per_day(freq: str) -> int:
    """
    根据 pandas 频率字符串计算一天内的样本数。

    当前项目要求一天能被频率整除，例如：
    - 5min -> 288
    - 15min -> 96
    - 1h -> 24
    - 1D -> 1
    """
    day_minutes = 24 * 60
    step_minutes = resolve_freq_step_minutes(freq)
    quotient = day_minutes / step_minutes

    if int(round(quotient)) != quotient:
        raise ValueError(
            f"Frequency '{freq}' does not evenly divide one day. "
            "Please use a fixed frequency that evenly partitions 24 hours."
        )

    return int(quotient)




# 测试代码 main 函数
def main():
    res = resolve_freq_step_minutes(freq="1min")
    print(res)
    res = resolve_freq_step_minutes(freq="5min")
    print(res)
    res = resolve_freq_step_minutes(freq="10min")
    print(res)
    res = resolve_freq_step_minutes(freq="1h")
    print(res)
    res = resolve_freq_step_minutes(freq="1d")
    print(res)

if __name__ == "__main__":
    main()
