import calendar
from typing import List
from typing import Tuple

import pandas as pd


def get_month_weekday_names(month: int, weekday: int) -> Tuple[str, str]:
    """
    Map integer month and weekday to their respective names.

    Args:
        month (int): Month for filtering (0=January, ..., 11=December).
        weekday (int): Weekday for filtering (0=Monday, ..., 6=Sunday).

    Returns:
        Tuple[str, str]: (Month Name, Weekday Name)
    """
    month_name = calendar.month_name[month + 1]  # month is 0-indexed
    weekday_name = calendar.day_name[weekday]  # weekday is 0=Monday
    return month_name, weekday_name


def get_hourly_ticks(timestamps: pd.DatetimeIndex) -> Tuple[List[int], List[str]]:
    """
    Generate hourly tick positions and labels.

    Args:
        timestamps (pd.DatetimeIndex): DatetimeIndex of timestamps.

    Returns:
        Tuple[List[int], List[str]]: (Tick Positions, Tick Labels)
    """
    hourly_positions = list(
        range(0, len(timestamps), 4)
    )  # Every 4 intervals (15 min each)
    hourly_labels = [timestamps[i].strftime("%H:%M") for i in hourly_positions]
    return hourly_positions, hourly_labels
