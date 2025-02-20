import calendar
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch


def generate_title(context_vars):
    """
    Generate a plot title based on the provided context variables.

    Args:
        context_vars (dict): Dictionary of context variables and their values.

    Returns:
        str: Generated title string.
    """
    title_elements = []
    for var_name, value in context_vars.items():
        # Convert variable names and values to readable format
        if var_name == "month":
            month_name = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ][value]
            title_elements.append(f"{month_name}")
        elif var_name == "weekday":
            weekday_name = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ][value]
            title_elements.append(f"{weekday_name}s")
        else:
            # For other variables, display as "Variable: Value"
            title_elements.append(f"{var_name.capitalize()}: {value}")

    title = " | ".join(title_elements)
    return title


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
