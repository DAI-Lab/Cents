from typing import Callable, Tuple

import numpy as np
import pandas as pd


def K(x_i: float, y_j: float, sigma: float) -> float:
    """
    Gaussian (RBF) kernel function.
    """
    return np.exp((-np.abs(x_i - y_j) ** 2) / (2 * (sigma**2)))


def max_mean_discrepancy(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two time series.

    Args:
        X: Time series data 1.
        Y: Time series data 2.
        sigma: Bandwidth of the RBF kernel.

    Returns:
        The MMD value between the two time series.
    """
    N = len(X)
    M = len(Y)

    term_1 = sum([sum([K(x_i, x_j, sigma) for x_j in X]) for x_i in X])
    term_2 = sum([sum([K(x_i, y_j, sigma) for y_j in Y]) for x_i in X])
    term_3 = sum([sum([K(y_i, y_j, sigma) for y_j in Y]) for y_i in Y])

    return ((1 / N**2) * term_1 - (2 / (M * N)) * term_2 + (1 / M**2) * term_3) ** 0.5


def dynamic_time_warping_dist(
    X: np.ndarray, Y: np.ndarray, norm: Callable[[float], float] = np.linalg.norm
) -> Tuple[np.ndarray, float]:
    """
    Compute the Dynamic Time Warping (DTW) distance between two time series.

    Args:
        X: Time series data 1.
        Y: Time series data 2.
        norm: Norm function to compute distances (default is Euclidean norm).

    Returns:
        A tuple containing the DTW distance matrix and the final DTW distance.
    """
    N = len(X)
    M = len(Y)
    D = np.zeros((N + 1, M + 1), dtype=float)
    D[:, 0] = np.inf
    D[0, :] = np.inf
    D[0][0] = 0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            m = min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
            D[i][j] = norm(X[i - 1] - Y[j - 1]) + m

    return D, D[N][M]


def get_period_bounds(
    df: pd.DataFrame, timeseries_colname: str, month: int, weekday: int
) -> Tuple[float, float]:
    df = df.loc[(df["month"] == month) & (df["weekday"] == weekday)].copy()
