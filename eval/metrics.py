from typing import Callable, Tuple

import numpy as np
import pandas as pd
import scipy

from eval.t2vec.t2vec import TS2Vec


def dynamic_time_warping_dist(
    X: np.ndarray, Y: np.ndarray, norm: Callable[[np.ndarray], float] = np.linalg.norm
) -> Tuple[np.ndarray, float]:
    """
    Compute the Dynamic Time Warping (DTW) distance between two time series.

    Args:
        X: Time series data 1 with shape (n_timeseries, timeseries_length, n_dimensions).
        Y: Time series data 2 with shape (n_timeseries, timeseries_length, n_dimensions).
        norm: Norm function to compute distances (default is Euclidean norm).

    Returns:
        A tuple containing the mean and standard deviation of the DTW distances.
    """
    assert X.shape == Y.shape, "Input arrays must have the same shape"

    n_timeseries = X.shape[0]
    dtw_distances = []

    for idx in range(n_timeseries):
        xi = X[idx]
        yi = Y[idx]

        N = len(xi)
        M = len(yi)
        D = np.zeros((N + 1, M + 1), dtype=float)
        D[:, 0] = np.inf
        D[0, :] = np.inf
        D[0][0] = 0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                m = min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
                D[i][j] = norm(xi[i - 1] - yi[j - 1]) + m

        dtw_distances.append(D[N][M])

    return np.mean(dtw_distances)


def get_period_bounds(
    df: pd.DataFrame, timeseries_colname: str, month: int, weekday: int
) -> Tuple[float, float]:
    df = df.loc[(df["month"] == month) & (df["weekday"] == weekday)].copy()
    array_timeseries = np.array(df[timeseries_colname].to_list())
    min_values = np.min(array_timeseries, axis=0)
    max_values = np.max(array_timeseries, axis=0)
    return min_values, max_values


def calculate_period_bound_mse(
    timeseries_array: np.ndarray,
    df: pd.DataFrame,
    timeseries_colname: str,
) -> Tuple[float, float]:
    n_timeseries = timeseries_array.shape[0]
    mse_list = []

    for i in range(n_timeseries):
        row = df.iloc[i]
        month = row["month"]
        weekday = row["weekday"]

        min_bounds, max_bounds = get_period_bounds(
            df, timeseries_colname, month, weekday
        )

        timeseries = timeseries_array[i]
        mse = 0.0
        for j, value in enumerate(timeseries):
            if value < min_bounds[j]:
                mse += (value - min_bounds[j]) ** 2
            elif value > max_bounds[j]:
                mse += (value - max_bounds[j]) ** 2
        mse /= len(timeseries)
        mse_list.append(mse)

    return np.mean(mse_list), np.std(mse_list)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def Context_FID(ori_data, generated_data):
    model = TS2Vec(
        input_dims=ori_data.shape[-1],
        device=0,
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=50000,
    )
    model.fit(ori_data, verbose=False)
    ori_represenation = model.encode(ori_data, encoding_window="full_series")
    gen_represenation = model.encode(generated_data, encoding_window="full_series")
    idx = np.random.permutation(ori_data.shape[0])
    ori_represenation = ori_represenation[idx]
    gen_represenation = gen_represenation[idx]
    results = calculate_fid(ori_represenation, gen_represenation)
    return results
