from functools import partial
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from dtaidistance import dtw
from sklearn.linear_model import Ridge
from sklearn.metrics import mutual_info_score, r2_score

from cents.eval.eval_utils import (
    gaussian_kernel_matrix,
    get_period_bounds,
    maximum_mean_discrepancy,
)
from cents.eval.t2vec.t2vec import TS2Vec


def dynamic_time_warping_dist(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    """
    Compute the Dynamic Time Warping (DTW) distance between two multivariate time series.

    Args:
        X: Time series data 1 with shape (n_timeseries, timeseries_length, n_dimensions).
        Y: Time series data 2 with shape (n_timeseries, timeseries_length, n_dimensions).

    Returns:
        Tuple[float, float]: The mean and standard deviation of DTW distances between time series pairs.
    """
    assert (X.shape[0], X.shape[2]) == (
        Y.shape[0],
        Y.shape[2],
    ), "Input arrays must have the same shape!"

    n_timeseries, _, n_dimensions = X.shape
    dtw_distances = []

    for i in range(n_timeseries):
        distances = [
            dtw.distance(X[i, :, dim], Y[i, :, dim]) ** 2 for dim in range(n_dimensions)
        ]
        dtw_distances.append(np.sqrt(sum(distances)))

    dtw_distances = np.array(dtw_distances)
    return np.mean(dtw_distances), np.std(dtw_distances)


def calculate_period_bound_mse(
    real_dataframe: pd.DataFrame, synthetic_timeseries: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate the Mean Squared Error (MSE) between synthetic and real time series data, considering period bounds.

    Args:
        real_dataframe: DataFrame containing real time series data.
        synthetic_timeseries: The synthetic time series data.

    Returns:
        Tuple[float, float]: The mean and standard deviation of the period-bound MSE.
    """
    mse_list = []
    n_dimensions = synthetic_timeseries.shape[-1]

    for idx, (_, row) in enumerate(real_dataframe.iterrows()):
        month, weekday = row["month"], row["weekday"]

        mse = 0.0
        for dim_idx in range(n_dimensions):
            min_bounds, max_bounds = get_period_bounds(real_dataframe, month, weekday)
            syn_timeseries = synthetic_timeseries[idx, :, dim_idx]

            for j in range(len(syn_timeseries)):
                value = syn_timeseries[j]
                if value < min_bounds[j, dim_idx]:
                    mse += (value - min_bounds[j, dim_idx]) ** 2
                elif value > max_bounds[j, dim_idx]:
                    mse += (value - max_bounds[j, dim_idx]) ** 2

        mse /= len(syn_timeseries) * n_dimensions
        mse_list.append(mse)

    return np.mean(mse_list), np.std(mse_list)


def calculate_mmd(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two sets of time series.

    Args:
        X: First set of time series data (n_samples, seq_len, n_features).
        Y: Second set of time series data (same shape as X).

    Returns:
        Tuple[float, float]: The mean and standard deviation of the MMD scores.
    """
    assert (X.shape[0], X.shape[2]) == (
        Y.shape[0],
        Y.shape[2],
    ), "Input arrays must have the same shape!"

    n_timeseries, _, n_dimensions = X.shape
    discrepancies = []
    sigmas = [1]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=np.array(sigmas))

    for i in range(n_timeseries):
        distances = []
        for dim in range(n_dimensions):
            x = np.expand_dims(X[i, :, dim], axis=-1)
            y = np.expand_dims(Y[i, :, dim], axis=-1)
            dist = maximum_mean_discrepancy(x, y, gaussian_kernel)
            distances.append(dist**2)

        mmd = np.sqrt(sum(distances))
        discrepancies.append(mmd)

    discrepancies = np.array(discrepancies)
    return np.mean(discrepancies), np.std(discrepancies)


def calculate_fid(act1: np.ndarray, act2: np.ndarray) -> float:
    """
    Calculate the Fréchet Inception Distance (FID) between two sets of feature representations.

    Args:
        act1: Feature representations of dataset 1.
        act2: Feature representations of dataset 2.

    Returns:
        float: FID score between the two feature sets.
    """
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def Context_FID(ori_data: np.ndarray, generated_data: np.ndarray) -> float:
    """
    Calculate the FID score between original and generated data representations using TS2Vec embeddings.

    Args:
        ori_data: Original time series data.
        generated_data: Generated time series data.

    Returns:
        float: FID score between the original and generated data representations.
    """
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


def compute_mig(
    embeddings: np.ndarray,
    context_vars: Dict[str, np.ndarray],
    n_bins: int = 10,
) -> Tuple[float, Dict[str, float]]:
    """
    Mutual-Information Gap (MIG) with robust binning.

    Args:
        embeddings : (N, D) float array
        context_vars : dict[str, (N,) int array]
        n_bins : number of equal-width bins for each latent dim

    Returns:
        overall_mig : float
        per_var      : dict[str, float]
    """
    N, D = embeddings.shape
    per_var: Dict[str, float] = {}
    for name, labels in context_vars.items():
        # build MI vector over latent dims
        mi_vec = []
        for d in range(D):
            # skip degenerate dimensions
            if np.allclose(embeddings[:, d], embeddings[0, d]):
                mi_vec.append(0.0)
                continue
            edges = np.histogram_bin_edges(embeddings[:, d], bins=n_bins)
            codes = np.digitize(embeddings[:, d], bins=edges[1:-1], right=False)
            mi_vec.append(mutual_info_score(labels, codes))
        mi = np.asarray(mi_vec)

        # if MI is all zeros, MIG is zero
        if mi.max() == 0.0:
            per_var[name] = 0.0
            continue

        top2 = np.sort(mi)[-2:]
        entropy = mutual_info_score(labels, labels) + 1e-12
        per_var[name] = (top2[1] - top2[0]) / entropy  # (largest - second) / H

    overall = float(np.mean(list(per_var.values()))) if per_var else 0.0
    return overall, per_var


def compute_sap(
    embeddings: np.ndarray,
    context_vars: Dict[str, np.ndarray],
    reg_strength: float = 1e-3,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the Separability-Attribute-Predictability (SAP) score.

    Args:
        embeddings : (N, D) float array
            Latent codes h for N samples and D dimensions.

        context_vars : dict[str, (N,) int array]
            Mapping of context variable names to discrete labels.

        reg_strength : float, default 1e-3
            ℓ2-regularisation strength for the ridge regressors that predict
            the factor labels from *one* latent coordinate at a time.

    Returns:
        overall_sap : float
        Mean SAP score across factors.

        per_var : dict[str, float]
            SAP score for each individual context variable.
    """
    N, D = embeddings.shape
    per_var = {}

    for name, labels in context_vars.items():
        # Convert labels to a float vector for regression (one-vs-rest works too)
        y = labels.astype(float)
        scores = []

        for d in range(D):
            # fit 1-D ridge regressor  h_d  ->  y
            model = Ridge(alpha=reg_strength, fit_intercept=True)
            model.fit(embeddings[:, [d]], y)
            y_pred = model.predict(embeddings[:, [d]])
            scores.append(r2_score(y, y_pred))  # goodness of fit

        top2 = np.sort(scores)[-2:]  # best & second-best
        per_var[name] = top2[1] - top2[0]  # SAP_i

    overall = float(np.mean(list(per_var.values()))) if per_var else 0.0
    return overall, per_var
