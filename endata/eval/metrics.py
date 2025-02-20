from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from dtaidistance import dtw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from endata.eval.loss import gaussian_kernel_matrix, maximum_mean_discrepancy
from endata.eval.t2vec.t2vec import TS2Vec


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


def get_period_bounds(
    df: pd.DataFrame, month: int, weekday: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the minimum and maximum bounds for time series values within a specified month and weekday.

    Args:
        df: DataFrame containing time series data.
        month: The month to filter on.
        weekday: The weekday to filter on.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays containing the minimum and maximum values for each timestamp.
    """
    df_filtered = df[(df["month"] == month) & (df["weekday"] == weekday)].copy()
    array_timeseries = np.array(df_filtered["timeseries"].to_list())
    min_values = np.min(array_timeseries, axis=0)
    max_values = np.max(array_timeseries, axis=0)
    return min_values, max_values


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


def visualization(
    ori_data: np.ndarray, generated_data: np.ndarray, analysis: str, compare: int = 3000
):
    analysis_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:analysis_sample_no]
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    no, seq_len, dim = ori_data.shape
    plots = []
    for d in range(dim):
        prep_data = np.array([ori_data[i, :, d] for i in range(analysis_sample_no)])
        prep_data_hat = np.array(
            [generated_data[i, :, d] for i in range(analysis_sample_no)]
        )
        colors = ["red"] * analysis_sample_no + ["blue"] * analysis_sample_no
        if analysis == "pca":
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(prep_data)
            pca_hat_results = pca.transform(prep_data_hat)
            f, ax = plt.subplots(1)
            ax.scatter(
                pca_results[:, 0],
                pca_results[:, 1],
                c=colors[:analysis_sample_no],
                alpha=0.2,
            )
            ax.scatter(
                pca_hat_results[:, 0],
                pca_hat_results[:, 1],
                c=colors[analysis_sample_no:],
                alpha=0.2,
            )
            font_size = 18
            ax.tick_params(axis="both", which="major", labelsize=font_size)
            ax.set_xlabel("PC1", fontsize=font_size)
            ax.set_ylabel("PC2", fontsize=font_size)
            leg = ax.legend(["Real", "Synthetic"])
            leg.prop.set_size(font_size)
            plots.append(f)
        elif analysis == "tsne":
            prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
            tsne = TSNE(
                n_components=2,
                learning_rate="auto",
                init="pca",
                verbose=0,
                perplexity=5,
                n_iter=300,
                early_exaggeration=5.0,
            )
            tsne_results = tsne.fit_transform(prep_data_final)
            f, ax = plt.subplots(1)
            ax.scatter(
                tsne_results[:analysis_sample_no, 0],
                tsne_results[:analysis_sample_no, 1],
                c=colors[:analysis_sample_no],
                alpha=0.2,
            )
            ax.scatter(
                tsne_results[analysis_sample_no:, 0],
                tsne_results[analysis_sample_no:, 1],
                c=colors[analysis_sample_no:],
                alpha=0.2,
            )
            font_size = 18
            ax.tick_params(axis="both", which="major", labelsize=font_size)
            ax.set_xlabel("t-SNE dim 1", fontsize=font_size)
            ax.set_ylabel("t-SNE dim 2", fontsize=font_size)
            leg = ax.legend(["Real", "Synthetic"])
            leg.prop.set_size(font_size)
            plots.append(f)
        elif analysis == "kernel":
            f, ax = plt.subplots(1)
            sns.kdeplot(data=prep_data.flatten(), fill=True, color="red", ax=ax)
            sns.kdeplot(
                data=prep_data_hat.flatten(),
                fill=True,
                color="blue",
                ax=ax,
                linestyle="--",
            )
            font_size = 18
            ax.tick_params(axis="both", which="major", labelsize=font_size)
            ax.set_xlabel("kWh", fontsize=font_size)
            ax.set_ylabel("Density", fontsize=font_size)
            leg = ax.legend(["Real", "Synthetic"])
            leg.prop.set_size(font_size)
            plots.append(f)
    return plots


def plot_syn_and_real_comparison(
    df: pd.DataFrame, syn_df: pd.DataFrame, context_vars: dict, dimension: int = 0
):
    cpu_context_vars = {}
    for k, v in context_vars.items():
        if isinstance(v, torch.Tensor):
            v = v[0].cpu().item()
        cpu_context_vars[k] = v
    fields = list(cpu_context_vars.keys())
    condition = df[fields].eq(pd.Series(cpu_context_vars)).all(axis=1)
    filtered_df = df[condition]
    array_data = np.array([ts[:, dimension] for ts in filtered_df["timeseries"]])
    if array_data.size == 0:
        return None, None
    min_values = np.min(array_data, axis=0)
    max_values = np.max(array_data, axis=0)
    syn_condition = syn_df[fields].eq(pd.Series(cpu_context_vars)).all(axis=1)
    syn_filtered_df = syn_df[syn_condition]
    if syn_filtered_df.empty:
        return None, None
    syn_values = np.array([ts[:, dimension] for ts in syn_filtered_df["timeseries"]])
    timestamps = pd.date_range(start="00:00", end="23:45", freq="15min")
    hourly_positions = np.arange(0, len(timestamps), 4)
    hourly_labels = [timestamps[i].strftime("%H:%M") for i in hourly_positions]
    fig_range, ax_range = plt.subplots(figsize=(15, 6))
    ax_range.fill_between(
        range(len(timestamps)),
        min_values,
        max_values,
        color="gray",
        alpha=0.5,
        label="Range of real time series",
    )
    synthetic_label_used = False
    for index in range(syn_values.shape[0]):
        ax_range.plot(
            range(len(timestamps)),
            syn_values[index],
            color="blue",
            marker="o",
            markersize=2,
            linestyle="-",
            alpha=0.6,
            label="Synthetic time series" if not synthetic_label_used else None,
        )
        synthetic_label_used = True
    font_size = 22
    ax_range.tick_params(axis="both", which="major", labelsize=font_size)
    ax_range.set_xlabel("Time of day", fontsize=font_size)
    ax_range.set_ylabel("kWh", fontsize=font_size)
    leg_range = ax_range.legend()
    leg_range.prop.set_size(font_size)
    ax_range.set_xticks(hourly_positions)
    ax_range.set_xticklabels(hourly_labels, rotation=45)
    fig_closest, ax_closest = plt.subplots(figsize=(15, 6))
    synthetic_plotted = False
    real_plotted = False
    for index in range(syn_values.shape[0]):
        syn_ts = syn_values[index]
        min_dtw_distance = float("inf")
        closest_real_ts = None
        for real_ts in array_data:
            distance = dtw.distance(syn_ts, real_ts)
            if distance < min_dtw_distance:
                min_dtw_distance = distance
                closest_real_ts = real_ts
        ax_closest.plot(
            range(len(timestamps)),
            syn_ts,
            color="blue",
            marker="o",
            markersize=2,
            linestyle="-",
            alpha=0.6,
            label="Synthetic time series" if not synthetic_plotted else None,
        )
        synthetic_plotted = True
        if closest_real_ts is not None:
            ax_closest.plot(
                range(len(timestamps)),
                closest_real_ts,
                color="red",
                marker="x",
                markersize=2,
                linestyle="--",
                alpha=0.6,
                label="Real time series" if not real_plotted else None,
            )
            real_plotted = True
    ax_closest.tick_params(axis="both", which="major", labelsize=font_size)
    ax_closest.set_xlabel("Time of day", fontsize=font_size)
    ax_closest.set_ylabel("kWh", fontsize=font_size)
    leg_closest = ax_closest.legend()
    leg_closest.prop.set_size(font_size)
    ax_closest.set_xticks(hourly_positions)
    ax_closest.set_xticklabels(hourly_labels, rotation=45)
    fig_range.tight_layout()
    fig_closest.tight_layout()
    return fig_range, fig_closest
