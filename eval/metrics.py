from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from dtaidistance import dtw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from eval.loss import gaussian_kernel_matrix, maximum_mean_discrepancy
from eval.t2vec.t2vec import TS2Vec
from eval.utils import generate_title, get_hourly_ticks, get_month_weekday_names


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
    """
    Visualize the original and generated data using PCA, t-SNE, or KDE plots.

    Args:
        ori_data: Original data (n_samples, seq_len, n_features).
        generated_data: Generated data (same shape as ori_data).
        analysis: 'pca', 'tsne', or 'kernel' for PCA, t-SNE, or KDE visualization.
        compare: Number of samples to compare.
    """
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

        # Visualization
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
                label="Original",
            )
            ax.scatter(
                pca_hat_results[:, 0],
                pca_hat_results[:, 1],
                c=colors[analysis_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
            ax.legend()
            ax.set_title(f"PCA plot for Dimension {d}")
            ax.set_xlabel("x-pca")
            ax.set_ylabel("y-pca")
            plt.show()
            plots.append(f)

        elif analysis == "tsne":
            prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
            tsne = TSNE(
                n_components=2,
                learning_rate="auto",
                init="pca",
                verbose=1,
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
                label="Original",
            )
            ax.scatter(
                tsne_results[analysis_sample_no:, 0],
                tsne_results[analysis_sample_no:, 1],
                c=colors[analysis_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
            ax.legend()
            ax.set_title(f"t-SNE plot for dimension {d}")
            plt.show()
            plots.append(f)

        elif analysis == "kernel":
            f, ax = plt.subplots(1)
            sns.kdeplot(
                data=prep_data.flatten(),
                fill=True,
                color="red",
                label="Original",
                ax=ax,
            )
            sns.kdeplot(
                data=prep_data_hat.flatten(),
                fill=True,
                color="blue",
                label="Synthetic",
                ax=ax,
                linestyle="--",
            )
            ax.legend()
            ax.set_title(f"KDE plot for Dimension {d}")
            plt.show()
            plots.append(f)

    return plots


def plot_syn_and_real_comparison(
    df: pd.DataFrame, syn_df: pd.DataFrame, conditioning_vars: dict, dimension: int = 0
):
    """
    Plot the range of real data and synthetic time series, along with the closest real time series using DTW.
    The x-axis will display hourly timestamps instead of every 15 minutes.

    Args:
        df (pd.DataFrame): DataFrame containing real time series data.
        syn_df (pd.DataFrame): DataFrame containing synthetic time series data.
        conditioning_vars (dict): Dictionary of conditioning variables and their values for filtering.
        dimension (int, optional): Time series dimension to plot. Defaults to 0.
    """
    # Filter real data based on conditioning variables
    condition = (
        df[list(conditioning_vars.keys())].eq(pd.Series(conditioning_vars)).all(axis=1)
    )
    filtered_df = df[condition]

    array_data = np.array([ts[:, dimension] for ts in filtered_df["timeseries"]])

    if array_data.size == 0:
        print(f"No real data for conditioning variables: {conditioning_vars}")
        return

    min_values = np.min(array_data, axis=0)
    max_values = np.max(array_data, axis=0)

    # Filter synthetic data with the same conditioning variables
    syn_condition = (
        syn_df[list(conditioning_vars.keys())]
        .eq(pd.Series(conditioning_vars))
        .all(axis=1)
    )
    syn_filtered_df = syn_df[syn_condition]

    if syn_filtered_df.empty:
        print(f"No synthetic data for conditioning variables: {conditioning_vars}")
        return

    syn_values = np.array([ts[:, dimension] for ts in syn_filtered_df["timeseries"]])

    # Generate timestamps at 15-minute intervals
    timestamps = pd.date_range(start="00:00", end="23:45", freq="15min")

    # Get hourly tick positions and labels
    hourly_positions = np.arange(0, len(timestamps), 4)
    hourly_labels = [timestamps[i].strftime("%H:%M") for i in hourly_positions]

    # Generate plot title based on conditioning variables
    title = ", ".join([f"{key}={value}" for key, value in conditioning_vars.items()])

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 14), sharex=True)

    # Plot 1: Real data range and synthetic time series
    axes[0].fill_between(
        range(len(timestamps)),
        min_values,
        max_values,
        color="gray",
        alpha=0.5,
        label="kWh load range of real data",
    )
    for index in range(syn_values.shape[0]):
        axes[0].plot(
            range(len(timestamps)),
            syn_values[index],
            color="blue",
            marker="o",
            markersize=2,
            linestyle="-",
            alpha=0.6,
            label="Synthetic time series" if index == 0 else None,
        )
    axes[0].set_title(f"Range of Real Data and Synthetic Time Series ({title})")
    axes[0].set_ylabel("Electric load in kWh")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Plot 2: Synthetic time series and closest real time series
    synthetic_plotted = False
    real_plotted = False

    for index in range(syn_values.shape[0]):
        syn_ts = syn_values[index]

        # Find the closest real time series using DTW
        min_dtw_distance = float("inf")
        closest_real_ts = None

        for real_ts in array_data:
            distance = dtw.distance(syn_ts, real_ts)
            if distance < min_dtw_distance:
                min_dtw_distance = distance
                closest_real_ts = real_ts

        # Plot synthetic time series
        axes[1].plot(
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

        # Plot closest real time series
        if closest_real_ts is not None:
            axes[1].plot(
                range(len(timestamps)),
                closest_real_ts,
                color="red",
                marker="x",
                markersize=2,
                linestyle="--",
                alpha=0.6,
                label="Closest real time series" if not real_plotted else None,
            )
            real_plotted = True

    axes[1].set_title(f"Synthetic vs Closest Real Time Series ({title})")
    axes[1].set_xlabel("Time of day")
    axes[1].set_ylabel("Electric load in kWh")
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Set x-ticks for both plots
    axes[1].set_xticks(hourly_positions)
    axes[1].set_xticklabels(hourly_labels, rotation=45)

    plt.tight_layout()
    return plt.gcf()
