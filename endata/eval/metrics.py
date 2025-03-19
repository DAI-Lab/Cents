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
    df: pd.DataFrame,
    month: int,
    weekday: int,
    time_column: str = "timeseries",  # Make column name configurable
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the minimum and maximum bounds for time series values within a specified month and weekday.

    Args:
        df: DataFrame containing time series data
        month: The month to filter on
        weekday: The weekday to filter on
        time_column: Name of the column containing time series data

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays containing the minimum and maximum values for each timestamp
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame")

    df_filtered = df[(df["month"] == month) & (df["weekday"] == weekday)].copy()
    if df_filtered.empty:
        return None, None

    # Get maximum length across all sequences
    max_length = max(ts.shape[0] for ts in df_filtered[time_column])

    # Pad all sequences to max length
    padded_sequences = []
    for ts in df_filtered[time_column]:
        if ts.shape[0] < max_length:
            padded = np.pad(
                ts,
                ((0, max_length - ts.shape[0]), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )
        else:
            padded = ts
        padded_sequences.append(padded)

    array_timeseries = np.array(padded_sequences)

    # Calculate min/max excluding NaN values
    min_values = np.nanmin(array_timeseries, axis=0)
    max_values = np.nanmax(array_timeseries, axis=0)

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
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    analysis: str,
    compare: int = 3000,
    value_label: str = "Value",
):
    """
    Create visualizations comparing original and generated time series data.

    Args:
        ori_data: Original time series data
        generated_data: Generated time series data
        analysis: Type of analysis ('pca', 'tsne', or 'kernel')
        compare: Maximum number of samples to compare
        value_label: Label for the value axis
    """
    analysis_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:analysis_sample_no]
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    # Get maximum length across all samples
    max_length = max(
        max(ts.shape[0] for ts in ori_data), max(ts.shape[0] for ts in generated_data)
    )

    # Pad all sequences to max length
    padded_ori = np.zeros((analysis_sample_no, max_length, ori_data.shape[-1]))
    padded_gen = np.zeros((analysis_sample_no, max_length, generated_data.shape[-1]))

    for i in range(analysis_sample_no):
        ori_len = ori_data[i].shape[0]
        gen_len = generated_data[i].shape[0]
        padded_ori[i, :ori_len] = ori_data[i]
        padded_gen[i, :gen_len] = generated_data[i]

    no, seq_len, dim = padded_ori.shape
    plots = []

    for d in range(dim):
        prep_data = np.array([padded_ori[i, :, d] for i in range(analysis_sample_no)])
        prep_data_hat = np.array(
            [padded_gen[i, :, d] for i in range(analysis_sample_no)]
        )

        if analysis == "pca":
            # Remove any NaN values before PCA
            valid_mask = ~np.isnan(prep_data).any(axis=1) & ~np.isnan(
                prep_data_hat
            ).any(axis=1)
            if not np.any(valid_mask):
                continue

            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(prep_data[valid_mask])
            pca_hat_results = pca.transform(prep_data_hat[valid_mask])

            f, ax = plt.subplots(1)
            ax.scatter(
                pca_results[:, 0],
                pca_results[:, 1],
                c="red",
                alpha=0.2,
            )
            ax.scatter(
                pca_hat_results[:, 0],
                pca_hat_results[:, 1],
                c="blue",
                alpha=0.2,
            )

        elif analysis == "tsne":
            # Remove any NaN values before t-SNE
            valid_mask = ~np.isnan(prep_data).any(axis=1) & ~np.isnan(
                prep_data_hat
            ).any(axis=1)
            if not np.any(valid_mask):
                continue

            prep_data_final = np.concatenate(
                (prep_data[valid_mask], prep_data_hat[valid_mask]), axis=0
            )

            # Adjust perplexity based on data size
            n_samples = prep_data_final.shape[0]
            perplexity = min(5, n_samples - 1) if n_samples > 1 else 1

            tsne = TSNE(
                n_components=2,
                learning_rate="auto",
                init="pca",
                verbose=0,
                perplexity=perplexity,
                n_iter=300,
                early_exaggeration=5.0,
            )
            tsne_results = tsne.fit_transform(prep_data_final)

            f, ax = plt.subplots(1)
            n_valid = np.sum(valid_mask)
            ax.scatter(
                tsne_results[:n_valid, 0],
                tsne_results[:n_valid, 1],
                c="red",
                alpha=0.2,
            )
            ax.scatter(
                tsne_results[n_valid:, 0],
                tsne_results[n_valid:, 1],
                c="blue",
                alpha=0.2,
            )

        elif analysis == "kernel":
            # For kernel density estimation, we'll use only the valid values
            valid_ori = prep_data[~np.isnan(prep_data)]
            valid_gen = prep_data_hat[~np.isnan(prep_data_hat)]

            if len(valid_ori) == 0 or len(valid_gen) == 0:
                continue

            f, ax = plt.subplots(1)
            sns.kdeplot(data=valid_ori, fill=True, color="red", ax=ax)
            sns.kdeplot(
                data=valid_gen,
                fill=True,
                color="blue",
                ax=ax,
                linestyle="--",
            )

        # Set common plot properties
        font_size = 18
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.set_xlabel(
            (
                "PC1"
                if analysis == "pca"
                else "t-SNE dim 1" if analysis == "tsne" else value_label
            ),
            fontsize=font_size,
        )
        ax.set_ylabel(
            (
                "PC2"
                if analysis == "pca"
                else "t-SNE dim 2" if analysis == "tsne" else "Density"
            ),
            fontsize=font_size,
        )
        leg = ax.legend(["Real", "Synthetic"])
        leg.prop.set_size(font_size)
        plots.append(f)

    return plots


def plot_syn_and_real_comparison(
    df: pd.DataFrame, syn_df: pd.DataFrame, context_vars: dict, dimension: int = 0
):
    """
    Plot comparison between synthetic and real time series data.

    Args:
        df: DataFrame containing real time series data
        syn_df: DataFrame containing synthetic time series data
        context_vars: Dictionary of context variables to filter data
        dimension: Dimension of time series to plot
    """
    cpu_context_vars = {}
    for k, v in context_vars.items():
        if isinstance(v, torch.Tensor):
            v = v[0].cpu().item()
        cpu_context_vars[k] = v

    fields = list(cpu_context_vars.keys())
    condition = df[fields].eq(pd.Series(cpu_context_vars)).all(axis=1)
    filtered_df = df[condition]

    if filtered_df.empty:
        return None, None

    # Get the maximum length from both real and synthetic data
    real_lengths = [ts.shape[0] for ts in filtered_df["timeseries"]]
    syn_lengths = [ts.shape[0] for ts in syn_df["timeseries"]]
    max_length = max(max(real_lengths), max(syn_lengths))

    # Extract and pad data if necessary
    array_data = []
    for ts in filtered_df["timeseries"]:
        if ts.shape[0] < max_length:
            padded = np.pad(
                ts[:, dimension],
                (0, max_length - ts.shape[0]),
                mode="constant",
                constant_values=np.nan,
            )
        else:
            padded = ts[:, dimension]
        array_data.append(padded)
    array_data = np.array(array_data)

    # Calculate min/max excluding NaN values
    min_values = np.nanmin(array_data, axis=0)
    max_values = np.nanmax(array_data, axis=0)

    # Filter synthetic data
    syn_condition = syn_df[fields].eq(pd.Series(cpu_context_vars)).all(axis=1)
    syn_filtered_df = syn_df[syn_condition]

    if syn_filtered_df.empty:
        return None, None

    # Extract and pad synthetic data
    syn_values = []
    for ts in syn_filtered_df["timeseries"]:
        if ts.shape[0] < max_length:
            padded = np.pad(
                ts[:, dimension],
                (0, max_length - ts.shape[0]),
                mode="constant",
                constant_values=np.nan,
            )
        else:
            padded = ts[:, dimension]
        syn_values.append(padded)
    syn_values = np.array(syn_values)

    # Create time step labels
    time_steps = np.arange(1, max_length + 1)
    # Show every 4th label to avoid overcrowding
    label_positions = np.arange(0, max_length, 4)
    time_labels = [f"t={i+1}" for i in label_positions]

    # Create range plot
    fig_range, ax_range = plt.subplots(figsize=(15, 6))
    ax_range.fill_between(
        time_steps,
        min_values,
        max_values,
        color="gray",
        alpha=0.5,
        label="Range of real time series",
    )

    # Plot synthetic data
    synthetic_label_used = False
    for syn_ts in syn_values:
        valid_mask = ~np.isnan(syn_ts)
        if np.any(valid_mask):
            ax_range.plot(
                time_steps[valid_mask],
                syn_ts[valid_mask],
                color="blue",
                marker="o",
                markersize=2,
                linestyle="-",
                alpha=0.6,
                label="Synthetic time series" if not synthetic_label_used else None,
            )
            synthetic_label_used = True

    # Set plot properties
    font_size = 22
    ax_range.tick_params(axis="both", which="major", labelsize=font_size)
    ax_range.set_xlabel("Time step", fontsize=font_size)
    ax_range.set_ylabel("Value", fontsize=font_size)
    leg_range = ax_range.legend()
    leg_range.prop.set_size(font_size)
    ax_range.set_xticks(label_positions + 1)
    ax_range.set_xticklabels(time_labels, rotation=45)

    # Create closest match plot
    fig_closest, ax_closest = plt.subplots(figsize=(15, 6))
    synthetic_plotted = False
    real_plotted = False

    for syn_ts in syn_values:
        valid_mask = ~np.isnan(syn_ts)
        if not np.any(valid_mask):
            continue

        syn_valid = syn_ts[valid_mask]
        min_dtw_distance = float("inf")
        closest_real_ts = None

        for real_ts in array_data:
            real_valid = real_ts[valid_mask]
            if not np.any(~np.isnan(real_valid)):
                continue
            distance = dtw.distance(syn_valid, real_valid)
            if distance < min_dtw_distance:
                min_dtw_distance = distance
                closest_real_ts = real_ts

        if closest_real_ts is not None:
            ax_closest.plot(
                time_steps[valid_mask],
                syn_valid,
                color="blue",
                marker="o",
                markersize=2,
                linestyle="-",
                alpha=0.6,
                label="Synthetic time series" if not synthetic_plotted else None,
            )
            synthetic_plotted = True

            ax_closest.plot(
                time_steps[valid_mask],
                closest_real_ts[valid_mask],
                color="red",
                marker="x",
                markersize=2,
                linestyle="--",
                alpha=0.6,
                label="Real time series" if not real_plotted else None,
            )
            real_plotted = True

    # Set plot properties
    ax_closest.tick_params(axis="both", which="major", labelsize=font_size)
    ax_closest.set_xlabel("Time step", fontsize=font_size)
    ax_closest.set_ylabel("Value", fontsize=font_size)
    leg_closest = ax_closest.legend()
    leg_closest.prop.set_size(font_size)
    ax_closest.set_xticks(label_positions + 1)
    ax_closest.set_xticklabels(time_labels, rotation=45)

    fig_range.tight_layout()
    fig_closest.tight_layout()
    return fig_range, fig_closest
