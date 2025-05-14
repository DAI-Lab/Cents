import calendar
import datetime
import pickle
import random
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from dtaidistance import dtw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
      ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError("Both inputs should be matrices.")

    if x.shape[1] != y.shape[1]:
        raise ValueError("The number of features should be the same.")

    norm = lambda x: np.sum(np.square(x), 1)

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    # x will be num_samples x num_features x 1,
    # and y will be 1 x num_features x num_samples (after broadcasting).
    # After the substraction we will get a
    # num_x_samples x num_features x num_y_samples matrix.
    # The resulting dist will be of shape num_y_samples x num_x_samples.
    # and thus we need to transpose it again.
    return np.transpose(norm(np.expand_dims(x, 2) - np.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    """Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1.0 / (2.0 * (np.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = np.matmul(beta, np.reshape(dist, (1, -1)))

    return np.reshape(np.sum(np.exp(-s), 0), np.shape(dist))


def maximum_mean_discrepancy(x, y, kernel):
    r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = np.mean(kernel(x, x))
    cost += np.mean(kernel(y, y))
    cost -= 2 * np.mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = np.where(cost > 0, cost, 0)
    return cost


def mmd_loss(source_samples, target_samples, weight=1.0):
    """Computes the MMD loss for each pair of corresponding samples."""
    assert (
        source_samples.shape == target_samples.shape
    ), "Shapes of source and target samples must match."
    num_samples = source_samples.shape[0]
    mmd_values = np.zeros(num_samples)
    sigmas = [1]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=np.array(sigmas))

    for i in range(num_samples):
        mmd_values[i] = maximum_mean_discrepancy(
            np.expand_dims(source_samples[i], axis=1),
            np.expand_dims(target_samples[i], axis=1),
            kernel=gaussian_kernel,
        )
    mmd_values = np.maximum(1e-4, mmd_values) * weight
    return mmd_values


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


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


def pkl_save(name, var):
    with open(name, "wb") as f:
        pickle.dump(var, f)


def pkl_load(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[: x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B * T, False, dtype=np.bool)
    ele_sel = np.random.choice(B * T, size=int(B * T * p), replace=False)
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res


def name_with_datetime(prefix="default"):
    now = datetime.now()
    return prefix + "_" + now.strftime("%Y%m%d_%H%M%S")


def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None,
):
    import torch

    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == "cuda":
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]


def create_visualizations(
    real_data_df: pd.DataFrame,
    syn_data_df: pd.DataFrame,
    dataset: Any,
    model: Any,
    num_samples: int = 100,
    num_runs: int = 1,
) -> Dict:
    """
    Create and store visualizations of the generated data.

    Args:
        real_data_df (pd.DataFrame): DataFrame containing real data
        syn_data_df (pd.DataFrame): DataFrame containing synthetic data
        dataset (Any): The dataset object
        model (Any): The trained model
        num_samples (int): Number of samples to generate per run
        num_runs (int): Number of visualization runs to perform

    Returns:
        Dict: Dictionary containing the generated visualizations
    """
    visualizations = {}

    real_data_array = np.stack(real_data_df["timeseries"])
    _, seq_len, dim = real_data_array.shape
    for i in range(num_runs):
        sample_index = np.random.randint(low=0, high=real_data_df.shape[0])
        sample_row = real_data_df.iloc[sample_index]
        context_vars_sample = {
            var_name: torch.tensor(
                [sample_row[var_name]] * num_samples,
                dtype=torch.long,
            )
            for var_name in dataset.context_vars
        }
        generated_samples = model.generate(context_vars_sample).cpu().numpy()
        if generated_samples.ndim == 2:
            generated_samples = generated_samples.reshape(
                generated_samples.shape[0], -1, generated_samples.shape[1]
            )
        generated_samples_df = pd.DataFrame(
            {
                var_name: [sample_row[var_name]] * num_samples
                for var_name in dataset.context_vars
            }
        )
        generated_samples_df["timeseries"] = list(generated_samples)

        normalization_keys = (
            dataset.normalization_group_keys
            if hasattr(dataset, "normalization_group_keys")
            else []
        )
        missing_keys = [
            key for key in normalization_keys if key not in generated_samples_df.columns
        ]
        if missing_keys:
            for key in missing_keys:
                if key in sample_row:
                    generated_samples_df[key] = sample_row[key]
                else:
                    raise ValueError(
                        f"Sample row does not contain required key: '{key}'."
                    )
        generated_samples_df = dataset.inverse_transform(generated_samples_df)
        range_fig, closest_fig = plot_syn_and_real_comparison(
            real_data_df,
            generated_samples_df,
            context_vars_sample,
            dimension=0,
        )
        if range_fig is not None:
            visualizations[f"RangePlot_{i}"] = range_fig
            plt.close(range_fig)
        if closest_fig is not None:
            visualizations[f"ClosestPlot_{i}"] = closest_fig
            plt.close(closest_fig)
        if dim > 1:
            syn_sample_0 = generated_samples[0, :, 0]
            real_sample_0 = (
                sample_row["timeseries"][:, 0]
                if sample_row["timeseries"].ndim == 2
                else sample_row["timeseries"]
            )
            syn_sample_1 = generated_samples[0, :, 1]
            real_sample_1 = (
                sample_row["timeseries"][:, 1]
                if sample_row["timeseries"].ndim == 2
                else None
            )
            fig_multi, axes_multi = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
            axes_multi[0].plot(syn_sample_0, label="Synthetic", color="blue")
            axes_multi[0].plot(real_sample_0, label="Real", color="red")
            font_size = 12
            axes_multi[0].tick_params(axis="both", which="major", labelsize=font_size)
            axes_multi[0].set_xlabel("Timestep", fontsize=font_size)
            axes_multi[0].set_ylabel("kWh", fontsize=font_size)
            leg_0 = axes_multi[0].legend()
            leg_0.prop.set_size(font_size)
            if real_sample_1 is not None:
                axes_multi[1].plot(syn_sample_1, label="Synthetic", color="blue")
                axes_multi[1].plot(real_sample_1, label="Real", color="red")
            else:
                axes_multi[1].plot(syn_sample_1, label="Synthetic", color="blue")
            axes_multi[1].tick_params(axis="both", which="major", labelsize=font_size)
            axes_multi[1].set_xlabel("Timestep", fontsize=font_size)
            axes_multi[1].set_ylabel("kWh", fontsize=font_size)
            leg_1 = axes_multi[1].legend()
            leg_1.prop.set_size(font_size)
            visualizations[f"MultiDim_Chart_{i}"] = fig_multi
            plt.close(fig_multi)
    syn_data_array = np.stack(syn_data_df["timeseries"])
    kde_plots = visualization(real_data_array, syn_data_array, "kernel")
    tsne_plots = visualization(real_data_array, syn_data_array, "tsne")
    if kde_plots is not None:
        for i, plot in enumerate(kde_plots):
            visualizations[f"KDE_Dim_{i}"] = plot
    if tsne_plots is not None:
        for i, plot in enumerate(tsne_plots):
            visualizations[f"TSNE_Dim_{i}"] = plot


# def evaluate_pv_shift(self, dataset: Any, model: Any):
#     avg_shift = dataset.compute_average_pv_shift()
#     if avg_shift is None or np.allclose(avg_shift, 0.0):
#         return
#     test_contexts = dataset.sample_shift_test_contexts()
#     n_sampled = len(test_contexts)
#     n_pv1_missing = sum(1 for c in test_contexts if c["missing_pv"] == 1)
#     n_pv0_missing = sum(1 for c in test_contexts if c["missing_pv"] == 0)

#     print(f"[Shift Contexts] Sampled: {n_sampled}.")
#     print(f"[Shift Contexts] PV=1 is missing in {n_pv1_missing} of these contexts.")
#     print(f"[Shift Contexts] PV=0 is missing in {n_pv0_missing} of these contexts.")
#     if len(test_contexts) == 0:
#         return
#     present_ctx_list = []
#     missing_ctx_list = []
#     present_pv_values = []
#     for cinfo in test_contexts:
#         base_ctx = cinfo["base_context"]
#         present_pv = cinfo["present_pv"]
#         missing_pv = cinfo["missing_pv"]
#         ctx_p = dict(base_ctx)
#         ctx_m = dict(base_ctx)
#         ctx_p["has_solar"] = present_pv
#         ctx_m["has_solar"] = missing_pv
#         present_ctx_list.append(ctx_p)
#         missing_ctx_list.append(ctx_m)
#         present_pv_values.append(present_pv)
#     present_ctx_tensors = {}
#     missing_ctx_tensors = {}
#     all_keys = present_ctx_list[0].keys()
#     for k in all_keys:
#         present_ctx_tensors[k] = torch.tensor(
#             [pc[k] for pc in present_ctx_list], dtype=torch.long, device=self.device
#         )
#         missing_ctx_tensors[k] = torch.tensor(
#             [mc[k] for mc in missing_ctx_list], dtype=torch.long, device=self.device
#         )
#     with torch.no_grad():
#         syn_ts_present = model.generate(present_ctx_tensors)
#         syn_ts_missing = model.generate(missing_ctx_tensors)
#     syn_ts_present = syn_ts_present.cpu().numpy()
#     syn_ts_missing = syn_ts_missing.cpu().numpy()
#     if syn_ts_present.ndim == 3 and syn_ts_present.shape[-1] == 1:
#         syn_ts_present = syn_ts_present[:, :, 0]
#         syn_ts_missing = syn_ts_missing[:, :, 0]
#     shifts = []
#     for i, pv_val in enumerate(present_pv_values):
#         shift_i = syn_ts_missing[i] - syn_ts_present[i]
#         if pv_val == 1:
#             shift_i = -shift_i
#         shifts.append(shift_i)
#     shifts = np.array(shifts)
#     avg_shift = np.asarray(avg_shift).reshape(-1)
#     l2_values = []
#     for i in range(shifts.shape[0]):
#         diff = shifts[i] - avg_shift
#         l2 = np.sqrt((diff**2).sum())
#         l2_values.append(l2)
#     mean_l2 = np.mean(l2_values)
#     wandb.log({"Shift_L2": mean_l2})

#     def find_context_matched_shift(dataset, cinfo):
#         base_ctx = cinfo["base_context"]
#         city_val = base_ctx.get("city", None)
#         btype_val = base_ctx.get("building_type", None)
#         df = dataset.data.copy()
#         mask = pd.Series([True] * len(df))
#         if city_val is not None and "city" in df.columns:
#             mask = mask & (df["city"] == city_val)
#         if btype_val is not None and "building_type" in df.columns:
#             mask = mask & (df["building_type"] == btype_val)
#         df_matched = df[mask]
#         if df_matched.empty:
#             return None
#         df_pv0 = df_matched[df_matched["has_solar"] == 0]
#         df_pv1 = df_matched[df_matched["has_solar"] == 1]
#         if df_pv0.empty or df_pv1.empty:
#             return None
#         ts_pv0 = np.stack(df_pv0["timeseries"].values, axis=0)
#         ts_pv1 = np.stack(df_pv1["timeseries"].values, axis=0)
#         mean_pv0 = ts_pv0.mean(axis=0)
#         mean_pv1 = ts_pv1.mean(axis=0)
#         mean_pv0_dim0 = mean_pv0[:, 0]
#         mean_pv1_dim0 = mean_pv1[:, 0]
#         real_shift = mean_pv1_dim0 - mean_pv0_dim0
#         return real_shift

#     matched_shifts = []
#     for cinfo in test_contexts:
#         matched = find_context_matched_shift(dataset, cinfo)
#         matched_shifts.append(matched)
#     n_plots = min(6, shifts.shape[0])
#     for j, idx in enumerate(
#         np.random.choice(shifts.shape[0], size=n_plots, replace=False)
#     ):
#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax.plot(avg_shift, label="Real shift", color="red")
#         ax.plot(shifts[idx], label="Synthetic shift", color="blue", linestyle="--")
#         matched_s = matched_shifts[idx]
#         if matched_s is not None:
#             ax.plot(
#                 matched_s,
#                 label="Context-matched shift",
#                 color="green",
#                 linestyle=":",
#             )
#         font_size = 12
#         ax.tick_params(axis="both", which="major", labelsize=font_size)
#         ax.set_xlabel("Timestep", fontsize=font_size)
#         ax.set_ylabel("kWh", fontsize=font_size)
#         leg = ax.legend()
#         leg.prop.set_size(font_size)
#         fig.tight_layout()
#         wandb.log({f"ShiftPlot_{j}": wandb.Image(fig)})
#         plt.close(fig)


def flatten_log_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """
    Flatten a dictionary of log values into a single dictionary of floats.

    Args:
        d (Dict[str, Any]): The dictionary to flatten
        prefix (str): The prefix to add to the keys
    Returns:
        Dict[str, float]: A flattened dictionary of floats
    """
    flat = {}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_log_dict(v, prefix=name + "/"))
        else:
            flat[name] = float(v)
    return flat
