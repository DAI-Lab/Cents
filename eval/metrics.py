from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from dtaidistance import dtw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from eval.t2vec.t2vec import TS2Vec


def dynamic_time_warping_dist(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the Dynamic Time Warping (DTW) distance between two multivariate time series.

    Args:
        X: Time series data 1 with shape (n_timeseries, timeseries_length, n_dimensions).
        Y: Time series data 2 with shape (n_timeseries, timeseries_length, n_dimensions).
        norm: Norm function to compute distances (default is Euclidean norm).

    Returns:
        A tuple containing the mean and standard deviation of the DTW distances.
    """
    assert (X.shape[0], X.shape[2]) == (
        Y.shape[0],
        Y.shape[2],
    ), "Input arrays must have the same n_timeseries and n_dimensions!"

    n_timeseries, _, n_dimensions = X.shape

    dtw_distances = []

    for i in range(n_timeseries):

        distances = []
        for dim in range(n_dimensions):
            dist = dtw.distance(X[i, :, dim], Y[i, :, dim])
            distances.append(dist**2)

        dtw_distance = np.sqrt(sum(distances))
        dtw_distances.append(dtw_distance)

    dtw_distances = np.array(dtw_distances)
    return np.mean(dtw_distances), np.std(dtw_distances)


def get_period_bounds(
    df: pd.DataFrame, timeseries_colname: str, month: int, weekday: int
) -> Tuple[float, float]:
    df = df.loc[(df["month"] == month) & (df["weekday"] == weekday)].copy()
    array_timeseries = np.array(df[timeseries_colname].to_list())
    min_values = np.min(array_timeseries, axis=0)
    max_values = np.max(array_timeseries, axis=0)
    return min_values, max_values


def calculate_period_bound_mse(
    synthetic_timeseries: np.ndarray, real_dataframe: pd.DataFrame, columns: List[str]
) -> Tuple[float, float]:
    mse_list = []

    for idx, row in real_dataframe.iterrows():
        month = row["month"]
        weekday = row["weekday"]

        mse = 0.0
        for dim_idx, colname in enumerate(columns):
            min_bounds, max_bounds = get_period_bounds(
                real_dataframe, colname, month, weekday
            )
            timeseries = synthetic_timeseries[idx, :, dim_idx]

            for j in range(len(timeseries)):
                value = timeseries[j]
                if value < min_bounds[j]:
                    mse += (value - min_bounds[j]) ** 2
                elif value > max_bounds[j]:
                    mse += (value - max_bounds[j]) ** 2

        mse /= len(timeseries)
        mse_list.append(mse)

    return np.mean(mse_list), np.std(mse_list)


def calculate_fid(act1, act2):

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

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


def visualization(ori_data, generated_data, analysis, compare=3000):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
     - ori_data: original data
     - generated_data: generated synthetic data
     - analysis: tsne or pca or kernel
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len]
            )
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (
                    prep_data_hat,
                    np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]),
                )
            )

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + [
        "blue" for i in range(anal_sample_no)
    ]

    if analysis == "pca":
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            pca_hat_results[:, 0],
            pca_hat_results[:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()
        plt.title("PCA plot")
        plt.xlabel("x-pca")
        plt.ylabel("y_pca")
        plt.show()

    elif analysis == "tsne":

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()

        plt.title("t-SNE plot")
        plt.xlabel("x-tsne")
        plt.ylabel("y_tsne")
        plt.show()

    elif analysis == "kernel":

        # Visualization parameter
        # colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

        f, ax = plt.subplots(1)
        sns.distplot(
            prep_data,
            hist=False,
            kde=True,
            kde_kws={"linewidth": 5},
            label="Original",
            color="red",
        )
        sns.distplot(
            prep_data_hat,
            hist=False,
            kde=True,
            kde_kws={"linewidth": 5, "linestyle": "--"},
            label="Synthetic",
            color="blue",
        )
        # Plot formatting

        # plt.legend(prop={'size': 22})
        plt.legend()
        plt.xlabel("Data Value")
        plt.ylabel("Data Density Estimate")
        # plt.rcParams['pdf.fonttype'] = 42

        # plt.savefig(str(args.save_dir)+"/"+args.model1+"_histo.png", dpi=100,bbox_inches='tight')
        # plt.ylim((0, 12))
        plt.show()
        plt.close()
