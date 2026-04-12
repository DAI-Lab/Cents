import warnings
from functools import partial
from itertools import product
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from dtaidistance import dtw
from scipy.stats import f as f_dist
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mutual_info_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.tsa.tsatools import lagmat

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
        ori_data: Original time series data (N, seq_len) or (N, seq_len, dims).
        generated_data: Generated time series data, same shape convention.

    Returns:
        float: FID score between the original and generated data representations.
    """
    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)
    # TS2Vec expects (n_instance, n_timestamps, n_features); ensure 3D
    if ori_data.ndim == 2:
        ori_data = ori_data[:, :, np.newaxis]
    if generated_data.ndim == 2:
        generated_data = generated_data[:, :, np.newaxis]
    if ori_data.ndim != 3 or generated_data.ndim != 3:
        warnings.warn(
            f"Context_FID: expected 2D or 3D arrays, got ori_data.ndim={ori_data.ndim}, generated_data.ndim={generated_data.ndim}; returning nan."
        )
        return float("nan")
    # Require at least one non–all-NaN row so TS2Vec.fit() does not infinite-loop
    ori_valid = ~np.isnan(ori_data).all(axis=2).all(axis=1)
    n_valid = int(ori_valid.sum())
    if n_valid == 0:
        warnings.warn(
            "Context_FID: ori_data has no valid (non–all-NaN) rows; returning nan."
        )
        return float("nan")
    # Allow single-sample (TS2Vec will get 1 batch); only reject when 0 valid
    if np.isnan(ori_data).any() or np.isnan(generated_data).any():
        warnings.warn(
            "Context_FID: ori_data or generated_data contain NaN; FID may be unreliable."
        )
    model = TS2Vec(
        input_dims=ori_data.shape[-1],
        device=0,
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=50000,
    )

    fit_log = model.fit(ori_data, verbose=False)
    model.fit(ori_data, verbose=False)

    ori_rep = model.encode(ori_data, encoding_window="full_series")
    gen_rep = model.encode(generated_data, encoding_window="full_series")

    idx = np.random.permutation(ori_data.shape[0])
    ori_rep = ori_rep[idx]
    gen_rep = gen_rep[idx]

    if not np.isfinite(ori_rep).all() or not np.isfinite(gen_rep).all():
        return float("nan")

    return calculate_fid(ori_rep, gen_rep)


def compute_cfs(
    x_real: np.ndarray,
    x_synth: np.ndarray,
    c: np.ndarray,
    n_folds: int = 5,
) -> float:
    """
    Compute Context Faithfulness Score (CFS).

    Trains a classifier to distinguish real (x, c) pairs from synthetic (x, c) pairs
    using cross-validation, then returns 2 * |AUROC - 0.5|.

    Args:
        x_real:  (N, T, D_x) real time series
        x_synth: (N, T, D_x) synthetic time series
        c:       (N, T, D_c) shared context (same for real and synthetic)
        n_folds: number of cross-validation folds

    Returns:
        float: CFS in [0, 1]. 0 = indistinguishable (perfect), 1 = fully separable (failed).
    """
    N = x_real.shape[0]

    real_pairs = np.concatenate([x_real, c], axis=-1)   # (N, T, D_x+D_c)
    synth_pairs = np.concatenate([x_synth, c], axis=-1)  # (N, T, D_x+D_c)

    # Mean pool over time → fixed-size vectors
    X_real_enc = real_pairs.mean(axis=1)   # (N, D_x+D_c)
    X_synth_enc = synth_pairs.mean(axis=1) # (N, D_x+D_c)

    X_all = np.concatenate([X_real_enc, X_synth_enc], axis=0)  # (2N, D)
    y_all = np.concatenate([np.ones(N), np.zeros(N)])           # (2N,)

    # Drop rows with NaN
    valid = ~np.isnan(X_all).any(axis=1)
    X_all = X_all[valid]
    y_all = y_all[valid]

    if len(X_all) < 2 * n_folds or len(np.unique(y_all)) < 2:
        warnings.warn("compute_cfs: insufficient valid samples; returning nan.")
        return float("nan")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    auroc_scores = []
    for train_idx, val_idx in skf.split(X_all, y_all):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_all[train_idx], y_all[train_idx])
        proba = clf.predict_proba(X_all[val_idx])[:, 1]
        auroc_scores.append(roc_auc_score(y_all[val_idx], proba))

    mean_auroc = float(np.mean(auroc_scores))
    return float(2.0 * abs(mean_auroc - 0.5))


def _build_lag_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Return (T-max_lag, max_lag) lag matrix with rows [x_{t-1}, ..., x_{t-L}]."""
    return lagmat(x, maxlag=max_lag, trim="forward", original="ex")[max_lag:]


def _compute_f_stats_batch(
    x_arr: np.ndarray,
    c_arr: np.ndarray,
    max_lag: int,
    pairs: List[Tuple[int, int]],
) -> np.ndarray:
    """Compute mean F-statistic per sample for the given (dx, dc) pairs."""
    N = x_arr.shape[0]
    f_per_sample = []
    for i in range(N):
        f_vals = []
        for dx, dc in pairs:
            xi = x_arr[i, :, dx]
            ci = c_arr[i, :, dc]

            if np.isnan(xi).any() or np.isnan(ci).any():
                continue

            X_x = _build_lag_matrix(xi, max_lag)  # (T-max_lag, max_lag)
            X_c = _build_lag_matrix(ci, max_lag)   # (T-max_lag, max_lag)
            y = xi[max_lag:]

            ones = np.ones((len(y), 1))
            X_r = np.hstack([ones, X_x])              # restricted: own lags only
            X_u = np.hstack([ones, X_x, X_c])         # unrestricted: own + c lags

            beta_r, _, _, _ = np.linalg.lstsq(X_r, y, rcond=None)
            beta_u, _, _, _ = np.linalg.lstsq(X_u, y, rcond=None)

            rss_r = float(np.sum((y - X_r @ beta_r) ** 2))
            rss_u = float(np.sum((y - X_u @ beta_u) ** 2))

            df1 = max_lag
            df2 = len(y) - 2 * max_lag - 1

            if df2 <= 0 or rss_u < 1e-12:
                continue

            F = ((rss_r - rss_u) / df1) / (rss_u / df2)
            f_vals.append(F)

        if f_vals:
            f_per_sample.append(float(np.mean(f_vals)))

    return np.array(f_per_sample)


def compute_gcp(
    x_real: np.ndarray,
    x_synth: np.ndarray,
    c_real: np.ndarray,
    c_synth: np.ndarray,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> Tuple[float, Dict]:
    """
    Compute Granger Causality Preservation (GCP).

    Measures how well synthetic data preserves the Granger-causal structure from
    context c → signal x, via Wasserstein distance between F-statistic distributions.

    Args:
        x_real:  (N, T, D_x) real signal
        x_synth: (N, T, D_x) synthetic signal
        c_real:  (N, T, D_c) context for real data
        c_synth: (N, T, D_c) context for synthetic data
        max_lag: maximum lag order (capped at T // 10 for short series)
        alpha:   significance threshold for diagnostic sig-rate computation

    Returns:
        gcp:         float >= 0. 0 = perfect preservation. Higher = more divergence.
        diagnostics: dict with sig_rate_real, sig_rate_synth, sig_rate_delta
    """
    N, T, D_x = x_real.shape
    D_c = c_real.shape[-1]

    # Cap lag for short series
    max_lag = min(max_lag, max(1, T // 10))

    pairs = list(product(range(D_x), range(D_c)))

    F_real = _compute_f_stats_batch(x_real, c_real, max_lag, pairs)
    F_synth = _compute_f_stats_batch(x_synth, c_synth, max_lag, pairs)

    if len(F_real) == 0 or len(F_synth) == 0:
        warnings.warn("compute_gcp: no valid F-statistics computed; returning nan.")
        return float("nan"), {}

    gcp = float(wasserstein_distance(F_real, F_synth))

    # Diagnostics: significance rates
    df2 = T - 2 * max_lag - 1
    if df2 > 0:
        pvals_real = f_dist.sf(F_real, max_lag, df2)
        pvals_synth = f_dist.sf(F_synth, max_lag, df2)
        sig_real = float(np.mean(pvals_real < alpha))
        sig_synth = float(np.mean(pvals_synth < alpha))
    else:
        sig_real = sig_synth = float("nan")

    diagnostics = {
        "sig_rate_real": sig_real,
        "sig_rate_synth": sig_synth,
        "sig_rate_delta": float(abs(sig_real - sig_synth)) if not np.isnan(sig_real) else float("nan"),
    }

    return gcp, diagnostics


def compute_context_recovery_score(
    real_data: np.ndarray,
    syn_data: np.ndarray,
    context_labels: Dict[str, np.ndarray],
    continuous_vars: Optional[List[str]] = None,
    test_ratio: float = 0.2,
) -> Tuple[float, Dict[str, Dict]]:
    """
    Context Recovery Score: measures whether static context is reflected in generated outputs.

    Trains a predictor f: timeseries -> context_label on real data, then evaluates
    accuracy on synthetic data conditioned on the same labels.  High score means
    the model correctly encodes the conditioning variable in the output.

    For categorical variables: classification accuracy (chance = 1/n_classes).
    For continuous variables: R² (0 = no recovery, 1 = perfect).

    Args:
        real_data:      (N, T, D) real time series
        syn_data:       (N, T, D) synthetic time series (same conditioning as real)
        context_labels: {name: (N,) array of integer labels or float values}
        continuous_vars: names of continuous context variables; all others treated as categorical
        test_ratio:     fraction of real data held out for real_baseline evaluation

    Returns:
        overall_score: mean synth_score across all context variables
        per_var: {name: {"synth_score": float, "real_baseline": float, "type": str}}
    """
    if continuous_vars is None:
        continuous_vars = []

    N, T, D = real_data.shape

    def _features(data: np.ndarray) -> np.ndarray:
        """Mean + std pool over time → (N, 2*D) fixed-size representation."""
        return np.concatenate([data.mean(axis=1), data.std(axis=1)], axis=-1)

    X_real = _features(real_data)
    X_synth = _features(syn_data)

    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    test_size = max(1, int(N * test_ratio))
    train_idx = idx[test_size:]
    test_idx = idx[:test_size]

    per_var: Dict[str, Dict] = {}
    scores: List[float] = []

    for name, labels in context_labels.items():
        labels_f = labels.astype(float)
        if np.isnan(labels_f).any():
            per_var[name] = {"synth_score": float("nan"), "real_baseline": float("nan"), "type": "unknown"}
            continue

        if name in continuous_vars:
            y = labels_f
            clf = Ridge(alpha=1e-3, fit_intercept=True)
            clf.fit(X_real[train_idx], y[train_idx])
            real_baseline = float(r2_score(y[test_idx], clf.predict(X_real[test_idx])))
            synth_score = float(r2_score(y, clf.predict(X_synth)))
            score_type = "r2"
        else:
            y = labels.astype(int)
            n_classes = len(np.unique(y))
            if n_classes < 2:
                per_var[name] = {"synth_score": float("nan"), "real_baseline": float("nan"), "type": "accuracy"}
                continue
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_real[train_idx], y[train_idx])
            real_baseline = float(np.mean(clf.predict(X_real[test_idx]) == y[test_idx]))
            synth_score = float(np.mean(clf.predict(X_synth) == y))
            score_type = "accuracy"

        per_var[name] = {
            "synth_score": synth_score,
            "real_baseline": real_baseline,
            "type": score_type,
        }
        scores.append(synth_score)

    overall = float(np.mean(scores)) if scores else float("nan")
    return overall, per_var


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
