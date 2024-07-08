import numpy as np
import pandas as pd


def K(x_i, y_j, sigma):
    return np.exp((-np.abs(x_i - y_j) ** 2) / (2 * (sigma ** 2)))

def max_mean_discrepancy(X, Y, sigma):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two time series.

    The formula for MMD is given by:

    MMD = sqrt( (1/N^2) * sum_{i=1}^N sum_{j=1}^N K(x_i, x_j)
                - (2/(M*N)) * sum_{i=1}^N sum_{j=1}^M K(x_i, y_j)
                + (1/M^2) * sum_{i=1}^M sum_{j=1}^M K(y_i, y_j) )

    where:
    - K(a, b) is the kernel function, in this case the Gaussian (RBF) kernel.
    - N is the number of samples in X.
    - M is the number of samples in Y.
    - x_i and x_j are samples from the first time series X.
    - y_i and y_j are samples from the second time series Y.

    Parameters:
    X (np.ndarray): Time series data 1 with shape (n_samples, n_features)
    Y (np.ndarray): Time series data 2 with shape (m_samples, n_features)
    sigma (float): The bandwidth of the RBF kernel.

    Returns:
    float: The MMD value between the two time series.
    """
    N = len(X)
    M = len(Y)

    term_1 = sum([sum([K(x_i, x_j) for x_j in X]) for x_i in X])
    term_2 = sum([sum([K(x_i, y_j) for y_j in Y]) for x_i in X])
    term_3 = sum([sum([K(y_i, y_j) for y_j in Y]) for y_i in Y])

    return ((1 / N ** 2) * term_1 - (2 / M * N) * term_2 + (1 / M ** 2) * term_3) ** 0.5
