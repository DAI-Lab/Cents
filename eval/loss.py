from functools import partial

import numpy as np


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
