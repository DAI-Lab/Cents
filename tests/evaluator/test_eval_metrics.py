import numpy as np
import pytest

from cents.eval.eval import calculate_mmd, dynamic_time_warping_dist


def test_dtw_identical_zero():
    X = np.zeros((2, 3, 1))
    Y = np.zeros((2, 3, 1))
    mean, std = dynamic_time_warping_dist(X, Y)
    assert mean == pytest.approx(0.0)
    assert std == pytest.approx(0.0)


def test_dtw_known_case():
    X = np.array([[[0], [1], [2]]])  # shape (1,3,1)
    Y = np.array([[[0], [2], [4]]])  # shape (1,3,1)
    mean, std = dynamic_time_warping_dist(X, Y)
    assert mean == pytest.approx(np.sqrt(5))
    assert std == pytest.approx(0.0)


def test_dtw_shape_mismatch_raises():
    X = np.zeros((1, 3, 1))
    Y = np.zeros((2, 3, 1))
    with pytest.raises(AssertionError):
        dynamic_time_warping_dist(X, Y)


def test_mmd_identical_zero():
    X = np.zeros((2, 4, 1))
    Y = np.zeros((2, 4, 1))
    mean, std = calculate_mmd(X, Y)
    assert mean == pytest.approx(0.0)
    assert std == pytest.approx(0.0)


def test_mmd_simple_nonzero():
    X = np.zeros((1, 5, 1))
    Y = np.ones((1, 5, 1))
    mean, std = calculate_mmd(X, Y)
    assert mean > 0
    assert std >= 0


def test_mmd_shape_mismatch_raises():
    X = np.zeros((1, 5, 1))
    Y = np.zeros((2, 5, 1))
    with pytest.raises(AssertionError):
        calculate_mmd(X, Y)
