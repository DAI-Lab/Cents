import numpy as np
import pandas as pd
import pytest
import torch


@pytest.mark.parametrize("dimension", ["1D", "2D"])
def test_inverse_transform_roundtrip(
    dimension,
    raw_dataset_1d,
    normalized_dataset_1d,
    raw_dataset_2d,
    normalized_dataset_2d,
):
    """
    Verify that after auto-normalization, we can call inverse_transform to
    recover (approximately) the original data.
    """

    if dimension == "1D":
        raw_ds = raw_dataset_1d
        norm_ds = normalized_dataset_1d
    else:
        raw_ds = raw_dataset_2d
        norm_ds = normalized_dataset_2d

    normalized_copy = norm_ds.data.copy()
    recovered_df = norm_ds.inverse_transform(normalized_copy, merged=True)

    raw_df = raw_ds.data.copy()

    assert len(raw_df) == len(recovered_df), "Row count mismatch."

    for idx in range(len(raw_df)):
        original_ts = raw_df.iloc[idx]["timeseries"]
        recovered_ts = recovered_df.iloc[idx]["timeseries"]

        assert original_ts.shape == recovered_ts.shape, "Shape mismatch."

        np.testing.assert_allclose(
            original_ts,
            recovered_ts,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"Time series mismatch at row {idx} for dimension={dimension}",
        )


def test_dataset_length(
    raw_dataset_1d, normalized_dataset_1d, raw_dataset_2d, normalized_dataset_2d
):
    """Check that each dataset has correct length."""
    assert len(raw_dataset_1d) == 10
    assert len(normalized_dataset_1d) == 10
    assert len(raw_dataset_2d) == 10
    assert len(normalized_dataset_2d) == 10


def test_get_item_shape_1d(raw_dataset_1d):
    ts, cond = raw_dataset_1d[0]
    assert ts.shape == (16, 1)
    assert isinstance(cond, dict)


def test_get_item_shape_2d(raw_dataset_2d):
    ts, cond = raw_dataset_2d[0]
    assert ts.shape == (16, 2)
    assert isinstance(cond, dict)
