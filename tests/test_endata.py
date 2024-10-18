"""Tests for `endata` package."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from datasets.openpower import OpenPowerDataManager
from datasets.pecanstreet import PecanStreetDataManager
from generator.gan.acgan import ACGAN
from generator.options import Options

TEST_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data_config.yaml"
)


class TestGenerator(unittest.TestCase):
    """Test Generators."""

    def test_acgan_output_shape(self):

        # opt = Options(model_name="acgan")
        # opt.device = "cpu"  # Use CPU for testing purposes
        # model = ACGAN(opt)
        # noise = torch.randn(opt.batch_size, opt.noise_dim).to(opt.device)
        # gen_ts = model.generator(noise, model.sample_conditioning_vars(None, opt.batch_size, random=True))
        # assert gen_ts.shape == (opt.batch_size, opt.seq_len, opt.input_dim)
        pass


class TestDataset(unittest.TestCase):
    """Test Dataset Functionality."""

    def test_data_manager(self):
        data_manager = PecanStreetDataManager(
            config_path=TEST_CONFIG_PATH, include_generation=False
        )

        data_manager = PecanStreetDataManager(
            config_path=TEST_CONFIG_PATH, include_generation=True
        )

        assert data_manager.data.shape[0] > 0, "Dataframe not loaded correctly"
        assert "timeseries" in data_manager.data.columns, "No timeseries column found."

        ts_shape = data_manager.data.timeseries.iloc[0].shape
        assert ts_shape == (96, 2)

        op_data_manager = OpenPowerDataManager()

        assert op_data_manager.data.shape[0] > 0, "Dataframe not loaded correctly"
        assert "timeseries" in data_manager.data.columns, "No timeseries column found."

        ts_shape = data_manager.data.timeseries.iloc[0].shape
        assert ts_shape == (96, 2)

    def test_ps_reverse_transform(self):
        data_manager = PecanStreetDataManager(
            config_path=TEST_CONFIG_PATH,
            include_generation=False,
            normalize=False,
            threshold=None,
        )
        global_normalized_data_manager = PecanStreetDataManager(
            config_path=TEST_CONFIG_PATH,
            include_generation=False,
            normalize=True,
            threshold=None,
            normalization_method="global",
        )
        grouped_normalized_data_manager = PecanStreetDataManager(
            config_path=TEST_CONFIG_PATH,
            include_generation=False,
            normalize=True,
            threshold=None,
            normalization_method="group",
        )

        original_dataset = data_manager.create_all_user_dataset()
        global_normalized_dataset = (
            global_normalized_data_manager.create_all_user_dataset()
        )
        grouped_normalized_dataset = (
            grouped_normalized_data_manager.create_all_user_dataset()
        )

        original_data = original_dataset.data
        global_normalized_data = global_normalized_dataset.data
        grouped_normalized_data = grouped_normalized_dataset.data

        global_transformed = global_normalized_dataset.inverse_transform(
            global_normalized_data
        )
        grouped_transformed = grouped_normalized_dataset.inverse_transform(
            grouped_normalized_data
        )

        mse_list: List[float] = []

        for idx in range(len(original_data)):
            unnormalized_timeseries = original_data.iloc[idx]["timeseries"]
            global_transformed_timeseries = global_transformed.iloc[idx]["timeseries"]
            grouped_transformed_timeseries = grouped_transformed.iloc[idx]["timeseries"]

            global_mse = mean_squared_error(
                unnormalized_timeseries, global_transformed_timeseries
            )
            grouped_mse = mean_squared_error(
                unnormalized_timeseries, grouped_transformed_timeseries
            )
            mse_list.append(global_mse)
            mse_list.append(grouped_mse)

        avg_mse = np.mean(mse_list)
        assert avg_mse < 0.001

    def test_op_reverse_transform(self):
        data_manager = OpenPowerDataManager(normalize=False, include_generation=True)
        normalized_data_manager = OpenPowerDataManager(
            normalize=True, include_generation=True
        )

        original_dataset = data_manager.create_pv_user_dataset()
        normalized_dataset = normalized_data_manager.create_pv_user_dataset()

        original_data = original_dataset.data.sort_values(
            by=["dataid", "year", "month", "weekday"]
        )
        normalized_data = normalized_dataset.data.sort_values(
            by=["dataid", "year", "month", "weekday"]
        )

        transformed = normalized_dataset.inverse_transform(normalized_data)

        mse_list: List[float] = []

        for idx in range(len(original_data)):
            unnormalized_timeseries = original_data.iloc[idx]["timeseries"]
            transformed_timeseries = transformed.iloc[idx]["timeseries"]

            mse = mean_squared_error(unnormalized_timeseries, transformed_timeseries)

            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        assert avg_mse < 0.001


if __name__ == "__main__":
    test = TestDataset()
    test.test_op_reverse_transform()
