"""Tests for `endata` package."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import mean_squared_error

from datasets.openpower import OpenPowerDataManager
from datasets.pecanstreet import PecanStreetDataManager
from generator.gan.acgan import ACGAN

TEST_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data_config.yaml"
)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDataset(unittest.TestCase):
    """Test Dataset Functionality."""

    def test_openpower_data_manager(self):
        """Test OpenPowerDataManager and OpenPowerDataset."""
        # Initialize the config for OpenPowerDataManager
        with initialize_config_dir(
            config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
        ):
            cfg = compose(config_name="openpower", overrides=[])

        data_manager = OpenPowerDataManager(cfg=cfg)
        dataset = data_manager.create_dataset()
        self.assertGreater(len(dataset), 0, "Dataset should not be empty.")

        timeseries, conditioning_vars = dataset[0]
        self.assertIsInstance(
            timeseries, torch.Tensor, "Timeseries should be a torch.Tensor."
        )
        expected_shape = (96, 2) if data_manager.include_generation else (96, 1)
        self.assertEqual(
            timeseries.shape,
            expected_shape,
            f"Timeseries shape should be {expected_shape}.",
        )

    def test_pecanstreet_data_manager(self):
        """Test PecanStreetDataManager and PecanStreetDataset."""
        # Initialize the config for PecanStreetDataManager
        with initialize_config_dir(
            config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
        ):
            cfg = compose(config_name="pecanstreet", overrides=[])

        data_manager = PecanStreetDataManager(cfg=cfg)
        dataset = data_manager.create_dataset()

        # Basic assertions
        self.assertGreater(len(dataset), 0, "Dataset should not be empty.")

        # Test that timeseries and conditioning variables are correct
        timeseries, conditioning_vars = dataset[0]
        self.assertIsInstance(
            timeseries, torch.Tensor, "Timeseries should be a torch.Tensor."
        )
        expected_shape = (96, 2) if data_manager.include_generation else (96, 1)
        self.assertEqual(
            timeseries.shape,
            expected_shape,
            f"Timeseries shape should be {expected_shape}.",
        )

        # Check conditioning variables
        self.assertIsInstance(
            conditioning_vars, dict, "Conditioning variables should be a dictionary."
        )
        expected_keys = [
            "month",
            "weekday",
            "building_type",
            "car1",
            "city",
            "state",
            "has_solar",
            "total_square_footage",
            "house_construction_year",
        ]
        for key in expected_keys:
            self.assertIn(
                key, conditioning_vars, f"Conditioning variable '{key}' missing."
            )
            self.assertIsInstance(
                conditioning_vars[key],
                torch.Tensor,
                f"Conditioning variable '{key}' should be a torch.Tensor.",
            )

    def test_openpower_inverse_transform(self):
        """Test inverse transformation for OpenPowerDataset."""
        with initialize_config_dir(
            config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
        ):
            cfg_normalized = compose(
                config_name="openpower", overrides=["normalize=True"]
            )
            cfg_unnormalized = compose(
                config_name="openpower", overrides=["normalize=False"]
            )

        data_manager_normalized = OpenPowerDataManager(cfg=cfg_normalized)
        data_manager_unnormalized = OpenPowerDataManager(cfg=cfg_unnormalized)
        dataset_normalized = data_manager_normalized.create_dataset()
        dataset_unnormalized = data_manager_unnormalized.create_dataset()

        transformed_df = dataset_normalized.inverse_transform(
            dataset_normalized.data
        ).sort_values(by=["dataid", "year", "month", "date_day"])
        original_df = dataset_unnormalized.data.sort_values(
            by=["dataid", "year", "month", "date_day"]
        )

        mse_list: List[float] = []

        for idx in range(len(original_df)):
            original_timeseries = original_df.iloc[idx]["timeseries"]
            transformed_timeseries = transformed_df.iloc[idx]["timeseries"]

            mse = mean_squared_error(
                original_timeseries.flatten(), transformed_timeseries.flatten()
            )
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        self.assertLess(
            avg_mse, 0.001, f"Average MSE should be less than 0.001, but got {avg_mse}."
        )

    def test_pecanstreet_inverse_transform(self):
        """Test inverse transformation for PecanStreetDataset."""
        with initialize_config_dir(
            config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
        ):
            cfg_normalized = compose(
                config_name="pecanstreet", overrides=["normalize=True"]
            )
            cfg_unnormalized = compose(
                config_name="pecanstreet", overrides=["normalize=False"]
            )

        data_manager_normalized = PecanStreetDataManager(cfg=cfg_normalized)
        data_manager_unnormalized = PecanStreetDataManager(cfg=cfg_unnormalized)
        dataset_normalized = data_manager_normalized.create_dataset()
        dataset_unnormalized = data_manager_unnormalized.create_dataset()

        transformed_df = dataset_normalized.inverse_transform(dataset_normalized.data)
        original_df = dataset_unnormalized.data

        mse_list: List[float] = []

        for idx in range(len(original_df)):
            original_timeseries = original_df.iloc[idx]["timeseries"]
            transformed_timeseries = transformed_df.iloc[idx]["timeseries"]

            mse = mean_squared_error(
                original_timeseries.flatten(), transformed_timeseries.flatten()
            )
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        self.assertLess(
            avg_mse, 0.001, f"Average MSE should be less than 0.001, but got {avg_mse}."
        )


if __name__ == "__main__":
    unittest.main()
