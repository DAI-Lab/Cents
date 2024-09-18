#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `endata` package."""

import os
import unittest
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from datasets.pecanstreet import PecanStreetDataManager
from eval.evaluator import Evaluator
from generator.gan.acgan import ACGAN
from generator.options import Options

TEST_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data_config.yaml"
    )


class TestGenerator(unittest.TestCase):
    """Test ACGAN Generator."""

    def test_generator_output_shape(self):
        pass

class TestDataset(unittest.TestCase):
    """Test Dataset Functionality."""

    def test_data_manager(self):
        data_manager = PecanStreetDataManager(config_path=TEST_CONFIG_PATH, include_generation=False)
        assert data_manager.data.shape[0] > 0, "Dataframe not loaded correctly"
        assert "timeseries" in data_manager.data.columns
        
        ts_shape = data_manager.data.timeseries.iloc[0].shape
        assert ts_shape == (96, 1)

    def test_reverse_transform(self):
        data_manager = PecanStreetDataManager(config_path=TEST_CONFIG_PATH, include_generation=False, normalize=False, threshold=None)
        normalized_data_manager = PecanStreetDataManager(config_path=TEST_CONFIG_PATH, include_generation=False, normalize=True, threshold=None, normalization_method="global")

        original_dataset = data_manager.create_all_user_dataset()
        normalized_dataset = normalized_data_manager.create_all_user_dataset()

        original_data = original_dataset.data
        normalized_data = normalized_dataset.data

        transformed = normalized_dataset.inverse_transform(normalized_data)

        mse_list: List[float] = []

        for idx in range(len(original_data)):
            unnormalized_timeseries = original_data.iloc[idx]["timeseries"]
            transformed_timeseries = transformed.iloc[idx]["timeseries"]

            assert (
                unnormalized_timeseries.shape == transformed_timeseries.shape
            ), "Shape mismatch between transformed and unnormalized timeseries"

            mse = mean_squared_error(unnormalized_timeseries, transformed_timeseries)
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        assert avg_mse < 0.01
        

    
