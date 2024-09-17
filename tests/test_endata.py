#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `endata` package."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from datasets.pecanstreet import PecanStreetDataset
from eval.evaluator import Evaluator
from generator.gan.acgan import ACGAN
from generator.options import Options


class TestGenerator(unittest.TestCase):
    """Test ACGAN Generator."""

    def test_generator_output_shape(self):
        opt = Options("acgan")
        model = ACGAN(opt)
        noise = torch.randn(opt.batch_size, opt.noise_dim).to(
            opt.device
        )  # Batch of 32 samples with noise_dim=128
        month_labels = torch.randint(0, 12, (opt.batch_size,)).to(opt.device)
        day_labels = torch.randint(0, 7, (opt.batch_size,)).to(opt.device)

        generated_data = model.generator(noise, month_labels, day_labels).to(opt.device)
        self.assertEqual(
            generated_data.shape, (opt.batch_size, opt.seq_len, opt.input_dim)
        )

class TestDataset(unittest.TestCase):
    """Test Dataset Functionality."""

    def test_reverse_transform(self):
        pass

    
