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
        pass

class TestDataset(unittest.TestCase):
    """Test Dataset Functionality."""

    def test_reverse_transform(self):
        pass

    
