import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from datasets.timeseries_dataset import TimeSeriesDataset
from endata.data_generator import DataGenerator
from generator.diffcharge.diffusion import DDPM
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.gan.acgan import ACGAN

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Trainer:
    """
    A class to handle training of generative models.
    """

    def __init__(
        self,
        model_name: str,
        dataset: TimeSeriesDataset,
        overrides: Optional[List[str]] = [],
    ):
        """
        Initialize the Trainer with the model name, dataset, and any configuration overrides.

        Args:
            model_name (str): The name of the generative model ('acgan', 'diffusion_ts', etc.).
            dataset (TimeSeriesDataset): The dataset to train on.
            overrides (List[str], optional): Configuration overrides.
        """
        self.model_name = model_name
        self.overrides = overrides
        self.dataset = dataset
        self.cfg = self._load_config()
        self.model = None
        self._initialize_model()

    def _load_config(self) -> DictConfig:
        """
        Load and merge the global config and model-specific config, then apply overrides.

        Returns:
            DictConfig: The merged configuration.
        """
        config_dir = os.path.join(ROOT_DIR, "config")

        self.overrides = [f"model={self.model_name}"] + self.overrides

        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="config", overrides=self.overrides)

        dataset_name = self.dataset.name if hasattr(self.dataset, "name") else "default"

        if self.dataset.cfg:
            cfg.dataset = self.dataset.cfg
        else:
            dataset_config_path = os.path.join(
                ROOT_DIR, f"config/dataset/{dataset_name}.yaml"
            )
            if os.path.exists(dataset_config_path):
                with initialize_config_dir(
                    config_dir=os.path.join(ROOT_DIR, "config/dataset"),
                    version_base=None,
                ):
                    dataset_cfg = compose(config_name=dataset_name, overrides=None)
                cfg.dataset = dataset_cfg
            else:
                print(
                    f"Warning: No config found for dataset {dataset_name}, using default dataset config."
                )

        return cfg

    def _initialize_model(self):
        """
        Initialize the model based on the model name and parameters.
        """
        model_dict = {
            "acgan": ACGAN,
            "diffusion_ts": Diffusion_TS,
            "diffcharge": DDPM,
        }
        if self.model_name in model_dict:
            model_class = model_dict[self.model_name]
            self.model = model_class(self.cfg).to(self.cfg.device)
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

    def fit(self):
        """
        Train the model.
        """
        if not self.dataset:
            raise ValueError("Dataset not specified or None.")

        self.model.train_model(self.dataset)

    def get_data_generator(self):
        """
        Create a data generator from the trained model.
        """
        data_generator = DataGenerator(
            model_name=self.model_name,
            cfg=self.cfg,
            model=self.model,
            conditioning_var_codes=(
                self.dataset.conditioning_var_codes
                if hasattr(self.dataset, "conditioning_var_codes")
                else None
            ),
        )
        return data_generator
