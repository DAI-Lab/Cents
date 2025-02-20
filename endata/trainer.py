# trainer.py

import os
from typing import List, Optional

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from endata.data_generator import DataGenerator
from endata.datasets.timeseries_dataset import TimeSeriesDataset
from endata.generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from endata.generator.gan.acgan import ACGAN

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Trainer:
    def __init__(self, model_name: str, dataset: TimeSeriesDataset, cfg=None):
        self.model_name = model_name
        self.dataset = dataset
        self.cfg = self._load_config() if not cfg else cfg
        self.model = None
        self._initialize_model()

    def _load_config(self) -> DictConfig:
        config_dir = os.path.join(ROOT_DIR, "config")
        self.overrides = [f"model={self.model_name}"]
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
        model_dict = {
            "acgan": ACGAN,
            "diffusion_ts": Diffusion_TS,
        }
        if self.model_name in model_dict:
            model_class = model_dict[self.model_name]
            self.model = model_class(self.cfg).to(self.cfg.device)
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

    def fit(self):
        if not self.dataset:
            raise ValueError("Dataset not specified or None.")
        self.model.train_model(self.dataset)

    def get_data_generator(self):
        data_generator = DataGenerator(
            model_name=self.model_name,
            context_var_codes=(
                self.dataset.context_var_codes
                if hasattr(self.dataset, "context_var_codes")
                else {}
            ),
            cfg=self.cfg,
        )
        data_generator.load_model(
            self.dataset.name,
            self.cfg.dataset,
            model=self.model,
            normalizer=self.dataset._normalizer,
        )
        return data_generator
