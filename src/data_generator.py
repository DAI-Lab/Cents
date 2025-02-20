import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.datasets.utils import convert_generated_data_to_df
from src.generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from src.generator.gan.acgan import ACGAN
from src.generator.normalizer import Normalizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class DataGenerator:
    def __init__(
        self,
        model_name: str,
        context_var_codes: dict = None,
        cfg: DictConfig = None,
    ):
        self.model_name = model_name
        self.context_var_codes = context_var_codes
        if cfg:
            self.cfg = cfg
        else:
            self.cfg = self._load_config()

        self.context_var_buffer = {}

    def _load_config(self) -> DictConfig:
        config_dir = os.path.join(ROOT_DIR, "config")
        self.overrides = [
            f"model={self.model_name}",
            "wandb_enabled=False",
        ]
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="config", overrides=self.overrides)
        return cfg

    def _init_model(self, model_path: str):
        model_dict = {
            "acgan": ACGAN,
            "diffusion_ts": Diffusion_TS,
        }
        if self.model_name in model_dict:
            model_class = model_dict[self.model_name]
            self.model = model_class(self.cfg).to(self.cfg.device)
            self.model.load(model_path)
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

    def _set_dataset_config(self, dataset_name: str, cfg: DictConfig = None):

        if cfg:
            self.cfg.dataset = cfg
            self.dataset_name = dataset_name
            return

        config_dir = os.path.join(ROOT_DIR, "config/dataset")
        dataset_config_path = os.path.join(config_dir, f"{dataset_name}.yaml")
        if os.path.exists(dataset_config_path):
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                dataset_cfg = compose(config_name=dataset_name, overrides=None)
            self.cfg.dataset = dataset_cfg
            self.dataset_name = dataset_name
        else:
            print(
                f"Warning: No config found for dataset {dataset_name}, using default dataset config."
            )
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                default_cfg = compose(config_name="default", overrides=None)
            self.cfg.dataset = default_cfg
            self.dataset_name = "default"

    def get_context_var_codes(self):
        if not hasattr(self, "cfg"):
            raise ValueError(
                "Config not set. Please set 'self.dataset_name' to the dataset name or call self.set_dataset_config()."
            )

        if self.context_var_codes is not None:
            return self.context_var_codes

        context_codes_path = os.path.join(
            ROOT_DIR, "data", self.dataset_name, "context_var_codes.json"
        )
        if not os.path.exists(context_codes_path):
            print(f"No context variable codes found at {context_codes_path}.")
            return {}
        else:
            with open(context_codes_path, "r") as f:
                context_var_codes = json.load(f)
            for outer_key, inner_dict in context_var_codes.items():
                context_var_codes[outer_key] = {
                    int(k): v for k, v in inner_dict.items()
                }
            return context_var_codes

    def set_model_context_vars(self, context_vars):
        if not self.cfg.dataset.context_vars:
            raise ValueError(
                "context variables are not set in the dataset configuration."
            )
        for var_name, code in context_vars.items():
            if var_name not in self.cfg.dataset.context_vars:
                raise ValueError(f"Invalid context variable: {var_name}")
            possible_values = list(range(self.cfg.dataset.context_vars[var_name]))
            if code not in possible_values:
                raise ValueError(
                    f"Invalid code '{code}' for context variable '{var_name}'. Possible values: {possible_values}"
                )
        self.context_var_buffer = {
            key: torch.tensor(value, dtype=torch.long, device=self.cfg.device)
            for key, value in context_vars.items()
        }

    def generate(self, num_samples=100):

        if not self.context_var_buffer:
            raise ValueError(
                f"The following context variables need to be set using set_model_context_vars(): {list(self.cfg.dataset.context_vars.keys())}"
            )

        context_vars = {}
        for var_name, code in self.context_var_buffer.items():
            context_vars[var_name] = torch.full(
                (num_samples,), code, dtype=torch.long, device=self.cfg.device
            )

        data = self.model.generate(context_vars)
        df = convert_generated_data_to_df(data, self.context_var_buffer, decode=False)

        if self.normalizer:
            inv_data = self.normalizer._inverse_transform(df)
            return inv_data
        else:
            return df

    def load_model(
        self,
        dataset_name: str,
        dataset_cfg: str = None,
        model: Any = None,
        normalizer: Any = None,
        model_path: Any = None,
        normalizer_path: Any = None,
    ):

        self.dataset_name = dataset_name

        if not self.cfg.dataset:
            self._set_dataset_config(dataset_name, dataset_cfg)

        if model:
            self.model = model
        if normalizer:
            self.normalizer = normalizer

        if not model and not model_path:
            model_path = self._get_model_checkpoint_path()
            self._init_model(model_path)

        if not normalizer and not normalizer_path:
            normalizer_path = self._get_normalizer_checkpoint_path()
            self._init_normalizer(normalizer_path)

    def _init_normalizer(self, normalizer_path: str):

        self.normalizer = Normalizer(
            dataset_cfg=self.cfg.dataset, dataset=None, normalizer_path=normalizer_path
        )

    def _get_model_checkpoint_path(self):
        project_root = str(Path(__file__).resolve().parent)
        checkpoint_dir = os.path.join(
            project_root, f"checkpoints/{self.dataset_name}/{self.model_name}"
        )
        time_series_dims = self.cfg.dataset.time_series_dims
        if self.model_name == "diffusion_ts":
            checkpoint_name = (
                f"diffusion_ts_checkpoint_dim_{time_series_dims}_cond_01_5000.pt"
            )
        elif self.model_name == "acgan":
            checkpoint_name = (
                f"acgan_checkpoint_dim_{time_series_dims}_cond_01_aux_1_5000.pt"
            )
        else:
            raise ValueError(f"No model checkpoint found for {self.model_name}.")
        return os.path.join(checkpoint_dir, checkpoint_name)

    def _get_normalizer_checkpoint_path(self):
        project_root = str(Path(__file__).resolve().parent)
        checkpoint_dir = os.path.join(
            project_root, f"checkpoints/{self.dataset_name}/normalizer"
        )
        time_series_dims = self.cfg.dataset.time_series_dims
        scale = self.cfg.dataset.scale
        return os.path.join(
            checkpoint_dir,
            f"{self.dataset_name}_dim_{time_series_dims}_scale_{scale}_normalizer.pt",
        )
