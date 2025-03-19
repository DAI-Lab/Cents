import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import botocore
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from endata.datasets.timeseries_dataset import TimeSeriesDataset
from endata.datasets.utils import convert_generated_data_to_df
from endata.eval.evaluator import Evaluator
from endata.generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from endata.generator.gan.acgan import ACGAN
from endata.generator.normalizer import Normalizer
from endata.utils.device import get_device

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class DataGenerator:
    def __init__(
        self,
        model_name: str,
        context_var_codes: dict = None,
        cfg: DictConfig = None,
        model_overrides: dict = None,
    ):
        self.model_name = model_name
        self.context_var_codes = context_var_codes
        if cfg:
            self.cfg = cfg
        else:
            self.cfg = self._load_config()

        self.context_var_buffer = {}

        if model_overrides:
            self.cfg.model = OmegaConf.merge(
                self.cfg.model, OmegaConf.create(model_overrides)
            )

        self.device = get_device(self.cfg.get("device", None))

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
            self.model = model_class(self.cfg).to(self.device)
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
            key: torch.tensor(value, dtype=torch.long, device=self.device)
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
                (num_samples,), code, dtype=torch.long, device=self.device
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

    def download_from_s3(self, bucket: str, s3_key: str, local_path: str) -> str:
        """
        Check if a file exists locally; if not, download it from S3.

        Args:
            bucket (str): Name of the S3 bucket.
            s3_key (str): S3 key (path) for the file.
            local_path (str): Local path where the file should be stored.

        Returns:
            str: The local path to the downloaded file.
        """
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3 = boto3.client(
                "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
            )
            print(f"Downloading {s3_key} to {local_path}...")
            s3.download_file(bucket, s3_key, local_path)
        else:
            print(f"Using cached file at {local_path}")
        return local_path

    def _get_model_checkpoint_path(self) -> str:
        dimensions = self.cfg.dataset.time_series_dims
        checkpoint_name = f"{self.model_name}_dim_{dimensions}.pt"
        bucket = "dai-watts"
        s3_key = f"{self.dataset_name}/{self.model_name}/{checkpoint_name}"

        local_cache_dir = os.path.join(
            Path.home(),
            ".cache",
            "endata",
            "checkpoints",
            self.dataset_name,
            self.model_name,
        )
        local_filename = checkpoint_name
        local_path = os.path.join(local_cache_dir, local_filename)
        return self.download_from_s3(bucket, s3_key, local_path)

    def _get_normalizer_checkpoint_path(self) -> str:
        dimensions = self.cfg.dataset.time_series_dims
        scale = self.cfg.dataset.scale
        checkpoint_name = f"normalizer_dim_{dimensions}_scale_{scale}.pt"
        bucket = "dai-watts"
        s3_key = f"{self.dataset_name}/normalizer/{checkpoint_name}"
        local_cache_dir = os.path.join(
            Path.home(),
            ".cache",
            "endata",
            "checkpoints",
            self.dataset_name,
            self.model_name,
        )
        local_filename = checkpoint_name
        local_path = os.path.join(local_cache_dir, local_filename)
        return self.download_from_s3(bucket, s3_key, local_path)

    def evaluate(self, dataset: TimeSeriesDataset, eval_config: dict = None):
        """
        Evaluate the model's performance on a dataset.

        Args:
            dataset: The dataset to evaluate on
            user_id: Optional user ID to evaluate on specific user
            eval_config: Optional configuration overrides for evaluation
        """
        if not hasattr(self, "model"):
            raise ValueError("Model not loaded. Call load_model() first.")

        if eval_config:
            self.cfg.evaluator = OmegaConf.merge(
                self.cfg.evaluator, OmegaConf.create(eval_config)
            )

        if not self.evaluator:
            self.evaluator = Evaluator(self.cfg, dataset)

        self.evaluator.evaluate_model(model=self.model)
