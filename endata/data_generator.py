import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from datasets.utils import convert_generated_data_to_df
from generator.diffcharge.diffusion import DDPM
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.gan.acgan import ACGAN

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataGenerator:
    """
    A wrapper class for generative models focusing on data generation.
    """

    def __init__(
        self,
        model_name: str,
        cfg: DictConfig = None,
        model: nn.Module = None,
        conditioning_var_codes: dict = None,
        overrides: Optional[List[str]] = [],
    ):
        """
        Initialize the DataGenerator with the model name and any configuration overrides.

        Args:
            model_name (str): The name of the generative model ('acgan', 'diffusion_ts', etc.).
            overrides (List[str], optional): Configuration overrides.
        """
        self.model_name = model_name
        self.overrides = overrides
        if cfg:
            self.cfg = cfg
            self._set_dataset_config(self.cfg.dataset.name, cfg.dataset)
        else:
            self._load_config()
        if model:
            self.model = model
        else:
            self._initialize_model()
        self.conditioning_var_buffer = {}
        self.conditioning_var_codes = (
            conditioning_var_codes if conditioning_var_codes is not None else {}
        )

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

    def _set_dataset_config(self, dataset_name: str, cfg: DictConfig = None):
        """
        Set the dataset configuration based on the dataset name without loading the actual data.

        Args:
            dataset_name (str): The name of the dataset ('pecanstreet', 'openpower', etc.).
            cfg (DictConfig): Optionally directly takes a config object that is set.
        """
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
        self._initialize_model()
        self.conditioning_var_codes = self.get_conditioning_var_codes()

    def get_conditioning_var_codes(self):
        """
        Get conditioning variable mappings available for the current dataset config.
        """
        if not hasattr(self, "dataset_name"):
            raise ValueError(
                "Dataset name not set. Please set 'self.dataset_name' to the dataset name or call self.set_dataset_config()."
            )

        if self.conditioning_var_codes:
            return self.conditioning_var_codes

        conditioning_codes_path = os.path.join(
            ROOT_DIR, "data", self.dataset_name, "conditioning_var_codes.json"
        )

        if not os.path.exists(conditioning_codes_path):
            print(f"No conditioning variable codes found at {conditioning_codes_path}.")
            return {}
        else:
            with open(conditioning_codes_path, "r") as f:
                conditioning_var_codes = json.load(f)

            formatted_codes = json.dumps(conditioning_var_codes, indent=4)
            print(formatted_codes)

            for outer_key, inner_dict in conditioning_var_codes.items():
                conditioning_var_codes[outer_key] = {
                    int(k): v for k, v in inner_dict.items()
                }

            return conditioning_var_codes

    def set_model_conditioning_vars(self, conditioning_vars):
        """
        Set conditioning variables for data generation.

        Args:
            conditioning_vars (Dict[str, int]): Sets the column name and code of the conditioning variables.
        """
        if not self.cfg.dataset.conditioning_vars:
            raise ValueError(
                "Conditioning variables are not set in the dataset configuration."
            )

        for var_name, code in conditioning_vars.items():
            if var_name not in self.cfg.dataset.conditioning_vars:
                raise ValueError(f"Invalid conditioning variable: {var_name}")
            possible_values = list(
                range(0, self.cfg.dataset.conditioning_vars[var_name])
            )
            if code not in possible_values:
                raise ValueError(
                    f"Invalid code '{code}' for conditioning variable '{var_name}'. Possible values: {possible_values}"
                )

        self.conditioning_var_buffer = {
            key: torch.tensor(value, dtype=torch.long, device=self.cfg.device)
            for key, value in conditioning_vars.items()
        }

    def generate(self, num_samples=100):
        """
        Generate data using the trained model.

        Args:
            num_samples (int): The number of timeseries to generate.

        Returns:
            Generated data in a data frame.
        """
        if not self.conditioning_var_buffer:
            raise ValueError(
                f"The following conditioning variables need to be set using set_model_conditioning_vars(): {list(self.cfg.dataset.conditioning_vars.keys())}"
            )

        conditioning_vars = {}
        for var_name, code in self.conditioning_var_buffer.items():
            conditioning_vars[var_name] = torch.full(
                (num_samples,), code, dtype=torch.long, device=self.cfg.device
            )

        data = self.model.generate(conditioning_vars)
        df = convert_generated_data_to_df(
            data, self.conditioning_var_buffer, self.conditioning_var_codes
        )
        return df

    def load_model(self, dataset_name: str):
        """
        Load the model from a checkpoint file.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot load checkpoint.")

        self._set_dataset_config(dataset_name)

        checkpoint_path = self._get_model_checkpoint_path()
        self.model.load(checkpoint_path)

    def _get_model_checkpoint_path(self):
        """
        Returns the checkpoint path for the data generator's model type.
        """
        project_root = str(Path(__file__).resolve().parent.parent)
        checkpoint_dir = os.path.join(
            project_root, f"checkpoints/models/{self.dataset_name}"
        )

        if self.model_name == "diffusion_ts":
            checkpoint_name = "diffusion_ts_checkpoint_1000.pt"
        elif self.model_name == "acgan":
            checkpoint_name = "acgan_checkpoint_200.pt"
        elif self.model_name == "diffcharge":
            checkpoint_name = "diffcharge_checkpoint_1000.pt"
        else:
            raise ValueError(f"No model checkpoint found for {self.model_name}.")

        return os.path.join(checkpoint_dir, checkpoint_name)
