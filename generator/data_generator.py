import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from datasets.timeseries_dataset import TimeSeriesDataset
from generator.diffcharge.diffusion import DDPM
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.gan.acgan import ACGAN


class DataGenerator:
    """
    A wrapper class for generative models.
    """

    @hydra.main(version_base=None, config_path="config", config_name="config")
    def __init__(self, model_name: str, overrides: Optional[List[str]] = None):
        """
        Initialize the wrapper with the model name and model parameters.

        Args:
            model_name (str): The name of the generative model ('acgan', 'diffusion_ts', etc.).
        """
        self.model_name = model_name
        self.cfg = self._load_config()
        self.model = None
        self._initialize_model()

    def _load_config(self) -> DictConfig:
        """
        Load and merge the global config and model-specific config, then apply overrides.

        Returns:
            DictConfig: The merged configuration.
        """
        project_root = Path(__file__).resolve().parent.parent
        config_dir = project_root / "config"
        global_config_path = config_dir / "config.yaml"
        model_config_path = config_dir / "models" / f"{self.model_name}.yaml"

        if not global_config_path.exists():
            raise FileNotFoundError(
                f"Global config file not found at {global_config_path}"
            )
        global_cfg = OmegaConf.load(global_config_path)

        if not model_config_path.exists():
            raise FileNotFoundError(
                f"Model config file not found at {model_config_path}"
            )
        model_cfg = OmegaConf.load(model_config_path)
        merged_cfg = OmegaConf.merge(global_cfg, model_cfg)

        if self.overrides:
            overrides_conf = OmegaConf.from_dotlist(self.overrides)
            merged_cfg = OmegaConf.merge(merged_cfg, overrides_conf)

        return merged_cfg

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
            self.model = model_class(self.cfg)
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

    def fit(self, dataset):
        """
        Train the model on the given dataset.

        Args:
            df (Any): Input data. Should be a compatible dataset object or pandas DataFrame.
        """
        sample_timeseries, sample_cond_vars = dataset[0]
        expected_seq_len = self.model.cfg.model.seq_len  # Access via cfg
        assert (
            sample_timeseries.shape[0] == expected_seq_len
        ), f"Expected timeseries length {expected_seq_len}, but got {sample_timeseries.shape[0]}"

        if self.model.conditioning_vars:
            for var in self.model.conditioning_vars:
                assert (
                    var in sample_cond_vars.keys()
                ), f"Set conditioning variable '{var}' specified in model.conditioning_vars not found in dataset"

        expected_input_dim = self.model.cfg.model.input_dim

        assert sample_timeseries.shape == (
            expected_seq_len,
            expected_input_dim,
        ), f"Expected timeseries shape ({expected_seq_len}, {expected_input_dim}), but got {sample_timeseries.shape}"

        self.model.train_model(dataset)

    def get_model_conditioning_vars(self):
        """
        Get conditioning variable mappings available for the current dataset config.
        """
        if not self.dataset or not self.cfg.dataset.conditioning_vars:
            return {}
        elif self.dataset:
            return self.dataset.get_conditioning_variables_integer_mapping()

    def set_model_conditioning_vars(self, conditioning_vars):
        """
        Set conditioning variables model training and generation is conditioned on.

        Args:
            conditioning_vars (Dict[str, int]): Sets the column name and number of categories of the conditioning column.
        """
        if not self.dataset:
            raise ValueError(
                "You need to set a dataset before setting conditioning variables!"
            )

        conditioning_vars = {
            key: torch.tensor(value, dtype=torch.long())
            for key, value in conditioning_vars.items()
        }

        self.conditioning_var_buffer = conditioning_vars

    def generate(self):
        """
        Generate data using the trained model.

        Args:
            conditioning_vars: The conditioning variables for generation.

        Returns:
            Generated data.
        """
        if (
            not self.conditioning_var_buffer
            and self.dataset.conditioning_vars is not None
        ):
            raise ValueError(
                f"The following conditioning variables need to be set using set_model_conditioning_variables(): {self.dataset.conditioning_vars.keys()}"
            )

        return self.model.generate(self.conditioning_var_buffer)

    def sample_conditioning_vars(self, dataset, num_samples, random=False):
        """
        Sample conditioning variables from the dataset.

        Args:
            dataset: The dataset to sample from.
            num_samples (int): Number of samples to generate.
            random (bool): Whether to sample randomly or from the dataset.

        Returns:
            conditioning_vars: Dictionary of conditioning variables.
        """
        return self.model.sample_conditioning_vars(dataset, num_samples, random)

    def load_model(self):
        """
        Load the model, optimizer, and EMA model from a checkpoint file.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot load checkpoint.")

        checkpoint_path = self._get_model_checkpoint_path()
        self.model.load(checkpoint_path)

    def _prepare_dataset(
        self, df: pd.DataFrame, timeseries_colname: str, conditioning_vars: Dict = None
    ):
        """
        Convert a pandas DataFrame into the required dataset format.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            dataset: The dataset in the required format.
        """
        if isinstance(df, torch.utils.data.Dataset):
            return df
        elif isinstance(df, pd.DataFrame):
            dataset = TimeSeriesDataset(
                dataframe=df,
                time_series_column_name=timeseries_colname,
                conditioning_vars=conditioning_vars,
            )
            return dataset
        else:
            raise TypeError("Input X must be a Dataset or a DataFrame.")

    def _get_model_checkpoint_path(self):
        """
        Returns the checkpoint path for the data generator's model type.
        """
        project_root = str(Path(__file__).resolve().parent.parent)
        checkpoint_dir = os.path.join(project_root, "checkpoints/models/")

        if self.model_name == "diffusion_ts":
            checkpoint_name = "diffusion_ts_checkpoint_1000.pt"
        elif self.model_name == "acgan":
            checkpoint_name = "acgan_checkpoint_200.pt"
        elif self.model_name == "diffcharge":
            checkpoint_name == "diffcharge_checkpoint_1000.pt"
        else:
            raise ValueError(f"No model checkpoint found for {self.model_name}.")
        return os.path.join(checkpoint_dir, checkpoint_name)
