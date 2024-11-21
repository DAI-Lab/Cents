import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from datasets.timeseries_dataset import TimeSeriesDataset
from datasets.utils import convert_generated_data_to_df
from generator.diffcharge.diffusion import DDPM
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.gan.acgan import ACGAN

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataGenerator:
    """
    A wrapper class for generative models.
    """

    def __init__(
        self,
        model_name: str,
        dataset: Optional[TimeSeriesDataset] = None,
        overrides: Optional[List[str]] = [],
    ):
        """
        Initialize the wrapper with the model name and model parameters.

        Args:
            model_name (str): The name of the generative model ('acgan', 'diffusion_ts', etc.).
        """
        self.model_name = model_name
        self.overrides = overrides
        self.cfg = self._load_config()
        self.set_dataset(dataset)
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
        if not hasattr(self, "dataset") or not self.dataset:
            raise ValueError(
                "Dataset not specified or None. Set the generator dataset using set_dataset()."
            )

        self.model.train_model(self.dataset)

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
            key: torch.tensor(value, dtype=torch.long)
            for key, value in conditioning_vars.items()
        }

        self.conditioning_var_buffer = conditioning_vars

    def generate(self, num_samples=100):
        """
        Generate data using the trained model.

        Args:
            num_samples: The number of timeseries to generate for the current conditioning_var_buffer.

        Returns:
            Generated data in a data frame.
        """
        if (
            not self.conditioning_var_buffer
            and self.cfg.dataset.conditioning_vars is not None
        ):
            raise ValueError(
                f"The following conditioning variables need to be set using set_model_conditioning_variables(): {self.cfg.dataset.conditioning_vars.keys()}"
            )

        conditioning_vars = {}
        for var_name, code in self.conditioning_var_buffer.items():
            conditioning_vars[var_name] = torch.full(
                (num_samples,), code, dtype=torch.long, device=self.cfg.device
            )
        data = self.model.generate(conditioning_vars)
        return convert_generated_data_to_df(
            data,
            self.conditioning_var_buffer,
            mapping=self.dataset.get_conditioning_var_codes(),
        )

    def set_dataset(self, dataset: TimeSeriesDataset):
        """
        Set the dataset for the DataGenerator.

        Args:
            dataset (TimeSeriesDataset): The dataset to be used.
        """
        self.dataset = dataset

        if hasattr(dataset, "name"):
            if self.dataset.name == "pecanstreet":
                with initialize_config_dir(
                    config_dir=os.path.join(ROOT_DIR, "config/dataset"),
                    version_base=None,
                ):
                    pecanstreet_cfg = compose(config_name="pecanstreet", overrides=None)
                    self.cfg.dataset = pecanstreet_cfg

            elif self.dataset.name == "openpower":
                with initialize_config_dir(
                    config_dir=os.path.join(ROOT_DIR, "config/dataset"),
                    version_base=None,
                ):
                    openpower_cfg = compose(config_name="openpower", overrides=None)
                    self.cfg.dataset = openpower_cfg
            else:
                if dataset.cfg:
                    self.cfg.dataset = dataset.cfg
                else:
                    print(
                        f"Warning: No cfg found for dataset {self.dataset.name}, setting default dataset config."
                    )
                    with initialize_config_dir(
                        config_dir=os.path.join(ROOT_DIR, "config/dataset"),
                        version_base=None,
                    ):
                        default_cfg = compose(config_name="default", overrides=None)
                        self.cfg.dataset = default_cfg

                if (
                    self.dataset.conditioning_vars
                    and not self.cfg.dataset.conditioning_vars
                ):
                    if not self.cfg.conditioning_vars:
                        print(
                            f"Warning: Found conditioning_vars {self.dataset.conditioning_vars} but dataset config conditioning_vars are not set."
                        )

        self._initialize_model()  # re init with dataset update

    def sample_conditioning_vars(self, num_samples, random=False):
        """
        Sample conditioning variables from the dataset.

        Args:
            num_samples (int): Number of samples to generate.
            random (bool): Whether to sample randomly or from the dataset.

        Returns:
            conditioning_vars: Dictionary of conditioning variables.
        """
        return self.model.sample_conditioning_vars(self.dataset, num_samples, random)

    def load_model(self):
        """
        Load the model, optimizer, and EMA model from a checkpoint file.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot load checkpoint.")

        if self.dataset is None:
            raise ValueError(
                "Dataset is not set. Cannot ensure compatibility when loading the model."
            )

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
                conditioning_var_column_names=conditioning_vars,
            )
            return dataset
        else:
            raise TypeError("Input X must be a Dataset or a DataFrame.")

    def _get_model_checkpoint_path(self):
        """
        Returns the checkpoint path for the data generator's model type.
        """
        project_root = str(Path(__file__).resolve().parent.parent)
        checkpoint_dir = os.path.join(
            project_root, f"checkpoints/models/{self.dataset.name}"
        )

        if self.model_name == "diffusion_ts":
            checkpoint_name = "diffusion_ts_checkpoint_1000.pt"
        elif self.model_name == "acgan":
            checkpoint_name = "acgan_checkpoint_200.pt"
        elif self.model_name == "diffcharge":
            checkpoint_name == "diffcharge_checkpoint_1000.pt"
        else:
            raise ValueError(f"No model checkpoint found for {self.model_name}.")
        return os.path.join(checkpoint_dir, checkpoint_name)
