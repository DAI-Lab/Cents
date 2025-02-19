import os
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from datasets.timeseries_dataset import TimeSeriesDataset
from endata.trainer import Trainer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SimpleTestDataset1D(TimeSeriesDataset):
    """A minimal example dataset with one time series column."""

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 16,
        normalize: bool = True,
        scale: bool = True,
    ):
        super().__init__(
            data=data,
            time_series_column_names=["time_series_col1"],
            seq_len=seq_len,
            conditioning_var_column_names=["conditioning_var"],
            normalize=normalize,
            scale=scale,
        )

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if (
            "time_series_col1" not in data.columns
            or "conditioning_var" not in data.columns
        ):
            raise ValueError("Missing required columns in data.")

        data["time_series_col1"] = data["time_series_col1"].apply(
            lambda x: np.array(x).reshape(-1, 1) if isinstance(x, list) else x
        )
        return data


class SimpleTestDataset2D(TimeSeriesDataset):
    """A minimal example dataset with two time series columns."""

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 16,
        normalize: bool = True,
        scale: bool = True,
    ):
        super().__init__(
            data=data,
            time_series_column_names=["time_series_col1", "time_series_col2"],
            seq_len=seq_len,
            conditioning_var_column_names=["conditioning_var"],
            normalize=normalize,
            scale=scale,
        )

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"time_series_col1", "time_series_col2", "conditioning_var"}
        if not required_cols.issubset(data.columns):
            raise ValueError("Missing required columns in data.")

        for col in ["time_series_col1", "time_series_col2"]:
            data[col] = data[col].apply(
                lambda x: np.array(x).reshape(-1, 1) if isinstance(x, list) else x
            )
        return data


@pytest.fixture
def raw_df_1d():
    """A small DataFrame for a 1D test dataset (untransformed)."""
    num_samples = 10
    seq_len = 16
    df = pd.DataFrame(
        {
            "time_series_col1": [
                np.random.rand(seq_len).tolist() for _ in range(num_samples)
            ],
            "conditioning_var": np.random.choice(["a", "b"], size=num_samples),
        }
    )
    return df


@pytest.fixture
def raw_df_2d():
    """A small DataFrame for a 2D test dataset (untransformed)."""
    num_samples = 10
    seq_len = 16
    df = pd.DataFrame(
        {
            "time_series_col1": [
                np.random.rand(seq_len).tolist() for _ in range(num_samples)
            ],
            "time_series_col2": [
                np.random.rand(seq_len).tolist() for _ in range(num_samples)
            ],
            "conditioning_var": np.random.choice(["a", "b"], size=num_samples),
        }
    )
    return df


@pytest.fixture
def raw_dataset_1d(raw_df_1d):
    """Dataset with normalize=False (keeps data in raw form)."""
    ds = SimpleTestDataset1D(raw_df_1d, normalize=False, scale=False)
    return ds


@pytest.fixture
def normalized_dataset_1d(raw_df_1d):
    """Dataset with normalize=True (automatically normalizes on init)."""
    ds = SimpleTestDataset1D(raw_df_1d, normalize=True, scale=True)
    return ds


@pytest.fixture
def raw_dataset_2d(raw_df_2d):
    ds = SimpleTestDataset2D(raw_df_2d, normalize=False, scale=False)
    return ds


@pytest.fixture
def normalized_dataset_2d(raw_df_2d):
    ds = SimpleTestDataset2D(raw_df_2d, normalize=True, scale=True)
    return ds


def load_top_level_config() -> DictConfig:
    config_dir = os.path.join(ROOT_DIR, "test", "test_configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="test_config", overrides=[])
    return cfg


def load_dataset_config(case: str) -> DictConfig:
    config_dir = os.path.join(ROOT_DIR, "test", "test_configs", "dataset")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        ds_cfg = compose(config_name=case, overrides=[])
    OmegaConf.set_struct(ds_cfg, False)
    return ds_cfg


def load_model_config(model_name: str) -> DictConfig:
    config_dir = os.path.join(ROOT_DIR, "test", "test_configs", "model")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        model_cfg = compose(config_name=model_name, overrides=[])
    return model_cfg


@pytest.fixture
def dataset_cfg_1d() -> DictConfig:
    ds_cfg = load_dataset_config("test1d")
    return ds_cfg


@pytest.fixture
def dataset_cfg_2d() -> DictConfig:
    ds_cfg = load_dataset_config("test2d")
    return ds_cfg


@pytest.fixture
def model_cfg_diffusion() -> DictConfig:
    return load_model_config("diffusion_ts")


@pytest.fixture
def model_cfg_acgan() -> DictConfig:
    return load_model_config("acgan")


@pytest.fixture
def full_cfg_1d(dataset_cfg_1d) -> DictConfig:
    top_cfg = load_top_level_config()
    full_cfg = OmegaConf.merge(top_cfg, {"dataset": dataset_cfg_1d})
    return full_cfg


@pytest.fixture
def full_cfg_2d(dataset_cfg_2d) -> DictConfig:
    top_cfg = load_top_level_config()
    full_cfg = OmegaConf.merge(top_cfg, {"dataset": dataset_cfg_2d})
    return full_cfg


@pytest.fixture
def dummy_trainer_diffusion_1d(full_cfg_1d, model_cfg_diffusion, normalized_dataset_1d):
    OmegaConf.set_struct(full_cfg_1d, False)
    merged_cfg = OmegaConf.merge(full_cfg_1d, {"model": model_cfg_diffusion})
    trainer = Trainer(
        model_name="diffusion_ts", dataset=normalized_dataset_1d, cfg=merged_cfg
    )
    return trainer


@pytest.fixture
def dummy_trainer_acgan_1d(full_cfg_1d, model_cfg_acgan, normalized_dataset_1d):
    OmegaConf.set_struct(full_cfg_1d, False)
    merged_cfg = OmegaConf.merge(full_cfg_1d, {"model": model_cfg_acgan})
    trainer = Trainer(model_name="acgan", dataset=normalized_dataset_1d, cfg=merged_cfg)
    return trainer


@pytest.fixture
def dummy_trainer_diffusion_2d(full_cfg_2d, model_cfg_diffusion, normalized_dataset_2d):
    OmegaConf.set_struct(full_cfg_2d, False)
    merged_cfg = OmegaConf.merge(full_cfg_2d, {"model": model_cfg_diffusion})
    trainer = Trainer(
        model_name="diffusion_ts", dataset=normalized_dataset_2d, cfg=merged_cfg
    )
    return trainer


@pytest.fixture
def dummy_trainer_acgan_2d(full_cfg_2d, model_cfg_acgan, normalized_dataset_2d):
    OmegaConf.set_struct(full_cfg_2d, False)
    merged_cfg = OmegaConf.merge(full_cfg_2d, {"model": model_cfg_acgan})
    trainer = Trainer(model_name="acgan", dataset=normalized_dataset_2d, cfg=merged_cfg)
    return trainer
