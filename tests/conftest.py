import os
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from cents.utils.config_loader import load_yaml

from cents.datasets.timeseries_dataset import TimeSeriesDataset
from cents.trainer import Trainer

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
            context_var_column_names=["context_var"],
            normalize=normalize,
            scale=scale,
        )

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if "time_series_col1" not in data.columns or "context_var" not in data.columns:
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
            context_var_column_names=["context_var"],
            normalize=normalize,
            scale=scale,
        )

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"time_series_col1", "time_series_col2", "context_var"}
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
            "context_var": np.random.choice(["a", "b"], size=num_samples),
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
            "context_var": np.random.choice(["a", "b"], size=num_samples),
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
    path = os.path.join(ROOT_DIR, "tests", "test_configs", "test_config.yaml")
    return load_yaml(path)


def load_dataset_config(case: str) -> DictConfig:
    path = os.path.join(ROOT_DIR, "tests", "test_configs", "dataset", f"{case}.yaml")
    ds_cfg = load_yaml(path)
    OmegaConf.set_struct(ds_cfg, False)
    return ds_cfg


def load_model_config(model_type: str) -> DictConfig:
    path = os.path.join(ROOT_DIR, "tests", "test_configs", "model", f"{model_type}.yaml")
    return load_yaml(path)


def load_trainer_config(trainer_name: str) -> DictConfig:
    path = os.path.join(ROOT_DIR, "tests", "test_configs", "trainer", f"{trainer_name}.yaml")
    return load_yaml(path)


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
def trainer_cfg_normalizer() -> DictConfig:
    return load_trainer_config("normalizer")


@pytest.fixture
def trainer_cfg_diffusion() -> DictConfig:
    return load_trainer_config("diffusion_ts")


@pytest.fixture
def trainer_cfg_acgan() -> DictConfig:
    return load_trainer_config("acgan")


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
def dummy_trainer_diffusion_1d(
    full_cfg_1d,
    model_cfg_diffusion,
    normalized_dataset_1d,
    trainer_cfg_diffusion,
    tmp_path,
):
    OmegaConf.set_struct(full_cfg_1d, False)
    merged_cfg = OmegaConf.merge(full_cfg_1d, {"model": model_cfg_diffusion})
    merged_cfg = OmegaConf.merge(merged_cfg, {"trainer": trainer_cfg_diffusion})
    merged_cfg.run_dir = str(tmp_path)
    trainer = Trainer(
        model_type="diffusion_ts", dataset=normalized_dataset_1d, cfg=merged_cfg
    )
    return trainer


@pytest.fixture
def dummy_trainer_acgan_1d(
    full_cfg_1d, model_cfg_acgan, normalized_dataset_1d, trainer_cfg_acgan, tmp_path
):
    OmegaConf.set_struct(full_cfg_1d, False)
    merged_cfg = OmegaConf.merge(full_cfg_1d, {"model": model_cfg_acgan})
    merged_cfg = OmegaConf.merge(merged_cfg, {"trainer": trainer_cfg_acgan})
    merged_cfg.run_dir = str(tmp_path)
    trainer = Trainer(model_type="acgan", dataset=normalized_dataset_1d, cfg=merged_cfg)
    return trainer


@pytest.fixture
def dummy_trainer_diffusion_2d(
    full_cfg_2d,
    model_cfg_diffusion,
    normalized_dataset_2d,
    trainer_cfg_diffusion,
    tmp_path,
):
    OmegaConf.set_struct(full_cfg_2d, False)
    merged_cfg = OmegaConf.merge(full_cfg_2d, {"model": model_cfg_diffusion})
    merged_cfg = OmegaConf.merge(merged_cfg, {"trainer": trainer_cfg_diffusion})
    merged_cfg.run_dir = str(tmp_path)
    trainer = Trainer(
        model_type="diffusion_ts", dataset=normalized_dataset_2d, cfg=merged_cfg
    )
    return trainer


@pytest.fixture
def dummy_trainer_acgan_2d(
    full_cfg_2d, model_cfg_acgan, normalized_dataset_2d, trainer_cfg_acgan, tmp_path
):
    OmegaConf.set_struct(full_cfg_2d, False)
    merged_cfg = OmegaConf.merge(full_cfg_2d, {"model": model_cfg_acgan})
    merged_cfg = OmegaConf.merge(merged_cfg, {"trainer": trainer_cfg_acgan})
    merged_cfg.run_dir = str(tmp_path)
    trainer = Trainer(model_type="acgan", dataset=normalized_dataset_2d, cfg=merged_cfg)
    return trainer
