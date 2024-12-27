import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from datasets.timeseries_dataset import TimeSeriesDataset
from datasets.utils import encode_conditioning_variables

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PecanStreetDataset(TimeSeriesDataset):
    """
    A dataset class for handling and preprocessing PecanStreet time series data,
    including normalization, handling PV data, and user-specific data retrieval.

    Attributes:
        cfg (DictConfig): The hydra config file
    """

    def __init__(self, cfg: DictConfig = None):
        if not cfg:
            with initialize_config_dir(
                config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
            ):
                cfg = compose(config_name="pecanstreet", overrides=None)

        self.cfg = cfg
        self.name = cfg.name
        self.geography = cfg.geography
        self.normalize = cfg.normalize
        self.threshold = (-1 * int(cfg.threshold), int(cfg.threshold))
        self.include_generation = cfg.include_generation
        self._load_data()
        self._set_user_flags()

        time_series_column_names = ["grid"]

        if self.include_generation:
            time_series_column_names.append("solar")

        conditioning_vars = list(self.cfg.conditioning_vars.keys())
        # normalization_group_keys = ["dataid", "month", "weekday"]
        # normalization_group_keys = ["month", "weekday"]
        normalization_group_keys = []

        super().__init__(
            data=self.data,
            entity_column_name="dataid",
            time_series_column_names=time_series_column_names,
            conditioning_var_column_names=conditioning_vars,
            seq_len=self.cfg.seq_len,
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            normalization_group_keys=normalization_group_keys,
        )

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the csv files into a pandas dataframe object.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))
        metadata_csv_path = os.path.join(path, "metadata.csv")

        if not os.path.exists(metadata_csv_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_csv_path}")

        self.metadata = pd.read_csv(
            metadata_csv_path, usecols=self.cfg.metadata_columns
        )

        if "solar" in self.metadata.columns:  # naming conflicts
            self.metadata.rename(columns={"solar": "has_solar"}, inplace=True)

        if self.geography:
            data_file_name = f"15minute_data_{self.geography}.csv"
            data_file_path = os.path.join(path, data_file_name)
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found at {data_file_path}")
            self.data = pd.read_csv(data_file_path)[self.cfg.data_columns]
        else:
            data_files = [
                os.path.join(path, "15minute_data_newyork.csv"),
                os.path.join(path, "15minute_data_california.csv"),
                os.path.join(path, "15minute_data_austin.csv"),
            ]
            for data_file in data_files:
                if not os.path.exists(data_file):
                    raise FileNotFoundError(f"Data file not found at {data_file}")
            self.data = pd.concat(
                [pd.read_csv(data_file) for data_file in data_files],
                axis=0,
            )[self.cfg.data_columns]

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the dataset by adding date-related columns, sorting, filtering, and normalizing.

        Args:
            data (pd.DataFrame): Raw data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        data["local_15min"] = pd.to_datetime(data["local_15min"], utc=True)
        data["month"] = data["local_15min"].dt.month_name()
        data["weekday"] = data["local_15min"].dt.day_name()
        data["date_day"] = data["local_15min"].dt.day
        data = data.sort_values(by=["local_15min"]).copy()
        data = data[~data["grid"].isna()]

        grouped_data = (
            data.groupby(["dataid", "month", "date_day", "weekday"])["grid"]
            .apply(np.array)
            .reset_index()
        )
        filtered_data = grouped_data[
            grouped_data["grid"].apply(len) == self.cfg.seq_len
        ].reset_index(drop=True)

        if self.include_generation:
            solar_data = self._preprocess_solar(data)
            filtered_data = pd.merge(
                filtered_data,
                solar_data,
                how="left",
                on=["dataid", "month", "weekday", "date_day"],
            )
        data = pd.merge(filtered_data, self.metadata, on="dataid", how="left")
        data = self._get_user_group_data(data)
        data = self._handle_missing_data(data)
        grouped_data.sort_values(by=["month", "weekday", "date_day"], inplace=True)
        return data

    def _preprocess_solar(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses solar data by filtering and normalizing if required.

        Args:
            data (pd.DataFrame): The raw data containing solar information.

        Returns:
            pd.DataFrame: Preprocessed solar data.
        """
        solar_data = (
            data[~data["solar"].isna()]
            .groupby(["dataid", "month", "date_day", "weekday"])["solar"]
            .apply(np.array)
            .reset_index()
        )
        solar_data = solar_data[solar_data["solar"].apply(len) == self.cfg.seq_len]
        return solar_data

    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data["car1"] = data["car1"].fillna("no")
        data["has_solar"] = data["has_solar"].fillna("no")
        data["house_construction_year"] = data["house_construction_year"].fillna(
            data["house_construction_year"].mean(skipna=True)
        )
        data["total_square_footage"] = data["total_square_footage"].fillna(
            data["total_square_footage"].mean(skipna=True)
        )
        assert data.isna().sum().sum() == 0, "Missing data remaining!"
        return data

    def _set_user_flags(self) -> Dict[int, bool]:
        """
        Sets user flags indicating whether a user has solar generation data.
        """
        self.user_flags = {
            user_id: self.metadata.loc[self.metadata["dataid"] == user_id]["has_solar"]
            .notna()
            .any()
            for user_id in self.data["dataid"].unique()
        }

    def _get_user_group_data(self, data: pd.DataFrame) -> "PecanStreetDataset":
        if self.cfg.user_id:
            return data[data["dataid"] == self.cfg.user_id].copy()

        if self.cfg.user_group == "pv_users":
            users = [user for user, has_pv in self.user_flags.items() if has_pv]
            return data[data["dataid"].isin(users)].copy()

        elif self.cfg.user_group == "non_pv_users":
            assert (
                self.include_generation == False
            ), "Include_generation must be set to False when working with the non pv user dataset!"
            users = [user for user, has_pv in self.user_flags.items() if not has_pv]
            return data[data["dataid"].isin(users)].copy()

        elif self.cfg.user_group == "all":
            assert (
                self.include_generation == False
            ), "Include_generation must be set to False when working with the entire dataset!"
            return data.copy()

        else:
            raise ValueError(f"User group {self.cfg.user_group} is not specified.")
