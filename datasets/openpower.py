import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datasets.timeseries_dataset import TimeSeriesDataset
from datasets.utils import encode_conditioning_variables

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OpenPowerDataset(TimeSeriesDataset):
    """
    A dataset class for handling and preprocessing OpenPower time series data,
    including normalization, handling PV data, and user-specific data retrieval.

    Attributes:
        cfg (DictConfig): The hydra config file
    """

    def __init__(self, cfg: DictConfig = None):
        if not cfg:
            with initialize_config_dir(
                config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
            ):
                cfg = compose(config_name="openpower", overrides=None)

        self.cfg = cfg
        self.name = cfg.name
        self.normalize = cfg.normalize
        self.threshold = (-1 * int(cfg.threshold), int(cfg.threshold))
        self.include_generation = cfg.include_generation
        self._load_data()

        time_series_column_names = ["grid_import"]

        if self.include_generation:
            assert (
                self.cfg.user_group == "pv_users"
            ), "If generation is to be included, the config defined user group should be 'pv_users'!"
            time_series_column_names.append("pv")

        conditioning_vars = list(self.cfg.conditioning_vars.keys())
        normalization_group_keys = ["dataid", "year", "month", "weekday", "date_day"]
        # normalization_group_keys = ["year", "month", "date_day"]

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

    def _load_data(self):
        """
        Loads and preprocesses the OpenPower dataset.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Dict[str, bool]]]: Preprocessed data and user mapping with PV and EV flags.
        """
        path = self.cfg.path
        columns = self.cfg.data_columns
        dataset_path = os.path.join(
            ROOT_DIR, path, "household_data_15min_singleindex.csv"
        )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Data file not found at {dataset_path}")
        self.data = pd.read_csv(dataset_path)[columns]

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the dataset, including melting grid import and PV data,
        combining them into a single timeseries column, and handling missing values.

        Args:
            data (pd.DataFrame): The raw data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        # Melt grid_import and pv columns
        grid_import_columns = [col for col in data.columns if "grid_import" in col]
        pv_columns = [col for col in data.columns if "pv" in col]

        df_grid_import = data.melt(
            id_vars=["utc_timestamp"],
            value_vars=grid_import_columns,
            var_name="user",
            value_name="grid_import",
        )

        df_pv = data.melt(
            id_vars=["utc_timestamp"],
            value_vars=pv_columns,
            var_name="user",
            value_name="pv",
        )

        # Extracting user IDs
        df_grid_import["dataid"] = df_grid_import["user"].str.extract(
            r"residential(\d+)_grid_import"
        )
        df_pv["dataid"] = df_pv["user"].str.extract(r"residential(\d+)_pv")

        # Dropping unnecessary columns
        df_grid_import = df_grid_import.drop(columns=["user"])
        df_pv = df_pv.drop(columns=["user"])

        df_combined = pd.merge(
            df_grid_import, df_pv, on=["utc_timestamp", "dataid"], how="outer"
        )

        df_combined = df_combined.loc[~df_combined.grid_import.isna()].copy()
        df_combined = df_combined.sort_values(by=["dataid", "utc_timestamp"])

        df_combined["utc_timestamp"] = pd.to_datetime(
            df_combined["utc_timestamp"], utc=True
        )
        df_combined["month"] = df_combined["utc_timestamp"].dt.month_name()
        df_combined["weekday"] = df_combined["utc_timestamp"].dt.day_name()
        df_combined["date_day"] = df_combined["utc_timestamp"].dt.day
        df_combined["year"] = df_combined["utc_timestamp"].dt.year.astype(str)
        df_combined = df_combined.sort_values(by=["utc_timestamp"]).copy()
        df_combined.drop(columns=["utc_timestamp"], inplace=True)

        self.group_keys = ["dataid", "year", "month", "date_day", "weekday"]

        grouped_grid_data = (
            df_combined.groupby(self.group_keys)["grid_import"]
            .apply(np.array)
            .reset_index()
        )

        grouped_pv_data = (
            df_combined.groupby(self.group_keys)["pv"].apply(np.array).reset_index()
        )

        if self.include_generation:
            grouped_pv_data = (
                df_combined.groupby(self.group_keys)["pv"]
                .apply(lambda x: x.to_numpy())
                .reset_index()
            )
            grouped_data = pd.merge(
                grouped_grid_data, grouped_pv_data, on=self.group_keys, how="outer"
            )
        else:
            grouped_data = grouped_grid_data

        grouped_data = self._set_and_add_user_flags(grouped_data)
        grouped_data = self._validate_time_series_lengths(grouped_data)
        return grouped_data

    def _extract_user_ids(self, columns: List[str]) -> List[str]:
        """
        Extracts unique user IDs from the data columns.

        Args:
            columns (List[str]): List of data columns.

        Returns:
            List[str]: List of unique user IDs.
        """
        user_ids = set()
        for col in columns:
            match = re.search(r"residential(\d+)", col)
            if match:
                user_ids.add(match.group(1))
        return sorted(list(user_ids), key=lambda x: int(x))

    def _set_and_add_user_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Maps each user ID to its corresponding PV and EV flags.

        Args:
            data (pd.DataFrame): The df the user flags are set to. Can't be self.data because it hasn't been written to self.data yet.

        Returns:
            Dict[str, Dict[str, bool]]: Mapping of user IDs to their PV and EV flags.
        """
        user_flags = self.cfg.user_flags
        ev_flags = self.cfg.ev_flags

        user_ids = self._extract_user_ids(self.cfg.data_columns)
        if len(user_ids) != len(user_flags) or len(user_ids) != len(ev_flags):
            raise ValueError(
                "Length of user_flags and ev_flags must match number of users."
            )
        user_mapping = {}
        for idx, user_id in enumerate(user_ids):
            user_mapping[user_id] = {
                "pv": user_flags[idx],
                "ev": ev_flags[idx],
            }
        self.user_flags = user_mapping

        data["pv"] = (
            data["dataid"].astype(str).map(lambda x: str(user_mapping[x]["pv"]))
        )
        data["ev"] = (
            data["dataid"].astype(str).map(lambda x: str(user_mapping[x]["ev"]))
        )
        return data

    def _validate_time_series_lengths(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates that all entries in time series columns have the correct length
        defined by cfg.seq_len. Drops rows with invalid lengths.

        Args:
            data (pd.DataFrame): The preprocessed data to validate.

        Returns:
            pd.DataFrame: The validated dataset with invalid rows removed.
        """
        valid_rows = []

        for idx, row in data.iterrows():
            is_valid = True
            for col in self.time_series_column_names:
                arr = row[col]
                # Check that the shape is either (seq_len,) or (seq_len, 1)
                if arr.ndim == 1 and len(arr) == self.cfg.seq_len:
                    continue  # Shape is valid
                elif arr.ndim == 2 and arr.shape == (self.cfg.seq_len, 1):
                    continue  # Shape is valid
                else:
                    is_valid = False
                    break

            if is_valid:
                valid_rows.append(row)

        # Create a new DataFrame with only valid rows
        validated_data = pd.DataFrame(valid_rows, columns=data.columns)
        return validated_data
