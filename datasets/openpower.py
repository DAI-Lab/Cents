import os
import re
import warnings
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OpenPowerDataManager:
    """
    A dataset manager class for handling and preprocessing OpenPower time series data,
    including normalization, handling PV data, and user-specific data retrieval.

    Attributes:
        config_path (str): Path to the dataset configuration YAML file.
        normalize (bool): Whether to apply normalization.
        threshold (Tuple[float, float], optional): Threshold for clipping data.
        include_generation (bool): Whether to include PV (solar) generation data.
        stats (Dict): Stores statistics like mean and standard deviation for normalization.
        data (pd.DataFrame): The processed data containing all users.
        user_mapping (Dict[str, Dict[str, bool]]): Mapping of user IDs to their PV and EV flags.
    """

    def __init__(
        self,
        config_path: str = "config/data_config.yaml",
        normalize: bool = True,
        threshold: Tuple[float, float] = None,
        include_generation: bool = True,
    ):
        """
        Initializes the OpenPowerDatasetManager object.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to "config/data_config.yaml".
            normalize (bool, optional): Whether to normalize the dataset. Defaults to True.
            threshold (Tuple[float, float], optional): Values to clip data. Defaults to None.
            include_generation (bool, optional): Whether to include PV generation data. Defaults to True.
        """
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.normalize = normalize
        self.threshold = threshold
        self.include_generation = include_generation

        # Read user_flags and ev_flags from config
        self.user_flags, self.ev_flags = self._get_user_flags()

        self.name = "openpower"
        self.stats = {}
        self.data, self.user_mapping = self.load_and_preprocess_data()

    def _get_dataset_info(self) -> Dict:
        """
        Retrieves dataset information from the configuration file.

        Returns:
            Dict: Dictionary containing path, data_columns, user_flags, and ev_flags.
        """
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        dataset_info = config.get("datasets", {}).get("openpower")
        if not dataset_info:
            raise ValueError(f"No dataset configuration found for {self.name}")
        return dataset_info

    def _get_user_flags(self) -> Tuple[List[bool], List[bool]]:
        """
        Retrieves user_flags and ev_flags from the configuration.

        Returns:
            Tuple[List[bool], List[bool]]: user_flags and ev_flags lists.
        """
        dataset_info = self._get_dataset_info()
        user_flags = dataset_info.get("user_flags")
        ev_flags = dataset_info.get("ev_flags")
        if user_flags is None or ev_flags is None:
            raise ValueError(
                "user_flags and ev_flags must be provided in the config file under openpower."
            )
        if len(user_flags) != 6 or len(ev_flags) != 6:
            raise ValueError(
                "user_flags and ev_flags lists must each have exactly 6 elements."
            )
        return user_flags, ev_flags

    def load_and_preprocess_data(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, bool]]]:
        """
        Loads and preprocesses the OpenPower dataset.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Dict[str, bool]]]: Preprocessed data and user mapping with PV and EV flags.
        """
        dataset_info = self._get_dataset_info()
        path = dataset_info["path"]
        columns = dataset_info["data_columns"]

        data = self._load_full_data(path, columns)
        data = self._preprocess_data(data)
        user_mapping = self._map_user_flags()
        data = self._add_user_flags(data, user_mapping)
        return data, user_mapping

    def _load_full_data(self, path: str, columns: List[str]) -> pd.DataFrame:
        """
        Loads the full dataset from the given path.

        Args:
            path (str): Path to the data directory.
            columns (List[str]): List of columns to load.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        dataset_path = os.path.join(
            ROOT_DIR, path, "household_data_15min_singleindex.csv"
        )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Data file not found at {dataset_path}")
        return pd.read_csv(dataset_path)[columns]

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the dataset, including melting grid import and PV data,
        combining them into a single timeseries column, and handling missing values.

        Args:
            data (pd.DataFrame): The raw data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        # Extract unique user IDs from the data_columns
        user_ids = self._extract_user_ids(data.columns)
        if len(user_ids) != len(self.user_flags):
            raise ValueError(
                "Number of user_flags does not match number of users extracted from data_columns."
            )

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

        # Time-based features
        df_combined["utc_timestamp"] = pd.to_datetime(
            df_combined["utc_timestamp"], utc=True
        )
        df_combined["month"] = df_combined["utc_timestamp"].dt.month - 1
        df_combined["weekday"] = df_combined["utc_timestamp"].dt.weekday.astype(int)
        df_combined["date_day"] = df_combined["utc_timestamp"].dt.day
        df_combined["year"] = df_combined["utc_timestamp"].dt.year.astype(int)
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

        if self.normalize:
            self.stats["grid_import"] = self._calculate_and_store_statistics(
                grouped_data, "grid_import"
            )
            grouped_data = self._normalize_and_scale(grouped_data, "grid_import")

            if self.include_generation:
                self.stats["pv"] = self._calculate_and_store_statistics(
                    grouped_pv_data, "pv"
                )
                grouped_data = self._normalize_and_scale(grouped_data, "pv")

        def valid_timeseries(ts):
            return ts.shape == (96, 1) or ts.shape == (96, 2)

        def combine_timeseries(row):
            grid_import = row["grid_import"]
            pv = row["pv"]

            if np.any(np.isnan(pv)):
                timeseries = grid_import.reshape(-1, 1)
            else:
                timeseries = np.column_stack((grid_import, pv))
            return timeseries

        grouped_data["timeseries"] = grouped_data.apply(combine_timeseries, axis=1)

        filtered_data = grouped_data[
            grouped_data["timeseries"].apply(valid_timeseries)
        ].reset_index(drop=True)
        # filtered_data = filtered_data.drop(columns=["grid_import", "pv"])

        return filtered_data

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

    def _map_user_flags(self) -> Dict[str, Dict[str, bool]]:
        """
        Maps each user ID to its corresponding PV and EV flags.

        Returns:
            Dict[str, Dict[str, bool]]: Mapping of user IDs to their PV and EV flags.
        """
        user_ids = self._extract_user_ids(self._get_dataset_info()["data_columns"])
        if len(user_ids) != len(self.user_flags) or len(user_ids) != len(self.ev_flags):
            raise ValueError(
                "Length of user_flags and ev_flags must match number of users."
            )
        user_mapping = {}
        for idx, user_id in enumerate(user_ids):
            user_mapping[user_id] = {
                "pv": self.user_flags[idx],
                "ev": self.ev_flags[idx],
            }
        return user_mapping

    def _add_user_flags(
        self, data: pd.DataFrame, user_mapping: Dict[str, Dict[str, bool]]
    ) -> pd.DataFrame:
        """
        Adds binary PV and EV flags to the dataset based on user_mapping.

        Args:
            data (pd.DataFrame): The dataset to augment.
            user_mapping (Dict[str, Dict[str, bool]]): Mapping of user IDs to their PV and EV flags.

        Returns:
            pd.DataFrame: The augmented dataset with PV and EV flags.
        """
        data["pv_flag"] = (
            data["dataid"].astype(str).map(lambda x: user_mapping[x]["pv"])
        )
        data["ev_flag"] = (
            data["dataid"].astype(str).map(lambda x: user_mapping[x]["ev"])
        )
        return data

    def _calculate_and_store_statistics(self, data: pd.DataFrame, column: str) -> Dict:
        """
        Calculates and stores statistics like mean and std for normalization.

        Args:
            data (pd.DataFrame): Data to calculate statistics for.
            column (str): Column name for which to calculate statistics.

        Returns:
            Dict: A dictionary containing the calculated statistics.
        """

        def calculate_stats(group):
            grid_import_values = np.concatenate(group[column].values)
            mean = np.mean(grid_import_values)
            std = np.std(grid_import_values)
            z_scores = (grid_import_values - mean) / (std + 1e-8)
            z_min = np.min(z_scores)
            z_max = np.max(z_scores)
            return pd.Series({"mean": mean, "std": std, "z_min": z_min, "z_max": z_max})

        if self.normalize:
            grouped_stats = data.groupby(["dataid", "year", "month", "weekday"]).apply(
                calculate_stats
            )
            return grouped_stats.to_dict(orient="index")
        else:
            return {}

    def _normalize_and_scale(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies standardization followed by min-max scaling to the data.

        Args:
            data (pd.DataFrame): The data to normalize and scale.
            column (str): The column on which normalization and scaling are applied.

        Returns:
            pd.DataFrame: The normalized and scaled data.
        """

        def normalize_and_scale_row(row):
            stats = self.stats[column][
                (row["dataid"], row["year"], row["month"], row["weekday"])
            ]

            mean = stats["mean"]
            std = stats["std"]
            z_min = stats["z_min"]
            z_max = stats["z_max"]

            values = np.array(row[column])
            standardized = (values - mean) / (std + 1e-8)

            if self.threshold:
                standardized = np.clip(standardized, *self.threshold)

            scaled = (standardized - z_min) / (z_max - z_min + 1e-8)
            return scaled

        data[column] = data.apply(normalize_and_scale_row, axis=1)
        return data

    def create_pv_user_dataset(self) -> "OpenPowerDataset":
        """
        Creates a dataset containing data from all users with PV data.

        Returns:
            OpenPowerDataset: The dataset containing all PV users.
        """
        pv_users = [
            user_id for user_id, flags in self.user_mapping.items() if flags["pv"]
        ]
        pv_data = self.data[self.data["dataid"].astype(str).isin(pv_users)].copy()

        pv_data = pv_data[
            pv_data["timeseries"].apply(lambda ts: ts.shape == (96, 2))
        ].copy()

        return OpenPowerDataset(
            data=pv_data,
            include_generation=True,
            stats=self.stats,
            user_mapping=self.user_mapping,
        )

    def create_non_pv_user_dataset(self) -> "OpenPowerDataset":
        """
        Creates a dataset containing data from all users without PV data.

        Returns:
            OpenPowerDataset: The dataset containing all non-PV users.
        """
        non_pv_users = [
            user_id for user_id, flags in self.user_mapping.items() if not flags["pv"]
        ]
        non_pv_data = self.data[
            self.data["dataid"].astype(str).isin(non_pv_users)
        ].copy()

        pv_data = pv_data[
            pv_data["timeseries"].apply(lambda ts: ts.shape == (96, 1))
        ].copy()

        return OpenPowerDataset(
            data=non_pv_data,
            include_generation=False,
            stats=self.stats,
            user_mapping=self.user_mapping,
        )

    def create_all_user_dataset(self) -> "OpenPowerDataset":
        """
        Creates a dataset containing data from all users without PV data.

        Returns:
            OpenPowerDataset: The dataset containing all non-PV users.
        """
        assert (
            self.include_generation == False
        ), "Include_generation must be set to False when working with the entire dataset!"

        return OpenPowerDataset(
            data=self.data,
            include_generation=self.include_generation,
            stats=self.stats,
            user_mapping=self.user_mapping,
        )


class OpenPowerDataset(Dataset):
    """
    A dataset class for handling individual user data from the OpenPower dataset.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        include_generation: bool,
        stats: Dict,
        user_mapping: Dict[str, Dict[str, bool]],
    ):
        """
        Initializes the OpenPowerDataset object.

        Args:
            data (pd.DataFrame): Time series data for all users.
            user_mapping (Dict[str, Dict[str, bool]]): Mapping of user IDs to their PV and EV flags.
        """
        self.data = self.validate_data(data)
        self.include_generation = include_generation
        self.user_mapping = user_mapping
        self.stats = stats

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the data to ensure consistency in the shapes of time series.

        Args:
            data (pd.DataFrame): DataFrame containing time series data.

        Returns:
            pd.DataFrame: The validated and possibly corrected time series data.
        """
        shapes = data["timeseries"].apply(
            lambda x: x.shape if isinstance(x, np.ndarray) else None
        )
        if len(shapes.unique()) > 2:
            print("Warning: More than two unique shapes found in timeseries")
        elif len(shapes.unique()) == 2:
            # Check if one shape is [96] and the other is [96, 2]
            expected_shapes = {(96,), (96, 2)}
            actual_shapes = set(shapes.unique())
            if not actual_shapes.issubset(expected_shapes):
                print("Warning: Inconsistent shapes found in timeseries")
        return data

    def inverse_transform_column(self, row: pd.Series, colname: str) -> np.ndarray:
        """
        Performs inverse transformation on a normalized and scaled column to retrieve original values.

        Args:
            row (pd.Series): A row from the DataFrame that contains the normalized and scaled data.
            colname (str): The column name of the normalized and scaled data (e.g., "grid" or "solar").

        Returns:
            np.ndarray: The original (un-normalized and un-scaled) time series data.
        """
        stats = self.stats[colname].get(
            (row["dataid"], row["year"], row["month"], row["weekday"])
        )
        if not stats:
            raise ValueError(
                f"No stats found for {colname} with dataid={row['dataid']}, month={row['month']}, weekday={row['weekday']}"
            )

        mean = stats["mean"]
        std = stats["std"]
        z_min = stats["z_min"]
        z_max = stats["z_max"]

        unscaled = row[colname] * (z_max - z_min + 1e-8) + z_min
        unnormalized = unscaled * (std + 1e-8) + mean
        return unnormalized

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies inverse transformation on the entire dataset to recover original time series values.

        Args:
            df (pd.DataFrame): The DataFrame containing normalized and scaled time series data.

        Returns:
            pd.DataFrame: The DataFrame with the original (un-normalized and un-scaled) time series.
        """

        def split_timeseries(row: np.ndarray) -> pd.Series:
            grid = row[:, 0].reshape(-1, 1)
            solar = row[:, 1].reshape(-1, 1) if row.shape[1] > 1 else None
            return pd.Series({"grid_import": grid, "pv": solar})

        if self.include_generation:
            df[["grid_import", "pv"]] = df["timeseries"].apply(split_timeseries)
        else:
            df["grid_import"] = df["timeseries"]

        df.drop(columns=["timeseries"], inplace=True)

        for idx, row in df.iterrows():
            df.at[idx, "grid_import"] = self.inverse_transform_column(
                row, "grid_import"
            )
            if self.include_generation and "pv" in df.columns and row["pv"] is not None:
                df.at[idx, "pv"] = self.inverse_transform_column(row, "pv")

        df = self._merge_columns_into_timeseries(df)
        df.sort_values(by=["dataid", "year", "month", "weekday"], inplace=True)
        return df

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - dataid: User ID tensor.
                - timeseries: Time series data tensor.
                - pv: Binary PV flag tensor.
                - ev: Binary EV flag tensor.
        """
        sample = self.data.iloc[idx]
        dataid = sample["dataid"]
        timeseries = sample["timeseries"]
        ev_flag = float(sample["ev_flag"])  # Binary flag: 1.0 or 0.0

        # Determine PV flag based on timeseries shape
        if isinstance(timeseries, np.ndarray):
            if timeseries.ndim == 1:
                # User does not have PV; timeseries is 1D (only grid_import)
                pv = torch.tensor(0.0, dtype=torch.float32)
            elif timeseries.ndim == 2 and timeseries.shape[1] == 2:
                # User has PV; timeseries is 2D (grid_import + pv)
                pv = torch.tensor(1.0, dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected timeseries shape: {timeseries.shape}")
        else:
            raise ValueError("timeseries must be a NumPy array")

        # Convert timeseries to tensor
        timeseries_tensor = torch.tensor(timeseries, dtype=torch.float32)

        # EV flag is already a separate binary value
        ev = torch.tensor(ev_flag, dtype=torch.float32)

        return timeseries_tensor, pv, ev

    def _merge_columns_into_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges grid and solar columns back into a single time series column.

        Args:
            df (pd.DataFrame): DataFrame containing separate grid and solar columns.

        Returns:
            pd.DataFrame: DataFrame with merged time series column.
        """
        if "pv" in df.columns:
            df["timeseries"] = df.apply(
                lambda row: (
                    row["grid_import"]
                    if not isinstance(row["pv"], (np.ndarray, list))
                    else np.column_stack((row["grid_import"], row["pv"]))
                ),
                axis=1,
            )
            df.drop(columns=["grid_import", "pv"], inplace=True)
        else:
            df["timeseries"] = df["grid_import"]
            df.drop(columns=["grid_import"], inplace=True)
        return df
