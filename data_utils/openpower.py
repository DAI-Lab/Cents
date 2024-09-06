import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OpenPowerDataset(Dataset):
    """
    A dataset class for handling and preprocessing OpenPower time series data,
    including normalization, handling grid and solar data, and user-specific data retrieval.

    Attributes:
        config_path (str): Path to the dataset configuration YAML file.
        normalize (bool): Whether to apply normalization.
        threshold (Tuple[float, float], optional): Threshold for clipping data.
        include_generation (bool): Whether to include solar generation data.
        stats (Dict): Stores statistics like mean and standard deviation for normalization.
        data (pd.DataFrame): The processed data.
        user_flags (Dict): Flags indicating user-specific settings from the dataset.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        normalize: bool = True,
        threshold: Tuple[float, float] = None,
        include_generation: bool = True,
    ):
        """
        Initializes the OpenPowerDataset object.

        Args:
            config_path (str): Path to the configuration file.
            normalize (bool, optional): Whether to normalize the dataset. Defaults to True.
            threshold (Tuple[float, float], optional): Values to clip data. Defaults to None.
            include_generation (bool, optional): Whether to include solar generation data. Defaults to True.
        """
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.normalize = normalize
        self.threshold = threshold
        self.include_generation = include_generation
        self.name = "openpower"
        self.stats = {}
        self.data, self.user_flags = self.load_and_preprocess_data()

    def _get_dataset_info(self) -> Tuple[str, List[str], Dict]:
        """
        Retrieves dataset path, column information, and user flags from the configuration file.

        Returns:
            Tuple[str, List[str], Dict]: Dataset path, columns, and user flags.
        """
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        dataset_info = config["datasets"].get(self.name)
        if not dataset_info:
            raise ValueError(f"No dataset configuration found for {self.name}")
        return dataset_info["path"], dataset_info["columns"], dataset_info["user_flags"]

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Loads and preprocesses the OpenPower dataset.

        Returns:
            Tuple[pd.DataFrame, Dict]: Preprocessed data and user flags.
        """
        path, columns, user_flags = self._get_dataset_info()
        data = self._load_full_data(path, columns)
        data = self._preprocess_data(data)
        return data, user_flags

    def _load_full_data(self, path: str, columns: List[str]) -> pd.DataFrame:
        """
        Loads the full dataset from the given path.

        Args:
            path (str): Path to the data file.
            columns (List[str]): List of columns to load.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        data_file_path = f"/{path}household_data_15min_singleindex.csv"
        return pd.read_csv(data_file_path)[columns]

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the dataset, including melting grid import and solar data,
        and handling missing values.

        Args:
            data (pd.DataFrame): The raw data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        # Melting grid import columns
        grid_import_columns = [col for col in data.columns if "grid_import" in col]
        df_grid_import = data.melt(
            id_vars=["utc_timestamp"],
            value_vars=grid_import_columns,
            var_name="user",
            value_name="grid",
        )

        # Melting solar columns
        pv_columns = [col for col in data.columns if "pv" in col]
        df_pv = data.melt(
            id_vars=["utc_timestamp"],
            value_vars=pv_columns,
            var_name="user",
            value_name="solar",
        )

        # Extracting user IDs
        df_grid_import["dataid"] = df_grid_import["user"].str.extract(
            r"residential(\d+)_grid_import"
        )
        df_pv["dataid"] = df_pv["user"].str.extract(r"residential(\d+)_pv")

        # Dropping unnecessary columns
        df_grid_import = df_grid_import.drop(columns=["user"])
        df_pv = df_pv.drop(columns=["user"])

        # Merging grid and solar data
        df_combined = pd.merge(
            df_grid_import, df_pv, on=["utc_timestamp", "dataid"], how="outer"
        )
        df_combined = df_combined.loc[~df_combined.grid.isna()].copy()
        df_combined = df_combined.sort_values(by=["dataid", "utc_timestamp"])

        # Time-based features
        df_combined["utc_timestamp"] = pd.to_datetime(
            df_combined["utc_timestamp"], utc=True
        )
        df_combined["month"] = df_combined["utc_timestamp"].dt.month - 1
        df_combined["weekday"] = df_combined["utc_timestamp"].dt.weekday
        df_combined["date_day"] = df_combined["utc_timestamp"].dt.day

        # Group by time-based columns for analysis
        grouped_data = (
            df_combined.groupby(["dataid", "month", "date_day", "weekday"])["grid"]
            .apply(np.array)
            .reset_index()
        )

        # Filter out rows that don't have exactly 96 samples (15-minute intervals in a day)
        filtered_data = grouped_data[grouped_data["grid"].apply(len) == 96].reset_index(
            drop=True
        )

        # Apply normalization if needed
        if self.normalize:
            self.stats["grid"] = self._calculate_and_store_statistics(
                filtered_data, "grid"
            )
            filtered_data = self._apply_normalization(filtered_data, "grid")

        # Include solar generation data
        if self.include_generation:
            solar_data = self._preprocess_solar(df_combined)
            filtered_data = pd.merge(
                filtered_data,
                solar_data,
                how="left",
                on=["dataid", "month", "weekday", "date_day"],
            )

        return self._merge_columns_into_timeseries(filtered_data).sort_values(
            by=["dataid", "month", "weekday"]
        )

    def _calculate_and_store_statistics(self, data: pd.DataFrame, column: str) -> Dict:
        """
        Calculates and stores statistics like mean, std, min, and max for normalization.

        Args:
            data (pd.DataFrame): Data to calculate statistics for.
            column (str): Column name for which to calculate statistics.

        Returns:
            Dict: A dictionary containing the calculated statistics.
        """

        def calculate_stats(group):
            all_values = np.concatenate(group[column].values)
            mean = np.mean(all_values)
            std = np.std(all_values)
            normalized = (all_values - mean) / (std + 1e-8)
            norm_min = normalized.min()
            norm_max = normalized.max()
            return pd.Series(
                {"mean": mean, "std": std, "norm_min": norm_min, "norm_max": norm_max}
            )

        grouped_stats = data.groupby(["dataid", "month", "weekday"]).apply(
            calculate_stats
        )
        return grouped_stats.to_dict(orient="index")

    def _apply_normalization(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Normalizes and scales the data based on the calculated statistics.

        Args:
            data (pd.DataFrame): Data to normalize.
            column (str): Column to normalize.

        Returns:
            pd.DataFrame: The normalized data.
        """

        def normalize_and_scale(row):
            stats = self.stats[column][(row["dataid"], row["month"], row["weekday"])]
            mean, std = stats["mean"], stats["std"]
            norm_min, norm_max = stats["norm_min"], stats["norm_max"]

            values = np.array(row[column])
            normalized = (values - mean) / (std + 1e-8)

            if self.threshold:
                values = np.clip(values, *self.threshold)

            scaled = (normalized - norm_min) / ((norm_max - norm_min) + 1e-8)
            return scaled

        data[column] = data.apply(normalize_and_scale, axis=1)
        return data

    def _preprocess_solar(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the solar data, including normalization.

        Args:
            data (pd.DataFrame): The dataset containing solar data.

        Returns:
            pd.DataFrame: Preprocessed solar data.
        """
        solar_data = (
            data[~data["solar"].isna()]
            .groupby(["dataid", "month", "date_day", "weekday"])["solar"]
            .apply(np.array)
            .reset_index()
        )
        solar_data = solar_data[solar_data["solar"].apply(len) == 96]

        if self.normalize:
            valid_stats = self._calculate_and_store_statistics(solar_data, "solar")
            self.stats["solar"] = valid_stats
            solar_data = self._apply_normalization(solar_data, "solar")

        return solar_data

    @staticmethod
    def _merge_columns_into_timeseries(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges grid and solar columns into a single timeseries column.

        Args:
            df (pd.DataFrame): DataFrame containing separate grid and solar columns.

        Returns:
            pd.DataFrame: DataFrame with a single timeseries column.
        """

        def ensure_two_dimensional(arr):
            if arr.ndim == 1:
                return np.expand_dims(arr, axis=-1)
            return arr

        if "solar" in df.columns:
            df["timeseries"] = df.apply(
                lambda row: (
                    ensure_two_dimensional(row["grid"])
                    if not isinstance(row["solar"], (np.ndarray, list))
                    else np.column_stack(
                        (
                            ensure_two_dimensional(row["grid"]),
                            ensure_two_dimensional(row["solar"]),
                        )
                    )
                ),
                axis=1,
            )
            df.drop(columns=["grid", "solar"], inplace=True)
        else:
            df["timeseries"] = df["grid"].apply(ensure_two_dimensional)
            df.drop(columns=["grid"], inplace=True)
        return df

    def create_user_dataset(self, user_id: int) -> "OpenPowerUserDataset":
        """
        Creates a dataset specific to a user based on the user ID.

        Args:
            user_id (int): ID of the user.

        Returns:
            OpenPowerUserDataset: A dataset object for the specific user.

        Raises:
            ValueError: If the user ID is not found in the dataset.
        """
        if user_id not in self.user_flags:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        user_data = self.data[self.data["dataid"] == user_id].copy()
        return OpenPowerUserDataset(
            user_data, self.stats, self.user_flags[user_id], self.include_generation
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The time series, month, and weekday as tensors.
        """
        sample = self.data.iloc[idx]
        time_series = sample["timeseries"]
        month = sample["month"]
        day = sample["weekday"]
        return (
            torch.tensor(time_series, dtype=torch.float32).to(device),
            torch.tensor(month, dtype=torch.long).to(device),
            torch.tensor(day, dtype=torch.long).to(device),
        )


class OpenPowerUserDataset(Dataset):
    """
    A dataset class for handling time series data for individual users in the OpenPower dataset.

    Attributes:
        data (pd.DataFrame): Processed time series data for the user.
        stats (Dict): Statistics for normalization.
        is_pv_user (bool): Whether the user has solar PV data.
        include_generation (bool): Whether solar generation data is included.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        stats: dict,
        is_pv_user: bool,
        include_generation: bool,
    ):
        """
        Initializes the OpenPowerUserDataset object.

        Args:
            data (pd.DataFrame): The time series data for the user.
            stats (Dict): Dictionary containing statistics for normalization.
            is_pv_user (bool): Indicates if the user has solar PV data.
            include_generation (bool): Whether to include solar generation data.
        """
        self.data = self.validate_data(data)
        self.stats = stats
        self.is_pv_user = is_pv_user
        self.include_generation = include_generation

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the time series data to ensure consistency in shapes.

        Args:
            data (pd.DataFrame): The dataset to validate.

        Returns:
            pd.DataFrame: The validated dataset.
        """
        shapes = data["timeseries"].apply(
            lambda x: x.shape if isinstance(x, np.ndarray) else None
        )
        if len(shapes.unique()) > 1:
            print("Warning: Inconsistent shapes found in timeseries")
            data["timeseries"] = data["timeseries"].apply(
                lambda x: (
                    np.expand_dims(x[:, 0], axis=-1)
                    if isinstance(x, np.ndarray) and x.ndim > 1
                    else x
                )
            )
        return data

    def inverse_transform_column(self, row: pd.Series, colname: str) -> np.ndarray:
        """
        Inverses the normalization applied to a column for a specific row.

        Args:
            row (pd.Series): The row containing the normalized data.
            colname (str): The name of the column to inverse transform.

        Returns:
            np.ndarray: The inverse-transformed data.
        """
        stats = self.stats[colname].get((row["dataid"], row["month"], row["weekday"]))
        if not stats:
            raise ValueError(
                f"No stats found for {colname} with dataid={row['dataid']}, month={row['month']}, weekday={row['weekday']}"
            )

        norm_min = stats["norm_min"]
        norm_max = stats["norm_max"]
        mean = stats["mean"]
        std = stats["std"]

        scaled = row[colname]
        unscaled = scaled * (norm_max - norm_min + 1e-8) + norm_min
        return unscaled * (std + 1e-8) + mean

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies inverse transformation to the dataset, converting normalized values to original values.

        Args:
            df (pd.DataFrame): The DataFrame with normalized data.

        Returns:
            pd.DataFrame: The DataFrame with original values.
        """

        def split_timeseries(row: np.ndarray) -> pd.Series:
            grid = row[:, 0].reshape(-1, 1)
            solar = row[:, 1].reshape(-1, 1) if row.shape[1] > 1 else None
            return pd.Series({"grid": grid, "solar": solar})

        if self.include_generation and self.is_pv_user:
            df[["grid", "solar"]] = df["timeseries"].apply(split_timeseries)
        else:
            df["grid"] = df["timeseries"]

        df.drop(columns=["timeseries"], inplace=True)

        for idx, row in df.iterrows():
            df.at[idx, "grid"] = self.inverse_transform_column(row, "grid")
            if (
                self.include_generation
                and "solar" in df.columns
                and row["solar"] is not None
            ):
                df.at[idx, "solar"] = self.inverse_transform_column(row, "solar")

        df = self._merge_columns_into_timeseries(df)
        df.sort_values(by=["dataid", "month", "weekday"], inplace=True)
        return df

    @staticmethod
    def _merge_columns_into_timeseries(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges grid and solar columns back into a single timeseries column.

        Args:
            df (pd.DataFrame): DataFrame with separate grid and solar columns.

        Returns:
            pd.DataFrame: DataFrame with merged timeseries column.
        """
        if "solar" in df.columns:
            df["timeseries"] = df.apply(
                lambda row: (
                    row["grid"]
                    if not isinstance(row["solar"], (np.ndarray, list))
                    else np.column_stack((row["grid"], row["solar"]))
                ),
                axis=1,
            )
            df.drop(columns=["grid", "solar"], inplace=True)
        else:
            df["timeseries"] = df["grid"]
            df.drop(columns=["grid"], inplace=True)
        return df

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The time series, month, and weekday as tensors.
        """
        sample = self.data.iloc[idx]
        time_series = sample["timeseries"]
        month = sample["month"]
        day = sample["weekday"]

        return (
            torch.tensor(time_series, dtype=torch.float32).to(device),
            torch.tensor(month, dtype=torch.long).to(device),
            torch.tensor(day, dtype=torch.long).to(device),
        )
