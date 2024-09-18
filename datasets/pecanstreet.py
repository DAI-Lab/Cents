import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from datasets.utils import encode_categorical_variables

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PecanStreetDataManager:
    def __init__(
        self,
        geography: str = None,
        config_path: str = "config/config.yaml",
        normalize: bool = False,
        threshold: Union[Tuple[float, float], None] = None,
        include_generation: bool = False,
        normalization_method: str = "group",
    ):
        self.geography = geography
        self.config_path = os.path.abspath(config_path)
        self.normalize = normalize
        self.threshold = threshold
        self.include_generation = include_generation
        self.normalization_method = normalization_method.lower()
        assert self.normalization_method in ['group', 'global'], "normalization_method must be 'group' or 'global'"
        self.name = "pecanstreet"
        self.stats = {}
        self.data, self.metadata, self.user_flags = self.load_and_preprocess_data()

    def _get_dataset_info(self) -> Tuple[str, List[str]]:
        """
        Retrieves dataset path and column information from the config file.

        Returns:
            Tuple[str, List[str]]: The dataset path and a list of relevant columns.

        Raises:
            ValueError: If dataset configuration is not found for the given dataset name.
        """
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        dataset_info = config["datasets"].get(self.name)
        if not dataset_info:
            raise ValueError(f"No dataset configuration found for {self.name}")

        # Resolve the dataset path relative to the config file location
        config_dir = os.path.dirname(self.config_path)
        dataset_path = os.path.join(config_dir, dataset_info["path"])

        return dataset_path, dataset_info["data_columns"], dataset_info["metadata_columns"]

    def load_and_preprocess_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, bool]]:
        """
        Loads and preprocesses the data, including filtering, normalization and handling of categorical metadata.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict[int, bool]]: Processed data, metadata, and user flags.
        """
        path, data_columns, metadata_columns = self._get_dataset_info()
        metadata_csv_path = os.path.join(path, "metadata.csv")
        if not os.path.exists(metadata_csv_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_csv_path}")
        metadata = pd.read_csv(metadata_csv_path, usecols=metadata_columns)
        data = self._load_full_data(path, data_columns)
        user_flags = self._set_user_flags(metadata, data)
        data = self._preprocess_data(data)
        data = pd.merge(data, metadata, on="dataid", how="left")
        data, mappings = encode_categorical_variables(data.fillna("no"))
        self.mappings = mappings
        return data, metadata, user_flags

    def _load_full_data(self, path: str, columns: List[str]) -> pd.DataFrame:
        """
        Loads the full dataset, optionally filtered by geography.

        Args:
            path (str): The dataset directory path.
            columns (List[str]): List of columns to load from the dataset.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if self.geography:
            data_file_name = f"15minute_data_{self.geography}.csv"
            data_file_path = os.path.join(path, data_file_name)
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found at {data_file_path}")
            return pd.read_csv(data_file_path)[columns]
        else:
            data_files = [
                os.path.join(path, "15minute_data_newyork.csv"),
                os.path.join(path, "15minute_data_california.csv"),
                os.path.join(path, "15minute_data_austin.csv"),
            ]
            for data_file in data_files:
                if not os.path.exists(data_file):
                    raise FileNotFoundError(f"Data file not found at {data_file}")
            return pd.concat(
                [pd.read_csv(data_file) for data_file in data_files],
                axis=0,
            )[columns]

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the dataset by adding date-related columns, sorting, filtering, and normalizing.

        Args:
            data (pd.DataFrame): Raw data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        data["local_15min"] = pd.to_datetime(data["local_15min"], utc=True)
        data["month"] = data["local_15min"].dt.month - 1
        data["weekday"] = data["local_15min"].dt.weekday
        data["date_day"] = data["local_15min"].dt.day
        data = data.sort_values(by=["local_15min"]).copy()
        data = data[~data["grid"].isna()]

        grouped_data = (
            data.groupby(["dataid", "month", "date_day", "weekday"])["grid"]
            .apply(np.array)
            .reset_index()
        )
        filtered_data = grouped_data[grouped_data["grid"].apply(len) == 96].reset_index(
            drop=True
        )

        if self.normalize:
            if self.normalization_method == 'group':
                self.stats["grid"] = self._calculate_and_store_statistics_group(
                    filtered_data, "grid"
                )
                filtered_data = self._apply_normalization_group(filtered_data, "grid")
            elif self.normalization_method == 'global':
                self.stats["grid"] = self._calculate_and_store_statistics_global(
                    filtered_data, "grid"
                )
                filtered_data = self._apply_normalization_global(filtered_data, "grid")

        if self.include_generation:
            solar_data = self._preprocess_solar(data)
            filtered_data = pd.merge(
                filtered_data,
                solar_data,
                how="left",
                on=["dataid", "month", "weekday", "date_day"],
            )

        return self._merge_columns_into_timeseries(filtered_data).sort_values(
            by=["dataid", "month", "weekday"]
        )

    def _calculate_and_store_statistics_group(self, data: pd.DataFrame, column: str) -> Dict:
        """
        Calculates and stores statistical data for group normalization, such as mean and standard deviation.

        Args:
            data (pd.DataFrame): The data on which to calculate statistics.
            column (str): The column for which to calculate statistics.

        Returns:
            Dict: A dictionary containing the statistics.
        """

        def calculate_stats(group):
            all_values = np.concatenate(group[column].values)
            mean = np.mean(all_values)
            std = np.std(all_values)
            return pd.Series({"mean": mean, "std": std})

        grouped_stats = data.groupby(["dataid", "month", "weekday"]).apply(
            calculate_stats
        )
        return grouped_stats.to_dict(orient="index")

    def _apply_normalization_group(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies group normalization to the data.

        Args:
            data (pd.DataFrame): The data to normalize.
            column (str): The column on which normalization is applied.

        Returns:
            pd.DataFrame: The normalized data.
        """

        def normalize(row):
            stats = self.stats[column][(row["dataid"], row["month"], row["weekday"])]
            mean, std = stats["mean"], stats["std"]
            values = np.array(row[column])
            normalized = (values - mean) / (std + 1e-8)
            if self.threshold:
                normalized = np.clip(normalized, *self.threshold)
            return normalized

        data[column] = data.apply(normalize, axis=1)
        return data

    def _calculate_and_store_statistics_global(self, data: pd.DataFrame, column: str) -> Dict:
        """
        Calculates and stores statistical data for global normalization.

        Args:
            data (pd.DataFrame): The data on which to calculate statistics.
            column (str): The column for which to calculate statistics.

        Returns:
            Dict: A dictionary containing the global mean and standard deviation.
        """
        all_values = np.concatenate(data[column].values)
        mean = np.mean(all_values)
        std = np.std(all_values)
        return {"mean": mean, "std": std}

    def _apply_normalization_global(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies global normalization to the data.

        Args:
            data (pd.DataFrame): The data to normalize.
            column (str): The column on which normalization is applied.

        Returns:
            pd.DataFrame: The normalized data.
        """

        mean = self.stats[column]["mean"]
        std = self.stats[column]["std"]

        def normalize(values):
            normalized = (values - mean) / (std + 1e-8)
            if self.threshold:
                normalized = np.clip(normalized, *self.threshold)
            return normalized

        data[column] = data[column].apply(normalize)
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
        solar_data = solar_data[solar_data["solar"].apply(len) == 96]

        if self.normalize:
            if self.normalization_method == 'group':
                valid_stats = self._calculate_and_store_statistics_group(solar_data, "solar")
                self.stats["solar"] = valid_stats
                solar_data = self._apply_normalization_group(solar_data, "solar")
            elif self.normalization_method == 'global':
                valid_stats = self._calculate_and_store_statistics_global(solar_data, "solar")
                self.stats["solar"] = valid_stats
                solar_data = self._apply_normalization_global(solar_data, "solar")

        return solar_data

    @staticmethod
    def _merge_columns_into_timeseries(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges grid and solar columns into a single time-series column.

        Args:
            df (pd.DataFrame): DataFrame with grid and optional solar data.

        Returns:
            pd.DataFrame: DataFrame with a single merged time-series column.
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

    @staticmethod
    def _set_user_flags(metadata: pd.DataFrame, data: pd.DataFrame) -> Dict[int, bool]:
        """
        Sets user flags indicating whether a user has solar generation data.

        Args:
            metadata (pd.DataFrame): Metadata containing information about users.
            data (pd.DataFrame): The main dataset.

        Returns:
            Dict[int, bool]: A dictionary mapping user IDs to a boolean indicating if they have solar data.
        """
        return {
            user_id: metadata.loc[metadata["dataid"] == user_id]["solar"].notna().any()
            for user_id in data["dataid"].unique()
        }

    def create_user_dataset(self, user_id: int) -> "PecanStreetDataset":
        """
        Creates a dataset for a specific user based on the user ID.

        Args:
            user_id (int): The user ID for which to create a dataset.

        Returns:
            PecanStreetDataset: The dataset specific to the given user.

        Raises:
            ValueError: If the user ID is not found in the dataset.
        """
        if user_id not in self.user_flags:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        user_data = self.data[self.data["dataid"] == user_id].copy()
        return PecanStreetDataset(
            user_data,
            self.stats,
            self.user_flags[user_id],
            self.include_generation,
            self.metadata,
        )
    

    def create_non_pv_user_dataset(self) -> "PecanStreetDataset":
        """
        Creates a dataset for all users without solar generation data (non-PV users).

        Returns:
            PecanStreetDataset: The dataset containing all non-PV users.
        """
        non_pv_users = [user for user, has_pv in self.user_flags.items() if not has_pv]
        non_pv_data = self.data[self.data["dataid"].isin(non_pv_users)].copy()

        return PecanStreetDataset(
            data=non_pv_data,
            stats=self.stats,
            is_pv_user=False,
            include_generation=False,
            metadata=self.metadata,
        )
    
    
    def create_all_pv_user_dataset(self) -> "PecanStreetDataset":
        """
        Creates a dataset for all users with solar generation data (PV users).

        Returns:
            PecanStreetDataset: The dataset containing all PV users.
        """
        pv_users = [user for user, has_pv in self.user_flags.items() if has_pv]
        pv_data = self.data[self.data["dataid"].isin(pv_users)].copy()

        return PecanStreetDataset(
            data=pv_data,
            stats=self.stats,
            is_pv_user=True,
            include_generation=True,
            metadata=self.metadata,
        )

    def create_all_user_dataset(self) -> "PecanStreetDataset":
        """
        Creates a dataset for all users without solar generation data (non-PV users).

        Returns:
            PecanStreetDataset: The dataset containing all non-PV users.
        """
        assert self.include_generation == False, "Include_generation must be set to False when working with the entire dataset!"

        return PecanStreetDataset(
            data=self.data,
            stats=self.stats,
            is_pv_user=False,
            include_generation=False,
            metadata=self.metadata,
        )


class PecanStreetDataset(Dataset):
    """
    A dataset class for individual users from the Pecan Street dataset.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        stats: dict,
        is_pv_user: bool,
        include_generation: bool,
        metadata: pd.DataFrame,
        normalization_method: str = "global",
    ):
        """
        Initializes the PecanStreetDataset object.

        Args:
            data (pd.DataFrame): Time series data for a specific user or group of users.
            stats (dict): Dictionary containing normalization statistics for the dataset.
            is_pv_user (bool): Whether the user has solar PV data.
            include_generation (bool): Whether to include solar generation data.
            metadata (pd.DataFrame): Metadata for the dataset (e.g., user locations, PV info).
            normalization_method (str, optional): Normalization method used ('group' or 'global'). Defaults to 'global'.
        """
        self.data = self.validate_data(data)
        self.stats = stats
        self.is_pv_user = is_pv_user
        self.include_generation = include_generation
        self.metadata = metadata
        self.normalization_method = normalization_method.lower()
        assert self.normalization_method in ['group', 'global'], "normalization_method must be 'group' or 'global'"

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
        Performs inverse transformation on a normalized column to retrieve original values.

        Args:
            row (pd.Series): A row from the DataFrame that contains the normalized data.
            colname (str): The column name of the normalized data (e.g., "grid" or "solar").

        Returns:
            np.ndarray: The original (un-normalized) time series data.
        """
        if self.normalization_method == 'group':
            stats = self.stats[colname].get((row["dataid"], row["month"], row["weekday"]))
            if not stats:
                raise ValueError(
                    f"No stats found for {colname} with dataid={row['dataid']}, month={row['month']}, weekday={row['weekday']}"
                )
            mean = stats["mean"]
            std = stats["std"]
        elif self.normalization_method == 'global':
            stats = self.stats[colname]
            mean = stats["mean"]
            std = stats["std"]
        else:
            raise ValueError("Invalid normalization_method")

        scaled = row[colname]
        unscaled = scaled * (std + 1e-8) + mean
        return unscaled

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies inverse transformation on the entire dataset to recover original time series values.

        Args:
            df (pd.DataFrame): The DataFrame containing normalized time series data.

        Returns:
            pd.DataFrame: The DataFrame with the original (un-normalized) time series.
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
        Merges grid and solar columns back into a single time series column.

        Args:
            df (pd.DataFrame): DataFrame containing separate grid and solar columns.

        Returns:
            pd.DataFrame: DataFrame with merged time series column.
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
            int: Number of samples in the dataset.
        """
        return len(self.data)   

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Retrieves a single sample from the dataset.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
                - time_series: The time series data tensor.
                - conditioning_vars: Dictionary of conditioning variables.
        """
        sample = self.data.iloc[idx]
        time_series = sample["timeseries"]

        # Extract conditioning variables
        conditioning_vars = {
            'month': torch.tensor(sample['month'], dtype=torch.long),
            'weekday': torch.tensor(sample['weekday'], dtype=torch.long),
            'building_type': torch.tensor(sample['building_type'], dtype=torch.long),
            'car1': torch.tensor(sample['car1'], dtype=torch.long),
            'city': torch.tensor(sample['city'], dtype=torch.long),
            'state': torch.tensor(sample['state'], dtype=torch.long),
            'solar': torch.tensor(sample['solar'], dtype=torch.long)
        }

        return (
            torch.tensor(time_series, dtype=torch.float32),
            conditioning_vars
        )
    
def generate_first_differenced_dataset(dataset):
    """
    Generate the first differenced dataset from a PecanStreetDataset.
    
    Args:
        dataset (PecanStreetDataset): The original dataset.
    
    Returns:
        PecanStreetDataset: A new dataset with differenced time series.
    """
    differenced_data = dataset.data.copy()
    
    def difference_timeseries(series):
        diff = np.diff(series, axis=0)
        return np.pad(diff, ((1, 0), (0, 0)), mode='constant')
    
    differenced_data['timeseries'] = differenced_data['timeseries'].apply(difference_timeseries)
    
    # Create a new dataset with the differenced data
    differenced_dataset = PecanStreetDataset(
        data=differenced_data,
        stats=dataset.stats, 
        is_pv_user=dataset.is_pv_user,
        include_generation=dataset.include_generation,
        metadata=dataset.metadata
    )
    
    return differenced_dataset


def inverse_transform_differenced_dataset(differenced_dataset, original_dataset):
    """
    Transform the differenced dataset back to original values.
    
    Args:
        differenced_dataset (PecanStreetDataset): The differenced dataset.
        original_dataset (PecanStreetDataset): The original dataset to get the first value.
    
    Returns:
        PecanStreetDataset: A new dataset with the original time series reconstructed.
    """
    reconstructed_data = differenced_dataset.data.copy()
    
    def reconstruct_timeseries(row):
        diff_series = row['timeseries']
        original_first_value = original_dataset.data[
            (original_dataset.data['dataid'] == row['dataid']) &
            (original_dataset.data['month'] == row['month']) &
            (original_dataset.data['weekday'] == row['weekday'])
        ]['timeseries'].iloc[0][0]
        
        reconstructed = np.zeros_like(diff_series)
        reconstructed[0] = original_first_value
        np.cumsum(diff_series[1:], axis=0, out=reconstructed[1:])
        reconstructed[1:] += reconstructed[0]
        
        return reconstructed
    
    reconstructed_data['timeseries'] = reconstructed_data.apply(reconstruct_timeseries, axis=1)
    
    reconstructed_dataset = PecanStreetDataset(
        data=reconstructed_data,
        stats=original_dataset.stats,
        is_pv_user=original_dataset.is_pv_user,
        include_generation=original_dataset.include_generation,
        metadata=original_dataset.metadata
    )
    
    return reconstructed_dataset
    

    
