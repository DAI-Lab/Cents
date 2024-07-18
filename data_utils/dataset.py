import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PecanStreetDataset(Dataset):
    def __init__(
        self,
        geography: str = None,
        config_path: str = "config/config.yaml",
        normalize=False,
        user_id=None,
    ):
        """
        Initialize the PecanStreetDataset with a specific geography, configuration path and user.

        Args:
            geography: Geography of the dataset (e.g., 'newyork').
            config_path: Path to the configuration file.
            normalize: Flag indicating whether data is normalized.
            user_id: ID of the desired user. If None, all user data is loaded.
        """
        self.geography = geography
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.normalize = normalize
        self.user_id = user_id
        self.name = "pecanstreet"
        self.path, self.columns = self._get_dataset_info()
        self.data = self.load_data()

    def _get_dataset_info(self) -> Tuple[str, List[str]]:
        """
        Retrieve dataset information from the configuration file.

        Returns:
            A tuple containing the path and columns of the dataset.
        """
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        dataset_info = config["datasets"].get(self.name)
        if not dataset_info:
            raise ValueError(f"No dataset configuration found for {self.name}")
        return dataset_info["path"], dataset_info["columns"]

    def load_data(self) -> pd.DataFrame:
        """
        Load the Pecan Street dataset.

        Returns:
            A pandas DataFrame containing the dataset.
        """
        self.metadata = pd.read_csv(f"/{self.path}metadata.csv")
        if self.geography:
            data_file_path = f"/{self.path}15minute_data_{self.geography}.csv"
            try:
                data = pd.read_csv(data_file_path)
                data = data[self.columns]
            except Exception as e:
                print(f"Failed to load data from {data_file_path}: {e}")
                return pd.DataFrame()
        else:
            ny_file_path = f"/{self.path}15minute_data_newyork.csv"
            cali_file_path = f"/{self.path}15minute_data_california.csv"
            austin_file_path = f"/{self.path}15minute_data_austin.csv"

            try:
                ny_data = pd.read_csv(ny_file_path)
                cali_data = pd.read_csv(cali_file_path)
                austin_data = pd.read_csv(austin_file_path)
                data = pd.concat([ny_data, cali_data, austin_data], axis=0)
                data = data[self.columns]
            except Exception as e:
                print(f"Failed to load full dataset!")
                return pd.DataFrame()

        data = self.preprocess_data(data)
        return data

    def preprocess_data(self, data: pd.DataFrame, threshold=(-2, 2)) -> pd.DataFrame:
        """
        Normalize the 'grid' column in the data based on groupings of 'dataid', 'month', and 'weekday'.

        Args:
            data (pd.DataFrame): The input data with columns 'dataid', 'month', 'weekday', and 'grid'.

        Returns:
            pd.DataFrame: The normalized data.
        """
        if self.user_id:
            data = data[data["dataid"] == self.user_id].copy()

            if not len(data):
                raise ValueError(f"No data found for user {self.user_id}!")

            self.user_metadata = self.metadata.loc[
                self.metadata["dataid"] == self.user_id
            ].copy()
            self.is_pv_user = self.user_metadata["pv"].notna().any()

        data["local_15min"] = pd.to_datetime(data["local_15min"], utc=True)
        data["month"] = data["local_15min"].dt.month - 1
        data["weekday"] = data["local_15min"].dt.weekday
        data["date_day"] = data["local_15min"].dt.day
        data = data.sort_values(by=["local_15min"], axis=0)

        # Drop rows where grid is na
        data = data[~data["grid"].isna()].copy()

        if not all(
            col in data.columns for col in ["dataid", "month", "weekday", "grid"]
        ):
            raise ValueError(
                "Input data must contain 'dataid', 'month', 'weekday', and 'grid' columns"
            )

        grouped_data = data.groupby(["dataid", "month", "date_day", "weekday"])
        grouped_data = grouped_data["grid"].apply(list).reset_index()

        # Remove columns without full day of available data
        filtered_data = grouped_data[grouped_data["grid"].apply(len) == 96]

        if self.normalize:

            def calculate_stats(group):
                all_values = np.concatenate(group["grid"].values)
                return pd.Series(
                    {
                        "mean": np.mean(all_values),
                        "std": np.std(all_values),
                        "min": np.min(all_values),
                        "max": np.max(all_values),
                    }
                )

            # per user weekday month mean std min and max value

            grouped_stats = (
                filtered_data.groupby(["dataid", "month", "weekday"])
                .apply(calculate_stats)
                .reset_index()
            )
            filtered_data = filtered_data.merge(
                grouped_stats, on=["dataid", "month", "weekday"]
            )

            def normalize_and_scale(grid, mean, std, min_val, max_val):
                normalized = [(val - mean) / std for val in grid]
                thresholded = np.clip(normalized, threshold[0], threshold[1])
                return [
                    (val - threshold[0]) / (threshold[1] - threshold[0])
                    for val in thresholded
                ]

            filtered_data["grid"] = filtered_data.apply(
                lambda row: normalize_and_scale(
                    row["grid"], row["mean"], row["std"], row["min"], row["max"]
                ),
                axis=1,
            )

            # Store statistics for inverse transformation
            self.stats = grouped_stats.set_index(
                ["dataid", "month", "weekday"]
            ).to_dict("index")

            filtered_data = filtered_data.drop(columns=["mean", "std", "min", "max"])

        return filtered_data

    def inverse_transform(
        self, preprocessed_data: pd.DataFrame, dataid, month, weekday
    ) -> np.array:
        """
        Convert a preprocessed time series back to its original scale.

        Args:
            preprocessed_data (pd.DataFrame): The preprocessed data.
            dataid: The dataid of the time series.
            month: The month of the time series.
            weekday: The weekday of the time series.

        Returns:
            np.array: The time series in its original scale.
        """
        if not self.normalize:
            return preprocessed_data["grid"].values

        stats = self.stats.get((dataid, month, weekday))
        if stats is None:
            raise ValueError(
                f"No statistics found for dataid={dataid}, month={month}, weekday={weekday}"
            )

        mean, std = stats["mean"], stats["std"]
        low, high = self.threshold

        def inverse_transform_single(val):
            # Inverse min-max scaling
            unscaled = val * (high - low) + low
            # Inverse normalization
            return unscaled * std + mean

        return np.array(
            [inverse_transform_single(val) for val in preprocessed_data["grid"]]
        )

    def filter_by_user(self, user_id: int) -> None:
        """
        Filter the dataset for a specific user.

        Args:
            user_id: The desired user's id.

        Returns:
            A pandas DataFrame with the selected user data.
        """
        user_data = self.data[self.data["dataid"] == user_id].copy()
        user_data.reset_index(inplace=True)
        user_data = user_data[["grid", "month", "weekday"]]
        self.data = user_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        time_series = sample["grid"]
        month = sample["month"]
        day = sample["weekday"]
        return (
            torch.tensor(time_series, dtype=torch.float32),
            torch.tensor(month, dtype=torch.long),
            torch.tensor(day, dtype=torch.long),
        )


def prepare_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_dataset(dataset: Dataset, val_split: float = 0.2):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    return train_dataset, val_dataset
