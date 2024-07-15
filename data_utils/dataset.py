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
    ):
        """
        Initialize the PecanStreetDataset with a specific geography, configuration path and user.

        Args:
            geography: Geography of the dataset (e.g., 'newyork').
            config_path: Path to the configuration file.
            normalize: Flag indicating whether data is normalized.
        """
        self.geography = geography
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.normalize = normalize
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

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the 'grid' column in the data based on groupings of 'dataid', 'month', and 'weekday'.

        Args:
            data (pd.DataFrame): The input data with columns 'dataid', 'month', 'weekday', and 'grid'.

        Returns:
            pd.DataFrame: The normalized data.
        """
        data["month"] = pd.to_datetime(data["local_15min"]).dt.month
        data["weekday"] = pd.to_datetime(data["local_15min"]).dt.weekday
        data["date_day"] = pd.to_datetime(data["local_15min"]).dt.day

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
                    {"mean": np.mean(all_values), "std": np.std(all_values)}
                )

            grouped_stats = (
                filtered_data.groupby(["dataid", "month", "weekday"])
                .apply(calculate_stats)
                .reset_index()
            )
            filtered_data = filtered_data.merge(
                grouped_stats, on=["dataid", "month", "weekday"]
            )

            def normalize(grid, mean, std):
                return [(val - mean) / std for val in grid]

            filtered_data["grid"] = filtered_data.apply(
                lambda row: normalize(row["grid"], row["mean"], row["std"]), axis=1
            )
            filtered_data = filtered_data.drop(columns=["mean", "std"])

        return filtered_data

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
            torch.tensor(time_series),
            torch.tensor(month, dtype=torch.long),
            torch.tensor(day, dtype=torch.long),
        )


def prepare_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_dataset(dataset: Dataset, val_split: float = 0.2, random_seed: int = 42):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed),
    )
    return train_dataset, val_dataset
