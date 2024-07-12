import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PecanStreetDataset(Dataset):
    def __init__(
        self, geography: str = "newyork", config_path: str = "config/config.yaml"
    ):
        """
        Initialize the PecanStreetDataset with a specific geography and configuration path.

        Args:
            geography: Geography of the dataset (e.g., 'newyork').
            config_path: Path to the configuration file.
        """
        self.geography = geography
        self.config_path = os.path.join(ROOT_DIR, config_path)
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
        data_file_path = f"/{self.path}15minute_data_{self.geography}.csv"
        try:
            data = pd.read_csv(data_file_path)
            data = data[self.columns]
        except Exception as e:
            print(f"Failed to load data from {data_file_path}: {e}")
            return pd.DataFrame()

        data["month"] = pd.to_datetime(data["local_15min"]).dt.month
        data["day"] = pd.to_datetime(data["local_15min"]).dt.weekday
        return data

    def get_user_data(self, user_id: int) -> pd.DataFrame:
        """
        Select data from a specific user.

        Args:
            user_id: The desired user's id.

        Returns:
            A pandas DataFrame with the selected user data.
        """
        user_data = self.data[self.data["dataid"] == user_id].copy()
        user_data.reset_index(inplace=True)
        user_data = user_data[["grid", "month", "day"]]
        return user_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        time_series = sample["grid"].astype(np.float32)
        month = sample["month"]
        day = sample["day"]
        return (
            torch.tensor(time_series),
            torch.tensor(month, dtype=torch.long),
            torch.tensor(day, dtype=torch.long),
        )


def prepare_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
