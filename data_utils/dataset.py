import os
from typing import List, Tuple

import pandas as pd
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset:
    def __init__(self, name: str, config_path: str = "config/config.yaml"):
        """
        Initialize the Dataset with a name and configuration path.

        Args:
            name: Name of the dataset.
            config_path: Path to the configuration file.
        """
        self.name = name
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.path, self.columns = self._get_dataset_info()
        self.data = self.get_data()

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
        Load the dataset. To be implemented by subclasses.

        Returns:
            A pandas DataFrame containing the dataset.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Format the dataset by selecting specific columns.

        Args:
            data: The raw dataset.

        Returns:
            A pandas DataFrame with the selected columns.
        """
        return data[self.columns]

    def get_data(self) -> pd.DataFrame:
        """
        Get the formatted dataset.

        Returns:
            A pandas DataFrame with the formatted dataset.
        """
        data = self.load_data()
        return self.format_data(data)


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
        super().__init__(name="pecanstreetdata", config_path=config_path)

    def load_data(self) -> pd.DataFrame:
        """
        Load the Pecan Street dataset.

        Returns:
            A pandas DataFrame containing the dataset.
        """
        data_file_path = f"/{self.path}15minute_data_{self.geography}.csv"
        try:
            data = pd.read_csv(data_file_path)
            return data
        except Exception as e:
            print(f"Failed to load data from {data_file_path}: {e}")
            return pd.DataFrame()

    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Format the Pecan Street dataset by selecting specific columns.

        Args:
            data: The raw dataset.

        Returns:
            A pandas DataFrame with the selected columns.
        """
        data = super().format_data(data)
        return data
