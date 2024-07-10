import yaml
import pandas as pd

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Dataset:
    def __init__(self, name, config_path="config/config.yaml"):
        self.name = name
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.path, self.columns = self._get_dataset_info()
        self.data = self.get_data()

    def _get_dataset_info(self):
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        dataset_info = config["datasets"].get(self.name)
        if not dataset_info:
            raise ValueError(f"No dataset configuration found for {self.name}")
        return dataset_info["path"], dataset_info["columns"]

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method")

    def format_data(self, data):
        return data[self.columns]

    def get_data(self):
        data = self.load_data()
        return self.format_data(data)


class PecanStreetDataset(Dataset):
    def __init__(self, geography="newyork", config_path="config/config.yaml"):
        self.geography = geography
        super().__init__(name="pecanstreetdata", config_path=config_path)
        
    def load_data(self):
        data_file_path = f"/{self.path}15minute_data_{self.geography}.csv"
        try:
            data = pd.read_csv(data_file_path)
            return data
        except Exception as e:
            print(f"Failed to load data from {data_file_path}: {e}")
            return None
        
    def format_data(self, data):
        data = super().format_data(data)
        return data
    