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
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        normalize=True,
        threshold=None,
        include_generation=True,
    ):
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.normalize = normalize
        self.threshold = threshold
        self.include_generation = include_generation
        self.name = "openpower"
        self.stats = {}
        self.data, self.metadata, self.user_flags = self.load_and_preprocess_data()

    def _get_dataset_info(self) -> Tuple[str, List[str]]:
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        dataset_info = config["datasets"].get(self.name)
        if not dataset_info:
            raise ValueError(f"No dataset configuration found for {self.name}")
        return dataset_info["path"], dataset_info["columns"]

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        path, columns = self._get_dataset_info()
        data = self._load_full_data(path, columns)
        data = self._preprocess_data(data)
        return data

    def _load_full_data(self, path: str, columns: List[str]) -> pd.DataFrame:
        data_file_path = f"/{path}household_data_15min.csv"
        return pd.read_csv(data_file_path)[columns]

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
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

        if self.threshold:
            data["grid"] = np.clip(data["grid"], *self.threshold)
            if self.include_generation and "solar" in data.columns:
                data["solar"] = np.clip(data["solar"], *self.threshold)

        if self.normalize:
            self.stats["grid"] = self._calculate_and_store_statistics(
                filtered_data, "grid"
            )
            filtered_data = self._apply_normalization(filtered_data, "grid")

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

    def _calculate_and_store_statistics(self, data: pd.DataFrame, column: str) -> Dict:
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
