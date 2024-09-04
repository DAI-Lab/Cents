import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PecanStreetDataset(Dataset):
    def __init__(
        self,
        geography: str = None,
        config_path: str = "config/config.yaml",
        normalize=False,
        threshold=None,
        include_generation=False,
    ):
        self.geography = geography
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.normalize = normalize
        self.threshold = threshold
        self.include_generation = include_generation
        self.name = "pecanstreet"
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
        metadata = pd.read_csv(f"/{path}metadata.csv")
        data = self._load_full_data(path, columns)
        user_flags = self._set_user_flags(metadata, data)
        data = self._preprocess_data(data)
        return data, metadata, user_flags

    def _load_full_data(self, path: str, columns: List[str]) -> pd.DataFrame:
        if self.geography:
            data_file_path = f"/{path}15minute_data_{self.geography}.csv"
            return pd.read_csv(data_file_path)[columns]
        else:
            return pd.concat(
                [
                    pd.read_csv(f"/{path}15minute_data_newyork.csv"),
                    pd.read_csv(f"/{path}15minute_data_california.csv"),
                    pd.read_csv(f"/{path}15minute_data_austin.csv"),
                ],
                axis=0,
            )[columns]

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

        grouped_stats = data.groupby(["dataid", "month", "weekday"]).apply(
            calculate_stats
        )
        return grouped_stats.to_dict(orient="index")

    def _apply_normalization(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
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

    @staticmethod
    def _normalize_and_scale(values, mean, std):
        normalized = (np.array(values) - mean) / (std + 1e-8)
        norm_min_val, norm_max_val = normalized.min(), normalized.max()
        scaled = (normalized - norm_min_val) / ((norm_max_val - norm_min_val) + 1e-8)
        return scaled, norm_min_val, norm_max_val

    def _preprocess_solar(self, data: pd.DataFrame) -> pd.DataFrame:
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
    def _set_user_flags(metadata: pd.DataFrame, data: pd.DataFrame) -> dict:
        return {
            user_id: metadata.loc[metadata["dataid"] == user_id]["solar"].notna().any()
            for user_id in data["dataid"].unique()
        }

    def create_user_dataset(self, user_id: int):
        if user_id not in self.user_flags:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        user_data = self.data[self.data["dataid"] == user_id].copy()
        return PecanStreetUserDataset(
            user_data,
            self.stats,
            self.user_flags[user_id],
            self.include_generation,
            self.metadata,
        )

    def create_all_pv_user_dataset(self):
        pv_users = [user for user, has_pv in self.user_flags.items() if has_pv]
        pv_data = self.data[self.data["dataid"].isin(pv_users)].copy()

        return PecanStreetUserDataset(
            data=pv_data,
            stats=self.stats,
            is_pv_user=True,
            include_generation=True,
            metadata=self.metadata,
        )

    def create_non_pv_user_dataset(self):
        non_pv_users = [user for user, has_pv in self.user_flags.items() if not has_pv]
        non_pv_data = self.data[self.data["dataid"].isin(non_pv_users)].copy()

        return PecanStreetUserDataset(
            data=non_pv_data,
            stats=self.stats,
            is_pv_user=False,
            include_generation=False,
            metadata=self.metadata,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        time_series = sample["timeseries"]
        month = sample["month"]
        day = sample["weekday"]
        return (
            torch.tensor(time_series, dtype=torch.float32).to(device),
            torch.tensor(month, dtype=torch.long).to(device),
            torch.tensor(day, dtype=torch.long).to(device),
        )


class PecanStreetUserDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        stats: dict,
        is_pv_user: bool,
        include_generation: bool,
        metadata: pd.DataFrame,
    ):
        self.data = self.validate_data(data)
        self.stats = stats
        self.is_pv_user = is_pv_user
        self.include_generation = include_generation
        self.metadata = metadata
        self.include_user_metadata()

    def include_user_metadata(self):
        self.data = pd.merge(
            left=self.data,
            right=self.metadata[["dataid", "city", "pv", "car1"]],
            on="dataid",
        )

    def validate_data(self, data):
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
        self.is_pv_user = False
        return data

    def inverse_transform_column(self, row, colname):
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

    def inverse_transform(self, df):
        def split_timeseries(row):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        time_series = sample["timeseries"]
        month = sample["month"]
        day = sample["weekday"]

        return (
            torch.tensor(time_series, dtype=torch.float32).to(device),
            torch.tensor(month, dtype=torch.long).to(device),
            torch.tensor(day, dtype=torch.long).to(device),
        )
