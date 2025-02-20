import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from endata.datasets.utils import encode_context_variables
from endata.generator.normalizer import Normalizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TimeSeriesDataset(Dataset, ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        time_series_column_names: Any,
        seq_len: int,
        context_var_column_names: Any = None,
        normalize: bool = True,
        scale: bool = True,
    ):
        self.time_series_column_names = (
            time_series_column_names
            if isinstance(time_series_column_names, list)
            else [time_series_column_names]
        )
        self.time_series_dims = len(self.time_series_column_names)
        self.context_vars = context_var_column_names or []
        self.seq_len = seq_len

        if not hasattr(self, "cfg"):
            with initialize_config_dir(
                config_dir=os.path.join(ROOT_DIR, "config", "dataset"),
                version_base=None,
            ):
                overrides = [
                    f"seq_len={seq_len}",
                    f"time_series_dims={len(self.time_series_column_names)}",
                ]
                cfg = compose(config_name="default", overrides=overrides)
                cfg.time_series_columns = self.time_series_column_names
                self.numeric_context_bins = cfg.numeric_context_bins
                context_vars = self._get_context_var_dict(data)
                cfg.context_vars = context_vars
                self.cfg = cfg

        self.numeric_context_bins = self.cfg.numeric_context_bins
        if not hasattr(self, "threshold"):
            self.threshold = (-self.cfg.threshold, self.cfg.threshold)

        if not hasattr(self, "name"):
            self.name = "custom"

        self.normalize = normalize
        self.scale = scale
        self.data = self._preprocess_data(data)

        if self.context_vars:
            self.data, self.context_var_codes = self._encode_context_vars(self.data)
        self._save_context_var_codes()

        if self.normalize:
            self._init_normalizer()
            self.data = self._normalizer._transform()

        self.data = self.merge_timeseries_columns(self.data)
        self.data = self.data.reset_index()
        self.data = self.get_frequency_based_rarity()
        self.data = self.get_clustering_based_rarity()
        self.data = self.get_combined_rarity()

    @abstractmethod
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        timeseries = torch.tensor(sample["timeseries"], dtype=torch.float32)
        context_vars_dict = {
            var: torch.tensor(sample[var], dtype=torch.long)
            for var in self.context_vars
        }
        return timeseries, context_vars_dict

    def split_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timeseries" not in df.columns:
            raise ValueError("Missing 'timeseries' column.")
        first_timeseries = df["timeseries"].iloc[0]
        if not isinstance(first_timeseries, np.ndarray):
            raise ValueError("'timeseries' entries must be numpy arrays.")
        n_dim = first_timeseries.shape[1]
        if n_dim != len(self.time_series_column_names):
            raise ValueError(
                "Mismatch between time series column names and data shape."
            )
        for idx, col_name in enumerate(self.time_series_column_names):
            df[col_name] = df["timeseries"].apply(lambda x: x[:, idx])
        df = df.drop(columns=["timeseries"])
        return df

    def merge_timeseries_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [
            col_name
            for col_name in self.time_series_column_names
            if col_name not in df.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        for col_name in self.time_series_column_names:
            for idx, arr in df[col_name].items():
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr)
                    df.at[idx, col_name] = arr
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                    df.at[idx, col_name] = arr
                elif arr.ndim == 2:
                    if arr.shape[0] != self.seq_len:
                        raise ValueError("Incorrect sequence length in column.")
                    if arr.shape[1] != 1:
                        raise ValueError("Incorrect dimension in column.")
                else:
                    raise ValueError("Array must have shape (seq_len, 1).")

        def merge_row(row):
            arrays = [row[col_name] for col_name in self.time_series_column_names]
            return np.hstack(arrays)

        df["timeseries"] = df.apply(merge_row, axis=1)
        df = df.drop(columns=self.time_series_column_names)
        return df

    def inverse_transform(
        self, data: pd.DataFrame, merged: bool = True
    ) -> pd.DataFrame:
        if not self.normalize:
            return data
        df = self.split_timeseries(data)
        df = self._normalizer._inverse_transform(df)
        if merged:
            df = self.merge_timeseries_columns(df)
        return df

    def _encode_context_vars(self, data: pd.DataFrame):
        columns_to_encode = self.context_vars
        encoded_data, context_codes = encode_context_variables(
            data=data,
            columns_to_encode=columns_to_encode,
            bins=self.numeric_context_bins,
        )
        return encoded_data, context_codes

    def _get_context_var_dict(self, data: pd.DataFrame) -> Dict[str, int]:
        context_var_dict = {}
        for var_name in self.context_vars:
            if pd.api.types.is_numeric_dtype(data[var_name]):
                binned = pd.cut(
                    data[var_name],
                    bins=self.numeric_context_bins,
                    include_lowest=True,
                )
                num_categories = binned.nunique()
                context_var_dict[var_name] = num_categories
            else:
                num_categories = data[var_name].astype("category").nunique()
                context_var_dict[var_name] = num_categories
        return context_var_dict

    def get_context_var_codes(self):
        return self.context_var_codes

    def _save_context_var_codes(self):
        dataset_dir = os.path.join(ROOT_DIR, "data", self.name)
        os.makedirs(dataset_dir, exist_ok=True)
        context_codes_path = os.path.join(dataset_dir, "context_var_codes.json")
        with open(context_codes_path, "w") as f:
            json.dump(self.context_var_codes, f, indent=4)

    def sample_random_context_vars(self):
        context_vars = {}
        context_var_dict = self._get_context_var_dict(self.data)
        for var_name, num_categories in context_var_dict.items():
            context_vars[var_name] = torch.randint(
                0, num_categories, dtype=torch.long, device=self.device
            )
        return context_vars

    def get_context_var_combination_rarities(self, coverage_threshold=0.95):
        grouped = self.data.groupby(self.context_vars).size().reset_index(name="count")
        grouped = grouped.sort_values(by="count", ascending=False)
        grouped["coverage"] = grouped["count"].cumsum() / self.data.shape[0]
        grouped["rare"] = grouped["coverage"] > coverage_threshold
        return grouped

    def get_frequency_based_rarity(self) -> pd.DataFrame:
        freq_counts = (
            self.data.groupby(self.context_vars).size().reset_index(name="count")
        )
        threshold = freq_counts["count"].quantile(0.1)
        freq_counts["is_frequency_rare"] = freq_counts["count"] < threshold
        self.data = self.data.merge(
            freq_counts[self.context_vars + ["is_frequency_rare"]],
            on=self.context_vars,
            how="left",
        )
        return self.data

    def get_clustering_based_rarity(self) -> pd.DataFrame:
        try:
            time_series_data = np.stack(self.data["timeseries"].values, axis=0)
        except ValueError as e:
            raise ValueError(f"Error stacking 'timeseries' data: {e}")
        num_samples, seq_len, n_dim = time_series_data.shape
        expected_n_dim = len(self.time_series_column_names)
        if n_dim != expected_n_dim:
            raise ValueError("Dimension mismatch in time series data.")
        features = self.extract_features(time_series_data)
        features_scaled = features
        kmeans = KMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        self.data["cluster"] = cluster_labels
        cluster_sizes = self.data["cluster"].value_counts().to_dict()
        size_threshold = np.percentile(list(cluster_sizes.values()), 100 * (0.9))
        self.data["is_pattern_rare"] = (
            self.data["cluster"].map(cluster_sizes) < size_threshold
        )
        return self.data

    def extract_features(self, time_series: np.ndarray) -> np.ndarray:
        num_samples, seq_len, n_dim = time_series.shape
        features = []
        for ts in time_series:
            mean = np.mean(ts, axis=0)
            std = np.std(ts, axis=0)
            max_val = np.max(ts, axis=0)
            min_val = np.min(ts, axis=0)
            skew = pd.Series(ts[:, 0]).skew()
            kurt = pd.Series(ts[:, 0]).kurtosis()
            peak_indices = np.argmax(ts, axis=0)
            feature_vector = np.concatenate(
                [mean, std, max_val, min_val, [skew], [kurt], peak_indices]
            )
            features.append(feature_vector)
        features = np.array(features)
        return features

    def get_combined_rarity(self) -> pd.DataFrame:
        if "is_frequency_rare" not in self.data.columns:
            self.data = self.get_frequency_based_rarity()
        if "is_pattern_rare" not in self.data.columns:
            self.data = self.get_clustering_based_rarity()
        self.data["is_rare"] = (
            self.data["is_frequency_rare"] & self.data["is_pattern_rare"]
        )
        return self.data

    @property
    def device(self):
        return (
            torch.device(self.cfg.device)
            if "device" in self.cfg
            else torch.device("cpu")
        )

    def _init_normalizer(self):
        normalizer_ckpt_path = os.path.join(
            ROOT_DIR,
            f"checkpoints/{self.name}/normalizer",
            f"{self.name}_dim_{self.time_series_dims}_scale_{self.scale}_normalizer.pt",
        )
        self._normalizer = Normalizer(
            dataset=self, dataset_cfg=self.cfg, normalizer_path=normalizer_ckpt_path
        )
