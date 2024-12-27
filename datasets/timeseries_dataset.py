import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from datasets.utils import encode_conditioning_variables
from generator.normalizer import Normalizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TimeSeriesDataset(Dataset, ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        time_series_column_names: Any,
        seq_len: int,
        normalization_group_keys: List = [],
        conditioning_var_column_names: Any = None,
        normalize: bool = True,
        scale: bool = True,
        cluster_n_clusters: int = 10,  # Number of clusters for K-Means
        cluster_features: List[str] = None,  # Features to use for clustering
        cluster_rarity_threshold: float = 0.1,  # Top 10% rare clusters
        **kwargs,
    ):
        """
        Base class for time series datasets with frequency and clustering-based rarity detection.
        """
        self.time_series_column_names = (
            time_series_column_names
            if isinstance(time_series_column_names, list)
            else [time_series_column_names]
        )
        self.conditioning_vars = conditioning_var_column_names or []
        self.seq_len = seq_len
        self.cluster_n_clusters = cluster_n_clusters
        self.cluster_features = cluster_features or ["mean", "std", "max", "min"]
        self.cluster_rarity_threshold = cluster_rarity_threshold

        if not hasattr(self, "cfg"):
            with initialize_config_dir(
                config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
            ):
                overrides = [
                    f"seq_len={seq_len}",
                    f"input_dim={len(time_series_column_names)}",
                ]
                cfg = compose(config_name="default", overrides=overrides)
                self.numeric_conditioning_bins = cfg.numeric_conditioning_bins
                conditioning_vars = self._get_conditioning_var_dict(data)
                cfg.conditioning_vars = conditioning_vars
                self.cfg = cfg

        self.numeric_conditioning_bins = self.cfg.numeric_conditioning_bins

        if not hasattr(self, "threshold"):
            self.threshold = (-self.cfg.threshold, self.cfg.threshold)

        if not hasattr(self, "name"):
            self.name = "custom"

        self.normalize = normalize
        self.scale = scale
        self.use_learned_normalizer = self.cfg.use_learned_normalizer
        self.normalization_stats = {}
        self.normalization_group_keys = normalization_group_keys or []
        self.data = self._preprocess_data(data)

        if self.conditioning_vars:
            self.data, self.conditioning_var_codes = self._encode_conditioning_vars(
                self.data,
            )

        self._save_conditioning_var_codes()

        if self.normalize:
            self.data = self._normalize(self.data)

        self.data = self.merge_timeseries_columns(self.data)
        self.data = self.data.reset_index()

        self.data = self.get_frequency_based_rarity()
        self.data = self.get_clustering_based_rarity()
        self.data = self.get_combined_rarity()

    @abstractmethod
    def _preprocess_data(self):
        """
        Dataset-specific preprocessing logic. Should return complete and validated time series data in wide format, where timeseries columns contain arrays with specified
        sequence length of values in each row. Should also contain an entity identifier column, as well as optional conditioning var columns.
        """
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        timeseries = torch.tensor(sample["timeseries"], dtype=torch.float32)
        conditioning_vars_dict = {
            var: torch.tensor(sample[var], dtype=torch.long)
            for var in self.conditioning_vars
        }
        return timeseries, conditioning_vars_dict

    def split_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the 'timeseries' column into separate time series columns.
        Each dimension in the 'timeseries' array corresponds to a column in self.time_series_column_names.
        """
        if "timeseries" not in df.columns:
            raise ValueError("The DataFrame must contain a 'timeseries' column.")

        first_timeseries = df["timeseries"].iloc[0]
        if not isinstance(first_timeseries, np.ndarray):
            raise ValueError("Entries in 'timeseries' column must be numpy arrays.")

        n_dim = first_timeseries.shape[1]
        if n_dim != len(self.time_series_column_names):
            raise ValueError(
                f"The number of time series column names ({len(self.time_series_column_names)}) "
                f"does not match the number of dimensions in the 'timeseries' array ({n_dim})."
            )

        for idx, col_name in enumerate(self.time_series_column_names):
            df[col_name] = df["timeseries"].apply(lambda x: x[:, idx])
        df = df.drop(columns=["timeseries"])
        return df

    def merge_timeseries_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges multiple time series columns into a single 'timeseries' column.
        The resulting 'timeseries' column contains arrays of shape (seq_len, n_dim).

        Args:
            df (pd.DataFrame): DataFrame containing the time series columns specified in self.time_series_column_names.

        Returns:
            pd.DataFrame: DataFrame with a new 'timeseries' column and the individual time series columns removed.

        Raises:
            ValueError: If any required time series columns are missing or have inconsistent shapes.
        """
        missing_cols = [
            col_name
            for col_name in self.time_series_column_names
            if col_name not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"The following time series columns are missing from the DataFrame: {missing_cols}"
            )
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
                        raise ValueError(
                            f"Array in column '{col_name}' at index {idx} has incorrect sequence length "
                            f"({arr.shape[0]} instead of {self.seq_len})."
                        )
                    if arr.shape[1] != 1:
                        raise ValueError(
                            f"Array in column '{col_name}' at index {idx} has incorrect number of dimensions "
                            f"({arr.shape[1]} instead of 1)."
                        )
                else:
                    raise ValueError(
                        f"Array in column '{col_name}' at index {idx} must have shape ({self.seq_len}, 1), "
                        f"but has shape {arr.shape}."
                    )

        def merge_row(row):
            arrays = [row[col_name] for col_name in self.time_series_column_names]
            merged_array = np.hstack(arrays)  # Shape: (seq_len, n_dim)
            return merged_array

        df["timeseries"] = df.apply(merge_row, axis=1)
        df = df.drop(columns=self.time_series_column_names)
        return df

    def _calculate_and_store_statistics(self, data: pd.DataFrame) -> None:
        """
        Calculates and stores statistical data for both standardization and min-max scaling
        for each time series column.

        Args:
            data (pd.DataFrame): The data on which to calculate statistics.
        """

        def calculate_stats(group, column):
            all_values = np.concatenate(group[column].values)
            mean = np.mean(all_values)
            std = np.std(all_values)

            standardized = (all_values - mean) / (std + 1e-8)

            if self.cfg.scale:
                z_min = np.min(standardized)
                z_max = np.max(standardized)
                return pd.Series(
                    {"mean": mean, "std": std, "z_min": z_min, "z_max": z_max}
                )
            else:
                return pd.Series({"mean": mean, "std": std})

        for column in self.time_series_column_names:
            if self.normalization_group_keys:
                grouped_stats = data.groupby(self.normalization_group_keys).apply(
                    lambda group: calculate_stats(group, column)
                )
                self.normalization_stats[column] = grouped_stats.to_dict(orient="index")
            else:
                stats = calculate_stats(data, column)
                self.normalization_stats[column] = stats.to_dict()

    def _load_or_compute_normalization_stats(self):
        """
        Loads normalization statistics from a JSON file if it exists.
        Otherwise, computes the statistics and saves them to a JSON file.
        """
        stats_path = os.path.join(
            ROOT_DIR, "data", self.name, "normalization_stats.json"
        )

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                loaded_stats = json.load(f)
            self.normalization_stats = self._convert_keys_from_json(loaded_stats)
            print(f"Loaded normalization stats from {stats_path}")
        else:
            self._calculate_and_store_statistics(self.data)
            serializable_stats = self._convert_keys_to_json(self.normalization_stats)
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(serializable_stats, f, indent=4)
            print(f"Saved normalization stats to {stats_path}")

            global_stats = {}
            for column in self.time_series_column_names:
                all_values = np.concatenate(self.data[column].values)
                mean = np.mean(all_values)
                std = np.std(all_values)
                if self.cfg.scale:
                    z_min = np.min(all_values)
                    z_max = np.max(all_values)
                    global_stats[column] = {
                        "mean": mean,
                        "std": std,
                        "z_min": z_min,
                        "z_max": z_max,
                    }
                else:
                    global_stats[column] = {"mean": mean, "std": std}
            self.normalization_stats["global"] = global_stats

    def _convert_keys_to_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts list keys in the normalization_stats dictionary to JSON-compatible string representations.
        """
        if self.normalization_group_keys:
            converted_stats = {}
            for column, group_stats in stats.items():
                converted_stats[column] = {
                    json.dumps(k): v for k, v in group_stats.items()
                }
            return converted_stats
        else:
            return stats

    def _convert_keys_from_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts string keys back to list keys in the normalization_stats dictionary.
        """
        if self.normalization_group_keys:
            converted_stats = {}
            for column, group_stats in stats.items():
                converted_stats[column] = {}
                for k, v in group_stats.items():
                    try:
                        group_key = json.loads(k)
                    except json.JSONDecodeError:
                        group_key = k
                    converted_stats[column][tuple(group_key)] = v
            return converted_stats
        else:
            return stats

    def _normalize_and_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standardization followed by min-max scaling to the data.

        Args:
            data (pd.DataFrame): The data to normalize and scale.

        Returns:
            pd.DataFrame: The normalized and scaled data.
        """

        def normalize_and_scale_row(row, column):
            if self.normalization_group_keys:
                group_key = tuple(row[key] for key in self.normalization_group_keys)
                stats = self.normalization_stats[column].get(group_key)
                if not stats:
                    # Fallback to global stats
                    stats = self.normalization_stats[column].get("global")
                    if not stats:
                        raise ValueError(
                            f"No stats found for group {group_key} or global stats in column {column}"
                        )
            else:
                stats = self.normalization_stats[column]

            mean = stats["mean"]
            std = stats["std"]

            values = np.array(row[column])
            standardized = (values - mean) / (std + 1e-8)

            if hasattr(self, "threshold") and self.threshold:
                standardized = np.clip(standardized, *self.threshold)

            if self.cfg.scale:
                z_min = stats["z_min"]
                z_max = stats["z_max"]
                scaled = (standardized - z_min) / (z_max - z_min + 1e-8)
                return scaled
            else:
                return standardized

        for column in self.time_series_column_names:
            data[column] = data.apply(
                lambda row: normalize_and_scale_row(row, column), axis=1
            )

        return data

    def inverse_transform_column(self, row: pd.Series, column: str) -> np.ndarray:
        """
        Performs inverse transformation on a normalized and optionally scaled column
        to retrieve original values.

        Args:
            row (pd.Series): A row from the DataFrame that contains the normalized data.
            column (str): The column name of the normalized data.

        Returns:
            np.ndarray: The original (un-normalized and un-scaled) time series data.
        """
        if self.normalization_group_keys:
            group_key = tuple(row[key] for key in self.normalization_group_keys)
            stats = self.normalization_stats[column].get(group_key)
            if not stats:
                # Fallback to global stats
                stats = self.normalization_stats[column].get("global")
                if not stats:
                    raise ValueError(
                        f"No stats found for group {group_key} or global stats in column {column}"
                    )
        else:
            stats = self.normalization_stats[column]

        mean = stats["mean"]
        std = stats["std"]

        if self.cfg.scale:
            z_min = stats["z_min"]
            z_max = stats["z_max"]

        values = np.array(row[column])

        if self.cfg.scale:
            scaled = values * (z_max - z_min + 1e-8) + z_min
            unnormalized = scaled * std + mean
        else:
            unnormalized = values * std + mean

        return unnormalized

    def inverse_transform(
        self, data: pd.DataFrame, merged: bool = True
    ) -> pd.DataFrame:
        """
        Reverts normalization on the data.

        Args:
            data (pd.DataFrame): DataFrame containing the normalized data.
            merged (bool): Whether to return a df with the original timeseries columns or a single, multi-dimensional merged timeseries column.

        Returns:
            pd.DataFrame: DataFrame with the original (un-normalized and un-scaled) time series.
        """
        if self.use_learned_normalizer:
            data = self.learned_normalizer.inverse_transform(data, use_model=True)

        data = self.split_timeseries(data)

        if not self.use_learned_normalizer:
            for column in self.time_series_column_names:
                data[column] = data.apply(
                    lambda row: self.inverse_transform_column(row, column), axis=1
                )

        if merged:
            data = self.merge_timeseries_columns(data)

        return data

    def _encode_conditioning_vars(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Handles integer encoding of static and numeric conditioning variables. Bins numeric conditioning vars before encoding.

        Args:
            data (pd.DataFrame): DataFrame containing the decoded data.

        Returns:
            The input dataframe with integer encoded conditioning variables.
        """
        columns_to_encode = self.conditioning_vars
        encoded_data, conditioning_codes = encode_conditioning_variables(
            data=data,
            columns_to_encode=columns_to_encode,
            bins=self.numeric_conditioning_bins,
        )
        return encoded_data, conditioning_codes

    def _get_conditioning_var_dict(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Extracts the name and number of unique categories from the conditioning variables specified in the data.
        For numerical variables, bins the data into specified categories.

        Args:
            data (pd.DataFrame): The dataset.

        Returns:
            Dict[str, int]: Dictionary mapping variable names to number of categories.
        """
        conditioning_var_dict = {}

        for var_name in self.conditioning_vars:
            if pd.api.types.is_numeric_dtype(data[var_name]):
                binned = pd.cut(
                    data[var_name],
                    bins=self.numeric_conditioning_bins,
                    include_lowest=True,
                )
                num_categories = binned.nunique()
                conditioning_var_dict[var_name] = num_categories
            else:
                num_categories = data[var_name].astype("category").nunique()
                conditioning_var_dict[var_name] = num_categories

        return conditioning_var_dict

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes timeseries values according to dataset-specific groupings.
        """
        if self.use_learned_normalizer:
            self._init_learned_normalizer()
            return self.learned_normalizer.transform()
        else:
            self._load_or_compute_normalization_stats()
            return self._normalize_and_scale(data)

    def get_conditioning_var_codes(self):
        """
        Returns the integer mappings for conditioning variables.
        """
        return self.conditioning_var_codes

    def _save_conditioning_var_codes(self):
        """
        Saves the self.conditioning_var_codes dictionary to a JSON file in ROOT_DIR/data/{dataset_name} directory.
        If the directory doesn't exist, it creates one.
        """
        if not hasattr(self, "name"):
            raise ValueError(
                "Dataset name is not set. Please set 'self.name' to the dataset name."
            )

        dataset_dir = os.path.join(ROOT_DIR, "data", self.name)
        os.makedirs(dataset_dir, exist_ok=True)
        conditioning_codes_path = os.path.join(
            dataset_dir, "conditioning_var_codes.json"
        )

        with open(conditioning_codes_path, "w") as f:
            json.dump(self.conditioning_var_codes, f, indent=4)

    def sample_random_conditioning_vars(self):
        """
        Samples a random set of conditioning_vars.
        """
        conditioning_vars = {}
        conditioning_var_dict = self._get_conditioning_var_dict(self.data)
        for var_name, num_categories in conditioning_var_dict.items():
            conditioning_vars[var_name] = torch.randint(
                0,
                num_categories,
                dtype=torch.long,
                device=self.device,
            )
        return conditioning_vars

    def get_conditioning_var_combination_rarities(self, coverage_threshold=0.95):
        """
        Groups the dataset by all conditioning variables, counting the number of daily load profiles per combination.
        Computes rarity based on cumulative coverage in the data.

            args: coverage_threshold (float): The threshold used to determine binary rarity.
            returns: dataframe with combinations and rarity codes.
        """
        grouped = (
            self.data.groupby(self.conditioning_vars).size().reset_index(name="count")
        )
        grouped = grouped.sort_values(by="count", ascending=False)
        grouped["coverage"] = grouped["count"].cumsum() / self.data.shape[0]
        grouped["rare"] = grouped["coverage"] > coverage_threshold
        return grouped

    def get_frequency_based_rarity(self) -> pd.DataFrame:
        """
        Labels samples as frequency-rare based on their conditioning variable combinations.

        Returns:
            pd.DataFrame: DataFrame with a new column 'is_frequency_rare'.
        """
        freq_counts = (
            self.data.groupby(self.conditioning_vars).size().reset_index(name="count")
        )

        threshold = freq_counts["count"].quantile(self.cluster_rarity_threshold)

        freq_counts["is_frequency_rare"] = freq_counts["count"] < threshold

        self.data = self.data.merge(
            freq_counts[self.conditioning_vars + ["is_frequency_rare"]],
            on=self.conditioning_vars,
            how="left",
        )

        return self.data

    def get_clustering_based_rarity(self) -> pd.DataFrame:
        """
        Labels samples as pattern-rare based on clustering of their time series consumption profiles.

        Returns:
            pd.DataFrame: DataFrame with a new column 'is_pattern_rare'.
        """
        try:
            time_series_data = np.stack(
                self.data["timeseries"].values, axis=0
            )  # Shape: (num_samples, seq_len, n_dim)
        except ValueError as e:
            raise ValueError(f"Error stacking 'timeseries' data: {e}")

        num_samples, seq_len, n_dim = time_series_data.shape
        expected_n_dim = len(self.time_series_column_names)
        if n_dim != expected_n_dim:
            raise ValueError(f"Expected n_dim={expected_n_dim}, but got n_dim={n_dim}.")

        # Feature Extraction
        features = self.extract_features(time_series_data)

        # Standardize features
        # scaler = StandardScaler()
        # features_scaled = scaler.fit_transform(features)
        features_scaled = features

        kmeans = KMeans(n_clusters=self.cluster_n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        self.data["cluster"] = cluster_labels

        cluster_sizes = self.data["cluster"].value_counts().to_dict()

        size_threshold = np.percentile(
            list(cluster_sizes.values()), 100 * (1 - self.cluster_rarity_threshold)
        )

        self.data["is_pattern_rare"] = (
            self.data["cluster"].map(cluster_sizes) < size_threshold
        )

        return self.data

    def extract_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extracts features from time series data for clustering.

        Args:
            time_series (np.ndarray): Shape (num_samples, seq_len, n_dim)

        Returns:
            np.ndarray: Shape (num_samples, num_features)
        """
        num_samples, seq_len, n_dim = time_series.shape
        features = []

        for ts in time_series:
            # Statistical Features for each dimension
            mean = np.mean(ts, axis=0)
            std = np.std(ts, axis=0)
            max_val = np.max(ts, axis=0)
            min_val = np.min(ts, axis=0)
            skew = pd.Series(ts[:, 0]).skew()
            kurt = pd.Series(ts[:, 0]).kurtosis()

            # Time-Domain Features
            peak_indices = np.argmax(ts, axis=0)  # Index of peak per dimension

            feature_vector = np.concatenate(
                [mean, std, max_val, min_val, [skew], [kurt], peak_indices]
            )
            features.append(feature_vector)

        features = np.array(features)  # Shape: (num_samples, features_dim)
        return features

    def get_combined_rarity(self) -> pd.DataFrame:
        """
        Combines frequency-based and clustering-based rarity to label samples as 'rare'.

        Returns:
            pd.DataFrame: DataFrame with a new column 'is_rare'.
        """
        self.data["is_rare"] = (
            self.data["is_frequency_rare"] & self.data["is_pattern_rare"]
        )
        return self.data

    def _init_learned_normalizer(self):
        """
        Sets up (or loads) the Normalizer which includes a ConditioningModule + StatsHead.
        """
        self.learned_normalizer = Normalizer(
            dataset=self,
            cfg=self.cfg,
        )

        normalizer_ckpt_path = os.path.join(
            ROOT_DIR, "checkpoints/normalizer", f"{self.name}_normalizer.pt"
        )
        if os.path.exists(normalizer_ckpt_path):
            print(f"Loading existing normalizer from {normalizer_ckpt_path}")
            self.learned_normalizer.load(normalizer_ckpt_path)
        else:
            print("No learned normalizer checkpoint found, training new normalizer.")
            self.learned_normalizer.compute_group_stats()
            self.learned_normalizer.train_normalizer()
            self.learned_normalizer.save(path=normalizer_ckpt_path)
