# timeseries_dataset.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.utils import encode_conditioning_variables


class TimeSeriesDataset(Dataset, ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        entity_column_name: str,
        time_series_column_names: Any,
        conditioning_var_column_names: Any = None,
        seq_len: int = None,
        normalize: bool = True,
        scale: bool = True,
        normalization_group_keys: List = [],
        **kwargs,
    ):
        """
        Base class for time series datasets.
        """
        self.time_series_column_names = (
            time_series_column_names
            if isinstance(time_series_column_names, list)
            else [time_series_column_names]
        )
        self.conditioning_vars = conditioning_var_column_names or []
        self.seq_len = seq_len
        self.normalize = normalize
        self.scale = scale
        self.normalization_stats = {}
        self.normalization_group_keys = normalization_group_keys or []
        self.data = self._preprocess_data(data)

        if self.normalize:
            self._calculate_and_store_statistics(self.data)
            self.data = self._normalize_and_scale(self.data)

        if self.conditioning_vars:
            self.data, self.conditioning_codes = self.encode_conditioning_vars(
                self.data
            )

    @abstractmethod
    def _preprocess_data(self):
        """
        Dataset-specific preprocessing logic. Should return complete and validated time series data in wide format, where timeseries columns contain arrays with specified
        sequence length of values in each row. Should also contain an entity identifier column, as well as optional conditioning var columns.
        """
        pass

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        sample = self.processed_data.iloc[idx]
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
            ValueError: If any required time series columns are missing.
            AssertionError: If any arrays in the time series columns do not have the correct shape.
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
            for idx, arr in df[col_name].iteritems():
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr)
                    df.at[idx, col_name] = arr  # Update in case it was a list
                # Handle arrays with shape (seq_len,) and (seq_len, 1)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                    df.at[idx, col_name] = arr  # Update with reshaped array
                else:
                    raise ValueError(
                        f"Array in column '{col_name}' at index {idx} must have shape (seq_len,) or (seq_len, 1), "
                        f"but has shape {arr.shape}."
                    )
                assert arr.shape[0] == self.seq_len, (
                    f"Array in column '{col_name}' at index {idx} must have length {self.seq_len}, "
                    f"but has length {arr.shape[0]}."
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

            z_min = np.min(standardized)
            z_max = np.max(standardized)

            return pd.Series({"mean": mean, "std": std, "z_min": z_min, "z_max": z_max})

        for column in self.time_series_column_names:
            if self.normalization_group_keys:
                grouped_stats = data.groupby(self.normalization_group_keys).apply(
                    lambda group: calculate_stats(group, column)
                )
                self.normalization_stats[column] = grouped_stats.to_dict(orient="index")
            else:
                stats = calculate_stats(data, column)
                self.normalization_stats[column] = stats.to_dict()

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
                    raise ValueError(
                        f"No stats found for group {group_key} in column {column}"
                    )
            else:
                stats = self.normalization_stats[column]

            mean = stats["mean"]
            std = stats["std"]
            z_min = stats["z_min"]
            z_max = stats["z_max"]

            values = np.array(row[column])
            standardized = (values - mean) / (std + 1e-8)

            if self.threshold:
                standardized = np.clip(standardized, *self.threshold)

            scaled = (standardized - z_min) / (z_max - z_min + 1e-8)

            if self.scale:
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
                raise ValueError(
                    f"No stats found for group {group_key} in column {column}"
                )
        else:
            stats = self.normalization_stats[column]

        mean = stats["mean"]
        std = stats["std"]
        z_min = stats["z_min"]
        z_max = stats["z_max"]

        values = np.array(row[column])

        if self.scale:
            scaled = values * (z_max - z_min + 1e-8) + z_min
            unnormalized = scaled * std + mean
        else:
            unnormalized = values * std + mean

        return unnormalized

    def inverse_transform(
        self, data: pd.DataFrame, merged: bool = False
    ) -> pd.DataFrame:
        """
        Reverts normalization on the data.

        Args:
            data (pd.DataFrame): DataFrame containing the normalized data.
            merged (bool): Whether to return a df with the original timeseries columns or a single, multi-dimensional merged timeseries column.

        Returns:
            pd.DataFrame: DataFrame with the original (un-normalized and un-scaled) time series.
        """
        data = self.split_timeseries(data)

        for column in self.time_series_column_names:
            data[column] = data.apply(
                lambda row: self.inverse_transform_column(row, column), axis=1
            )

        if merged:
            data = self.merge_timeseries_columns(data)

        return data

    def encode_conditioning_vars(
        self, data: pd.DataFrame, numeric_conditioning_bins=5
    ) -> pd.DataFrame:
        """
        Handles integer encoding of static and numeric conditioning variables. Bins numeric conditioning vars before encoding.

        Args:
            data (pd.DataFrame): DataFrame containing the decoded data.
            numeric_conditioning_bins (int): Parameter specifying the number of bins for numeric conditioning variables.

        Returns:
            The input dataframe with integer encoded conditioning variables.
        """
        columns_to_encode = self.conditioning_vars
        encoded_data, conditioning_codes = encode_conditioning_variables(
            data=data,
            columns_to_encode=columns_to_encode,
            bins=numeric_conditioning_bins,
        )
        return encoded_data, conditioning_codes

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes timeseries values according to dataset-specific groupings.
        """
        return self._normalize_and_scale(data)

    def get_conditioning_var_codes(self):
        """
        Returns the integer mappings for conditioning variables.
        """
        pass
