import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from datasets.utils import encode_conditioning_variables


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_series_column_names: Union[str, List[str]],
        entity_column_name: str,
        timestamp_column_name: str,
        conditioning_vars: Optional[List[str]] = None,
        overrides: Optional[List[str]] = None,
    ):
        """
        Initializes the TimeSeriesDataset.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the time series data and optional conditioning variables.
            time_series_column_names (str or list of str): The name(s) of the column(s) containing the time series data.
            entity_column_name (str): The name of the column containing the unique entity identifier.
            timestamp_column_name (str, optional): The name of the column containing timestamps for sorting. Defaults to "timestamp".
            conditioning_vars (list of str, optional): List of column names to be used as conditioning variables. Defaults to None.
            overrides (list of str, optional): List of configuration overrides (e.g., ['model.seq_len=100']). Defaults to None.
        """
        self.dataframe = dataframe.copy()
        self.time_series_column_names = (
            time_series_column_names
            if isinstance(time_series_column_names, list)
            else [time_series_column_names]
        )
        self.entity_column_name = entity_column_name
        self.timestamp_column_name = timestamp_column_name
        self.conditioning_vars = conditioning_vars if conditioning_vars else []
        self.overrides = overrides if overrides is not None else []

        self.cfg = self._load_config()

        self.encoded_data, self.category_mapping = encode_conditioning_variables(
            self.dataframe
        )

        self.processed_data = self._preprocess_data()

        if self.cfg.dataset.normalize:
            self._normalize_data()

    def _load_config(self) -> DictConfig:
        """
        Loads and merges the default config.yaml and applies any overrides.

        Returns:
            DictConfig: The merged configuration.
        """
        project_root = Path(__file__).resolve().parent.parent
        config_dir = project_root / "config"
        default_config_path = config_dir / "config.yaml"

        if not default_config_path.exists():
            raise FileNotFoundError(
                f"Default config.yaml not found at {default_config_path}"
            )

        default_cfg = OmegaConf.load(default_config_path)

        if self.overrides:
            overrides_cfg = OmegaConf.from_dotlist(self.overrides)
            merged_cfg = OmegaConf.merge(default_cfg, overrides_cfg)
        else:
            merged_cfg = default_cfg

        return merged_cfg

    def _preprocess_data(self) -> pd.DataFrame:
        """
        Sorts the data by timestamp, groups by entity, normalizes time series columns per entity,
        merges time series columns into a single column containing numpy arrays,
        and slices into sequences of seq_len.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with sequences.
        """
        if self.timestamp_column_name not in self.encoded_data.columns:
            raise ValueError(
                f"Timestamp column '{self.timestamp_column_name}' not found in DataFrame."
            )

        self.encoded_data.sort_values(
            by=[self.entity_column_name, self.timestamp_column_name], inplace=True
        )

        self.normalization_stats = {}
        processed_sequences = []

        grouped = self.encoded_data.groupby(self.entity_column_name)

        for entity_id, group in grouped:
            # Reset index for the group
            group = group.reset_index(drop=True)
            group[self.time_series_column_names] = group[
                self.time_series_column_names
            ].astype(float)

            mean = group[self.time_series_column_names].mean()
            std = group[self.time_series_column_names].std()

            # Avoid division by zero
            std[std == 0] = 1.0

            self.normalization_stats[entity_id] = {
                col: {"mean": mean[col], "std": std[col]}
                for col in self.time_series_column_names
            }

            for col in self.time_series_column_names:
                group[col] = (group[col] - mean[col]) / std[col]

            group["timeseries"] = group[self.time_series_column_names].values.tolist()

            static_metadata = {}
            for var in self.conditioning_vars:
                # Assuming conditioning variables are static per entity
                static_metadata[var] = group[var].iloc[0]

            seq_len = self.cfg.dataset.seq_len
            num_sequences = len(group) // seq_len

            # Slice into sequences of length seq_len
            for i in range(num_sequences):
                seq = group.iloc[i * seq_len : (i + 1) * seq_len]

                # Extract timeseries data and ensure it's a numpy array with correct dimensions
                timeseries_data = np.array(
                    seq["timeseries"].tolist()
                )  # Shape: (seq_len, n_timeseries_columns)

                processed_sequences.append(
                    {
                        "entity": entity_id,
                        "timeseries": timeseries_data,
                        "conditioning_vars": static_metadata,
                    }
                )

        processed_data = pd.DataFrame(processed_sequences)
        return processed_data

    def inverse_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverts the normalization on the provided data using stored statistics.

        Args:
            data (pd.DataFrame): The DataFrame with normalized time series data.

        Returns:
            pd.DataFrame: The DataFrame with inverse normalized time series data,
                        with original time series columns restored.
        """
        if not hasattr(self, "normalization_stats"):
            raise AttributeError(
                "Normalization has not been applied or normalization statistics are missing."
            )

        inverse_normalized_rows = []

        for idx, row in data.iterrows():
            entity = row.get("entity")
            if entity not in self.normalization_stats:
                raise ValueError(
                    f"No normalization statistics found for entity '{entity}'."
                )

            means = np.array(
                [
                    self.normalization_stats[entity][col]["mean"]
                    for col in self.time_series_column_names
                ]
            )
            stds = np.array(
                [
                    self.normalization_stats[entity][col]["std"]
                    for col in self.time_series_column_names
                ]
            )

            timeseries_array = np.array(
                row["timeseries"]
            )  # Shape: (seq_len, n_timeseries_columns)

            original_timeseries = (
                timeseries_array * stds + means
            )  # Broadcasting over seq_len
            timeseries_df = pd.DataFrame(
                original_timeseries, columns=self.time_series_column_names
            )

            if "timestamps" in row:
                timeseries_df[self.timestamp_column_name] = row["timestamps"]

            timeseries_df[self.entity_column_name] = entity
            for var, value in row["conditioning_vars"].items():
                timeseries_df[var] = value

            inverse_normalized_rows.append(timeseries_df)

        inverse_normalized_df = pd.concat(inverse_normalized_rows, ignore_index=True)
        return inverse_normalized_df

    def get_conditioning_variables_integer_mapping(
        self,
    ) -> Optional[Dict[str, Dict[int, str]]]:
        """
        Returns a dictionary mapping column names to their integer code mappings.
        Includes predefined mappings for 'weekday' and 'month', and merges with any additional mappings
        present in self.category_mapping.

        Returns:
            Optional[Dict[str, Dict[int, str]]]: A dictionary where each key is a column name and its value is another
                                                 dictionary mapping integer codes to their corresponding string values.
                                                 Returns None if no category_mapping is present.
        """
        return self.category_mapping if hasattr(self, "category_mapping") else None

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.processed_data.iloc[idx]

        timeseries = torch.tensor(sample["timeseries"], dtype=torch.float32)

        conditioning_vars_dict = {}
        for var, code in sample["conditioning_vars"].items():
            conditioning_vars_dict[var] = torch.tensor(code, dtype=torch.long)

        return timeseries, conditioning_vars_dict
