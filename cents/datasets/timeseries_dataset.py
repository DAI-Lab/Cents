import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

from cents.datasets.utils import encode_context_variables
from cents.models.normalizer import Normalizer
from cents.utils.utils import _ckpt_name, get_normalizer_training_config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for time series data with optional context variables,
    normalization, and rarity-based filtering.

    This class handles:
    - Preprocessing raw DataFrame input.
    - Encoding context variables.
    - Normalizing time series via a trained Normalizer.
    - Merging split time series columns into a single array per sample.
    - Computing rarity flags based on frequency and clustering.

    Args:
        data (pd.DataFrame): Raw input dataframe containing time series and context columns.
        time_series_column_names (Union[str, List[str]]): Column name(s) for time series data.
        seq_len (int): Expected sequence length for each time series.
        context_var_column_names (Optional[Union[str, List[str]]]): Column name(s) for context variables.
        normalize (bool): If True, apply normalizer to data on init.
        scale (bool): If True, scale data in Normalizer.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_series_column_names: Any,
        seq_len: int,
        context_var_column_names: Any = None,
        normalize: bool = True,
        scale: bool = True,
        overrides: Dict[str, Any] = {},
    ):
        # Initialize basic attributes
        self.time_series_column_names = (
            time_series_column_names
            if isinstance(time_series_column_names, list)
            else [time_series_column_names]
        )
        self.time_series_dims = len(self.time_series_column_names)
        self.context_vars = context_var_column_names or []
        self.seq_len = seq_len

        # Load dataset-level config if not already set
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

        if self.scale:
            assert self.normalize, "Normalization must be enabled if scaling is enabled"

        # Preprocess and optionally encode context
        self.data = self._preprocess_data(data)
        if self.context_vars:
            self.data, self.context_var_codes = self._encode_context_vars(self.data)
        self._save_context_var_codes()

        if self.normalize:
            self._init_normalizer()
            self.data = self._normalizer.transform(self.data)

        self.data = self.merge_timeseries_columns(self.data)
        self.data = self.data.reset_index()
        self.data = self.get_frequency_based_rarity()
        self.data = self.get_clustering_based_rarity()
        self.data = self.get_combined_rarity()

    @abstractmethod
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw input DataFrame. Must be implemented
        by subclasses to reshape and validate time series and context columns.

        Args:
            data (pd.DataFrame): Raw input data.

        Returns:
            pd.DataFrame: Cleaned and formatted DataFrame.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the number of samples after preprocessing and filtering.

        Returns:
            int: Dataset length.
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieve a single sample for training.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - timeseries: Tensor of shape (seq_len, dims).
                - context_vars: Dict of context variable tensors.
        """
        sample = self.data.iloc[idx]
        timeseries = torch.tensor(sample["timeseries"], dtype=torch.float32)
        context_vars_dict = {
            var: torch.tensor(sample[var], dtype=torch.long)
            for var in self.context_vars
        }
        return timeseries, context_vars_dict

    def get_train_dataloader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 4
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for training.

        Args:
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.
            num_workers (int): Number of worker processes.

        Returns:
            DataLoader: Configured data loader.
        """
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def split_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand the merged 'timeseries' column into separate per-dimension columns.

        Args:
            df (pd.DataFrame): DataFrame containing a 'timeseries' np.ndarray column.

        Returns:
            pd.DataFrame: DataFrame with each dimension in its own column.

        Raises:
            ValueError: If input format is invalid.
        """
        if "timeseries" not in df.columns:
            raise ValueError("Missing 'timeseries' column.")
        first_ts = df["timeseries"].iloc[0]
        if not isinstance(first_ts, np.ndarray):
            raise ValueError("'timeseries' entries must be numpy arrays.")
        n_dim = first_ts.shape[1]
        if n_dim != len(self.time_series_column_names):
            raise ValueError("Mismatch between column names and data shape.")
        for idx, col_name in enumerate(self.time_series_column_names):
            df[col_name] = df["timeseries"].apply(lambda x: x[:, idx])
        return df.drop(columns=["timeseries"])

    def merge_timeseries_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine separate dimension columns back into a single 'timeseries' column.

        Args:
            df (pd.DataFrame): DataFrame with per-dimension columns.

        Returns:
            pd.DataFrame: DataFrame with merged 'timeseries' column.

        Raises:
            ValueError: If required columns are missing or malformed.
        """
        missing_cols = [c for c in self.time_series_column_names if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        for col in self.time_series_column_names:
            for idx, arr in df[col].items():
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr)
                    df.at[idx, col] = arr
                if arr.ndim == 1:
                    df.at[idx, col] = arr.reshape(-1, 1)
                elif arr.ndim == 2:
                    if arr.shape != (self.seq_len, 1):
                        raise ValueError("Incorrect array shape.")
                else:
                    raise ValueError("Array must have 2 dims.")
        df["timeseries"] = df.apply(
            lambda r: np.hstack([r[c] for c in self.time_series_column_names]), axis=1
        )
        return df.drop(columns=self.time_series_column_names)

    def inverse_transform(
        self, data: pd.DataFrame, merged: bool = True
    ) -> pd.DataFrame:
        """
        Apply inverse normalization and optionally re-merge time series.

        Args:
            data (pd.DataFrame): DataFrame with normalized values.
            merged (bool): If True, returns merged time series column.

        Returns:
            pd.DataFrame: Denormalized DataFrame.
        """
        if not self.normalize:
            return data
        df = self.split_timeseries(data)
        df = self._normalizer.inverse_transform(df)
        return self.merge_timeseries_columns(df) if merged else df

    def _encode_context_vars(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode and bin numeric or categorical context variables.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            Tuple of encoded DataFrame and mapping codes.
        """
        return encode_context_variables(
            data=data,
            columns_to_encode=self.context_vars,
            bins=self.numeric_context_bins,
        )

    def _get_context_var_dict(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Infer number of categories for each context variable.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            dict: {var_name: num_categories}
        """
        context_dict = {}
        for var in self.context_vars:
            if pd.api.types.is_numeric_dtype(data[var]):
                binned = pd.cut(
                    data[var], bins=self.numeric_context_bins, include_lowest=True
                )
                context_dict[var] = binned.nunique()
            else:
                context_dict[var] = data[var].astype("category").nunique()
        return context_dict

    def get_context_var_codes(self) -> Dict[Any, Any]:
        """
        Retrieve the encoding codes for context variables.

        Returns:
            dict: Mapping codes per variable.
        """
        return self.context_var_codes

    def _save_context_var_codes(self) -> None:
        """
        Persist context variable codes to JSON under project data directory.
        """
        dataset_dir = os.path.join(ROOT_DIR, "data", self.name)
        os.makedirs(dataset_dir, exist_ok=True)
        with open(os.path.join(dataset_dir, "context_var_codes.json"), "w") as f:
            json.dump(self.context_var_codes, f, indent=4)

    def sample_random_context_vars(self) -> Dict[str, torch.Tensor]:
        """
        Sample a random context vector for each variable.

        Returns:
            dict: Random context index tensors.
        """
        ctx = {}
        for var, n in self._get_context_var_dict(self.data).items():
            ctx[var] = torch.randint(0, n, (), dtype=torch.long)
        return ctx

    def get_context_var_combination_rarities(
        self, coverage_threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Compute rarity flags based on cumulative coverage threshold.

        Args:
            coverage_threshold (float): Fraction cutoff for "rare".

        Returns:
            pd.DataFrame: Groups with count, coverage, and rarity boolean.
        """
        grouped = self.data.groupby(self.context_vars).size().reset_index(name="count")
        grouped = grouped.sort_values(by="count", ascending=False)
        grouped["coverage"] = grouped["count"].cumsum() / len(self.data)
        grouped["rare"] = grouped["coverage"] > coverage_threshold
        return grouped

    def get_frequency_based_rarity(self) -> pd.DataFrame:
        """
        Label samples as rare if their context frequency is below 10th percentile.

        Returns:
            pd.DataFrame: DataFrame with 'is_frequency_rare' column.
        """
        freq = self.data.groupby(self.context_vars).size().reset_index(name="count")
        threshold = freq["count"].quantile(0.1)
        freq["is_frequency_rare"] = freq["count"] < threshold
        self.data = self.data.merge(
            freq[self.context_vars + ["is_frequency_rare"]],
            on=self.context_vars,
            how="left",
        )
        return self.data

    def get_clustering_based_rarity(self) -> pd.DataFrame:
        """
        Label samples as pattern-rare by KMeans over extracted features.

        Returns:
            pd.DataFrame: DataFrame with 'is_pattern_rare' column.
        """
        ts_data = np.stack(self.data["timeseries"].values)
        n_samples, seq_len, nd = ts_data.shape
        if nd != len(self.time_series_column_names):
            raise ValueError("Dimension mismatch in time series.")
        feats = self.extract_features(ts_data)
        labels = KMeans(n_clusters=10, random_state=42).fit_predict(feats)
        self.data["cluster"] = labels
        sizes = self.data["cluster"].value_counts().to_dict()
        cut = np.percentile(list(sizes.values()), 90)
        self.data["is_pattern_rare"] = self.data["cluster"].map(sizes) < cut
        return self.data

    def extract_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute summary statistics and peak indices per sample.

        Args:
            time_series (np.ndarray): Array shape (n_samples, seq_len, dims).

        Returns:
            np.ndarray: Feature matrix (n_samples, feature_dim).
        """
        features = []
        for ts in time_series:
            mean = ts.mean(axis=0)
            std = ts.std(axis=0)
            max_v = ts.max(axis=0)
            min_v = ts.min(axis=0)
            skew = pd.Series(ts[:, 0]).skew()
            kurt = pd.Series(ts[:, 0]).kurtosis()
            peaks = np.argmax(ts, axis=0)
            vec = np.concatenate([mean, std, max_v, min_v, [skew], [kurt], peaks])
            features.append(vec)
        return np.array(features)

    def get_combined_rarity(self) -> pd.DataFrame:
        """
        Compute joint rarity mask from frequency and pattern rarity.

        Returns:
            pd.DataFrame: DataFrame with 'is_rare' boolean column.
        """
        if "is_frequency_rare" not in self.data:
            self.get_frequency_based_rarity()
        if "is_pattern_rare" not in self.data:
            self.get_clustering_based_rarity()
        self.data["is_rare"] = (
            self.data["is_frequency_rare"] & self.data["is_pattern_rare"]
        )
        return self.data

    def _init_normalizer(self) -> None:
        """
        Initialize or load a cached Normalizer for this dataset.

        On first run, trains a new Normalizer and writes a single state dict to cache.
        On subsequent runs, loads that file. If loading fails, deletes the corrupted cache and retrains.
        """
        normalizer_dir = (
            Path.home() / ".cache" / "cents" / "checkpoints" / self.name / "normalizer"
        )
        normalizer_dir.mkdir(parents=True, exist_ok=True)
        cache_path = normalizer_dir / _ckpt_name(
            self.name, "normalizer", self.time_series_dims
        )

        ncfg = get_normalizer_training_config()
        self._normalizer = Normalizer(
            dataset_cfg=self.cfg,
            normalizer_training_cfg=ncfg,
            dataset=self,
        )

        # attempt to load existing state dict
        if cache_path.exists():
            try:
                state = torch.load(cache_path, map_location="cpu")
                sd = state.get("state_dict", state)
                self._normalizer.load_state_dict(sd)
                self._normalizer.eval()
                print(f"[Cents] Loaded normalizer from {cache_path}")
                return
            except Exception:
                try:
                    cache_path.unlink()
                except OSError:
                    pass

        # train and cache a single state dict
        print("[Cents] Training normalizer…")
        trainer = pl.Trainer(
            max_epochs=ncfg.n_epochs,
            accelerator=ncfg.accelerator,
            devices=ncfg.devices,
            strategy=ncfg.strategy,
            log_every_n_steps=ncfg.log_every_n_steps,
            logger=False,
        )
        trainer.fit(self._normalizer)
        torch.save(self._normalizer.state_dict(), cache_path)
        print(f"[Cents] Saved normalizer to {cache_path}")
