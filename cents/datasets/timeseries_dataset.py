import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from omegaconf import ListConfig, OmegaConf
import pickle

from cents.datasets.utils import encode_context_variables
from cents.models.normalizer import Normalizer
from cents.utils.config_loader import load_yaml, apply_overrides
from cents.utils.utils import _ckpt_name, get_normalizer_training_config, get_context_config

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
        skip_heavy_processing: bool = False,
        size: int = None,
        categorical_time_series: Dict[str, int] = None,
        force_retrain_normalizer: bool = False,
        run_dir: Any = None,
    ):
        # Initialize basic attributes
        # Handle OmegaConf ListConfig objects
        if not isinstance(time_series_column_names, list):
            time_series_column_names = list(time_series_column_names)
        if isinstance(context_var_column_names, ListConfig):
            context_var_column_names = list(context_var_column_names)

        self.time_series_column_names = time_series_column_names
        self.time_series_dims = self.cfg.time_series_dims
        self.context_vars = context_var_column_names or []
        self.seq_len = seq_len

        # Load dataset-level config if not already set
        if not hasattr(self, "cfg"):
            cfg = load_yaml(os.path.join(ROOT_DIR, "config", "dataset", "default.yaml"))
            # dynamic overrides for required fields
            dyn = [
                f"seq_len={seq_len}",
                f"time_series_dims={len(self.time_series_column_names)}",
            ]
            cfg = apply_overrides(cfg, dyn)
            cfg.time_series_columns = self.time_series_column_names
            self.numeric_context_bins = cfg.numeric_context_bins
            self.cfg = cfg

        self.context_var_dict = self.cfg.context_vars
        self.numeric_cols = [k for k, v in self.cfg.context_vars.items() if v[0] == "categorical" and v[1] is None]
        self.numeric_context_bins = self.cfg.numeric_context_bins

        for k in self.numeric_cols:
            self.context_var_dict[k] = ["categorical", self.numeric_context_bins]

        if not hasattr(self, "name"):
            self.name = "custom"

        # Add continuous variables to context_vars if specified
        self.continuous_vars = [k for k, v in self.cfg.context_vars.items() if v[0] == "continuous"]

        self.normalize = normalize
        self.scale = scale
        self.force_retrain_normalizer = force_retrain_normalizer
        self.run_dir = Path(run_dir) if run_dir is not None else None

        # Store categorical time series info
        self.categorical_time_series = categorical_time_series or {}

        # if self.scale:
        #     assert self.normalize, "Normalization must be enabled if scaling is enabled"

        # Preprocess and optionally encode context
        self.data = self._preprocess_data(data)

        if self.continuous_vars:
            self._normalize_continuous_vars()

        if size is not None:
            self.data = self.data.sample(size)
            print(f"Sampled {size} rows from dataset")
        if self.context_vars:
            self.data, self.context_var_codes = self._encode_context_vars(self.data)
        self._save_context_var_codes()


        self.context_cfg = get_context_config()
        self.dynamic_module_type = self.context_cfg.dynamic_context.type 
        self.static_module_type = self.context_cfg.static_context.type
        self.stats_head_type = self.context_cfg.normalizer.stats_head_type

        is_ddp_subprocess = self._is_ddp_subprocess()
        if self.normalize:
            self._init_normalizer()
            cache_path = self._get_normalization_cache_path()
            if cache_path.exists():
                print(f"[{'DDP Subprocess' if is_ddp_subprocess else 'Main Process'}] Loading pre-normalized data from cache", cache_path)
                with open(cache_path, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                # Normalize (only main process or if cache doesn't exist)
                if not is_ddp_subprocess:
                    print("[Main Process] Normalizing data...")
                self.data = self._normalizer.transform(self.data)
                # Save to cache for subprocesses (only main process)
                if not is_ddp_subprocess:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.data, f)
                    print(f"[Main Process] Cached normalized data for subprocesses")
        self.data = self.merge_timeseries_columns(self.data)
        self.data = self.data.reset_index()
        
        # Check if we should skip heavy processing for DDP
        if is_ddp_subprocess and skip_heavy_processing:
            print("skipped rarity computation for DDP compatibility")
            cache_path = self._get_rarity_cache_path()
            if self._load_rarity_cache(cache_path):
                self._rarity_computed = True
        else:
            print("Computing rarity features...")
            self.data = self.get_frequency_based_rarity()
            self.data = self.get_clustering_based_rarity()
            self.data = self.get_combined_rarity()
            rarity_cache_path = self._get_rarity_cache_path()
            self._save_rarity_cache(rarity_cache_path)
            self._rarity_computed = True
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
                    Categorical variables are long tensors, continuous variables are float tensors.
        """
        sample = self.data.iloc[idx]
        timeseries = torch.tensor(sample["timeseries"], dtype=torch.float32)
        
        continuous_vars = getattr(self.cfg, 'continuous_context_vars', None) or []
        context_vars_dict = {}
        for var in self.context_vars:
            if var in continuous_vars:
                # Continuous variables: keep as float
                val = sample[var]
                # Check for NaN/Inf in the data itself
                if isinstance(val, (float, int)) and (not isinstance(val, bool) and (np.isnan(val) or np.isinf(val))):
                    raise ValueError(
                        f"NaN/Inf detected in continuous variable '{var}' in dataset at index {idx}. "
                        f"Value: {val}. This should not happen if normalization was done correctly."
                    )
                context_vars_dict[var] = torch.tensor(val, dtype=torch.float32)
            else:
                # Categorical variables: use long
                context_vars_dict[var] = torch.tensor(sample[var], dtype=torch.long)
        return timeseries, context_vars_dict

    def __getstate__(self):
        """
        Make the dataset picklable for DataLoader worker processes by dropping
        non-picklable attributes (e.g., attached Lightning normalizer module).

        Note: Training workers only need access to the preprocessed `self.data`.
        The normalizer is not used during batching, so it is safe to omit here.
        """
        state = self.__dict__.copy()
        if state.get("_normalizer", None) is not None:
            state["_normalizer"] = None
        return state

    def get_train_dataloader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 8, persistent_workers: bool = True
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for training.

        Args:
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.
            num_workers (int): Number of worker processes.
            persistent_workers (bool): Whether to keep workers alive between epochs.

        Returns:
            DataLoader: Configured data loader.
        """

        self._normalize_continuous_vars()
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=persistent_workers
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
        
        # For categorical time series, ensure they remain as integers
        for col in self.time_series_column_names:
            if col in self.categorical_time_series:
                df[col] = df[col].apply(lambda x: x.astype(np.int32))
        
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

    def apply_pretrained_normalizer(self) -> None:
        """
        Transform self.data with the attached pretrained normalizer (e.g. after
        dataset._normalizer = data_generator.normalizer). Use when normalize=False
        was set to avoid training but you still want real data in normalized space.
        """
        if getattr(self, "_normalizer", None) is None:
            return
        df_split = self.split_timeseries(self.data.copy())
        self.data = self.merge_timeseries_columns(self._normalizer.transform(df_split))
        self.data = self.data.reset_index(drop=True)

    def _encode_context_vars(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode and bin numeric or categorical context variables.
        Continuous variables are kept as-is.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            Tuple of encoded DataFrame and mapping codes.
        """
        continuous_vars = [k for k, v in self.cfg.context_vars.items() if v[0] == "continuous"]
        time_series_cols = [k for k, v in self.cfg.context_vars.items() if v[0] == "time_series"]
        encoded_data, mapping = encode_context_variables(
            data=data,
            columns_to_encode=self.context_vars,
            bins=self.numeric_context_bins,
            numeric_cols=self.numeric_cols,
            continuous_vars=continuous_vars,
            time_series_cols=time_series_cols,
            categorical_time_series=self.categorical_time_series,
        )
        
        return encoded_data, mapping

    def _normalize_continuous_vars(self):
        """
        Normalize continuous context variables in the dataset using z-score normalization.
        This is done once during dataset initialization, so models receive pre-normalized values.
        """
        if not self.continuous_vars:
            return
        
        # Store stats for potential inverse transform if needed
        self.continuous_var_stats = {}
        
        for var_name in self.continuous_vars:
            if var_name in self.data.columns:
                values = self.data[var_name]
                # Compute mean and std
                mean_val = float(values.mean())
                std_val = float(values.std())
                
                # Avoid zero std
                if std_val < 1e-8:
                    std_val = 1.0
                    print(f"[Dataset] Warning: {var_name} has zero std, using std=1.0")
                
                # Store stats for reference and for sampling in original range
                self.continuous_var_stats[var_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': float(values.min()),
                    'max': float(values.max()),
                }
                
                # Normalize the values in-place: (x - mean) / std
                self.data[var_name] = (values - mean_val) / std_val
                
                print(f"[Dataset] Normalized {var_name}: mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                print(f"[Dataset] Warning: Continuous variable {var_name} not found in data columns")

    def _get_context_var_dict(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Infer number of categories for each context variable.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            dict: {var_name: num_categories}
        """
        context_dict = {}
        numeric_cols = getattr(self.cfg, 'numeric_cols', [])
        for var in self.context_vars:
            if var in numeric_cols:
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
        
        for var, info in self.context_var_dict.items():
            if info[0] == "categorical":
                ctx[var] = torch.randint(0, info[1], (), dtype=torch.long)
            elif info[0] == "continuous":
                # Sample from original [min, max] then z-score to match training data
                stats = getattr(self, "continuous_var_stats", {}).get(var)
                if stats is not None and "min" in stats and "max" in stats:
                    x_raw = np.random.uniform(stats["min"], stats["max"])
                    x_norm = (x_raw - stats["mean"]) / stats["std"]
                    ctx[var] = torch.tensor(x_norm, dtype=torch.float32)
                else:
                    raise ValueError(f"Continuous variable {var} has no stats")
            else:
                raise ValueError(f"Invalid context variable type: {info[0]}")
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

    def _ensure_rarity_computed(self):
        """
        Compute rarity features lazily to avoid DDP issues.
        """
        if not self._rarity_computed:
            cache_path = self._get_rarity_cache_path()
            if self._load_rarity_cache(cache_path):
                self._rarity_computed = True
            else:
                self.data = self.get_frequency_based_rarity()
                self.data = self.get_clustering_based_rarity()
                self.data = self.get_combined_rarity()
                self._save_rarity_cache(cache_path)
                self._rarity_computed = True

    def _get_rarity_cache_path(self) -> str:
        """Get cache file path for rarity features."""
        import hashlib
        context_cfg = get_context_config()
        context_module_type = context_cfg.static_context.type
        cache_key = f"{self.name}_{len(self.data)}_{self.seq_len}_{str(sorted(self.context_vars))}_{context_module_type or ''}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        if self.run_dir is not None:
            cache_dir = self.run_dir / "cache" / "rarity"
            cache_dir.mkdir(parents=True, exist_ok=True)
            return str(cache_dir / f"rarity_{cache_hash}.pkl")
        cache_dir = os.path.join(ROOT_DIR, "cache", "rarity")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"rarity_{cache_hash}.pkl")
    
    def _get_normalization_cache_path(self):
        """Get cache file path for normalized data."""
        import hashlib
        from pathlib import Path
        context_cfg = get_context_config()
        context_module_type = context_cfg.dynamic_context.type
        stats_head_type = context_cfg.normalizer.stats_head_type
        cache_key = f"{self.name}_{len(self.data)}_{self.seq_len}_{self.normalize}_{self.scale}_{context_module_type or ''}_{stats_head_type or ''}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        if self.run_dir is not None:
            cache_dir = self.run_dir / "cache" / "normalized_data"
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir / f"normalized_{cache_hash}.pkl"
        cache_dir = Path(ROOT_DIR) / "cache" / "normalized_data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"normalized_{cache_hash}.pkl"

    def _save_rarity_cache(self, cache_path: str) -> None:
        """Save rarity features to cache."""
        import pickle
        rarity_data = {
            'is_frequency_rare': self.data.get('is_frequency_rare'),
            'is_pattern_rare': self.data.get('is_pattern_rare'),
            'is_rare': self.data.get('is_rare'),
            'cluster': self.data.get('cluster')
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(rarity_data, f)
        print(f"Saved rarity cache to {cache_path}")

    def _load_rarity_cache(self, cache_path: str) -> bool:
        """Load rarity features from cache."""
        import pickle
        if not os.path.exists(cache_path):
            return False
        try:
            with open(cache_path, 'rb') as f:
                rarity_data = pickle.load(f)
            for col, data in rarity_data.items():
                if data is not None:
                    self.data[col] = data
            return True
        except Exception as e:
            print(f"Failed to load rarity cache: {e}")
            return False

    def _is_ddp_subprocess(self) -> bool:
        """Detect if we're running in a DDP subprocess."""
        import os
        # Check for DDP-related environment variables
        ddp_indicators = [
            'LOCAL_RANK' in os.environ,
            'WORLD_SIZE' in os.environ,
            'RANK' in os.environ,
            os.environ.get('CUDA_VISIBLE_DEVICES', '').count(',') > 0,  # Multiple GPUs
        ]
        return any(ddp_indicators)

    def _init_normalizer(self) -> None:
        """
        Initialize or load a cached Normalizer for this dataset.

        On first run, trains a new Normalizer and writes a single state dict to cache.
        On subsequent runs, loads that file. If loading fails, deletes the corrupted cache and retrains.
        """
        if self.run_dir is not None:
            normalizer_dir = self.run_dir / "normalizer"
        else:
            normalizer_dir = (
                Path.home() / ".cache" / "cents" / "checkpoints" / self.name / "normalizer"
            )
        normalizer_dir.mkdir(parents=True, exist_ok=True)

        ncfg = get_normalizer_training_config()
        if hasattr(self.cfg, "normalizer_use_global_stats_preprocessing"):
            ncfg = OmegaConf.merge(
                ncfg,
                OmegaConf.create({"use_global_stats_preprocessing": self.cfg.normalizer_use_global_stats_preprocessing}),
            )
        use_global = ncfg.get("use_global_stats_preprocessing", True)

        cache_path = normalizer_dir / _ckpt_name(
            self.name,
            "normalizer",
            self.time_series_dims,
            static_module_type=self.static_module_type,
            stats_head_type=self.stats_head_type,
            dynamic_module_type=self.dynamic_module_type,
            use_global_stats_preprocessing=use_global,
        )

        print(f"[Cents] cache_path: {cache_path}")

        self._normalizer = Normalizer(
            dataset_cfg=self.cfg,
            normalizer_training_cfg=ncfg,
            dataset=self,
            context_cfg=self.context_cfg,
        )

        # attempt to load existing state dict (unless force_retrain_normalizer is True)
        if cache_path.exists() and not self.force_retrain_normalizer:
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
        elif self.force_retrain_normalizer and cache_path.exists():
            print(f"[Cents] Force retrain enabled, ignoring cached normalizer at {cache_path}")

        # train and cache a single state dict
        print("[Cents] Training normalizer…")
        print(f"[Cents] devices: {ncfg.devices}")
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
