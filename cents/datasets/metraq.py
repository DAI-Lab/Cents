import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from cents.utils.config_loader import load_yaml, apply_overrides

from cents.datasets.timeseries_dataset import TimeSeriesDataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MetraqDataset(TimeSeriesDataset):
    """
    Dataset class for Metraq time series data.

    Handles loading and preprocessing including normalization and context variables.
    Data: https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality

    Attributes:
        cfg (DictConfig): Hydra config for the dataset.
        name (str): Dataset name.
        geography (str): Geographic region selector.
        normalize (bool): Whether to apply normalization.
        threshold (Tuple[int, int]): Range filter for grid values.
        include_generation (bool): If True, include solar series.
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        overrides: Optional[List[str]] = None,
        force_retrain_normalizer: bool = False,
        run_dir: Optional[str] = None,
    ):
        """
        Initialize and preprocess the PecanStreet dataset.

        Loads metadata and timeseries CSVs, then applies filtering,
        grouping, user-subsetting, and calls the base class for
        further preprocessing (normalization, merging, rarity flags).

        Args:
            cfg (Optional[DictConfig]): Override Hydra config; if None,
                load from `config/dataset/pecanstreet.yaml`.
            overrides (Optional[List[str]]): Override Hydra config; if None,
                load from `config/dataset/pecanstreet.yaml` and apply overrides.

        Raises:
            FileNotFoundError: If required CSV files are missing.
        """
        if cfg is None:
            cfg = load_yaml(os.path.join(ROOT_DIR, "config", "dataset", "metraq.yaml"))
            if overrides:
                cfg = apply_overrides(cfg, overrides)

        self.cfg = cfg
        self.name = cfg.name
        self.geography = cfg.geography
        self.normalize = cfg.normalize
        self.target_time_series_columns = cfg.time_series_columns

        self.threshold = (-1 * int(cfg.threshold), int(cfg.threshold))
        self.time_series_dims = cfg.time_series_dims

        self._load_data()
        
        ts_cols: List[str] = self.cfg.time_series_columns[: self.time_series_dims]

        self.context_time_series_columns = {k:v[1] for k,v in self.cfg.context_vars.items() if v[0] == "time_series"}
        self.context_series_names = list(self.context_time_series_columns.keys())

        super().__init__(
            data=self.data,
            time_series_column_names=ts_cols,
            context_var_column_names=list(self.cfg.context_vars.keys()),
            seq_len=self.cfg.seq_len,
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            skip_heavy_processing=cfg.get('skip_heavy_processing', False),
            size=cfg.get('max_samples', None),
            force_retrain_normalizer=force_retrain_normalizer,
            run_dir=run_dir,
        )

    def _load_data(self) -> None:
        """
        Populates self.data DataFrames.

        Raises:
            FileNotFoundError: If any required CSV file is missing.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path)) + "/metraq_aq_processed.csv"

        data = pd.read_csv(path)

        if self.geography:
            data = data.loc[data.sensor_name.isin(self.geography)].copy()

        self.data = data

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamps, assemble sequences of length seq_len, and merge metadata.

        Args:
            data (pd.DataFrame): Raw concatenated grid (and solar) rows.

        Returns:
            pd.DataFrame: One row per sequence, with array-valued 'grid' and
                optional 'solar' columns plus context and metadata fields.
        """
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["entry_date"], utc=True)
        data["year"] = data["timestamp"].dt.year
        data["month"] = data["timestamp"].dt.month_name()
        data["weekday"] = data["timestamp"].dt.day_name()
        data["day"] = data["timestamp"].dt.day

        ctx_ts = list(self.context_series_names)
        tgt_ts = list(self.target_time_series_columns)

        if "PM10" in data.columns and "PM2.5" in data.columns:
            pm10 = pd.to_numeric(data["PM10"], errors="coerce")
            pm25 = pd.to_numeric(data["PM2.5"], errors="coerce")
            data["PMcoarse"] = (pm10 - pm25).clip(lower=0.0)

        if "PMcoarse" in data.columns:
            tgt_ts = ["PMcoarse" if c == "PM10" else c for c in tgt_ts]

        # Decompose circular wind direction into Cartesian components so z-score
        # normalization is meaningful. WD=355° and WD=5° are 10° apart but would
        # get opposite signs after z-scoring — wind_u/wind_v avoids this.
        if "WD" in data.columns and "WS" in data.columns:
            wd_deg = pd.to_numeric(data["WD"], errors="coerce")
            ws = pd.to_numeric(data["WS"], errors="coerce").clip(lower=0.0)
            # Binary mask: 1 where WD is measured, 0 where it is missing
            data["wd_valid"] = wd_deg.notna().astype(np.int8)
            wd_deg = wd_deg.fillna(0.0)
            ws = ws.fillna(0.0)
            wd_rad = np.deg2rad(wd_deg)
            data["wind_u"] = ws * np.sin(wd_rad)
            data["wind_v"] = ws * np.cos(wd_rad)
            # Replace WS/WD in ctx_ts with wind_u/wind_v (handles legacy configs that
            # listed WS/WD; current config lists wind_u/wind_v/wd_valid directly).
            ctx_ts = [
                "wind_u" if c == "WS" else "wind_v" if c == "WD" else c
                for c in ctx_ts
            ]

        ts_cols = ctx_ts + tgt_ts

        missing = [c for c in ts_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required time-series columns after preprocessing: {missing}")
    
        data = data.sort_values(["sensor_name", "timestamp"])

        # print(data)

        group_keys = ["sensor_name", "year", "month", "day", "weekday"]

        # Continuous (scalar) context vars are constant per station — carry them through
        # with "first" so they survive the groupby without being collapsed into lists.
        static_continuous_cols = [
            k for k, v in self.cfg.context_vars.items()
            if v[0] == "continuous" and k in data.columns
        ]
        agg_dict = {c: list for c in ts_cols}
        agg_dict.update({c: "first" for c in static_continuous_cols})

        grouped = (
            data.groupby(group_keys, as_index=False, sort=False)
                .agg(agg_dict)
        )

        # print(grouped)

        for c in ts_cols:
            grouped[c] = grouped[c].map(np.asarray)

        len_col = tgt_ts[0] if len(tgt_ts) > 0 else ts_cols[0]
        grouped = grouped[grouped[len_col].apply(len) == self.cfg.seq_len].reset_index(drop=True)

        grouped = self._handle_missing_data(grouped)

        # print("POST CLEAN")
        # print(grouped)

        ctx_numeric = [c for c in ctx_ts if c not in self.categorical_time_series]

        log1p_channels = {"R"}
        binary_channels = {"wd_valid"}  # already in [0, 1] — skip z-scoring
        clip_bound = 5.0
        eps = 1e-8
        # Compute global mean/std per channel over all rows and timesteps
        ctx_stats = {}
        for c in ctx_numeric:
            # stacked shape: (N, L)
            X = np.stack(grouped[c].values).astype(np.float32)

            if c in binary_channels:
                # Already in [0, 1] — pass through without z-scoring
                grouped[c] = list(X)
                continue

            if c in log1p_channels:
                X = np.log1p(np.clip(X, a_min=0.0, a_max=None))

            mu = float(X.mean())
            sd = float(X.std())
            if sd < 1e-6:
                sd = 1.0  # avoid divide-by-zero; effectively makes it "center only"
            ctx_stats[c] = (mu, sd)

            Xn = (X - mu) / (sd + eps)
            Xn = np.clip(Xn, -clip_bound, clip_bound).astype(np.float32)

            grouped[c] = list(Xn)

        # (Optional) store for later inverse-transform / debugging
        self.context_ts_stats_ = ctx_stats

        # arrays -> tuples (hashable)
        for c in ts_cols:
            grouped[c] = grouped[c].map(tuple)

        return grouped

    def _handle_missing_data(self, data):
        numeric_series = [c for c in self.context_series_names if c not in self.categorical_time_series]

        mask = data[numeric_series].applymap(is_all_nan).any(axis=1) if numeric_series else pd.Series([False] * len(data))
        data = data[~mask]

        for col in numeric_series:
            data[col] = data[col].apply(fill_with_row_mean)

        # categorical time series must have no NaNs
        cat_cols = list(self.categorical_time_series.keys())
        if cat_cols:
            mask = data[cat_cols].applymap(is_any_nan).any(axis=1)
            data = data[~mask]

        # ensure no NaNs in target series columns
        for tcol in self.target_time_series_columns:
            # If you replaced PM10->PMcoarse in cfg, this remains correct
            if tcol in data.columns:
                data = data.loc[data[tcol].apply(lambda x: not np.isnan(np.asarray(x, dtype=float)).any())]

        def row_has_low_std(row, cols, thresh=0.01):
            for c in cols:
                arr = np.asarray(row[c], dtype=np.float32)
                if arr.std() < thresh:
                    return True
            return False

        mask = data.apply(
            lambda row: row_has_low_std(row, self.target_time_series_columns, thresh=0.01),
            axis=1
        )

        data = data[~mask]
        return data


def is_all_nan(arr):
    return pd.isna(arr).all()

def is_any_nan(arr):
    return pd.isna(arr).any()

def fill_with_row_mean(lst):
    s = pd.Series(lst, dtype=float)
    m = s.mean(skipna=True)
    return s.fillna(m).tolist()