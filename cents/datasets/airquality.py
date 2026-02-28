from ast import Str
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from cents.utils.config_loader import load_yaml, apply_overrides

from cents.datasets.timeseries_dataset import TimeSeriesDataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
# These are warnings for an error that is accounted for in the code
warnings.filterwarnings("ignore", category=RuntimeWarning) 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class AirQualityDataset(TimeSeriesDataset):
    def __init__(self, cfg: DictConfig = None,
                 overrides: Optional[List[str]] = None,
                 force_retrain_normalizer: bool = False,
                 run_dir: Optional[str] = None):
        """
        Initializes the AirQuality Dataset. Available at:
        https://doi.org/10.24432/C5RK5G.
        Hourly Air Quality at Multiple Sites in China, in many measures.
        Accompanying location and weather information.
        """

        if cfg is None:
            cfg = load_yaml(os.path.join(ROOT_DIR, "config", "dataset", "airquality.yaml"))
        
        if overrides:
            cfg = apply_overrides(cfg, overrides)
        
        self.cfg = cfg
        self.name = cfg.name
        self.normalize = cfg.normalize
        if isinstance(cfg.time_series_columns, str):
            cfg.time_series_columns = [cfg.time_series_columns]
        self.target_time_series_columns = cfg.time_series_columns
        self.geography = cfg.geography
        self.time_series_dims = cfg.time_series_dims

        self.city_names = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
                            "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]
        self.context_time_series_columns = {k:v[1] for k,v in self.cfg.context_vars.items() if v[0] == "time_series"}
        self.context_series_names = list(self.context_time_series_columns.keys())
        
        self.categorical_time_series = {
            k: v[1] for k, v in self.cfg.context_vars.items() 
            if v[0] == "time_series" and v[1] is not None
        }
        self._load_data()

        super().__init__(
            data=self.data,
            time_series_column_names=self.target_time_series_columns,
            context_var_column_names=list(self.cfg.context_vars.keys()),
            seq_len=self.cfg.seq_len,
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            skip_heavy_processing=cfg.get('skip_heavy_processing', False),
            size=cfg.get('max_samples', None),
            categorical_time_series=self.categorical_time_series,
            force_retrain_normalizer=force_retrain_normalizer,
            run_dir=run_dir,
        )

    def _load_data(self):
        """
        Loads in metadata and data for commercial energy dataset.
        Categorial CVs: Year, Month, Day, Location
        Context Time Series: Temperature, Pressure, Dewpoint, Precipitation, Wind
        """

        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))

        if not self.geography: self.geography = self.city_names

        self.geography = [self.geography] if isinstance(self.geography, str) else self.geography
        dfs = []
        for name in self.geography:
            fname = f"PRSA_Data_{name}_20130301-20170228.csv"
            data_path = os.path.join(path, fname)

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            
            dfs.append(pd.read_csv(data_path)[self.cfg.data_columns])
        
        self.data = pd.concat(dfs, axis=0)

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        # Timestamp + weekday
        data["timestamp"] = pd.to_datetime(data[["year", "month", "day", "hour"]])
        data["weekday"] = data["timestamp"].dt.day_name()

        # -------------------------
        # Engineer wind_u / wind_v
        # -------------------------
        wd_deg_map = {
            "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
            "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
            "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
            "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
        }

        has_wd = "wd" in data.columns
        has_wspm = "WSPM" in data.columns
        if has_wd and has_wspm:
            wd_clean = data["wd"].astype(str).str.strip().str.upper()
            wd_deg = wd_clean.map(wd_deg_map)

            # indicator can help if any weird labels slip in
            data["wd_valid"] = wd_deg.notna().astype(np.int8)

            theta = np.deg2rad(wd_deg.fillna(0.0).to_numpy(dtype=float))
            wspm = pd.to_numeric(data["WSPM"], errors="coerce").fillna(0.0)

            # u = speed * cos(theta), v = speed * sin(theta)
            # (note: choice of axes is arbitrary here; consistency matters more than convention)
            data["wind_u"] = wspm * np.cos(theta)
            data["wind_v"] = wspm * np.sin(theta)

            # Drop raw wind columns after engineering
            data.drop(columns=["wd", "WSPM"], inplace=True)
        else:
            # If one is missing, don't silently create nonsense
            # You can choose to raise instead if this should never happen.
            if "wd" in data.columns:
                data.drop(columns=["wd"], inplace=True)
            if "WSPM" in data.columns:
                data.drop(columns=["WSPM"], inplace=True)

        # -------------------------
        # Engineer PMcoarse
        # -------------------------
        if "PM10" in data.columns and "PM2.5" in data.columns:
            pm10 = pd.to_numeric(data["PM10"], errors="coerce")
            pm25 = pd.to_numeric(data["PM2.5"], errors="coerce")
            data["PMcoarse"] = (pm10 - pm25).clip(lower=0.0)

        # -------------------------
        # Choose time-series columns
        # -------------------------
        # Context TS columns come from cfg; targets come from cfg.time_series_columns
        ctx_ts = list(self.context_series_names)
        tgt_ts = list(self.target_time_series_columns)

        # Replace context wind variables: remove wd/WSPM if present, add wind_u/wind_v (+wd_valid if you want)
        ctx_ts = [c for c in ctx_ts if c not in ("wd", "WSPM")]
        for c in ("wind_u", "wind_v"):
            if c in data.columns and c not in ctx_ts:
                ctx_ts.append(c)
        # optional
        if "wd_valid" in data.columns and "wd_valid" not in ctx_ts:
            ctx_ts.append("wd_valid")

        # Replace PM10 target with PMcoarse if PM10 is in targets
        if "PMcoarse" in data.columns:
            tgt_ts = ["PMcoarse" if c == "PM10" else c for c in tgt_ts]
        # (optional) if you *also* want to drop PM10 if it still exists
        tgt_ts = [c for c in tgt_ts if c != "PM10"]



        ts_cols = ctx_ts + tgt_ts

        # Ensure all ts_cols exist
        missing = [c for c in ts_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required time-series columns after preprocessing: {missing}")

        # Sort
        data = data.sort_values(["station", "year", "month", "day", "hour"])

        # Month name mapping (keeps your categorical month encoding behavior)
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        data["month"] = data["month"].map(lambda x: months[x - 1])

        group_keys = ["station", "year", "month", "day", "weekday"]

        grouped = (
            data.groupby(group_keys, as_index=False, sort=False)
                .agg({c: list for c in ts_cols})
        )

        # lists -> numpy arrays
        for c in ts_cols:
            grouped[c] = grouped[c].map(np.asarray)
        

        # Keep only full-length sequences
        # Use the first target if possible, else fall back to first ts col
        len_col = tgt_ts[0] if len(tgt_ts) > 0 else ts_cols[0]
        grouped = grouped[grouped[len_col].apply(len) == self.cfg.seq_len].reset_index(drop=True)

        grouped = self._handle_missing_data(grouped)

        ctx_numeric = [c for c in ctx_ts if c not in self.categorical_time_series]
        # Optional: handle heavy-tailed / zero-inflated channels
        log1p_channels = {"RAIN"}  # add more if needed

        clip_bound = 5.0
        eps = 1e-8
        # Compute global mean/std per channel over all rows and timesteps
        ctx_stats = {}
        for c in ctx_numeric:
            # stacked shape: (N, L)
            X = np.stack(grouped[c].values).astype(np.float32)

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
