import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from cents.utils.config_loader import load_yaml, apply_overrides

from cents.datasets.timeseries_dataset import TimeSeriesDataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class WalmartDataset(TimeSeriesDataset):
    """
    Dataset class for Walmart M5 daily unit sales time series.

    Data: https://www.kaggle.com/competitions/m5-forecasting-accuracy

    Each sample is a 30-day non-overlapping window of daily unit sales for a
    single item-store pair, filtered to high-velocity items (top 20% by mean
    daily sales).  Static context encodes category, department, store, and
    state.  Dynamic context includes sell price, SNAP eligibility, and a
    binary calendar-event indicator.
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        overrides: Optional[List[str]] = None,
        force_retrain_normalizer: bool = False,
        run_dir: Optional[str] = None,
    ):
        if cfg is None:
            cfg = load_yaml(os.path.join(ROOT_DIR, "config", "dataset", "walmart.yaml"))
            if overrides:
                cfg = apply_overrides(cfg, overrides)

        self.cfg = cfg
        self.name = cfg.name
        self.normalize = cfg.normalize
        self.target_time_series_columns = list(cfg.time_series_columns)
        self.time_series_dims = cfg.time_series_dims
        self.geography = cfg.get("geography", None)

        self.context_time_series_columns = {
            k: v[1] for k, v in cfg.context_vars.items() if v[0] == "time_series"
        }
        self.context_series_names = list(self.context_time_series_columns.keys())
        # No categorical time series for this dataset (all dynamic vars are continuous/binary)
        self.categorical_time_series = {
            k: v[1] for k, v in cfg.context_vars.items()
            if v[0] == "time_series" and v[1] is not None
        }

        self._load_data()

        ts_cols: List[str] = self.target_time_series_columns[: self.time_series_dims]

        super().__init__(
            data=self.data,
            time_series_column_names=ts_cols,
            context_var_column_names=list(cfg.context_vars.keys()),
            seq_len=cfg.seq_len,
            normalize=cfg.normalize,
            scale=cfg.scale,
            skip_heavy_processing=cfg.get("skip_heavy_processing", False),
            size=cfg.get("max_samples", None),
            categorical_time_series=self.categorical_time_series,
            force_retrain_normalizer=force_retrain_normalizer,
            run_dir=run_dir,
        )

    def _load_data(self) -> None:
        """
        Loads and joins sales, calendar, and price CSVs; filters to
        high-velocity item-store pairs; constructs per-state SNAP flags
        and binary event indicators.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))

        # --- Sales: wide → long ---
        sales_path = os.path.join(path, "sales_train_evaluation.csv")
        if not os.path.exists(sales_path):
            raise FileNotFoundError(f"Sales file not found: {sales_path}")

        sales = pd.read_csv(sales_path)
        meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        day_cols = [c for c in sales.columns if c.startswith("d_")]
        sales_long = sales[meta_cols + day_cols].melt(
            id_vars=meta_cols, var_name="d", value_name="sales"
        )
        del sales  # free memory before merges

        # --- Calendar ---
        calendar = pd.read_csv(
            os.path.join(path, "calendar.csv"),
            usecols=["d", "date", "wm_yr_wk", "weekday", "month", "year",
                     "event_name_1", "snap_CA", "snap_TX", "snap_WI"],
        )
        calendar["date"] = pd.to_datetime(calendar["date"])
        sales_long = sales_long.merge(calendar, on="d", how="left")
        del calendar

        # --- Sell prices (weekly → broadcast to daily via merge, then ffill) ---
        prices = pd.read_csv(os.path.join(path, "sell_prices.csv"))
        sales_long = sales_long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        del prices

        sales_long = sales_long.sort_values(["id", "date"])
        sales_long["sell_price"] = (
            sales_long.groupby("id")["sell_price"]
            .transform(lambda x: x.ffill().bfill())
        )

        # --- State-specific SNAP eligibility (vectorised) ---
        sales_long["snap"] = np.where(
            sales_long["state_id"] == "CA", sales_long["snap_CA"],
            np.where(
                sales_long["state_id"] == "TX", sales_long["snap_TX"],
                sales_long["snap_WI"],
            ),
        ).astype(np.int8)

        # --- Binary calendar-event indicator ---
        sales_long["event_binary"] = sales_long["event_name_1"].notna().astype(np.int8)

        # --- Filter to high-velocity item-store pairs (top 20% by mean daily sales) ---
        mean_sales = sales_long.groupby("id")["sales"].mean()
        threshold = mean_sales.quantile(0.80)
        high_vel_ids = mean_sales.index[mean_sales >= threshold]
        sales_long = sales_long[sales_long["id"].isin(high_vel_ids)].copy()

        # --- Month name (consistent with other datasets) ---
        sales_long["month"] = sales_long["month"].map(lambda x: _MONTHS[int(x) - 1])

        # Drop columns that are no longer needed after engineering
        sales_long.drop(
            columns=["snap_CA", "snap_TX", "snap_WI", "event_name_1",
                     "wm_yr_wk", "d", "item_id"],
            errors="ignore",
            inplace=True,
        )

        self.data = sales_long.reset_index(drop=True)

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns non-overlapping 30-day window IDs within each item-store
        series, groups into windows, and z-score normalises continuous
        dynamic context channels.
        """
        data = data.copy()

        ts_ctx = list(self.context_series_names)   # ["sell_price", "snap", "event_binary"]
        tgt_ts = list(self.target_time_series_columns)  # ["sales"]
        all_ts = ts_ctx + tgt_ts

        # Assign window indices: non-overlapping blocks of seq_len days per series
        data = data.sort_values(["id", "date"]).reset_index(drop=True)
        data["_row"] = data.groupby("id").cumcount()
        data["_window"] = data["_row"] // self.cfg.seq_len

        group_keys = ["id", "_window"]
        static_cols = ["cat_id", "dept_id", "store_id", "state_id", "year", "month", "weekday"]

        agg_dict = {c: list for c in all_ts}
        agg_dict.update({c: "first" for c in static_cols})

        grouped = (
            data.groupby(group_keys, as_index=False, sort=False)
                .agg(agg_dict)
        )
        grouped.drop(columns=["_window"], inplace=True, errors="ignore")

        for c in all_ts:
            grouped[c] = grouped[c].map(np.asarray)

        # Keep only complete windows
        len_col = tgt_ts[0]
        grouped = grouped[grouped[len_col].apply(len) == self.cfg.seq_len].reset_index(drop=True)

        grouped = self._handle_missing_data(grouped)

        # Z-score normalise continuous dynamic context; pass binaries through
        binary_channels = {"snap", "event_binary"}
        clip_bound = 5.0
        eps = 1e-8
        ctx_stats = {}

        for c in ts_ctx:
            X = np.stack(grouped[c].values).astype(np.float32)

            if c in binary_channels:
                grouped[c] = list(X)
                continue

            mu = float(X.mean())
            sd = float(X.std())
            if sd < 1e-6:
                sd = 1.0
            ctx_stats[c] = (mu, sd)

            Xn = np.clip((X - mu) / (sd + eps), -clip_bound, clip_bound).astype(np.float32)
            grouped[c] = list(Xn)

        self.context_ts_stats_ = ctx_stats

        # Convert arrays → tuples (hashable, required by base class)
        for c in all_ts:
            grouped[c] = grouped[c].map(tuple)

        return grouped

    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        numeric_series = [
            c for c in self.context_series_names if c not in self.categorical_time_series
        ]

        # Drop windows where any numeric context series is entirely NaN
        if numeric_series:
            mask = data[numeric_series].applymap(is_all_nan).any(axis=1)
            data = data[~mask]

        # Fill isolated NaNs in numeric context series with within-window mean
        for col in numeric_series:
            data[col] = data[col].apply(fill_with_row_mean)

        # Drop windows with NaN in any categorical time series (none expected here)
        cat_cols = list(self.categorical_time_series.keys())
        if cat_cols:
            mask = data[cat_cols].applymap(is_any_nan).any(axis=1)
            data = data[~mask]

        # Drop windows with any NaN in target
        for tcol in self.target_time_series_columns:
            if tcol in data.columns:
                data = data.loc[
                    data[tcol].apply(
                        lambda x: not np.isnan(np.asarray(x, dtype=float)).any()
                    )
                ]

        # Drop near-constant windows (std < 0.01 → degenerate for diffusion)
        def _low_std(row, cols, thresh=0.01):
            return any(np.asarray(row[c], dtype=np.float32).std() < thresh for c in cols)

        mask = data.apply(lambda row: _low_std(row, self.target_time_series_columns), axis=1)
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
