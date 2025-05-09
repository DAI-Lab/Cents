import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from cents.datasets.timeseries_dataset import TimeSeriesDataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PecanStreetDataset(TimeSeriesDataset):
    """
    Dataset class for PecanStreet time series data.

    Handles loading, preprocessing, and user-specific subsetting of grid and
    optional solar time series, including normalization and context variables.

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
            with initialize_config_dir(
                config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
            ):
                cfg = compose(config_name="pecanstreet", overrides=overrides)

        self.cfg = cfg
        self.name = cfg.name
        self.geography = cfg.geography
        self.normalize = cfg.normalize
        self.threshold = (-1 * int(cfg.threshold), int(cfg.threshold))
        self.time_series_dims = cfg.time_series_dims

        self.cfg.time_series_columns = ["grid", "solar"]

        self.include_generation = self.time_series_dims > 1

        if self.time_series_dims > 1 and self.cfg.user_group in {"non_pv_users", "all"}:
            raise AssertionError(
                "time_series_dims > 1 requires solar readings; "
                "override `dataset.user_group` to `pv_users` or set time_series_dims = 1."
            )

        self._load_data()
        self._set_user_flags()

        ts_cols: List[str] = self.cfg.time_series_columns[: self.time_series_dims]

        super().__init__(
            data=self.data,
            time_series_column_names=ts_cols,
            context_var_column_names=list(self.cfg.context_vars.keys()),
            seq_len=self.cfg.seq_len,
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
        )

    def _load_data(self) -> None:
        """
        Load metadata and 15-minute grid (and solar) CSV files.

        Populates self.metadata and self.data DataFrames.

        Raises:
            FileNotFoundError: If any required CSV file is missing.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))

        meta_path = os.path.join(path, "metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        self.metadata = pd.read_csv(meta_path, usecols=self.cfg.metadata_columns)
        if "solar" in self.metadata.columns:
            self.metadata.rename(columns={"solar": "has_solar"}, inplace=True)

        if self.geography:
            fname = f"15minute_data_{self.geography}.csv"
            data_path = os.path.join(path, fname)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            self.data = pd.read_csv(data_path)[self.cfg.data_columns]
        else:
            files = ["newyork", "california", "austin"]
            dfs = []
            for region in files:
                fp = os.path.join(path, f"15minute_data_{region}.csv")
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"Data file not found at {fp}")
                dfs.append(pd.read_csv(fp))
            self.data = pd.concat(dfs, axis=0)[self.cfg.data_columns]

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
        data["local_15min"] = pd.to_datetime(data["local_15min"], utc=True)
        data["month"] = data["local_15min"].dt.month_name()
        data["weekday"] = data["local_15min"].dt.day_name()
        data["date_day"] = data["local_15min"].dt.day
        data = data.sort_values(by=["local_15min"]).dropna(subset=["grid"]).copy()

        grouped = (
            data.groupby(["dataid", "month", "date_day", "weekday"])["grid"]
            .apply(np.array)
            .reset_index()
        )
        grouped = grouped[grouped["grid"].apply(len) == self.cfg.seq_len].reset_index(
            drop=True
        )

        if self.include_generation:
            solar_df = self._preprocess_solar(data)
            grouped = pd.merge(
                grouped,
                solar_df,
                on=["dataid", "month", "date_day", "weekday"],
                how="left",
            )

        merged = pd.merge(grouped, self.metadata, on="dataid", how="left")
        merged = self._get_user_group_data(merged)
        merged = self._handle_missing_data(merged)
        return merged.sort_values(by=["month", "weekday", "date_day"]).reset_index(
            drop=True
        )

    def _preprocess_solar(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and filter solar sequences of length seq_len.

        Args:
            data (pd.DataFrame): Raw DataFrame including 'solar' values.

        Returns:
            pd.DataFrame: Same grouping keys plus array-valued 'solar'.
        """
        sd = (
            data.dropna(subset=["solar"])
            .groupby(["dataid", "month", "date_day", "weekday"])["solar"]
            .apply(np.array)
            .reset_index()
        )
        return sd[sd["solar"].apply(len) == self.cfg.seq_len].reset_index(drop=True)

    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaNs in categorical and numeric metadata columns.

        Args:
            data (pd.DataFrame): Merged sequence+metadata rows.

        Returns:
            pd.DataFrame: Fully imputed DataFrame.
        """
        df = data.copy()
        df["car1"] = df["car1"].fillna("no")
        df["has_solar"] = df["has_solar"].fillna("no")
        df["house_construction_year"] = df["house_construction_year"].fillna(
            df["house_construction_year"].mean()
        )
        df["total_square_footage"] = df["total_square_footage"].fillna(
            df["total_square_footage"].mean()
        )
        assert df.isna().sum().sum() == 0, "Missing data remaining!"
        return df

    def _set_user_flags(self) -> Dict[int, bool]:
        """
        Build a mapping from user_id to whether they have solar data.

        Returns:
            Dict[int, bool]: True if user ever has has_solar.
        """
        flags: Dict[int, bool] = {}
        for uid in self.data["dataid"].unique():
            flags[uid] = (
                self.metadata[self.metadata["dataid"] == uid]["has_solar"].notna().any()
            )
        self.user_flags = flags
        return flags

    def _get_user_group_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Subset rows by cfg.user_group ('pv_users','non_pv_users','all').

        Args:
            data (pd.DataFrame): Preprocessed sequence+metadata.

        Returns:
            pd.DataFrame: Filtered by user criteria.

        Raises:
            AssertionError: If config contradicts include_generation.
            ValueError: On unrecognized group.
        """
        grp = self.cfg.user_group
        if grp == "pv_users":
            users = [u for u, pv in self.user_flags.items() if pv]
            return data[data["dataid"].isin(users)].copy()
        if grp == "non_pv_users":
            assert (
                not self.include_generation
            ), "time_series_dims > 1 conflicts with non_pv_users"
            users = [u for u, pv in self.user_flags.items() if not pv]
            return data[data["dataid"].isin(users)].copy()
        if grp == "all":
            assert (
                not self.include_generation
            ), "time_series_dims > 1 conflicts with user_group='all'"
            return data.copy()
        raise ValueError(f"User group '{grp}' not recognized.")

    def compute_average_pv_shift(
        self,
        group_vars: Optional[List[str]] = None,
        pv_col: str = "has_solar",
        grid_col: str = "grid",
    ) -> np.ndarray:
        """
        Compute mean difference (pv=1 minus pv=0) per timestep across all contexts.

        Args:
            group_vars (Optional[List[str]]): Context variables (excluding pv_col). Defaults to all.
            pv_col (str): Column for PV state (0/1).
            grid_col (str): Column for grid series.

        Returns:
            np.ndarray: Shape (seq_len,), average shift or zeros if none.
        """
        if group_vars is None:
            group_vars = [v for v in self.context_vars if v != pv_col]
        shifts: List[np.ndarray] = []
        for _, sub in self.data.groupby(group_vars):
            states = sub[pv_col].unique()
            if set(states) >= {0, 1}:
                ts0 = np.stack(sub[sub[pv_col] == 0]["timeseries"].to_numpy())
                ts1 = np.stack(sub[sub[pv_col] == 1]["timeseries"].to_numpy())
                diff = ts1.mean(axis=0) - ts0.mean(axis=0)
                if diff.ndim == 2 and diff.shape[1] == 1:
                    diff = diff[:, 0]
                shifts.append(diff)
        if not shifts:
            return np.zeros(self.cfg.seq_len, dtype=float)
        return np.mean(np.stack(shifts, 0), axis=0)

    def sample_shift_test_contexts(
        self,
        group_vars: Optional[List[str]] = None,
        pv_col: str = "has_solar",
    ) -> List[Dict[str, Any]]:
        """
        Identify contexts present with only pv=0 or only pv=1 for test sampling.

        Args:
            group_vars (Optional[List[str]]): Context vars (excluding pv_col).
            pv_col (str): Column for PV state.

        Returns:
            List[Dict[str, Any]]: Each with keys 'base_context','present_pv','missing_pv'.
        """
        if group_vars is None:
            group_vars = [v for v in self.context_vars if v != pv_col]
        tests: List[Dict[str, Any]] = []
        for vals, sub in self.data.groupby(group_vars):
            uniq = sub[pv_col].unique()
            if len(uniq) == 1:
                present = int(uniq[0])
                missing = 1 - present
                base = {
                    gv: (vals if len(group_vars) == 1 else vals[i])
                    for i, gv in enumerate(group_vars)
                }
                base[pv_col] = present
                tests.append(
                    {
                        "base_context": base,
                        "present_pv": present,
                        "missing_pv": missing,
                    }
                )
        return tests
