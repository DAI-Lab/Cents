import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from endata.datasets.timeseries_dataset import TimeSeriesDataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PecanStreetDataset(TimeSeriesDataset):
    """
    A dataset class for handling and preprocessing PecanStreet time series data,
    including normalization, handling PV data, and user-specific data retrieval.

    Attributes:
        cfg (DictConfig): The hydra config file
    """

    def __init__(self, cfg: DictConfig = None):
        if not cfg:
            with initialize_config_dir(
                config_dir=os.path.join(ROOT_DIR, "config/dataset"), version_base=None
            ):
                cfg = compose(config_name="pecanstreet", overrides=None)

        self.cfg = cfg
        self.name = cfg.name
        self.geography = cfg.geography
        self.normalize = cfg.normalize
        self.threshold = (-1 * int(cfg.threshold), int(cfg.threshold))
        self.include_generation = cfg.include_generation
        self._load_data()
        self._set_user_flags()

        time_series_column_names = ["grid"]

        if self.include_generation:
            time_series_column_names.append("solar")

        context_vars = list(self.cfg.context_vars.keys())
        normalization_group_keys = []

        super().__init__(
            data=self.data,
            entity_column_name="dataid",
            time_series_column_names=time_series_column_names,
            context_var_column_names=context_vars,
            seq_len=self.cfg.seq_len,
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            normalization_group_keys=normalization_group_keys,
        )

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the csv files into a pandas dataframe object.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))
        metadata_csv_path = os.path.join(path, "metadata.csv")

        if not os.path.exists(metadata_csv_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_csv_path}")

        self.metadata = pd.read_csv(
            metadata_csv_path, usecols=self.cfg.metadata_columns
        )

        if "solar" in self.metadata.columns:  # naming conflicts
            self.metadata.rename(columns={"solar": "has_solar"}, inplace=True)

        if self.geography:
            data_file_name = f"15minute_data_{self.geography}.csv"
            data_file_path = os.path.join(path, data_file_name)
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found at {data_file_path}")
            self.data = pd.read_csv(data_file_path)[self.cfg.data_columns]
        else:
            data_files = [
                os.path.join(path, "15minute_data_newyork.csv"),
                os.path.join(path, "15minute_data_california.csv"),
                os.path.join(path, "15minute_data_austin.csv"),
            ]
            for data_file in data_files:
                if not os.path.exists(data_file):
                    raise FileNotFoundError(f"Data file not found at {data_file}")
            self.data = pd.concat(
                [pd.read_csv(data_file) for data_file in data_files],
                axis=0,
            )[self.cfg.data_columns]

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the dataset by adding date-related columns, sorting, filtering, and normalizing.

        Args:
            data (pd.DataFrame): Raw data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        data["local_15min"] = pd.to_datetime(data["local_15min"], utc=True)
        data["month"] = data["local_15min"].dt.month_name()
        data["weekday"] = data["local_15min"].dt.day_name()
        data["date_day"] = data["local_15min"].dt.day
        data = data.sort_values(by=["local_15min"]).copy()
        data = data[~data["grid"].isna()]

        grouped_data = (
            data.groupby(["dataid", "month", "date_day", "weekday"])["grid"]
            .apply(np.array)
            .reset_index()
        )
        filtered_data = grouped_data[
            grouped_data["grid"].apply(len) == self.cfg.seq_len
        ].reset_index(drop=True)

        if self.include_generation:
            solar_data = self._preprocess_solar(data)
            filtered_data = pd.merge(
                filtered_data,
                solar_data,
                how="left",
                on=["dataid", "month", "weekday", "date_day"],
            )
        data = pd.merge(filtered_data, self.metadata, on="dataid", how="left")
        data = self._get_user_group_data(data)
        data = self._handle_missing_data(data)
        grouped_data.sort_values(by=["month", "weekday", "date_day"], inplace=True)
        return data

    def _preprocess_solar(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses solar data by filtering and normalizing if required.

        Args:
            data (pd.DataFrame): The raw data containing solar information.

        Returns:
            pd.DataFrame: Preprocessed solar data.
        """
        solar_data = (
            data[~data["solar"].isna()]
            .groupby(["dataid", "month", "date_day", "weekday"])["solar"]
            .apply(np.array)
            .reset_index()
        )
        solar_data = solar_data[solar_data["solar"].apply(len) == self.cfg.seq_len]
        return solar_data

    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data["car1"] = data["car1"].fillna("no")
        data["has_solar"] = data["has_solar"].fillna("no")
        data["house_construction_year"] = data["house_construction_year"].fillna(
            data["house_construction_year"].mean(skipna=True)
        )
        data["total_square_footage"] = data["total_square_footage"].fillna(
            data["total_square_footage"].mean(skipna=True)
        )
        assert data.isna().sum().sum() == 0, "Missing data remaining!"
        return data

    def _set_user_flags(self) -> Dict[int, bool]:
        """
        Sets user flags indicating whether a user has solar generation data.
        """
        self.user_flags = {
            user_id: self.metadata.loc[self.metadata["dataid"] == user_id]["has_solar"]
            .notna()
            .any()
            for user_id in self.data["dataid"].unique()
        }

    def _get_user_group_data(self, data: pd.DataFrame) -> "PecanStreetDataset":
        if self.cfg.user_id:
            return data[data["dataid"] == self.cfg.user_id].copy()

        if self.cfg.user_group == "pv_users":
            users = [user for user, has_pv in self.user_flags.items() if has_pv]
            return data[data["dataid"].isin(users)].copy()

        elif self.cfg.user_group == "non_pv_users":
            assert (
                self.include_generation == False
            ), "Include_generation must be set to False when working with the non pv user dataset!"
            users = [user for user, has_pv in self.user_flags.items() if not has_pv]
            return data[data["dataid"].isin(users)].copy()

        elif self.cfg.user_group == "all":
            assert (
                self.include_generation == False
            ), "Include_generation must be set to False when working with the entire dataset!"
            return data.copy()

        else:
            raise ValueError(f"User group {self.cfg.user_group} is not specified.")

    def compute_average_pv_shift(
        self,
        group_vars: list = None,
        pv_col: str = "has_solar",
        grid_col: str = "grid",
    ) -> np.ndarray:
        """
        Computes an 'average' shift from pv=0 to pv=1 per timestep for houses/contexts
        that actually contain both pv=0 and pv=1 examples.

        Args:
            group_vars (list): The context variables we want to group by (excluding `pv`).
            pv_col (str): The name of the column indicating PV state (0 or 1).
            grid_col (str): The name of the column containing the main timeseries load.

        Returns:
            avg_shift (np.ndarray): shape (seq_len,), the average timeseries difference
                across all groups that have both pv=0 and pv=1.
        """
        if group_vars is None:
            group_vars = [v for v in self.context_vars if v != pv_col]

        df = self.data.copy()
        # We'll group by group_vars. Then within each group, we see if we have pv=0 and pv=1.
        grouped = df.groupby(group_vars)

        # We accumulate the difference (pv=1 - pv=0) for each group that has both states
        # Then average over all those differences
        shift_accumulator = []
        for group_vals, subdf in grouped:
            # subdf might contain rows with different pv states
            unique_pv_states = subdf[pv_col].unique()
            if len(set(unique_pv_states).intersection({0, 1})) == 2:
                # both pv=0 and pv=1 present
                # Let's compute the mean timeseries for pv=0 and for pv=1
                # We assume the "timeseries" column is shape (seq_len, n_dim)
                # If it's single-dim, n_dim=1. If multiple dims, adjust logic or pick dimension 0
                sub_pv0 = subdf[subdf[pv_col] == 0]["timeseries"]
                sub_pv1 = subdf[subdf[pv_col] == 1]["timeseries"]

                # Compute an average timeseries for each
                # shape: (seq_len, n_dim)
                mean_ts_pv0 = np.mean(np.stack(sub_pv0.to_numpy()), axis=0)
                mean_ts_pv1 = np.mean(np.stack(sub_pv1.to_numpy()), axis=0)

                # difference: shape (seq_len, n_dim)
                diff_ts = mean_ts_pv1 - mean_ts_pv0
                # If you only have 1 dimension for "grid", diff_ts is shape (seq_len,1).
                # let's flatten or pick dimension 0 if needed
                if diff_ts.ndim == 2 and diff_ts.shape[1] == 1:
                    diff_ts = diff_ts[:, 0]

                shift_accumulator.append(diff_ts)

        if len(shift_accumulator) == 0:
            print("Warning: Found no groups that contain both pv=0 and pv=1!")
            return np.zeros((self.seq_len,))

        # shape after stacking: (num_groups, seq_len)
        shift_matrix = np.stack(shift_accumulator, axis=0)
        avg_shift = np.mean(shift_matrix, axis=0)  # shape (seq_len,)

        return avg_shift

    def sample_shift_test_contexts(
        self, group_vars: list = None, pv_col: str = "has_solar"
    ) -> list:
        """
        Finds example contexts that only have pv=0 (no pv=1) so we can test generating pv=1 for them,
        or vice versa, or any other custom logic you want.

        Returns a small list of context dicts for test generation.

        Args:
            group_vars (list): The context variables we want to group by (excluding `pv`).
            pv_col (str): The name of the column for PV state.

        Returns:
            test_contexts (list of dict): Each dict has {var_name: category_index}
                                          that we can pass to a generator to sample from.
        """
        if group_vars is None:
            group_vars = [v for v in self.context_vars if v != pv_col]

        df = self.data.copy()
        grouped = df.groupby(group_vars)

        test_contexts = []
        for group_vals, subdf in grouped:
            # Check which PV states appear
            unique_pv_states = subdf[pv_col].unique()
            if len(unique_pv_states) == 1:
                # This group has only pv=0 or only pv=1
                # We can test generating the "missing" state
                the_pv_value_present = int(unique_pv_states[0])
                missing_pv = 1 - the_pv_value_present

                # Build a context dict. group_vals is something like (month='Jan', weekday='Mon', etc.)
                # We need to map them back to var_name => category index (which the model uses).
                ctx_dict = {}
                if len(group_vars) == 1:
                    ctx_dict[group_vars[0]] = (
                        group_vals  # if group_vals is just one val
                    )
                else:
                    for var_name, val in zip(group_vars, group_vals):
                        ctx_dict[var_name] = val

                # Add the known existing pv state
                ctx_dict[pv_col] = the_pv_value_present

                test_contexts.append(
                    {
                        "base_context": ctx_dict,
                        "present_pv": the_pv_value_present,
                        "missing_pv": missing_pv,
                    }
                )

        return test_contexts
