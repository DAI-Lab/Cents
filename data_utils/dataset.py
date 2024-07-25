import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PecanStreetDataset(Dataset):
    def __init__(
        self,
        geography: str = None,
        config_path: str = "config/config.yaml",
        normalize=False,
        threshold=None,
        include_generation=False,
        user_id=None,
    ):
        """
        Initialize the PecanStreetDataset with a specific geography, configuration path and user.

        Args:
            geography: Geography of the dataset (e.g., 'newyork').
            config_path: Path to the configuration file.
            normalize: Flag indicating whether data is normalized.
            threshold: Thresholds at which to clip the data.
            include_generation: Flag indicating whether to include solar data if available.
            user_id: ID of the desired user. If None, all user data is loaded.
        """
        self.geography = geography
        self.config_path = os.path.join(ROOT_DIR, config_path)
        self.normalize = normalize
        self.threshold = threshold
        self.include_generation = include_generation
        self.user_id = user_id
        self.is_pv_user = None
        self.name = "pecanstreet"
        self.stats = {}
        self.path, self.columns = self._get_dataset_info()
        self.data = self.load_data()

    def _get_dataset_info(self) -> Tuple[str, List[str]]:
        """
        Retrieve dataset information from the configuration file.

        Returns:
            A tuple containing the path and columns of the dataset.
        """
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        dataset_info = config["datasets"].get(self.name)
        if not dataset_info:
            raise ValueError(f"No dataset configuration found for {self.name}")
        return dataset_info["path"], dataset_info["columns"]

    def load_data(self) -> pd.DataFrame:
        """
        Load the Pecan Street dataset.

        Returns:
            A pandas DataFrame containing the dataset.
        """
        self.metadata = pd.read_csv(f"/{self.path}metadata.csv")
        if self.geography:
            data_file_path = f"/{self.path}15minute_data_{self.geography}.csv"
            try:
                data = pd.read_csv(data_file_path)
                data = data[self.columns]
            except Exception as e:
                print(f"Failed to load data from {data_file_path}: {e}")
                return pd.DataFrame()
        else:
            ny_file_path = f"/{self.path}15minute_data_newyork.csv"
            cali_file_path = f"/{self.path}15minute_data_california.csv"
            austin_file_path = f"/{self.path}15minute_data_austin.csv"

            try:
                ny_data = pd.read_csv(ny_file_path)
                cali_data = pd.read_csv(cali_file_path)
                austin_data = pd.read_csv(austin_file_path)
                data = pd.concat([ny_data, cali_data, austin_data], axis=0)
                data = data[self.columns]
            except Exception as e:
                print(f"Failed to load full dataset!")
                return pd.DataFrame()

        if self.user_id:
            data = data[data["dataid"] == self.user_id].copy()

            if not len(data):
                raise ValueError(f"No data found for user {self.user_id}!")

            self.user_metadata = self.metadata.loc[
                self.metadata["dataid"] == self.user_id
            ].copy()
            self.is_pv_user = self.user_metadata["pv"].notna().any()

        grid_data = self.preprocess_data(data, "grid", self.threshold)

        if self.include_generation and self.is_pv_user:
            grid_data["solar"] = self.preprocess_data(data, "solar", self.threshold)[
                "solar"
            ].copy()
        else:
            grid_data["timeseries"] = [
                np.array(g).reshape(-1, 1) for g in grid_data["grid"]
            ]

        grid_data = self.merge_columns_into_timeseries(grid_data)
        return grid_data.sort_values(by=["dataid", "month", "weekday"])

    def preprocess_data(
        self, data: pd.DataFrame, column: str, threshold=(-2, 2)
    ) -> pd.DataFrame:
        """
        Normalize the specified column in the data based on groupings of 'dataid', 'month', and 'weekday'.

        Args:
            data (pd.DataFrame): The input data.
            column (str): The column to preprocess.
            threshold (tuple): The normalization threshold.

        Returns:
            pd.DataFrame: The normalized data.
        """
        data["local_15min"] = pd.to_datetime(data["local_15min"], utc=True)
        data["month"] = data["local_15min"].dt.month - 1
        data["weekday"] = data["local_15min"].dt.weekday
        data["date_day"] = data["local_15min"].dt.day
        data = data.sort_values(by=["local_15min"], axis=0)

        data = data[~data[column].isna()].copy()

        if not all(
            col in data.columns for col in ["dataid", "month", "weekday", column]
        ):
            raise ValueError(
                f"Input data must contain at least 'dataid', 'month', 'weekday', and '{column}' columns"
            )

        grouped_data = data.groupby(["dataid", "month", "date_day", "weekday"])
        grouped_data = grouped_data[column].apply(list).reset_index()
        filtered_data = grouped_data[grouped_data[column].apply(len) == 96]

        if self.normalize:

            def calculate_stats(group):
                all_values = np.concatenate(group[column].values)
                return pd.Series(
                    {
                        "mean": np.mean(all_values),
                        "std": np.std(all_values),
                        "min": np.min(all_values),
                        "max": np.max(all_values),
                    }
                )

            grouped_stats = (
                filtered_data.groupby(["dataid", "month", "weekday"])
                .apply(calculate_stats)
                .reset_index()
            )
            filtered_data = filtered_data.merge(
                grouped_stats, on=["dataid", "month", "weekday"]
            )

            def normalize_and_scale(values, mean, std, min_val, max_val):
                normalized = np.array([(val - mean) / std for val in values])

                if self.threshold:
                    thresholded = np.clip(normalized, threshold[0], threshold[1])
                    return np.array(
                        [
                            (val - threshold[0]) / (threshold[1] - threshold[0])
                            for val in thresholded
                        ]
                    )
                else:
                    return np.array(
                        [(val - min_val) / (max_val - min_val) for val in normalized]
                    )

            filtered_data[column] = filtered_data.apply(
                lambda row: normalize_and_scale(
                    row[column], row["mean"], row["std"], row["min"], row["max"]
                ),
                axis=1,
            )

            # Store statistics for inverse transformation
            if column not in self.stats:
                self.stats[column] = {}

            self.stats[column] = grouped_stats.set_index(
                ["dataid", "month", "weekday"]
            ).to_dict("index")

            filtered_data = filtered_data.drop(columns=["mean", "std", "min", "max"])

        return filtered_data

    def inverse_transform(
        self,
        preprocessed_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert a preprocessed time series back to its original scale.

        Args:
            preprocessed_data (pd.DataFrame): The preprocessed data.

        Returns:
            pd.DataFrame: A DataFrame containing the time series in its original scale along with month and weekday.
        """

        def split_timeseries(row):
            grid = row[:, 0].reshape(-1, 1)
            solar = row[:, 1].reshape(-1, 1)
            return pd.Series({"grid": grid, "solar": solar})

        if not self.normalize:
            return preprocessed_data.copy()

        if self.include_generation and self.is_pv_user:
            preprocessed_data[["grid", "solar"]] = preprocessed_data[
                "timeseries"
            ].apply(split_timeseries)
        else:
            preprocessed_data["grid"] = preprocessed_data["timeseries"]

        preprocessed_data = preprocessed_data.drop(columns=["timeseries"])

        def inverse_transform_column(row, colname, stats):
            month = row["month"]
            weekday = row["weekday"]
            dataid = row["dataid"]

            col_stats = stats.get((dataid, month, weekday))

            if col_stats is None:
                raise ValueError(
                    f"No statistics found for dataid={dataid}, month={month}, weekday={weekday} for {colname}"
                )

            mean, std, min_val, max_val = (
                col_stats["mean"],
                col_stats["std"],
                col_stats["min"],
                col_stats["max"],
            )

            if self.threshold:
                low, high = self.threshold
            else:
                low, high = (min_val, max_val)

            unscaled = np.array([value * (high - low) + low for value in row[colname]])
            unnormalized = np.array([value * std + mean for value in unscaled])

            return unnormalized.squeeze()

        result_data = []

        for dataid in preprocessed_data["dataid"].unique():
            dataid_df = preprocessed_data[preprocessed_data["dataid"] == dataid]

            for _, row in dataid_df.iterrows():
                transformed_row = {
                    "dataid": dataid,
                    "month": row["month"],
                    "weekday": row["weekday"],
                    "grid": inverse_transform_column(row, "grid", self.stats["grid"]),
                }

                if "solar" in preprocessed_data.columns:
                    transformed_row["solar"] = inverse_transform_column(
                        row, "solar", self.stats["solar"]
                    )

                result_data.append(transformed_row)

        result_df = self.merge_columns_into_timeseries(pd.DataFrame(result_data))
        return result_df.sort_values(by=["dataid", "month", "weekday"])

    def merge_columns_into_timeseries(self, df):
        if self.include_generation and self.is_pv_user:
            df["timeseries"] = [
                np.vstack((g, s)).T for g, s in zip(df["grid"], df["solar"])
            ]
            df.drop(columns=["grid", "solar"], axis=1, inplace=True)
        else:
            df["timeseries"] = df["grid"]
            df.drop(columns=["grid"], axis=1, inplace=True)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        time_series = sample["timeseries"]

        month = sample["month"]
        day = sample["weekday"]
        return (
            torch.tensor(time_series, dtype=torch.float32).to(device),
            torch.tensor(month, dtype=torch.long).to(device),
            torch.tensor(day, dtype=torch.long).to(device),
        )


def prepare_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_dataset(dataset: Dataset, val_split: float = 0.2):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    return train_dataset, val_dataset
