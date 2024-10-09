import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, time_series_column_name, conditioning_vars=None):
        """
        Initializes the TimeSeriesDataset.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the time series data and optional conditioning variables.
            time_series_column_name (str): The name of the column containing the time series data.
            conditioning_vars (list of str, optional): List of column names to be used as conditioning variables.
        """
        self.data = dataframe.reset_index(drop=True)
        self.conditioning_vars = conditioning_vars or []
        self.time_series_column_name = time_series_column_name

        if self.time_series_column_name not in self.data.columns:
            raise ValueError(
                f"Time series column '{self.time_series_column_name}' not found in DataFrame."
            )

        for var in self.conditioning_vars:
            if var not in self.data.columns:
                raise ValueError(
                    f"Conditioning variable '{var}' not found in DataFrame."
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        time_series = sample[self.time_series_column_name]
        time_series = torch.tensor(time_series, dtype=torch.float32)

        conditioning_vars_dict = {}
        for var in self.conditioning_vars:
            value = sample[var]
            conditioning_vars_dict[var] = torch.tensor(value, dtype=torch.long)

        return time_series, conditioning_vars_dict
