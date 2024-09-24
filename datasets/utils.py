from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def check_inverse_transform(
    normalized_dataset: Dataset, unnormalized_dataset: Dataset
) -> float:
    """
    Checks if the inverse transform of the normalized dataset matches the unnormalized dataset by computing
    the Mean Squared Error (MSE) for each row of time series data.

    Args:
        normalized_dataset (Dataset): The dataset that has been normalized.
        unnormalized_dataset (Dataset): The original dataset before normalization.

    Returns:
        float: The average MSE between the transformed and original time series across all rows.
    """
    unnormalized_df = unnormalized_dataset.data
    transformed = normalized_dataset.inverse_transform(normalized_dataset.data)

    mse_list: List[float] = []

    for idx in range(len(unnormalized_df)):
        unnormalized_timeseries = unnormalized_df.iloc[idx]["timeseries"]
        transformed_timeseries = transformed.iloc[idx]["timeseries"]

        assert (
            unnormalized_timeseries.shape == transformed_timeseries.shape
        ), "Shape mismatch between transformed and unnormalized timeseries"

        mse = mean_squared_error(unnormalized_timeseries, transformed_timeseries)
        mse_list.append(mse)

    avg_mse = np.mean(mse_list)
    print(f"Average MSE over all rows: {avg_mse}")
    return avg_mse


def prepare_dataloader(
    dataset: Dataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """
    Prepares a DataLoader for batching the dataset.

    Args:
        dataset (Dataset): The dataset to be batched.
        batch_size (int): The size of each batch.
        shuffle (bool, optional): Whether to shuffle the dataset before batching. Defaults to True.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_dataset(dataset: Dataset, val_split: float = 0.1) -> Tuple[Dataset, Dataset]:
    """
    Splits a dataset into training and validation sets based on a validation split ratio.

    Args:
        dataset (Dataset): The dataset to be split.
        val_split (float, optional): The ratio of the validation set. Defaults to 0.1 (10%).

    Returns:
        Tuple[Dataset, Dataset]: The training and validation datasets.
    """
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    return train_dataset, val_dataset


def encode_conditioning_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes conditioning columns (e.g. city, total square footage), and converts it into integer encoded mappings. Discretizes numerical conditioning
    variables into categorical bins.

    Args:
        data (pd.DataFrame): The data whose cols are being encoded.

    Returns:
        data (pd.DataFrame): The data frame that now has integer codes where numerical and categorical values used to be.
    """
    for col in data.columns:

        if col in ["dataid", "timeseries", "month", "weekday", "date_day"]:
            continue

        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.cut(
                data[col], bins=5, labels=[0, 1, 2, 3, 4], include_lowest=True
            ).astype(int)
        else:
            data[col] = data[col].astype("category").cat.codes
    return data
