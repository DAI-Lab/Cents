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


def encode_categorical_variables(data: pd.DataFrame, columns: List[str]):
    """
    Encodes categorical variables in a DataFrame to integer codes.

    Args:
        data (pd.DataFrame): Input DataFrame containing categorical variables.
        columns (List[str]): List of column names to transform.

    Returns:
        df_encoded (pd.DataFrame): DataFrame with categorical variables encoded as integer codes.
        mappings (dict): Dictionary mapping column names to their category-to-code mappings.
    """
    df_encoded = data.copy()
    mappings = {}

    for col in columns:
        df_encoded[col] = df_encoded[col].astype("category")

        category_to_code = dict(enumerate(df_encoded[col].cat.categories))
        code_to_category = {v: k for k, v in category_to_code.items()}
        df_encoded[col] = df_encoded[col].cat.codes

        mappings[col] = {
            "category_to_code": {cat: code for code, cat in category_to_code.items()},
            "code_to_category": code_to_category,
        }

    return df_encoded


def encode_numerical_variables(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Takes numerical conditioning columns (e.g. total square footage), and converts it into integer encoded mappings.

    Args:
        data (pd.DataFrame): The data whose numerical cols are being encoded.
        columns (List[str]): The column names of numerical columns that need to be encoded.

    Returns:
        data (pd.DataFrame): The data frame that now has integer codes where numerical values used to be.
    """
    for col in columns:

        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col]

        if data[col].isnull().all():
            raise ValueError(f"Column '{col}' contains no valid numeric values.")

        data[col] = pd.cut(
            data[col], bins=5, labels=[0, 1, 2, 3, 4], include_lowest=True
        ).astype(int)

    return data
