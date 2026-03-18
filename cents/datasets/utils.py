from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset


def split_timeseries(
    df: pd.DataFrame,
    time_series_columns: List[str],
) -> pd.DataFrame:
    first = df["timeseries"].iloc[0]
    if not isinstance(first, np.ndarray):
        raise ValueError("‘timeseries’ must be numpy arrays")
    n_dim = first.shape[1]
    if n_dim != len(time_series_columns):
        raise ValueError("shape mismatch")
    for i, col in enumerate(time_series_columns):
        df[col] = df["timeseries"].apply(lambda x: x[:, i].tolist())
    return df.drop(columns=["timeseries"])


def merge_timeseries_columns(
    df: pd.DataFrame,
    time_series_columns: List[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in time_series_columns:
        df[col] = df[col].apply(lambda arr: np.asarray(arr).reshape(-1, 1))
    df["timeseries"] = df[time_series_columns].apply(
        lambda r: np.hstack([r[c] for c in time_series_columns]), axis=1
    )
    return df.drop(columns=time_series_columns)


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
    print(f"[Cents] Average MSE over all rows: {avg_mse}")
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


def encode_context_variables(
    data: pd.DataFrame, columns_to_encode: List[str], bins: int, 
    numeric_cols: List[str] = None, continuous_vars: List[str] = None,
    time_series_cols: List[str] = None,
    categorical_time_series: Dict[str, int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Any]]]:
    """
    Encodes specified columns in the DataFrame either by binning numeric columns
    or by converting categorical/string columns to integer codes. For 'weekday'
    and 'month' columns, encoding follows chronological order.
    Continuous variables are skipped and kept as-is.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        columns_to_encode (List[str]): List of column names to encode.
        bins (int): Number of bins for numeric columns.
        numeric_cols (List[str], optional): Columns to treat as numeric for binning.
        continuous_vars (List[str], optional): Columns to keep as continuous (skip encoding).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Dict[int, Any]]]:
            - Encoded DataFrame.
            - Mapping dictionary for each encoded column.
    """
    encoded_data = data.copy()
    mapping: Dict[str, Dict[int, Any]] = {}
    continuous_vars = continuous_vars or []

    # Define the chronological order for weekdays and months
    weekdays_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    months_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    for col in columns_to_encode:
        # Skip continuous variables - they should remain as float values
        if col in categorical_time_series:
            encoded_data[col], mapping[col] = encode_list_column(encoded_data[col])
        elif col in time_series_cols or col in continuous_vars:
            continue
        elif numeric_cols and col in numeric_cols:
            print("ENCODING NUMERIC COL", col)
            # Numeric column: Perform binning
            # Handle NaN values by filling with median before binning
            if encoded_data[col].isna().all():
                # Entire column is NaN - fill with 0 and create single bin
                print(f"  Warning: {col} is entirely NaN, filling with 0")
                encoded_data[col] = 0
            elif encoded_data[col].isna().any():
                print(f"  Warning: {col} has {encoded_data[col].isna().sum()} NaN values, filling with median")
                median_val = encoded_data[col].median()
                encoded_data[col] = encoded_data[col].fillna(median_val if pd.notna(median_val) else 0)
            
            binned = pd.cut(encoded_data[col], bins=bins, include_lowest=True, duplicates='drop')
            encoded_data[col] = binned.cat.codes  # Assign integer codes starting from 0
            bin_intervals = binned.cat.categories
            # Create the mapping from integer code to bin interval
            bin_mapping = {
                code: str(interval) for code, interval in enumerate(bin_intervals)
            }
            mapping[col] = bin_mapping
        else:
            # Categorical/String column
            if col.lower() == "weekday":
                # Ensure consistent capitalization
                encoded_data[col] = encoded_data[col].str.capitalize()
                # Define categorical type with specified order
                encoded_data[col] = pd.Categorical(
                    encoded_data[col], categories=weekdays_order, ordered=True
                )
                encoded_data[col] = encoded_data[col].cat.codes
                # Create the mapping from code to weekday
                weekday_mapping = {code: day for code, day in enumerate(weekdays_order)}
                mapping[col] = weekday_mapping
            elif col.lower() == "month":
                # Ensure consistent capitalization
                encoded_data[col] = encoded_data[col].str.capitalize()
                # Define categorical type with specified order
                encoded_data[col] = pd.Categorical(
                    encoded_data[col], categories=months_order, ordered=True
                )
                encoded_data[col] = encoded_data[col].cat.codes
                # Create the mapping from code to month
                month_mapping = {code: month for code, month in enumerate(months_order)}
                mapping[col] = month_mapping
            else:
                # For other categorical/string columns, perform standard encoding
                encoded_data[col] = encoded_data[col].astype("category").cat.codes
                categories = data[col].astype("category").cat.categories
                category_mapping = {
                    i: category for i, category in enumerate(categories)
                }
                mapping[col] = category_mapping

    return encoded_data, mapping


def convert_generated_data_to_df(
    data: torch.Tensor,
    context_vars: Dict[str, Any],
    mapping: Optional[Dict[str, Any]] = None,
    decode: bool = True,
) -> pd.DataFrame:
    """
    Convert generated time series data into a DataFrame with a single 'timeseries' column.
    """
    data_np = data.cpu().numpy()
    if data_np.ndim == 2:
        data_np = data_np.reshape(data_np.shape[0], -1, 1)
    elif data_np.ndim != 3:
        raise ValueError("Generated data must have 2 or 3 dimensions.")

    n_samples = data_np.shape[0]

    def _get_code_at(code: Any, i: int) -> Any:
        if isinstance(code, torch.Tensor):
            if code.dim() == 2 and code.shape[0] == n_samples:
                # Dynamic (time-series) context: return as numpy array
                return code[i].cpu().numpy()
            if code.dim() == 1 and code.shape[0] == n_samples:
                return code[i].item()
        return code.item() if isinstance(code, torch.Tensor) else code

    records = []
    for i in range(n_samples):
        record = {}
        for var, code in context_vars.items():
            v = _get_code_at(code, i)
            if decode:
                if mapping is None:
                    raise ValueError("Mapping must be provided when decode=True.")
                record[var] = mapping[var][v]
            elif isinstance(v, np.ndarray):
                record[var] = v.tolist()
            else:
                record[var] = v if isinstance(v, float) else int(v)
        record["timeseries"] = data_np[i]
        records.append(record)

    return pd.DataFrame.from_records(records)

def encode_list_column(series: pd.Series):
    count = 0
    for x in series:
        for t in x:
            if pd.isna(t):
                count += 1
    vocab = sorted({t for x in series for t in x})
    tok2id = {t: i for i, t in enumerate(vocab)}
    encoded = series.apply(lambda x: [tok2id[t] for t in x])
    encoded = encoded.map(tuple)
    mapping = dict(enumerate(vocab))  # id -> token
    return encoded, mapping