import numpy as np
from sklearn.metrics import mean_squared_error


@staticmethod
def check_inverse_transform(normalized_dataset, unnormalized_dataset):
    unnormalized_df = unnormalized_dataset.data
    transformed = normalized_dataset.inverse_transform()

    mse_list = []

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
