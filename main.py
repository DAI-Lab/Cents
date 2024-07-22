import numpy as np
import pandas as pd

from data_utils.dataset import PecanStreetDataset, prepare_dataloader, split_dataset
from eval.metrics import (
    Context_FID,
    calculate_period_bound_mse,
    dynamic_time_warping_dist,
)
from generator.acgan import ACGAN

if __name__ == "__main__":

    data = PecanStreetDataset(
        normalize=True, user_id=661, include_generation=False, threshold=(-2, 2)
    )
    train_dataset, val_dataset = split_dataset(data)
    model = ACGAN(
        input_dim=1,
        noise_dim=512,
        embedding_dim=512,
        window_length=96,
        learning_rate=1e-4,
        weight_path="runs/",
    )
    model.train(train_dataset, val_dataset, batch_size=32, num_epoch=100)

    _, _, syn_ts_df = model.generate_data_for_eval(data.data)
    unnormalized_syn = data.inverse_transform(syn_ts_df, 661, "grid")
    unnormalized_ori = data.inverse_transform(data.data, 661, "grid")
    syn = np.expand_dims(np.array(unnormalized_syn["grid"].tolist()), axis=-1)
    ori = np.expand_dims(np.array(unnormalized_ori["grid"].tolist()), axis=-1)

    print(f"Context FID: {Context_FID(ori, syn)}")
    print(f"Dynamic Time Warping Distance: {dynamic_time_warping_dist(ori, syn)}")
    print(
        f"Period bound MSE: {calculate_period_bound_mse(timeseries_array=syn, df=data.data, timeseries_colname='grid')}"
    )
