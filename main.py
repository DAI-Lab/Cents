import pandas as pd

from data_utils.dataset import PecanStreetDataset, prepare_dataloader, split_dataset
from eval.metrics import (
    Context_FID,
    calculate_period_bound_mse,
    dynamic_time_warping_dist,
)
from generator.acgan import ACGAN

if __name__ == "__main__":

    data = PecanStreetDataset(normalize=True, user_id=1642, include_generation=False)
    train_dataset, val_dataset = split_dataset(data)
    model = ACGAN(
        input_dim=1,
        noise_dim=64,
        embedding_dim=64,
        window_length=96,
        learning_rate=5e-4,
        weight_path="runs/",
    )
    model.train(train_dataset, val_dataset, batch_size=32, num_epoch=100)

    ori, syn = model.generate_data_for_eval(data.data)

    print(f"Context FID: {Context_FID(ori, syn)}")
    print(f"Dynamic Time Warping Distance: {dynamic_time_warping_dist(ori, syn)}")
    print(
        f"Period bound MSE: {calculate_period_bound_mse(timeseries_array=syn, df=data.data, timeseries_colname='grid')}"
    )
