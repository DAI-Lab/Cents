import numpy as np
import pandas as pd

from data_utils.dataset import PecanStreetDataset, prepare_dataloader, split_dataset
from generator.acgan import ACGAN


def main():
    data = PecanStreetDataset(
        normalize=True, user_id=661, include_generation=True, threshold=(-2, 2)
    )
    og_data = PecanStreetDataset(
        normalize=False, user_id=661, include_generation=True, threshold=(-2, 2)
    ).data

    inv_data = data.inverse_transform(data.data)
    a = 1
    # train_dataset, val_dataset = split_dataset(data)
    # model = ACGAN(
    #     input_dim=1,
    #     noise_dim=512,
    #     embedding_dim=512,
    #     window_length=96,
    #     learning_rate=1e-4,
    #     weight_path="runs/",
    # )


if __name__ == "__main__":
    main()
