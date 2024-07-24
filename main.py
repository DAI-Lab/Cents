import numpy as np
import pandas as pd

from data_utils.dataset import PecanStreetDataset, prepare_dataloader, split_dataset
from eval.evaluator import Evaluator
from generator.acgan import ACGAN


def main():
    full_dataset = PecanStreetDataset(
        normalize=True, user_id=None, include_generation=True, threshold=(-2, 2)
    )
    all_users = full_dataset.data.dataid.unique()

    for user in all_users:
        data = PecanStreetDataset(
            normalize=True, user_id=user, include_generation=True, threshold=(-2, 2)
        )
        train_dataset, val_dataset = split_dataset(data)
        model = ACGAN(
            input_dim=2,
            noise_dim=512,
            embedding_dim=512,
            window_length=96,
            learning_rate=1e-4,
            weight_path="runs/",
        )
        model.train(train_dataset, val_dataset, batch_size=32, num_epoch=50)
        user_evaluator = Evaluator(data, model, 2, "runs/")
        user_evaluator.evaluate_for_user(user)


if __name__ == "__main__":
    main()
