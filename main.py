import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from data_utils.dataset import PecanStreetDataset, split_dataset
from eval.evaluator import Evaluator
from generator.acgan import ACGAN
from generator.diffcharge.diffusion import DDPM
from generator.diffcharge.options import Options
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.timegan import TimeGAN

# from generator.timegan_.tgan import TimeGAN
# from generator.timegan_.options import Options


def evaluate_acgan():
    full_dataset = PecanStreetDataset(
        normalize=True, user_id=None, include_generation=True, threshold=(-5, 5)
    )
    all_users = full_dataset.data.dataid.unique()
    all_users = [661]

    for user in tqdm(all_users):
        print(f"Training for user {user}...")
        data = PecanStreetDataset(
            normalize=True, user_id=user, include_generation=True, threshold=(-5, 5)
        )
        train_dataset, val_dataset = split_dataset(data)

        input_dim = (
            int(data.is_pv_user) + 1
        )  # if user has available pv data, input dim is 2

        model = ACGAN(
            input_dim=input_dim,
            noise_dim=2,
            embedding_dim=256,
            window_length=96,
            learning_rate=1e-4,
            weight_path="runs/",
        )
        model.train(train_dataset, val_dataset, batch_size=32, num_epoch=200)
        user_evaluator = Evaluator(data, model, input_dim, f"runs/acgan/user_{user}")
        user_evaluator.evaluate_all_users()


def evaluate_diffusion_ts():
    full_dataset = PecanStreetDataset(
        normalize=True, user_id=None, include_generation=True, threshold=(-6, 6)
    )
    all_users = full_dataset.data.dataid.unique()
    all_users = [661]

    for user in tqdm(all_users):
        print(f"Training for user {user}...")
        data = PecanStreetDataset(
            normalize=True, user_id=user, include_generation=True, threshold=(-6, 6)
        )

        input_dim = (
            int(data.is_pv_user) + 1
        )  # if user has available pv data, input dim is 2

        model = Diffusion_TS(seq_length=96, feature_size=input_dim, d_model=96)
        model.train_model(data, batch_size=32)
        user_evaluator = Evaluator(
            data, model, input_dim, f"runs/diffusion_ts/user_{user}"
        )
        user_evaluator.evaluate_all_users()


def main():
    # evaluate_acgan()
    # evaluate_diffusion_ts()
    full_dataset = PecanStreetDataset(
        normalize=True, user_id=None, include_generation=True, threshold=(-4, 4)
    )
    all_users = full_dataset.data.dataid.unique()
    all_users = [3687]
    all_users = [661]

    for user in tqdm(all_users):
        print(f"Training for user {user}...")
        data = PecanStreetDataset(
            normalize=True, user_id=user, include_generation=True, threshold=(-4, 4)
        )
        train_dataset, val_dataset = split_dataset(data)

        input_dim = (
            int(data.is_pv_user) + 1
        )  # if user has available pv data, input dim is 2

        model = DDPM(Options(model_name="diffusion", isTrain=True))
        model.train(train_dataset, val_dataset)
        user_evaluator = Evaluator(
            data, model, input_dim, f"runs/diffcharge/user_{user}"
        )
        user_evaluator.evaluate_all_users()


if __name__ == "__main__":
    main()
