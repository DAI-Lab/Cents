import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from eval.metrics import (
    Context_FID,
    calculate_fid,
    calculate_period_bound_mse,
    dynamic_time_warping_dist,
    visualization,
)


class Evaluator:
    def __init__(self, real_dataset, synthetic_dataset, log_dir="logs"):
        self.real_dataset = real_dataset
        self.real_df = real_dataset.data
        self.synthetic_df = synthetic_dataset
        self.writer = SummaryWriter(log_dir)
        self.metrics = {"dtw": [], "mse": [], "fid": []}

    def evaluate_user(self, user_id):
        real_user_data = self.real_df[self.real_df["user_id"] == user_id]
        syn_user_data = self.synthetic_df[self.synthetic_df["user_id"] == user_id]

        if real_user_data.empty or syn_user_data.empty:
            return

        # Get inverse transformed data
        real_user_inv = self.real_dataset.inverse_transform(real_user_data)
        syn_user_inv = self.inverse_transform(syn_user_data)

        dtw_mean, dtw_std = dynamic_time_warping_dist(real_user_inv, syn_user_inv)
        self.writer.add_scalar(f"DTW/{user_id}/mean", dtw_mean)
        self.writer.add_scalar(f"DTW/{user_id}/std", dtw_std)
        self.metrics["dtw"].append((user_id, dtw_mean, dtw_std))

        # Compute Period Bound MSE
        mse_mean, mse_std = calculate_period_bound_mse(
            syn_user_inv, real_user_data, "grid"
        )
        self.writer.add_scalar(f"MSE/{user_id}/mean", mse_mean)
        self.writer.add_scalar(f"MSE/{user_id}/std", mse_std)
        self.metrics["mse"].append((user_id, mse_mean, mse_std))

        # Compute Context FID
        fid_score = Context_FID(
            real_user_data["grid"].values, syn_user_data["syn"].values
        )
        self.writer.add_scalar(f"FID/{user_id}", fid_score)
        self.metrics["fid"].append((user_id, fid_score))

        # Visualization
        visualization(
            real_user_data["grid"].values, syn_user_data["syn"].values, "tsne"
        )
        visualization(real_user_data["grid"].values, syn_user_data["syn"].values, "pca")
        visualization(
            real_user_data["grid"].values, syn_user_data["syn"].values, "kernel"
        )

    def evaluate_all_users(self):
        user_ids = self.real_df["user_id"].unique()
        for user_id in user_ids:
            self.evaluate_user(user_id)
        self.writer.flush()
