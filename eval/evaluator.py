import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from eval.metrics import (
    Context_FID,
    calculate_period_bound_mse,
    dynamic_time_warping_dist,
    visualization,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, real_dataset, model, n_dimensions, log_dir="logs"):
        self.real_dataset = real_dataset
        self.real_df = real_dataset.data
        self.n_dimensions = n_dimensions
        self.synthetic_df = self.generate_data_for_eval(
            model, n_dimensions=n_dimensions
        )
        self.writer = SummaryWriter(log_dir)
        self.metrics = {"dtw": [], "mse": [], "fid": []}

    def evaluate_for_user(self, user_id):

        real_user_data = self.real_df[self.real_df["user_id"] == user_id]
        syn_user_data = self.synthetic_df[self.synthetic_df["user_id"] == user_id]

        relevant_cols = ["grid"]

        if self.n_dimensions == 2:
            relevant_cols.append("solar")

        assert (
            real_user_data.shape == syn_user_data.shape
        ), "Real and synthetic dataframes need to have similar size."

        # Get inverse transformed data
        real_user_data_inv = self.real_dataset.inverse_transform(real_user_data)
        syn_user_data_inv = self.inverse_transform(syn_user_data)

        real_data_array = np.expand_dims(
            np.array(real_user_data[relevant_cols].to_list()), axis=-1
        )
        syn_data_array = np.expand_dims(
            np.array(syn_user_data[relevant_cols].to_list()), axis=-1
        )
        real_data_array_inv = np.expand_dims(
            np.array(real_user_data_inv[relevant_cols].to_list()), axis=-1
        )
        syn_data_array_inv = np.expand_dims(
            np.array(syn_user_data_inv[relevant_cols].to_list()), axis=-1
        )

        # Compute dtw using original scale data
        dtw_mean, dtw_std = dynamic_time_warping_dist(
            real_data_array_inv, syn_data_array_inv
        )
        self.writer.add_scalar(f"DTW/{user_id}/mean", dtw_mean)
        self.writer.add_scalar(f"DTW/{user_id}/std", dtw_std)
        self.metrics["dtw"].append((user_id, dtw_mean, dtw_std))

        # Compute Period Bound MSE using original scale data and dataframe
        mse_mean, mse_std = calculate_period_bound_mse(
            syn_data_array_inv, real_user_data_inv, columns=relevant_cols
        )
        self.writer.add_scalar(f"MSE/{user_id}/mean", mse_mean)
        self.writer.add_scalar(f"MSE/{user_id}/std", mse_std)
        self.metrics["mse"].append((user_id, mse_mean, mse_std))

        # Compute Context FID using normalized and scaled data
        for _, row in syn_user_data.iterrows():
            for col in relevant_cols:
                gen_daily_profile = row[col]
                fid_score = Context_FID(real_data_array, gen_daily_profile)
                self.writer.add_scalar(f"FID/{col}/{user_id}", fid_score)
                self.metrics["fid"].append((user_id, col, fid_score))

        # for col in relevant_cols:
        #     visualization(
        #         real_user_data_inv[col].values, syn_user_data_inv[col].values, "tsne"
        #     )
        #     visualization(
        #         real_user_data_inv[col].values, syn_user_data_inv[col].values, "kernel"
        #     )
        #     self.writer.add_figure()

    def evaluate_all_users(self):
        user_ids = self.real_df["user_id"].unique()
        for user_id in user_ids:
            self.evaluate_user(user_id)
        self.writer.flush()

    def generate_data_for_eval(self, model, n_dimensions):
        syn_ts = []

        for _, row in self.real_df.iterrows():
            month_label = torch.tensor([row["month"]]).to(device)
            day_label = torch.tensor([row["weekday"]]).to(device)
            gen_ts = model.generate([month_label, day_label]).squeeze().cpu().numpy()

            if n_dimensions == 2:
                gen_grid, gen_solar = gen_ts[0], gen_ts[1]
                syn_ts.append(
                    (row["month"], row["weekday"], row["date_day"], gen_grid, gen_solar)
                )
            else:
                syn_ts.append((row["month"], row["weekday"], row["date_day"], gen_ts))

        columns = ["month", "weekday", "date_day", "solar"]

        if n_dimensions == 2:
            columns.append("solar")

        syn_ts_df = pd.DataFrame(syn_ts, columns=columns)
        return syn_ts_df
