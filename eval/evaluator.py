import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from eval.discriminative_metric import discriminative_score_metrics
from eval.metrics import (
    Context_FID,
    calculate_mmd,
    calculate_period_bound_mse,
    dynamic_time_warping_dist,
    plot_range_with_syn_values,
    plot_syn_with_closest_real_ts,
    visualization,
)
from eval.predictive_metric import predictive_score_metrics
from generator.acgan import ACGAN
from generator.diffcharge.diffusion import DDPM
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.options import Options

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, real_dataset, model_name, log_dir="runs"):
        self.real_dataset = real_dataset
        self.model_name = model_name
        self.writer = SummaryWriter(f"{log_dir}/{model_name}")
        self.metrics = {
            "dtw": [],
            "mmd": [],
            "mse": [],
            "fid": [],
            "discr": [],
            "pred": [],
        }

    def evaluate_for_user(self, user_id):
        user_dataset = self.real_dataset.create_user_dataset(user_id)
        model = self.get_trained_model_for_user(self.model_name, user_dataset)
        user_log_dir = f"{self.writer.log_dir}/user_{user_id}"
        user_writer = SummaryWriter(user_log_dir)

        print("----------------------")
        print(f"Starting evaluation for user {user_id}")
        print("----------------------")

        self.run_eval(user_dataset, model, user_writer, user_id)

    def run_eval(self, dataset, model, writer, user_id=None):

        real_user_data = dataset.data

        syn_user_data = self.generate_data_for_eval(model, dataset.data)
        syn_user_data["dataid"] = real_user_data["dataid"]

        # Get inverse transformed data
        real_user_data_inv = dataset.inverse_transform(real_user_data)
        syn_user_data_inv = dataset.inverse_transform(syn_user_data)

        real_data_array = np.stack(real_user_data["timeseries"])
        syn_data_array = np.stack(syn_user_data["timeseries"])
        real_data_array_inv = np.stack(real_user_data_inv["timeseries"])
        syn_data_array_inv = np.stack(syn_user_data_inv["timeseries"])

        # Compute dtw using original scale data
        dtw_mean, dtw_std = dynamic_time_warping_dist(
            real_data_array_inv, syn_data_array_inv
        )
        writer.add_scalar("DTW/mean", dtw_mean)
        writer.add_scalar("DTW/std", dtw_std)
        self.metrics["dtw"].append((user_id, dtw_mean, dtw_std))

        # Compute maximum mean discrepancy between real and synthetic data for all daily load profiles and get mean
        mmd_mean, mmd_std = calculate_mmd(real_data_array_inv, syn_data_array_inv)
        writer.add_scalar("MMD/mean", mmd_mean)
        writer.add_scalar("MMD/std", mmd_std)
        self.metrics["mmd"].append((user_id, mmd_mean, mmd_std))

        # Compute Period Bound MSE using original scale data and dataframe
        mse_mean, mse_std = calculate_period_bound_mse(
            syn_data_array_inv, real_user_data_inv
        )
        writer.add_scalar("MSE/mean", mse_mean)
        writer.add_scalar("MSE/std", mse_std)
        self.metrics["mse"].append((user_id, mse_mean, mse_std))

        # Compute Context FID using normalized and scaled data
        print(f"Training TS2Vec for user {user_id}...")
        fid_score = Context_FID(real_data_array, syn_data_array)
        print("Done!")
        writer.add_scalar("FID/score", fid_score)
        self.metrics["fid"].append((user_id, fid_score))

        # Compute discriminative score using original scale data
        print(f"Computing discriminative score for user {user_id}...")
        discr_score, _, _ = discriminative_score_metrics(
            real_data_array_inv, syn_data_array_inv
        )
        print("Done!")
        writer.add_scalar("Discr/score", discr_score)
        self.metrics["discr"].append((user_id, discr_score))

        # Compute predictive score using original scale data
        print(f"Computing predictive score for user {user_id}...")
        pred_score = predictive_score_metrics(real_data_array_inv, syn_data_array_inv)
        print("Done!")
        writer.add_scalar("Pred/score", pred_score)
        self.metrics["pred"].append((user_id, pred_score))

        # Randomly select three months and three weekdays
        unique_months = real_user_data["month"].unique()
        unique_weekdays = real_user_data["weekday"].unique()
        selected_months = random.sample(list(unique_months), 1)
        selected_weekdays = random.sample(list(unique_weekdays), 1)

        # Add KDE plot
        plots = visualization(real_data_array_inv, syn_data_array_inv, "kernel")
        for i, plot in enumerate(plots):
            writer.add_figure(tag=f"KDE Dimension {i}", figure=plot)

        # Plot the range and synthetic values for each combination
        for month in selected_months:
            for weekday in selected_weekdays:
                fig = plot_range_with_syn_values(
                    real_user_data_inv, syn_user_data_inv, month, weekday
                )
                writer.add_figure(
                    tag=f"Range_Plot_Month_{month}_Weekday_{weekday}",
                    figure=fig,
                )

                fig2 = plot_syn_with_closest_real_ts(
                    real_user_data_inv, syn_user_data_inv, month, weekday
                )

                writer.add_figure(
                    tag=f"Closest_Real_TS_Plot_Month_{month}_Weekday_{weekday}",
                    figure=fig2,
                )

        writer.flush()
        writer.close()
        return syn_data_array_inv[:, :, 0], real_data_array_inv[:, :, 0]

    def evaluate_all_users(self):
        user_ids = self.real_dataset.data["dataid"].unique()
        syn_data = []
        real_data = []

        for user_id in user_ids:
            # if user_id == 27:
            syn_user_data, real_user_data = self.evaluate_for_user(user_id)
            syn_data.append(syn_user_data)
            real_data.append(real_user_data)

        syn_data = np.expand_dims(np.concatenate(syn_data, axis=0), axis=-1)
        real_data = np.expand_dims(np.concatenate(real_data, axis=0), axis=-1)

        plot = visualization(real_data, syn_data, "tsne")
        self.writer.add_figure(tag=f"TSNE", figure=plot)

        print("Final Results: \n")
        print("--------------------")
        dtw, mmd, mse, context_fid, discr_score, pred_score = self.get_summary_metrics()
        print(f"Mean User DTW: {dtw}")
        print(f"Mean User MMD: {mmd}")
        print(f"Mean User Bound MSE: {mse}")
        print(f"Mean User Context-FID: {context_fid}")
        print(f"Mean User Discriminative Score: {discr_score}")
        print(f"Mean User Predictive Score: {pred_score}")
        print("--------------------")
        self.writer.add_scalar(f"{self.model_name}/Mean_User_DTW", dtw)
        self.writer.add_scalar(f"{self.model_name}/Mean_User_MMD", mmd)
        self.writer.add_scalar(f"{self.model_name}/Mean_User_MSE", mse)
        self.writer.add_scalar(f"{self.model_name}/Mean_User_Context_FID", context_fid)
        self.writer.add_scalar(
            f"{self.model_name}/Mean_User_Discriminative_Score", discr_score
        )
        self.writer.add_scalar(
            f"{self.model_name}/Mean_User_Predictive_Score", pred_score
        )
        self.writer.flush()
        self.writer.close()

    def generate_data_for_eval(self, model, real_user_df):
        month_labels = torch.tensor(real_user_df["month"].values).to(device)
        day_labels = torch.tensor(real_user_df["weekday"].values).to(device)

        generated_ts = (
            model.generate(month_labels=month_labels, day_labels=day_labels)
            .cpu()
            .numpy()
        )

        if len(generated_ts.shape) == 2:
            generated_ts = generated_ts.reshape(
                generated_ts.shape[0], -1, generated_ts.shape[1]
            )

        syn_ts = []
        for idx, (_, row) in enumerate(real_user_df.iterrows()):
            gen_ts = generated_ts[idx]
            syn_ts.append((row["month"], row["weekday"], row["date_day"], gen_ts))

        columns = ["month", "weekday", "date_day", "timeseries"]
        syn_ts_df = pd.DataFrame(syn_ts, columns=columns)
        return syn_ts_df

    def evaluate_all_pv_users(self):
        dataset = self.real_dataset.create_all_pv_user_dataset()
        model = self.get_trained_model_for_user(self.model_name, dataset)
        user_log_dir = f"{self.writer.log_dir}/pv_users"
        user_writer = SummaryWriter(user_log_dir)

        print("----------------------")
        print(f"Starting evaluation for all pv users")
        print("----------------------")

        self.run_eval(dataset, model, user_writer, None)

    def evaluate_all_non_pv_users(self):
        dataset = self.real_dataset.create_non_pv_user_dataset()
        model = self.get_trained_model_for_user(self.model_name, dataset)
        user_log_dir = f"{self.writer.log_dir}/pv_users"
        user_writer = SummaryWriter(user_log_dir)

        print("----------------------")
        print(f"Starting evaluation for all non-pv users")
        print("----------------------")

        self.run_eval(dataset, model, user_writer, None)

    def get_summary_metrics(self):
        """
        Calculate the mean values for all users across all metrics.

        Returns:
            A tuple containing the mean values for dtw, mse, and fid.
        """
        metrics_summary = {
            "dtw": [],
            "mmd": [],
            "mse": [],
            "fid": [],
            "discr": [],
            "pred": [],
        }

        # Collect mean values for each metric
        for metric in metrics_summary.keys():
            mean_values = [value[1] for value in self.metrics[metric]]
            metrics_summary[metric] = np.mean(mean_values)

        return (
            metrics_summary["dtw"],
            metrics_summary["mmd"],
            metrics_summary["mse"],
            metrics_summary["fid"],
            metrics_summary["discr"],
            metrics_summary["pred"],
        )

    def get_trained_model_for_user(self, model_name, user_dataset):

        input_dim = (
            int(user_dataset.is_pv_user) + 1
        )  # if user has available pv data, input dim is 2

        opt = Options(model_name)
        opt.input_dim = input_dim

        if model_name == "acgan":
            model = ACGAN(opt)
            model.train_model(user_dataset)

        elif model_name == "diffcharge":
            model = DDPM(opt)
            model.train_model(user_dataset)

        elif model_name == "diffusion_ts":
            model = Diffusion_TS(opt)
            model.train_model(user_dataset)

        else:
            raise ValueError("Model name not recognized!")

        return model
