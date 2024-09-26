from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from eval.discriminative_metric import discriminative_score_metrics
from eval.metrics import Context_FID
from eval.metrics import calculate_mmd
from eval.metrics import calculate_period_bound_mse
from eval.metrics import dynamic_time_warping_dist
from eval.metrics import plot_range_with_syn_values
from eval.metrics import plot_syn_with_closest_real_ts
from eval.metrics import visualization
from eval.predictive_metric import predictive_score_metrics
from generator.diffcharge.diffusion import DDPM
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.gan.acgan import ACGAN
from generator.options import Options

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    """
    A class for evaluating generative models on time series data.
    """

    def __init__(self, real_dataset: Any, model_name: str, log_dir: str = "runs"):
        """
        Initialize the Evaluator.

        Args:
            real_dataset: The real dataset used for evaluation.
            model_name (str): The name of the generative model being evaluated.
            log_dir (str): Directory for storing TensorBoard logs.
        """
        self.real_dataset = real_dataset
        self.model_name = model_name
        self.log_dir = log_dir
        self.writer = SummaryWriter(f"{log_dir}/{model_name}")
        self.metrics: Dict[str, List] = {
            "dtw": [],
            "mmd": [],
            "mse": [],
            "fid": [],
            "discriminative": [],
            "predictive": [],
        }

    def evaluate_model(self, user_id: int = None, distinguish_rare: bool = False):
        """
        Evaluate the model for a specific user or the entire dataset.

        Args:
            user_id (int, optional): The ID of the user to evaluate. If None, evaluate on the entire dataset.
            distinguish_rare (bool): Whether to distinguish between rare and non-rare data samples.
        """
        # Prepare dataset and logging
        if user_id is not None:
            dataset = self.real_dataset.create_user_dataset(user_id)
            log_dir = f"{self.log_dir}/{self.model_name}/user_{user_id}"
        else:
            dataset = self.real_dataset
            log_dir = f"{self.log_dir}/{self.model_name}/all_users"

        writer = SummaryWriter(log_dir)
        model = self.get_trained_model(dataset)

        print("----------------------")
        if user_id is not None:
            print(f"Starting evaluation for user {user_id}")
        else:
            print("Starting evaluation for all users")
        print("----------------------")

        self.run_evaluation(dataset, model, writer, distinguish_rare)

    def run_evaluation(
        self, dataset: Any, model: Any, writer: SummaryWriter, distinguish_rare: bool
    ):
        """
        Run the evaluation process.

        Args:
            dataset: The dataset to evaluate.
            model: The trained model.
            writer (SummaryWriter): TensorBoard writer for logging results.
            distinguish_rare (bool): Whether to distinguish between rare and non-rare data samples.
        """
        if distinguish_rare:
            rare_indices, non_rare_indices = self.identify_rare_combinations(
                dataset, model
            )
            self.evaluate_subset(dataset, model, writer, rare_indices, data_type="rare")
            self.evaluate_subset(
                dataset, model, writer, non_rare_indices, data_type="non_rare"
            )
        else:
            all_indices = dataset.data.index.to_numpy()
            self.evaluate_subset(dataset, model, writer, all_indices, data_type="all")

        writer.flush()
        writer.close()

    def identify_rare_combinations(
        self, dataset: Any, model: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify indices of rare and non-rare conditioning combinations in the dataset.

        Args:
            dataset: The dataset containing conditioning variables.
            model: The trained model with the conditioning module.

        Returns:
            Tuple containing indices of rare and non-rare conditioning combinations.
        """
        conditioning_vars = {
            name: torch.tensor(
                dataset.data[name].values, dtype=torch.long, device=device
            )
            for name in model.categorical_dims.keys()
        }

        with torch.no_grad():
            embeddings = model.generator.conditioning_module(conditioning_vars)
            mahalanobis_distances = (
                model.generator.conditioning_module.compute_mahalanobis_distance(
                    embeddings
                )
            )

        # Define rarity threshold (e.g., top 20% as rare)
        rarity_threshold = torch.quantile(mahalanobis_distances, 0.8)
        rare_mask = mahalanobis_distances > rarity_threshold

        rare_indices = np.where(rare_mask.cpu().numpy())[0]
        non_rare_indices = np.where(~rare_mask.cpu().numpy())[0]

        return rare_indices, non_rare_indices

    def evaluate_subset(
        self,
        dataset: Any,
        model: Any,
        writer: SummaryWriter,
        indices: np.ndarray,
        data_type: str,
    ):
        """
        Evaluate the model on a subset of the data.

        Args:
            dataset: The dataset containing real data.
            model: The trained model to generate data.
            writer (SummaryWriter): TensorBoard writer for logging results.
            indices (np.ndarray): Indices of data to use.
            data_type (str): Label for the data subset ("rare", "non_rare", or "all").
        """
        real_data_subset = dataset.data.iloc[indices].reset_index(drop=True)
        conditioning_vars = {
            name: torch.tensor(
                real_data_subset[name].values, dtype=torch.long, device=device
            )
            for name in model.categorical_dims.keys()
        }

        generated_ts = model.generate(conditioning_vars).cpu().numpy()
        if generated_ts.ndim == 2:
            generated_ts = generated_ts.reshape(
                generated_ts.shape[0], -1, generated_ts.shape[1]
            )

        syn_data_subset = real_data_subset.copy()
        syn_data_subset["timeseries"] = list(generated_ts)

        real_data_inv = dataset.inverse_transform(real_data_subset)
        syn_data_inv = dataset.inverse_transform(syn_data_subset)

        # Convert to numpy arrays
        real_data_array = np.stack(real_data_inv["timeseries"])
        syn_data_array = np.stack(syn_data_inv["timeseries"])

        # Compute metrics
        self.compute_metrics(
            real_data_array, syn_data_array, real_data_subset, writer, data_type
        )

        # Create visualizations
        self.create_visualizations(
            real_data_inv, syn_data_inv, dataset, model, writer, data_type
        )

    def compute_metrics(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        dataset: Any,
        writer: SummaryWriter,
        data_type: str,
    ):
        """
        Compute evaluation metrics and log them.

        Args:
            real_data (np.ndarray): Real data array.
            syn_data (np.ndarray): Synthetic data array.
            dataset (Any): The original dataset.
            writer (SummaryWriter): TensorBoard writer for logging results.
            data_type (str): Label for the data subset ("rare", "non_rare", or "all").
        """
        # DTW
        dtw_mean, dtw_std = dynamic_time_warping_dist(real_data, syn_data)
        writer.add_scalar(f"DTW/{data_type}/mean", dtw_mean)
        writer.add_scalar(f"DTW/{data_type}/std", dtw_std)
        self.metrics["dtw"].append((data_type, dtw_mean, dtw_std))

        # MMD
        mmd_mean, mmd_std = calculate_mmd(real_data, syn_data)
        writer.add_scalar(f"MMD/{data_type}/mean", mmd_mean)
        writer.add_scalar(f"MMD/{data_type}/std", mmd_std)
        self.metrics["mmd"].append((data_type, mmd_mean, mmd_std))

        # MSE
        mse_mean, mse_std = calculate_period_bound_mse(dataset, syn_data)
        writer.add_scalar(f"MSE/{data_type}/mean", mse_mean)
        writer.add_scalar(f"MSE/{data_type}/std", mse_std)
        self.metrics["mse"].append((data_type, mse_mean, mse_std))

        # FID
        fid_score = Context_FID(real_data, syn_data)
        writer.add_scalar(f"FID/{data_type}/score", fid_score)
        self.metrics["fid"].append((data_type, fid_score))

        # Discriminative Score
        discr_score, _, _ = discriminative_score_metrics(real_data, syn_data)
        writer.add_scalar(f"Discriminative/{data_type}/score", discr_score)
        self.metrics["discriminative"].append((data_type, discr_score))

        # Predictive Score
        pred_score = predictive_score_metrics(real_data, syn_data)
        writer.add_scalar(f"Predictive/{data_type}/score", pred_score)
        self.metrics["predictive"].append((data_type, pred_score))

    def create_visualizations(
        self,
        real_data_df: pd.DataFrame,
        syn_data_df: pd.DataFrame,
        dataset: Any,
        model: Any,
        writer: SummaryWriter,
        data_type: str,
        num_samples: int = 100,
        num_runs: int = 3,
    ):
        """
        Create various visualizations for the evaluation results.

        Args:
            real_data_df (pd.DataFrame): Inverse-transformed real data.
            syn_data_df (pd.DataFrame): Inverse-transformed synthetic data.
            dataset (Any): The dataset object.
            model (Any): The trained model.
            writer (SummaryWriter): TensorBoard writer for logging visualizations.
            data_type (str): Label for the data subset ("rare", "non_rare", or "all").
            num_samples (int): Number of samples to generate for visualization.
            num_runs (int): Number of visualization runs.
        """
        for i in range(num_runs):
            # Sample a conditioning variable combination from real data
            sample_row = real_data_df.sample(n=1).iloc[0]
            conditioning_vars_sample = {
                var_name: torch.tensor(
                    [sample_row[var_name]] * num_samples,
                    dtype=torch.long,
                    device=device,
                )
                for var_name in model.categorical_dims.keys()
            }

            generated_samples = model.generate(conditioning_vars_sample).cpu().numpy()
            if generated_samples.ndim == 2:
                generated_samples = generated_samples.reshape(
                    generated_samples.shape[0], -1, generated_samples.shape[1]
                )

            generated_samples_df = pd.DataFrame(
                {
                    var_name: [sample_row[var_name]] * num_samples
                    for var_name in model.categorical_dims.keys()
                }
            )
            generated_samples_df["timeseries"] = list(generated_samples)
            generated_samples_df["dataid"] = sample_row[
                "dataid"
            ]  # required for inverse transform
            generated_samples_df = dataset.inverse_transform(generated_samples_df)

            # Extract month and weekday for plotting
            month = sample_row.get("month", None)
            weekday = sample_row.get("weekday", None)

            # Visualization 1: Plot range with synthetic values
            range_plot = plot_range_with_syn_values(
                real_data_df, generated_samples_df, month, weekday
            )
            writer.add_figure(f"Visualizations/{data_type}/Range_Plot_{i}", range_plot)

            # Visualization 2: Plot closest real signals with synthetic values
            closest_plot = plot_syn_with_closest_real_ts(
                real_data_df, generated_samples_df, month, weekday
            )
            writer.add_figure(
                f"Visualizations/{data_type}/Closest_Real_TS_{i}", closest_plot
            )

        # Visualization 3: KDE plots for real and synthetic data
        real_data_array = np.stack(real_data_df["timeseries"])
        syn_data_array = np.stack(syn_data_df["timeseries"])
        kde_plot = visualization(real_data_array, syn_data_array, "kernel")
        writer.add_figure(f"Visualizations/{data_type}/KDE", kde_plot)

    def get_trained_model(self, dataset: Any) -> Any:
        """
        Get a trained model for the dataset.

        Args:
            dataset: The dataset to train the model on.

        Returns:
            Any: The trained model.
        """
        input_dim = (
            int(dataset.include_generation) + 1
        )  # 2 if generation is included, otherwise 1

        opt = Options(self.model_name)
        opt.input_dim = input_dim

        model_dict = {
            "acgan": ACGAN,
            "diffcharge": DDPM,
            "diffusion_ts": Diffusion_TS,
            # TODO: Add LLMs potentially
        }

        if self.model_name in model_dict:
            model_class = model_dict[self.model_name]
            model = model_class(opt)
        else:
            raise ValueError("Model name not recognized!")

        model.train_model(dataset)
        return model

    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Calculate the mean values for all metrics.

        Returns:
            Dict[str, float]: Mean values for each metric.
        """
        metrics_summary = {}
        for metric_name, values in self.metrics.items():
            metric_values = [value[1] for value in values]  # value[0] is data_type
            metrics_summary[metric_name] = np.mean(metric_values)

        return metrics_summary

    def log_final_results(self):
        """
        Log the final evaluation results.
        """
        metrics_summary = self.get_summary_metrics()
        for metric_name, mean_value in metrics_summary.items():
            self.writer.add_scalar(f"Final_Results/{metric_name}", mean_value)
            print(f"{metric_name}: {mean_value:.4f}")

        self.writer.close()
