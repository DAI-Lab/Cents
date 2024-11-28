import datetime
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

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
from generator.diffcharge.diffusion import DDPM
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.gan.acgan import ACGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class Evaluator:
    """
    A class for evaluating generative models on time series data.
    """

    def __init__(self, cfg: DictConfig, real_dataset: Any):
        """
        Initialize the Evaluator.

        Args:
            real_dataset: The real dataset used for evaluation.
            model_name (str): The name of the generative model being evaluated.
            log_dir (str): Base directory for storing TensorBoard logs.
        """
        self.real_dataset = real_dataset
        self.cfg = cfg
        self.model_name = cfg.model.name
        self.metrics: Dict[str, List] = {
            "dtw": [],
            "mmd": [],
            "mse": [],
            "fid": [],
            "discriminative": [],
            "predictive": [],
        }

        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            dir=cfg.run_dir,
        )

    def evaluate_model(
        self,
        user_id: int = None,
        distinguish_rare: bool = False,
        model: Any = None,
    ):
        """
        Evaluate the model.

        Args:
            user_id (int, optional): The ID of the user to evaluate. If None, evaluate on the entire dataset.
            distinguish_rare (bool): Whether to distinguish between rare and non-rare data samples.
        """
        if user_id is not None:
            dataset = self.real_dataset.create_user_dataset(user_id)
        else:
            dataset = self.real_dataset

        if not model:
            model = self.get_trained_model(dataset)

        if user_id is not None:
            logger.info(f"Starting evaluation for user {user_id}")
        else:
            logger.info("Starting evaluation for all users")
        logger.info("----------------------")

        # Pass data_label to run_evaluation
        self.run_evaluation(dataset, model, distinguish_rare)

    def run_evaluation(self, dataset: Any, model: Any, distinguish_rare: bool):
        """
        Run the evaluation process.

        Args:
            dataset: The dataset to evaluate.
            model: The trained model.
            distinguish_rare (bool): Whether to distinguish between rare and non-rare data samples.
        """
        if distinguish_rare:
            rare_indices, non_rare_indices = self.identify_rare_combinations(
                dataset, model
            )
            self.evaluate_subset(dataset, model, rare_indices, data_type="rare")
            self.evaluate_subset(
                dataset,
                model,
                non_rare_indices,
            )
        else:
            all_indices = dataset.data.index.to_numpy()
            self.evaluate_subset(dataset, model, all_indices)

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
            embeddings = model.conditioning_module(conditioning_vars)
            mahalanobis_distances = (
                model.conditioning_module.compute_mahalanobis_distance(embeddings)
            )

        rarity_threshold = torch.quantile(mahalanobis_distances, 0.8)
        rare_mask = mahalanobis_distances > rarity_threshold

        rare_indices = np.where(rare_mask.cpu().numpy())[0]
        non_rare_indices = np.where(~rare_mask.cpu().numpy())[0]

        return rare_indices, non_rare_indices

    def evaluate_subset(
        self,
        dataset: Any,
        model: Any,
        indices: np.ndarray,
    ):
        """
        Evaluate the model on a subset of the data.

        Args:
            dataset: The dataset containing real data.
            model: The trained model to generate data.
            indices (np.ndarray): Indices of data to use.
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
        self.compute_metrics(real_data_array, syn_data_array, real_data_subset)

        # Generate plots
        self.create_visualizations(real_data_inv, syn_data_inv, dataset, model)

    def compute_metrics(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        real_data_subset: pd.DataFrame,
    ):
        """
        Compute evaluation metrics and log them.

        Args:
            real_data (np.ndarray): Real data array.
            syn_data (np.ndarray): Synthetic data array.
            real_data_subset (pd.DataFrame): Real data subset DataFrame.
        """
        # DTW
        logger.info(f"--- Starting DTW distance computation ---")
        dtw_mean, dtw_std = dynamic_time_warping_dist(real_data, syn_data)
        wandb.log({"DTW/mean": dtw_mean, "DTW/std": dtw_std})
        self.metrics["dtw"].append((dtw_mean, dtw_std))
        logger.info(f"--- DTW distance computation complete ---")
        logger.info("----------------------")

        # MMD
        logger.info(f"--- Starting MMD computation ---")
        mmd_mean, mmd_std = calculate_mmd(real_data, syn_data)
        wandb.log({"MMD/mean": mmd_mean, "MMD/std": mmd_std})
        self.metrics["mmd"].append((mmd_mean, mmd_std))
        logger.info(f"--- MMD computation complete ---")
        logger.info("----------------------")

        # MSE
        logger.info(f"--- Starting Bounded MSE computation ---")
        mse_mean, mse_std = calculate_period_bound_mse(real_data_subset, syn_data)
        wandb.log({"MSE/mean": mse_mean, "MSE/std": mse_std})
        self.metrics["mse"].append((mse_mean, mse_std))
        logger.info(f"--- Bounded MSE computation complete ---")
        logger.info("----------------------")

        # FID
        logger.info(f"--- Starting Context FID computation ---")
        fid_score = Context_FID(real_data, syn_data)
        wandb.log({"Context_FID": fid_score})
        self.metrics["fid"].append(fid_score)
        logger.info(f"--- Context FID computation complete ---")
        logger.info("----------------------")

        # Discriminative Score
        logger.info(f"--- Starting Discriminative Score computation ---")
        discr_score, _, _ = discriminative_score_metrics(real_data, syn_data)
        wandb.log({"Disc_Score": discr_score})
        self.metrics["discriminative"].append(discr_score)
        logger.info(f"--- Discriminative Score computation complete ---")
        logger.info("----------------------")

        # Predictive Score
        logger.info(f"--- Starting Predictive Score computation ---")
        pred_score = predictive_score_metrics(real_data, syn_data)
        wandb.log({"Pred_Score": pred_score})
        self.metrics["predictive"].append(pred_score)
        logger.info(f"--- Predictive Score computation complete ---")
        logger.info("----------------------")

    def create_visualizations(
        self,
        real_data_df: pd.DataFrame,
        syn_data_df: pd.DataFrame,
        dataset: Any,
        model: Any,
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
            num_samples (int): Number of samples to generate for visualization.
            num_runs (int): Number of visualization runs.
        """
        logger.info(f"--- Starting Visualizations ---")
        for i in range(num_runs):
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
            if range_plot is not None:
                wandb.log({f"Range_Plot_{i}": wandb.Image(range_plot)})

            # Visualization 2: Plot closest real signals with synthetic values
            closest_plot = plot_syn_with_closest_real_ts(
                real_data_df, generated_samples_df, month, weekday
            )
            if closest_plot is not None:
                wandb.log({f"Closest_Real_TS_{i}": wandb.Image(closest_plot)})

        # Visualization 3: KDE plots for real and synthetic data
        real_data_array = np.stack(real_data_df["timeseries"])
        syn_data_array = np.stack(syn_data_df["timeseries"])
        kde_plot = visualization(real_data_array, syn_data_array, "kernel")
        if kde_plot is None:
            wandb.log({f"KDE": wandb.Image(kde_plot)})
        logger.info(f"--- Visualizations complete! ---")

    def get_trained_model(self, dataset: Any) -> Any:
        """
        Get a trained model for the dataset.

        Args:
            dataset: The dataset to train the model on.

        Returns:
            Any: The trained model.
        """
        model_dict = {
            "acgan": ACGAN,
            "diffcharge": DDPM,
            "diffusion_ts": Diffusion_TS,
        }

        if self.model_name in model_dict:
            model_class = model_dict[self.model_name]
            model = model_class(self.cfg)
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
            if values:
                if isinstance(values[0], tuple):
                    # For metrics with mean and std
                    metric_values = [value[0] for value in values]  # value[0] is mean
                else:
                    metric_values = values
                metrics_summary[metric_name] = np.mean(metric_values)
            else:
                metrics_summary[metric_name] = None

        return metrics_summary
