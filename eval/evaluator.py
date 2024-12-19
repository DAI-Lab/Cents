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
from sklearn.metrics import f1_score, precision_score, recall_score

from eval.discriminative_metric import discriminative_score_metrics
from eval.metrics import (
    Context_FID,
    calculate_mmd,
    calculate_period_bound_mse,
    dynamic_time_warping_dist,
    plot_syn_and_real_comparison,
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
            for name in model.conditioning_var_n_categories.keys()
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
            for name in model.conditioning_var_n_categories.keys()
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
        # self.compute_metrics(real_data_array, syn_data_array, real_data_inv)

        # Generate plots
        # self.create_visualizations(real_data_inv, syn_data_inv, dataset, model)

        # Evaluate conditioning module rarity predictions
        self.evaluate_conditioning_module(model)

    def compute_metrics(
        self, real_data: np.ndarray, syn_data: np.ndarray, real_data_frame: pd.DataFrame
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
        mse_mean, mse_std = calculate_period_bound_mse(real_data_frame, syn_data)
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
        num_runs: int = 10,
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
            # Randomly select a sample from real_data_df
            sample_index = np.random.randint(low=0, high=real_data_df.shape[0])
            sample_row = real_data_df.iloc[sample_index]

            # Prepare conditioning variables for the model
            conditioning_vars_sample = {
                var_name: torch.tensor(
                    [sample_row[var_name]] * num_samples,
                    dtype=torch.long,
                    device=device,
                )
                for var_name in model.conditioning_var_n_categories.keys()
            }

            # Generate synthetic samples
            generated_samples = model.generate(conditioning_vars_sample).cpu().numpy()
            if generated_samples.ndim == 2:
                generated_samples = generated_samples.reshape(
                    generated_samples.shape[0], -1, generated_samples.shape[1]
                )

            # Create DataFrame for generated samples
            generated_samples_df = pd.DataFrame(
                {
                    var_name: [sample_row[var_name]] * num_samples
                    for var_name in model.conditioning_var_n_categories.keys()
                }
            )
            generated_samples_df["timeseries"] = list(generated_samples)
            generated_samples_df["dataid"] = sample_row["dataid"]

            # Check and fill missing normalization group keys
            normalization_keys = dataset.normalization_group_keys
            missing_keys = [
                key
                for key in normalization_keys
                if key not in generated_samples_df.columns
            ]

            if missing_keys:
                logger.warning(
                    f"Missing normalization group keys: {missing_keys}. Filling with sample_row values."
                )
                for key in missing_keys:
                    if key in sample_row:
                        generated_samples_df[key] = sample_row[key]
                        logger.info(
                            f"Filled missing key '{key}' with value '{sample_row[key]}'."
                        )
                    else:
                        raise ValueError(
                            f"Sample row does not contain required key: '{key}'."
                        )

            # Perform inverse transformation now that all keys are present
            generated_samples_df = dataset.inverse_transform(generated_samples_df)

            # Extract conditioning vars for visualization
            cond_vars_for_vis = {
                name: tensor[0].item()
                for name, tensor in conditioning_vars_sample.items()
            }

            # Visualization: Combined range and closest real time series plot
            comparison_plot = plot_syn_and_real_comparison(
                real_data_df, generated_samples_df, cond_vars_for_vis
            )
            if comparison_plot is not None:
                wandb.log({f"Comparison_Plot_{i}": wandb.Image(comparison_plot)})

        # Visualization 3: KDE plots for real and synthetic data
        real_data_array = np.stack(real_data_df["timeseries"])
        syn_data_array = np.stack(syn_data_df["timeseries"])
        kde_plots = visualization(real_data_array, syn_data_array, "kernel")
        if kde_plots is not None:
            for i, plot in enumerate(kde_plots):
                wandb.log({f"KDE_Dim_{i}": wandb.Image(plot)})

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

    def evaluate_conditioning_module(self, model: Any) -> Dict[str, float]:
        """
        Evaluate the computed rarities of conditional embeddings against frequency based ground truth rarity.
        """
        comb_rarity_df = self.real_dataset.get_conditioning_var_combination_rarities(
            coverage_threshold=0.8
        )
        conditioning_vars_list = list(self.cfg.dataset.conditioning_vars.keys())

        true_labels = []
        pred_labels = []

        for _, row in comb_rarity_df.iterrows():

            conditioning_vars = {
                var_name: torch.tensor(
                    [row[var_name]], dtype=torch.long, device=self.cfg.device
                )
                for var_name in conditioning_vars_list
            }

            with torch.no_grad():
                z, mu, logvar = model.conditioning_module(
                    conditioning_vars, sample=False
                )

                # Determine predicted rarity (True/False)
                pred_rare_mask = model.conditioning_module.is_rare(mu)
                pred_rare = pred_rare_mask.item()

            true_rare = bool(row["rare"])
            true_labels.append(true_rare)
            pred_labels.append(pred_rare)

        # 4. Compute classification metrics
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)

        metrics = {"precision": precision, "recall": recall, "f1_score": f1}

        wandb.log(
            {"Rarity_Precision": precision, "Rarity_Recall": recall, "Rarity_F1": f1}
        )
        return metrics
