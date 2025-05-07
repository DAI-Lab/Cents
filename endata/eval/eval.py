import datetime
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from endata.eval.discriminative_score import discriminative_score_metrics
from endata.eval.eval_metrics import (
    Context_FID,
    calculate_mmd,
    dynamic_time_warping_dist,
)
from endata.eval.predictive_score import predictive_score_metrics
from endata.models.acgan import ACGAN
from endata.models.diffusion_ts import Diffusion_TS
from endata.utils.utils import get_device

logger = logging.getLogger(__name__)


class Evaluator:
    """
    A class for evaluating generative models on time series data.

    This class handles the evaluation process, including metric computation,
    visualization generation, and results storage. It can evaluate models on
    either the entire dataset or specific users.

    Attributes:
        cfg (DictConfig): Configuration for the evaluation process
        real_dataset (Any): The real dataset used for evaluation
        model_name (str): Name of the model being evaluated
        results_dir (str): Directory where evaluation results are stored
        current_results (Dict): Dictionary containing the current evaluation results
    """

    def __init__(
        self, cfg: DictConfig, real_dataset: Any, results_dir: Optional[str] = None
    ):
        """
        Initialize the Evaluator.

        Args:
            cfg (DictConfig): Configuration for the evaluation process
            real_dataset (Any): The real dataset used for evaluation
            results_dir (Optional[str]): Directory to store results. If None, uses default location
        """
        self.real_dataset = real_dataset
        self.cfg = cfg
        self.model_name = cfg.model.name
        self.device = get_device(cfg.get("device", None))

        if results_dir is None:
            results_dir = os.path.join(
                Path.home(),
                ".cache",
                "endata",
                "results",
                self.model_name,
                self.real_dataset.name,
            )
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.current_results = {
            "metrics": {},
            "visualizations": {},
            "metadata": {
                "model_name": self.model_name,
                "dataset_name": self.real_dataset.name,
                "timestamp": datetime.now().isoformat(),
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
        }

    def evaluate_model(
        self,
        user_id: Optional[int] = None,
        model: Optional[Any] = None,
    ) -> Dict:
        """
        Evaluate the model and store results.

        Args:
            user_id (Optional[int]): The ID of the user to evaluate. If None, evaluate on the entire dataset.
            model (Optional[Any]): The model to evaluate. If None, will load or train a model.

        Returns:
            Dict: Dictionary containing the evaluation results
        """
        if user_id is not None:
            dataset = self.real_dataset.create_user_dataset(user_id)
        else:
            dataset = self.real_dataset

        if not model:
            model = self.get_trained_model(dataset)

        model.to(self.device)

        if user_id is not None:
            logger.info(f"[EnData] Starting evaluation for user {user_id}")
        else:
            logger.info("[EnData] Starting evaluation for all users")
        logger.info("----------------------")

        self.run_evaluation(dataset, model)
        self.save_results()

        return self.current_results

    def save_results(self) -> Tuple[str, str]:
        """
        Save the current evaluation results to disk.

        Returns:
            Tuple[str, str]: Paths to the saved results and metadata files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_file = os.path.join(self.results_dir, f"results_{timestamp}.json")
        metadata_file = os.path.join(self.results_dir, f"metadata_{timestamp}.json")

        with open(results_file, "w") as f:
            json.dump(self.current_results["metrics"], f, indent=2)
        with open(metadata_file, "w") as f:
            json.dump(self.current_results["metadata"], f, indent=2)

        logger.info(f"Saved evaluation results to {results_file}")
        return results_file, metadata_file

    def load_results(self, timestamp: Optional[str] = None) -> Dict:
        """
        Load evaluation results from disk.

        Args:
            timestamp (Optional[str]): Specific timestamp to load. If None, loads latest results.

        Returns:
            Dict: Dictionary containing the loaded results
        """
        if timestamp:
            results_file = os.path.join(self.results_dir, f"results_{timestamp}.json")
            metadata_file = os.path.join(self.results_dir, f"metadata_{timestamp}.json")
        else:
            # Get latest results
            result_files = glob.glob(os.path.join(self.results_dir, "results_*.json"))
            if not result_files:
                raise FileNotFoundError(f"No results found in {self.results_dir}")
            results_file = max(result_files)
            metadata_file = results_file.replace("results_", "metadata_")

        with open(results_file, "r") as f:
            metrics = json.load(f)
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        return {"metrics": metrics, "metadata": metadata}

    def compute_metrics(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        real_data_frame: pd.DataFrame,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute evaluation metrics and store them in current_results.

        Args:
            real_data (np.ndarray): Real data array (shape: [N, seq_len, dims])
            syn_data (np.ndarray): Synthetic data array (shape: [N, seq_len, dims])
            real_data_frame (pd.DataFrame): Real data subset (inverse-transformed)
            mask (Optional[np.ndarray]): Boolean array indicating which rows are "rare"
        """
        logger.info(f"--- Starting Full-Subset Metrics ---")

        metrics = {}

        # Compute and store metrics
        dtw_mean, dtw_std = dynamic_time_warping_dist(real_data, syn_data)
        metrics["DTW"] = {"mean": dtw_mean, "std": dtw_std}
        logger.info(f"[EnData] DTW completed")

        mmd_mean, mmd_std = calculate_mmd(real_data, syn_data)
        metrics["MMD"] = {"mean": mmd_mean, "std": mmd_std}
        logger.info(f"[EnData] MMD completed")

        fid_score = Context_FID(real_data, syn_data)
        metrics["Context_FID"] = fid_score
        logger.info(f"[EnData] Context-FID completed")

        discr_score, _, _ = discriminative_score_metrics(real_data, syn_data)
        metrics["Disc_Score"] = discr_score
        logger.info(f"[EnData] Discr Score completed")

        pred_score = predictive_score_metrics(real_data, syn_data)
        metrics["Pred_Score"] = pred_score
        logger.info(f"[EnData] Pred Score completed")

        self.current_results["metrics"] = metrics

        if mask is not None:
            logger.info("[EnData] Starting Rare-Subset Metrics")
            rare_metrics = {}
            rare_real_data = real_data[mask]
            rare_syn_data = syn_data[mask]
            rare_real_df = real_data_frame[mask].reset_index(drop=True)

            dtw_mean_r, dtw_std_r = dynamic_time_warping_dist(
                rare_real_data, rare_syn_data
            )
            rare_metrics["DTW"] = {"mean": dtw_mean_r, "std": dtw_std_r}
            logger.info(f"[EnData] DTW completed")

            mmd_mean_r, mmd_std_r = calculate_mmd(rare_real_data, rare_syn_data)
            rare_metrics["MMD"] = {"mean": mmd_mean_r, "std": mmd_std_r}
            logger.info(f"[EnData] MMD completed")

            mse_mean_r, mse_std_r = calculate_period_bound_mse(
                rare_real_df, rare_syn_data
            )
            rare_metrics["MSE"] = {"mean": mse_mean_r, "std": mse_std_r}
            logger.info(f"[EnData] BMSE completed")

            fid_score_r = Context_FID(rare_real_data, rare_syn_data)
            rare_metrics["Context_FID"] = fid_score_r
            logger.info(f"[EnData] Context-FID completed")

            discr_score_r, _, _ = discriminative_score_metrics(
                rare_real_data, rare_syn_data
            )
            rare_metrics["Disc_Score"] = discr_score_r
            logger.info(f"[EnData] Discr Score completed")

            pred_score_r = predictive_score_metrics(rare_real_data, rare_syn_data)
            rare_metrics["Pred_Score"] = pred_score_r
            logger.info(f"[EnData] Pred Score completed")

            logger.info("[EnData] Done computing Rare-Subset Metrics.")
            metrics["rare_subset"] = rare_metrics

    def get_trained_model(self, dataset: Any) -> Any:
        model_dict = {
            "acgan": ACGAN,
            "diffusion_ts": Diffusion_TS,
        }
        if self.model_name in model_dict:
            model_class = model_dict[self.model_name]
            model = model_class(self.cfg)
        else:
            raise ValueError("Model name not recognized!")
        if self.cfg.model_ckpt is not None:
            model.load(self.cfg.model_ckpt)
        else:
            model.train_model(dataset)
        return model

    def run_evaluation(self, dataset: Any, model: Any):
        """
        Run the evaluation process.

        Args:
            dataset: The dataset to evaluate.
            model: The trained model.
        """
        logger.info(
            f"Starting evaluation of {self.model_name} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
        )
        all_indices = dataset.data.index.to_numpy()
        self.evaluate_subset(dataset, model, all_indices)

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
        dataset.data = dataset.get_combined_rarity()
        real_data_subset = dataset.data.iloc[indices].reset_index(drop=True)
        context_vars = {
            name: torch.tensor(
                real_data_subset[name].values, dtype=torch.long, device=self.device
            )
            for name in dataset.context_vars
        }

        generated_ts = model.generate(context_vars).cpu().numpy()
        if generated_ts.ndim == 2:
            generated_ts = generated_ts.reshape(
                generated_ts.shape[0], -1, generated_ts.shape[1]
            )

        syn_data_subset = real_data_subset.copy()
        syn_data_subset["timeseries"] = list(generated_ts)

        real_data_inv = dataset.inverse_transform(real_data_subset)
        syn_data_inv = dataset.inverse_transform(syn_data_subset)

        real_data_array = np.stack(real_data_inv["timeseries"])
        syn_data_array = np.stack(syn_data_inv["timeseries"])

        if self.cfg.evaluator.eval_metrics:
            rare_mask = None

            if (
                self.cfg.evaluator.eval_context_sparse
                and "is_rare" in real_data_subset.columns
            ):
                rare_mask = real_data_subset["is_rare"].values

            self.compute_metrics(
                real_data_array, syn_data_array, real_data_inv, rare_mask
            )
