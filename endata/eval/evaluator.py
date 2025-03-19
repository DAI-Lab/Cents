import datetime
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from endata.eval.discriminative_metric import discriminative_score_metrics
from endata.eval.metrics import (
    Context_FID,
    calculate_mmd,
    calculate_period_bound_mse,
    dynamic_time_warping_dist,
    plot_syn_and_real_comparison,
    visualization,
)
from endata.eval.predictive_metric import predictive_score_metrics
from endata.generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from endata.generator.gan.acgan import ACGAN
from endata.utils.device import get_device

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 20,
    }
)


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
            logger.info(f"Starting evaluation for user {user_id}")
        else:
            logger.info("Starting evaluation for all users")
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
        logger.info(f"--- DTW completed ---")

        mmd_mean, mmd_std = calculate_mmd(real_data, syn_data)
        metrics["MMD"] = {"mean": mmd_mean, "std": mmd_std}
        logger.info(f"--- MMD completed ---")

        # mse_mean, mse_std = calculate_period_bound_mse(real_data_frame, syn_data)
        # metrics["MSE"] = {"mean": mse_mean, "std": mse_std}
        # logger.info(f"--- BMSE completed ---")

        fid_score = Context_FID(real_data, syn_data)
        metrics["Context_FID"] = fid_score
        logger.info(f"--- Context-FID completed ---")

        discr_score, _, _ = discriminative_score_metrics(real_data, syn_data)
        metrics["Disc_Score"] = discr_score
        logger.info(f"--- Discr Score completed ---")

        pred_score = predictive_score_metrics(real_data, syn_data)
        metrics["Pred_Score"] = pred_score
        logger.info(f"--- Pred Score completed ---")

        self.current_results["metrics"] = metrics

        if mask is not None:
            logger.info("--- Starting Rare-Subset Metrics ---")
            rare_metrics = {}
            rare_real_data = real_data[mask]
            rare_syn_data = syn_data[mask]
            rare_real_df = real_data_frame[mask].reset_index(drop=True)

            dtw_mean_r, dtw_std_r = dynamic_time_warping_dist(
                rare_real_data, rare_syn_data
            )
            rare_metrics["DTW"] = {"mean": dtw_mean_r, "std": dtw_std_r}
            logger.info(f"--- DTW completed ---")

            mmd_mean_r, mmd_std_r = calculate_mmd(rare_real_data, rare_syn_data)
            rare_metrics["MMD"] = {"mean": mmd_mean_r, "std": mmd_std_r}
            logger.info(f"--- MMD completed ---")

            mse_mean_r, mse_std_r = calculate_period_bound_mse(
                rare_real_df, rare_syn_data
            )
            rare_metrics["MSE"] = {"mean": mse_mean_r, "std": mse_std_r}
            logger.info(f"--- BMSE completed ---")

            fid_score_r = Context_FID(rare_real_data, rare_syn_data)
            rare_metrics["Context_FID"] = fid_score_r
            logger.info(f"--- Context-FID completed ---")

            discr_score_r, _, _ = discriminative_score_metrics(
                rare_real_data, rare_syn_data
            )
            rare_metrics["Disc_Score"] = discr_score_r
            logger.info(f"--- Discr Score completed ---")

            pred_score_r = predictive_score_metrics(rare_real_data, rare_syn_data)
            rare_metrics["Pred_Score"] = pred_score_r
            logger.info(f"--- Pred Score completed ---")

            logger.info("Done computing Rare-Subset Metrics.")
            metrics["rare_subset"] = rare_metrics

    def create_visualizations(
        self,
        real_data_df: pd.DataFrame,
        syn_data_df: pd.DataFrame,
        dataset: Any,
        model: Any,
        num_samples: int = 100,
        num_runs: int = 1,
    ) -> Dict:
        """
        Create and store visualizations of the generated data.

        Args:
            real_data_df (pd.DataFrame): DataFrame containing real data
            syn_data_df (pd.DataFrame): DataFrame containing synthetic data
            dataset (Any): The dataset object
            model (Any): The trained model
            num_samples (int): Number of samples to generate per run
            num_runs (int): Number of visualization runs to perform

        Returns:
            Dict: Dictionary containing the generated visualizations
        """
        visualizations = {}

        real_data_array = np.stack(real_data_df["timeseries"])
        _, seq_len, dim = real_data_array.shape
        for i in range(num_runs):
            sample_index = np.random.randint(low=0, high=real_data_df.shape[0])
            sample_row = real_data_df.iloc[sample_index]
            context_vars_sample = {
                var_name: torch.tensor(
                    [sample_row[var_name]] * num_samples,
                    dtype=torch.long,
                    device=self.device,
                )
                for var_name in model.context_var_n_categories.keys()
            }
            generated_samples = model.generate(context_vars_sample).cpu().numpy()
            if generated_samples.ndim == 2:
                generated_samples = generated_samples.reshape(
                    generated_samples.shape[0], -1, generated_samples.shape[1]
                )
            generated_samples_df = pd.DataFrame(
                {
                    var_name: [sample_row[var_name]] * num_samples
                    for var_name in model.context_var_n_categories.keys()
                }
            )
            generated_samples_df["timeseries"] = list(generated_samples)
            # generated_samples_df["dataid"] = sample_row["dataid"]
            normalization_keys = (
                dataset.normalization_group_keys
                if hasattr(dataset, "normalization_group_keys")
                else []
            )
            missing_keys = [
                key
                for key in normalization_keys
                if key not in generated_samples_df.columns
            ]
            if missing_keys:
                for key in missing_keys:
                    if key in sample_row:
                        generated_samples_df[key] = sample_row[key]
                    else:
                        raise ValueError(
                            f"Sample row does not contain required key: '{key}'."
                        )
            generated_samples_df = dataset.inverse_transform(generated_samples_df)
            range_fig, closest_fig = plot_syn_and_real_comparison(
                real_data_df,
                generated_samples_df,
                context_vars_sample,
                dimension=0,
            )
            if range_fig is not None:
                visualizations[f"RangePlot_{i}"] = range_fig
                plt.close(range_fig)
            if closest_fig is not None:
                visualizations[f"ClosestPlot_{i}"] = closest_fig
                plt.close(closest_fig)
            if dim > 1:
                syn_sample_0 = generated_samples[0, :, 0]
                real_sample_0 = (
                    sample_row["timeseries"][:, 0]
                    if sample_row["timeseries"].ndim == 2
                    else sample_row["timeseries"]
                )
                syn_sample_1 = generated_samples[0, :, 1]
                real_sample_1 = (
                    sample_row["timeseries"][:, 1]
                    if sample_row["timeseries"].ndim == 2
                    else None
                )
                fig_multi, axes_multi = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
                axes_multi[0].plot(syn_sample_0, label="Synthetic", color="blue")
                axes_multi[0].plot(real_sample_0, label="Real", color="red")
                font_size = 12
                axes_multi[0].tick_params(
                    axis="both", which="major", labelsize=font_size
                )
                axes_multi[0].set_xlabel("Timestep", fontsize=font_size)
                axes_multi[0].set_ylabel("kWh", fontsize=font_size)
                leg_0 = axes_multi[0].legend()
                leg_0.prop.set_size(font_size)
                if real_sample_1 is not None:
                    axes_multi[1].plot(syn_sample_1, label="Synthetic", color="blue")
                    axes_multi[1].plot(real_sample_1, label="Real", color="red")
                else:
                    axes_multi[1].plot(syn_sample_1, label="Synthetic", color="blue")
                axes_multi[1].tick_params(
                    axis="both", which="major", labelsize=font_size
                )
                axes_multi[1].set_xlabel("Timestep", fontsize=font_size)
                axes_multi[1].set_ylabel("kWh", fontsize=font_size)
                leg_1 = axes_multi[1].legend()
                leg_1.prop.set_size(font_size)
                visualizations[f"MultiDim_Chart_{i}"] = fig_multi
                plt.close(fig_multi)
        syn_data_array = np.stack(syn_data_df["timeseries"])
        kde_plots = visualization(real_data_array, syn_data_array, "kernel")
        tsne_plots = visualization(real_data_array, syn_data_array, "tsne")
        if kde_plots is not None:
            for i, plot in enumerate(kde_plots):
                visualizations[f"KDE_Dim_{i}"] = plot
        if tsne_plots is not None:
            for i, plot in enumerate(tsne_plots):
                visualizations[f"TSNE_Dim_{i}"] = plot

        self.current_results["visualizations"] = visualizations

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
            for name in model.context_var_n_categories.keys()
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

        if self.cfg.evaluator.eval_pv_shift:
            self.evaluate_pv_shift(dataset=dataset, model=model)

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

        if self.cfg.evaluator.eval_vis:
            self.create_visualizations(real_data_inv, syn_data_inv, dataset, model)

    def evaluate_pv_shift(self, dataset: Any, model: Any):
        avg_shift = dataset.compute_average_pv_shift()
        if avg_shift is None or np.allclose(avg_shift, 0.0):
            return
        test_contexts = dataset.sample_shift_test_contexts()
        n_sampled = len(test_contexts)
        n_pv1_missing = sum(1 for c in test_contexts if c["missing_pv"] == 1)
        n_pv0_missing = sum(1 for c in test_contexts if c["missing_pv"] == 0)

        print(f"[Shift Contexts] Sampled: {n_sampled}.")
        print(f"[Shift Contexts] PV=1 is missing in {n_pv1_missing} of these contexts.")
        print(f"[Shift Contexts] PV=0 is missing in {n_pv0_missing} of these contexts.")
        if len(test_contexts) == 0:
            return
        present_ctx_list = []
        missing_ctx_list = []
        present_pv_values = []
        for cinfo in test_contexts:
            base_ctx = cinfo["base_context"]
            present_pv = cinfo["present_pv"]
            missing_pv = cinfo["missing_pv"]
            ctx_p = dict(base_ctx)
            ctx_m = dict(base_ctx)
            ctx_p["has_solar"] = present_pv
            ctx_m["has_solar"] = missing_pv
            present_ctx_list.append(ctx_p)
            missing_ctx_list.append(ctx_m)
            present_pv_values.append(present_pv)
        present_ctx_tensors = {}
        missing_ctx_tensors = {}
        all_keys = present_ctx_list[0].keys()
        for k in all_keys:
            present_ctx_tensors[k] = torch.tensor(
                [pc[k] for pc in present_ctx_list], dtype=torch.long, device=self.device
            )
            missing_ctx_tensors[k] = torch.tensor(
                [mc[k] for mc in missing_ctx_list], dtype=torch.long, device=self.device
            )
        with torch.no_grad():
            syn_ts_present = model.generate(present_ctx_tensors)
            syn_ts_missing = model.generate(missing_ctx_tensors)
        syn_ts_present = syn_ts_present.cpu().numpy()
        syn_ts_missing = syn_ts_missing.cpu().numpy()
        if syn_ts_present.ndim == 3 and syn_ts_present.shape[-1] == 1:
            syn_ts_present = syn_ts_present[:, :, 0]
            syn_ts_missing = syn_ts_missing[:, :, 0]
        shifts = []
        for i, pv_val in enumerate(present_pv_values):
            shift_i = syn_ts_missing[i] - syn_ts_present[i]
            if pv_val == 1:
                shift_i = -shift_i
            shifts.append(shift_i)
        shifts = np.array(shifts)
        avg_shift = np.asarray(avg_shift).reshape(-1)
        l2_values = []
        for i in range(shifts.shape[0]):
            diff = shifts[i] - avg_shift
            l2 = np.sqrt((diff**2).sum())
            l2_values.append(l2)
        mean_l2 = np.mean(l2_values)
        wandb.log({"Shift_L2": mean_l2})

        def find_context_matched_shift(dataset, cinfo):
            base_ctx = cinfo["base_context"]
            city_val = base_ctx.get("city", None)
            btype_val = base_ctx.get("building_type", None)
            df = dataset.data.copy()
            mask = pd.Series([True] * len(df))
            if city_val is not None and "city" in df.columns:
                mask = mask & (df["city"] == city_val)
            if btype_val is not None and "building_type" in df.columns:
                mask = mask & (df["building_type"] == btype_val)
            df_matched = df[mask]
            if df_matched.empty:
                return None
            df_pv0 = df_matched[df_matched["has_solar"] == 0]
            df_pv1 = df_matched[df_matched["has_solar"] == 1]
            if df_pv0.empty or df_pv1.empty:
                return None
            ts_pv0 = np.stack(df_pv0["timeseries"].values, axis=0)
            ts_pv1 = np.stack(df_pv1["timeseries"].values, axis=0)
            mean_pv0 = ts_pv0.mean(axis=0)
            mean_pv1 = ts_pv1.mean(axis=0)
            mean_pv0_dim0 = mean_pv0[:, 0]
            mean_pv1_dim0 = mean_pv1[:, 0]
            real_shift = mean_pv1_dim0 - mean_pv0_dim0
            return real_shift

        matched_shifts = []
        for cinfo in test_contexts:
            matched = find_context_matched_shift(dataset, cinfo)
            matched_shifts.append(matched)
        n_plots = min(6, shifts.shape[0])
        for j, idx in enumerate(
            np.random.choice(shifts.shape[0], size=n_plots, replace=False)
        ):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(avg_shift, label="Real shift", color="red")
            ax.plot(shifts[idx], label="Synthetic shift", color="blue", linestyle="--")
            matched_s = matched_shifts[idx]
            if matched_s is not None:
                ax.plot(
                    matched_s,
                    label="Context-matched shift",
                    color="green",
                    linestyle=":",
                )
            font_size = 12
            ax.tick_params(axis="both", which="major", labelsize=font_size)
            ax.set_xlabel("Timestep", fontsize=font_size)
            ax.set_ylabel("kWh", fontsize=font_size)
            leg = ax.legend()
            leg.prop.set_size(font_size)
            fig.tight_layout()
            wandb.log({f"ShiftPlot_{j}": wandb.Image(fig)})
            plt.close(fig)
