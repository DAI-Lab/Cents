import datetime
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.model.name}_{real_dataset.name}_{cfg.dataset.user_group}_cond_weight_{cfg.model.cond_loss_weight}_{cfg.model.n_epochs}_{timestamp}"

        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            dir=cfg.run_dir,
            name=run_name,
        )

    def evaluate_model(
        self,
        user_id: int = None,
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
        self.run_evaluation(dataset, model)

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
        conditioning_vars = {
            name: torch.tensor(
                real_data_subset[name].values, dtype=torch.long, device=device
            )
            for name in model.context_var_n_categories.keys()
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

        if self.cfg.evaluator.eval_pv_shift:
            self.evaluate_pv_shift(dataset=dataset, model=model)

        if self.cfg.evaluator.eval_metrics:
            rare_mask = None

            if self.cfg.evaluator.eval_cond and "is_rare" in real_data_subset.columns:
                rare_mask = real_data_subset["is_rare"].values

            self.compute_metrics(
                real_data_array, syn_data_array, real_data_inv, rare_mask
            )

        if self.cfg.evaluator.eval_vis:
            self.create_visualizations(real_data_inv, syn_data_inv, dataset, model)

        if self.cfg.evaluator.eval_gmm:
            self.evaluate_embedding_space_with_gmm(dataset, model)

    def compute_metrics(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        real_data_frame: pd.DataFrame,
        mask: np.ndarray = None,
    ):
        """
        Compute evaluation metrics and log them.

        Args:
            real_data (np.ndarray): Real data array (shape: [N, seq_len, dims]).
            syn_data (np.ndarray): Synthetic data array (shape: [N, seq_len, dims]).
            real_data_frame (pd.DataFrame): Real data subset (inverse-transformed).
            mask (np.ndarray, optional): Boolean array of shape (N,) indicating which rows are "rare".
                                        If provided, we also compute metrics for that subset.
        """
        logger.info(f"--- Starting Full-Subset Metrics ---")

        dtw_mean, dtw_std = dynamic_time_warping_dist(real_data, syn_data)
        wandb.log({"DTW/mean": dtw_mean, "DTW/std": dtw_std})
        logger.info(f"--- DTW completed ---")

        mmd_mean, mmd_std = calculate_mmd(real_data, syn_data)
        wandb.log({"MMD/mean": mmd_mean, "MMD/std": mmd_std})
        logger.info(f"--- MMD completed ---")

        mse_mean, mse_std = calculate_period_bound_mse(real_data_frame, syn_data)
        wandb.log({"MSE/mean": mse_mean, "MSE/std": mse_std})
        logger.info(f"--- BMSE completed ---")

        fid_score = Context_FID(real_data, syn_data)
        wandb.log({"Context_FID": fid_score})
        logger.info(f"--- Context-FID completed ---")

        discr_score, _, _ = discriminative_score_metrics(real_data, syn_data)
        wandb.log({"Disc_Score": discr_score})
        logger.info(f"--- Discr Score completed ---")

        pred_score = predictive_score_metrics(real_data, syn_data)
        wandb.log({"Pred_Score": pred_score})
        logger.info(f"--- Pred Score completed ---")

        if mask is not None:
            logger.info("--- Starting Rare-Subset Metrics ---")

            rare_real_data = real_data[mask]
            rare_syn_data = syn_data[mask]
            rare_real_df = real_data_frame[mask].reset_index(drop=True)

            dtw_mean_r, dtw_std_r = dynamic_time_warping_dist(
                rare_real_data, rare_syn_data
            )
            wandb.log({"DTW_rare/mean": dtw_mean_r, "DTW_rare/std": dtw_std_r})
            logger.info(f"--- DTW completed ---")

            mmd_mean_r, mmd_std_r = calculate_mmd(rare_real_data, rare_syn_data)
            wandb.log({"MMD_rare/mean": mmd_mean_r, "MMD_rare/std": mmd_std_r})
            logger.info(f"--- MMD completed ---")

            mse_mean_r, mse_std_r = calculate_period_bound_mse(
                rare_real_df, rare_syn_data
            )
            wandb.log({"MSE_rare/mean": mse_mean_r, "MSE_rare/std": mse_std_r})
            logger.info(f"--- BMSE completed ---")

            fid_score_r = Context_FID(rare_real_data, rare_syn_data)
            wandb.log({"Context_FID_rare": fid_score_r})
            logger.info(f"--- Context-FID completed ---")

            discr_score_r, _, _ = discriminative_score_metrics(
                rare_real_data, rare_syn_data
            )
            wandb.log({"Disc_Score_rare": discr_score_r})
            logger.info(f"--- Discr Score completed ---")

            pred_score_r = predictive_score_metrics(rare_real_data, rare_syn_data)
            wandb.log({"Pred_Score_rare": pred_score_r})
            logger.info(f"--- Pred Score completed ---")

            logger.info("Done computing Rare-Subset Metrics.")

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
            sample_index = np.random.randint(low=0, high=real_data_df.shape[0])
            sample_row = real_data_df.iloc[sample_index]

            conditioning_vars_sample = {
                var_name: torch.tensor(
                    [sample_row[var_name]] * num_samples,
                    dtype=torch.long,
                    device=device,
                )
                for var_name in model.context_var_n_categories.keys()
            }

            generated_samples = model.generate(conditioning_vars_sample).cpu().numpy()
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
            generated_samples_df["dataid"] = sample_row["dataid"]

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

            generated_samples_df = dataset.inverse_transform(generated_samples_df)
            cond_vars_for_vis = {
                name: tensor[0].item()
                for name, tensor in conditioning_vars_sample.items()
            }

            comparison_plot = plot_syn_and_real_comparison(
                real_data_df, generated_samples_df, cond_vars_for_vis
            )
            if comparison_plot is not None:
                wandb.log({f"Comparison_Plot_{i}": wandb.Image(comparison_plot)})

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

        if self.cfg.model_ckpt is not None:
            model.load(self.cfg.model_ckpt)
        else:
            model.train_model(dataset)

        return model

    def evaluate_embedding_space_with_gmm(self, dataset: Any, model: Any):
        """
        1) Collect the entire dataset's embeddings (or user-specific if desired).
        2) Fit multiple GMMs with different numbers of components, log AIC/LL.
        3) Choose the best GMM (lowest AIC) and refit on embeddings.
        4) Use PCA and TSNE to project embeddings to 2D and 3D, color by GMM cluster.
        5) Log plots to Weights & Biases.
        """
        logger.info("Starting GMM evaluation in embedding space...")

        # -----------------------------
        # 1) Collect embeddings
        # -----------------------------
        # If you want the entire dataset, do:
        #   indices = dataset.data.index.to_numpy()
        # If you wanted a user-specific approach, adapt accordingly.
        all_indices = dataset.data.index.to_numpy()
        real_data_subset = dataset.data.iloc[all_indices].reset_index(drop=True)

        # Build conditioning vars (categoricals) for the entire subset
        conditioning_vars = {
            name: torch.tensor(
                real_data_subset[name].values, dtype=torch.long, device=device
            )
            for name in model.context_var_n_categories.keys()
        }

        # Forward pass to get embeddings
        with torch.no_grad():
            embeddings, _ = model.conditioning_module(
                conditioning_vars
            )  # shape: (N, D)

        # Move to CPU numpy
        embeddings_np = embeddings.cpu().numpy()

        # -----------------------------
        # 2) Fit GMMs (2,3,5,10 comps), log AIC & log-likelihood
        # -----------------------------
        logger.info("Fitting multiple GMMs to find best number of components...")
        possible_components = [2, 3, 5, 10]

        best_gmm = None
        best_n = None
        lowest_aic = float("inf")

        for n_comp in possible_components:
            gmm = GaussianMixture(
                n_components=n_comp, random_state=42, covariance_type="full"
            )
            gmm.fit(embeddings_np)

            # Log-likelihood on the entire embedding set
            # gmm.score(...) returns the average log-likelihood per sample
            ll = gmm.score(embeddings_np) * embeddings_np.shape[0]
            # AIC
            aic = gmm.aic(embeddings_np)
            # BIC (if you want to track it as well)
            bic = gmm.bic(embeddings_np)

            if aic < lowest_aic:
                best_gmm = gmm
                best_n = n_comp
                lowest_aic = aic

        logger.info(f"Best GMM has {best_n} components with AIC={lowest_aic:.2f}.")

        # -----------------------------
        # 3) Refit GMM with best # of comps
        # -----------------------------
        # (You can skip this if you want to reuse best_gmm directly,
        #  but let's be explicit in case you prefer a clean re-fit.)
        best_gmm = GaussianMixture(
            n_components=best_n, random_state=42, covariance_type="full"
        )
        best_gmm.fit(embeddings_np)

        # Assign cluster labels to each embedding
        cluster_labels = best_gmm.predict(embeddings_np)

        # -----------------------------
        # 4) PCA & TSNE -> 2D & 3D
        # -----------------------------
        logger.info("Projecting embeddings with PCA & t-SNE and plotting clusters...")

        # Helper function to scatter-plot and log to W&B
        def log_scatter_plot_2d(X_2d, labels, title_prefix="Plot"):
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
            plt.colorbar(sc, label="Cluster ID")
            plt.title(title_prefix)
            wandb.log({title_prefix: wandb.Image(plt)})
            plt.close()

        def log_scatter_plot_3d(X_3d, labels, title_prefix="3D Plot"):
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels, cmap="tab10", alpha=0.6
            )
            plt.title(title_prefix)
            fig.colorbar(sc, label="Cluster ID", shrink=0.5, aspect=10)
            wandb.log({title_prefix: wandb.Image(fig)})
            plt.close(fig)

        # ---- PCA 2D ----
        pca_2d = PCA(n_components=2, random_state=42).fit_transform(embeddings_np)
        log_scatter_plot_2d(pca_2d, cluster_labels, title_prefix="PCA_2D_GMM_Clusters")

        # ---- PCA 3D ----
        pca_3d = PCA(n_components=3, random_state=42).fit_transform(embeddings_np)
        log_scatter_plot_3d(pca_3d, cluster_labels, title_prefix="PCA_3D_GMM_Clusters")

        # ---- t-SNE 2D ----
        tsne_2d = TSNE(
            n_components=2, random_state=42, learning_rate="auto", init="pca"
        ).fit_transform(embeddings_np)
        log_scatter_plot_2d(
            tsne_2d, cluster_labels, title_prefix="TSNE_2D_GMM_Clusters"
        )

        # ---- t-SNE 3D (optional) ----
        tsne_3d = TSNE(
            n_components=3, random_state=42, learning_rate="auto", init="pca"
        ).fit_transform(embeddings_np)
        log_scatter_plot_3d(
            tsne_3d, cluster_labels, title_prefix="TSNE_3D_GMM_Clusters"
        )

        logger.info("GMM evaluation complete. Logged PCA & t-SNE cluster plots to W&B.")

    def evaluate_pv_shift(self, dataset: Any, model: Any):
        """
        Evaluate PV shift (pv=0 vs. pv=1) by generating all contexts in one pass.
        Compute a single L2 shift error, plot a few examples, and log to W&B.
        """
        logger.info("Starting PV shift evaluation...")
        avg_shift = dataset.compute_average_pv_shift()
        if avg_shift is None or np.allclose(avg_shift, 0.0):
            logger.warning(
                "No valid average shift could be computed. Skipping shift eval."
            )
            return

        test_contexts = dataset.sample_shift_test_contexts()
        if len(test_contexts) == 0:
            logger.warning("No shift contexts found. Skipping.")
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
                [pc[k] for pc in present_ctx_list], dtype=torch.long, device=device
            )
            missing_ctx_tensors[k] = torch.tensor(
                [mc[k] for mc in missing_ctx_list], dtype=torch.long, device=device
            )

        with torch.no_grad():
            syn_ts_present = model.generate(present_ctx_tensors)  # (N, seq_len, dim)
            syn_ts_missing = model.generate(missing_ctx_tensors)  # (N, seq_len, dim)

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

        logger.info(f"Overall shift L2 across all contexts: {mean_l2:.4f}")
        wandb.log({"Shift_L2": mean_l2})

        n_plots = min(3, shifts.shape[0])
        example_indices = np.random.choice(shifts.shape[0], size=n_plots, replace=False)
        for j, idx in enumerate(example_indices):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(avg_shift, label="Real avg shift", color="blue")
            ax.plot(shifts[idx], label="Synthetic shift", color="red", linestyle="--")
            ax.set_title(f"Shift Example {j+1}, L2={l2_values[idx]:.4f}")
            ax.legend()
            wandb.log({f"ShiftPlot_{j}": wandb.Image(fig)})
            plt.close(fig)

        logger.info("Finished PV shift evaluation.")
