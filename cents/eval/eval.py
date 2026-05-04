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
from omegaconf import DictConfig, OmegaConf

import wandb
from cents.eval.discriminative_score import discriminative_score_metrics
from cents.eval.eval_metrics import (
    Context_FID,
    calculate_banded_mse,
    calculate_mmd,
    compute_cfs,
    compute_context_recovery_score,
    compute_gcp,
    compute_mig,
    compute_sap,
    dynamic_time_warping_dist,
)
from cents.eval.eval_utils import flatten_log_dict
from cents.eval.predictive_score import predictive_score_metrics
from cents.models.acgan import ACGAN
from cents.models.diffusion_ts import Diffusion_TS
from cents.utils.utils import get_device

logger = logging.getLogger(__name__)


class Evaluator:
    """
    A class for evaluating generative models on time series data.

    This class handles the evaluation process, including metric computation,
    visualization generation, and results storage.

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
        print(f"[CENTS] Using device: {self.device} for evaluation")

        if results_dir is None:
            results_dir = cfg.evaluator.get("save_dir") or os.path.join(
                Path.home(),
                ".cache",
                "cents",
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
        model: Optional[Any] = None,
        data_generator: Optional[Any] = None,
    ) -> Dict:
        """
        Evaluate the model and store results.

        Args:
            model (Optional[Any]): The model to evaluate. If None, will load or train a model.
            data_generator (Optional[Any]): DataGenerator instance. If provided, will use its model and normalizer.

        Returns:
            Dict: Dictionary containing the evaluation results
        """
        dataset = self.real_dataset

        if data_generator is not None:
            if data_generator.model is not None:
                model = data_generator.model
                print(f"[Cents] Using pre-trained model from DataGenerator")
            if data_generator.normalizer is not None:
                dataset._normalizer = data_generator.normalizer
                print("[Cents] Using pre-trained normalizer from DataGenerator")
                if not getattr(dataset.cfg, "normalize", True):
                    dataset.apply_pretrained_normalizer()
                    print("[Cents] Normalized dataset with pretrained normalizer")
        else:
            if not model:
                model = self.get_trained_model(dataset)

        model.to(self.device)
        model.eval()

        logger.info("[Cents] Starting evaluation")
        logger.info("----------------------")

        self.run_evaluation(dataset, model)

        if self.cfg.evaluator.get("save_results", False):
            self.save_results()

        if self.cfg.get("wandb", {}).get("enabled", False) and wandb.run is not None:
            wandb.log(flatten_log_dict(self.current_results["metrics"]))

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

        logger.info(f"[Cents] Saved evaluation results to {results_file}")
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

    def compute_quality_metrics(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        real_data_frame: pd.DataFrame,
        mask: Optional[np.ndarray] = None,
        target: Optional[Dict] = None,
        log_prefix: str = "",
        context_arrays: Optional[Dict[str, np.ndarray]] = None,
        ts_column_names: Optional[list] = None,
        static_context_arrays: Optional[Dict[str, np.ndarray]] = None,
        continuous_vars: Optional[list] = None,
    ) -> Dict:
        """
        Compute evaluation metrics and store them in current_results (or in target if provided).

        Args:
            real_data (np.ndarray): Real data array (shape: [N, seq_len, dims])
            syn_data (np.ndarray): Synthetic data array (shape: [N, seq_len, dims])
            real_data_frame (pd.DataFrame): Real data subset (inverse-transformed)
            mask (Optional[np.ndarray]): Boolean array indicating which rows are "rare"
            target (Optional[Dict]): If set, write metrics into this dict instead of current_results (for normalized_domain).
            log_prefix (str): Prefix for log messages (e.g. "[normalized]").
            context_arrays (Optional[Dict[str, np.ndarray]]): Named dynamic context arrays (N, T) each.
            ts_column_names (Optional[list]): Names of the time-series output dimensions, aligned with real_data axis-2.
        """
        logger.info(f"[Cents] --- {log_prefix}Full-Subset Metrics ---")

        metrics = target if target is not None else {}

        # Compute and store metrics
        dtw_mean, dtw_std = dynamic_time_warping_dist(real_data, syn_data)
        metrics["DTW"] = {"mean": dtw_mean, "std": dtw_std}
        logger.info(f"[Cents] DTW completed")

        mmd_mean, mmd_std = calculate_mmd(real_data, syn_data)
        metrics["MMD"] = {"mean": mmd_mean, "std": mmd_std}
        logger.info(f"[Cents] MMD completed")

        banded_mse = calculate_banded_mse(real_data, syn_data)
        metrics["Banded_MSE"] = banded_mse
        logger.info(f"[Cents] Banded MSE completed")

        fid_score = Context_FID(real_data, syn_data)
        metrics["Context_FID"] = fid_score
        logger.info(f"[Cents] Context-FID completed")

        discr_score, _, _ = discriminative_score_metrics(real_data, syn_data)
        metrics["Disc_Score"] = discr_score
        logger.info(f"[Cents] Discr Score completed")

        trtr = self.cfg.evaluator.get("pred_score_trtr", False)
        pred_score = predictive_score_metrics(real_data, syn_data, trtr=trtr)
        metrics["Pred_Score"] = pred_score
        logger.info(f"[Cents] Pred Score completed")

        # CFS / GCP — only when dynamic context or multivariate signal is present
        cf_cfg = self.cfg.evaluator.get("eval_context_faithfulness", None)
        if cf_cfg and cf_cfg.get("enabled", False) and context_arrays is not None:
            cf_metrics = self._compute_context_faithfulness_metrics(
                real_data, syn_data, context_arrays, ts_column_names, cf_cfg
            )
            metrics["context_faithfulness"] = cf_metrics

        # Context Recovery Score — tests whether static context is encoded in generated outputs
        if self.cfg.evaluator.get("eval_context_recovery", False) and static_context_arrays:
            crs_score, crs_per_var = compute_context_recovery_score(
                real_data, syn_data, static_context_arrays, continuous_vars=continuous_vars
            )
            metrics["context_recovery"] = {"overall": crs_score, "per_var": crs_per_var}
            logger.info(f"[Cents] Context Recovery Score completed: {crs_score:.4f}")

        if mask is not None:
            logger.info("[Cents] Starting Rare-Subset Metrics")
            rare_metrics = {}
            rare_real_data = real_data[mask]
            rare_syn_data = syn_data[mask]
            rare_real_df = real_data_frame[mask].reset_index(drop=True)

            # Check if rare subset has any valid samples
            if len(rare_real_data) == 0:
                logger.warning("[Cents] Rare subset is empty - skipping rare subset metrics")
                rare_metrics = {
                    "DTW": {"mean": float('nan'), "std": float('nan')},
                    "MMD": {"mean": float('nan'), "std": float('nan')},
                    "Context_FID": float('nan'),
                    "Disc_Score": float('nan'),
                    "Pred_Score": float('nan')
                }
            else:
                logger.info(f"[Cents] Rare subset contains {len(rare_real_data)} samples")
                
                dtw_mean_r, dtw_std_r = dynamic_time_warping_dist(
                    rare_real_data, rare_syn_data
                )
                rare_metrics["DTW"] = {"mean": dtw_mean_r, "std": dtw_std_r}
                logger.info(f"[Cents] DTW completed")

                mmd_mean_r, mmd_std_r = calculate_mmd(rare_real_data, rare_syn_data)
                rare_metrics["MMD"] = {"mean": mmd_mean_r, "std": mmd_std_r}
                logger.info(f"[Cents] MMD completed")

                fid_score_r = Context_FID(rare_real_data, rare_syn_data)
                rare_metrics["Context_FID"] = fid_score_r
                logger.info(f"[Cents] Context-FID completed")

                discr_score_r, _, _ = discriminative_score_metrics(
                    rare_real_data, rare_syn_data
                )
                rare_metrics["Disc_Score"] = discr_score_r
                logger.info(f"[Cents] Discr Score completed")

                pred_score_r = predictive_score_metrics(rare_real_data, rare_syn_data)
                rare_metrics["Pred_Score"] = pred_score_r
                logger.info(f"[Cents] Pred Score completed")

            logger.info("[Cents] Done computing Rare-Subset Metrics.")
            metrics["rare_subset"] = rare_metrics

        if target is None:
            self.current_results["metrics"] = metrics
        return metrics

    def _compute_context_faithfulness_metrics(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        context_arrays: Dict[str, np.ndarray],
        ts_column_names: Optional[list],
        cf_cfg,
    ) -> Dict:
        """
        Compute CFS and GCP for configured (x_dim, c_dim) pairs.

        Each pair specifies one x dimension (by name from ts_column_names, or by index)
        and one c dimension (by name from context_arrays, or by name from ts_column_names
        for within-signal multivariate pairs).

        CFS is only computed when c comes from context_arrays (shared context).
        GCP is computed for all pairs.
        """
        results: Dict = {}
        max_lag: int = int(cf_cfg.get("gcp_max_lag", 5))
        pairs_cfg = cf_cfg.get("pairs", [])

        if not pairs_cfg:
            logger.warning("[Cents] eval_context_faithfulness.pairs is empty — skipping CFS/GCP.")
            return results

        ts_names = list(ts_column_names) if ts_column_names else []

        def _resolve_x(name) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            """Return (x_real_slice, x_synth_slice) of shape (N, T, 1)."""
            if isinstance(name, int):
                idx = name
            elif name in ts_names:
                idx = ts_names.index(name)
            else:
                logger.warning(f"[Cents] CFS/GCP: x dim '{name}' not found in ts_column_names {ts_names}; skipping.")
                return None
            return real_data[:, :, idx : idx + 1], syn_data[:, :, idx : idx + 1]

        def _resolve_c(name) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
            """Return (c_real, c_synth, is_shared) of shape (N, T, 1). is_shared=True if dynamic context."""
            if name in context_arrays:
                arr = context_arrays[name]  # (N, T)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]  # edge-case: (N,) → (N, 1)
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]  # (N, T, 1)
                return arr, arr, True  # same array for real and synth
            elif name in ts_names:
                idx = ts_names.index(name)
                c_r = real_data[:, :, idx : idx + 1]
                c_s = syn_data[:, :, idx : idx + 1]
                return c_r, c_s, False
            else:
                logger.warning(f"[Cents] CFS/GCP: c dim '{name}' not found in context_arrays or ts_column_names; skipping.")
                return None

        for pair in pairs_cfg:
            x_name = pair.get("x")
            c_name = pair.get("c")
            if x_name is None or c_name is None:
                logger.warning(f"[Cents] CFS/GCP: pair {pair} missing 'x' or 'c' key; skipping.")
                continue

            x_resolved = _resolve_x(x_name)
            c_resolved = _resolve_c(c_name)
            if x_resolved is None or c_resolved is None:
                continue

            x_r, x_s = x_resolved
            c_r, c_s, is_shared = c_resolved
            pair_key = f"{x_name}_vs_{c_name}"

            pair_result: Dict = {}

            # CFS — only meaningful when context is shared (dynamic context)
            if is_shared:
                try:
                    cfs = compute_cfs(x_r, x_s, c_r)
                    pair_result["CFS"] = cfs
                    logger.info(f"[Cents] CFS completed for {pair_key}: {cfs:.4f}")
                except Exception as e:
                    logger.warning(f"[Cents] CFS failed for {pair_key}: {e}")
                    pair_result["CFS"] = float("nan")
            else:
                logger.info(f"[Cents] CFS skipped for {pair_key} (c is a generated dim, not shared context).")

            # GCP — works for both shared and generated-dim context
            try:
                gcp, diag = compute_gcp(x_r, x_s, c_r, c_s, max_lag=max_lag)
                pair_result["GCP"] = gcp
                pair_result["GCP_diagnostics"] = diag
                logger.info(f"[Cents] GCP completed for {pair_key}: {gcp:.4f}")
            except Exception as e:
                logger.warning(f"[Cents] GCP failed for {pair_key}: {e}")
                pair_result["GCP"] = float("nan")
                pair_result["GCP_diagnostics"] = {}

            results[pair_key] = pair_result

        return results

    def compute_disentanglement_metrics(
        self,
        context_vars: Dict[str, torch.Tensor],
        model: Any,
    ) -> None:
        """
        Compute disentanglement metrics and store them in current_results.

        Args:
            context_vars (Dict[str, torch.Tensor]): Dictionary of context variables
            model (Any): The model to evaluate
        """
        logger.info("[Cents] --- Starting Disentanglement Metrics ---")

        with torch.no_grad():
            h, _ = model.context_module(context_vars)  # (N, D)

        emb_np = h.cpu().numpy()
        ctx_np = {k: v.cpu().numpy() for k, v in context_vars.items()}

        mig, mig_detail = compute_mig(emb_np, ctx_np)
        sap, sap_detail = compute_sap(emb_np, ctx_np)

        self.current_results["metrics"].setdefault("disentanglement", {})
        self.current_results["metrics"]["disentanglement"].update(
            {
                "MIG": {"mean": mig, **mig_detail},
                "SAP": {"mean": sap, **sap_detail},
            }
        )

        logger.info("[Cents] MIG completed")
        logger.info("[Cents] SAP completed")

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
            f"[Cents] Starting evaluation of {self.model_name} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
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
        continuous_vars = getattr(dataset, "continuous_vars", [])
        static_context_vars = {}
        for name in dataset.static_context_vars:
            vals = real_data_subset[name].values
            dtype = torch.float32 if name in continuous_vars else torch.long
            static_context_vars[name] = torch.tensor(vals, dtype=dtype, device=self.device)
        dynamic_context_vars = {}
        categorical_ts = getattr(dataset, "categorical_time_series", {})
        for name in dataset.dynamic_context_vars:
            vals = real_data_subset[name].values
            # Dynamic module expects tensors (training path uses torch.from_numpy in dataset __getitem__)
            if len(vals) and hasattr(vals[0], "__len__") and not isinstance(vals[0], (str, bytes)):
                arr = np.stack([np.asarray(v, dtype=np.float32 if name not in categorical_ts else np.int64) for v in vals])
            else:
                arr = np.asarray(vals, dtype=np.float32 if name not in categorical_ts else np.int64)
            dtype = torch.long if name in categorical_ts else torch.float32
            dynamic_context_vars[name] = torch.tensor(arr, dtype=dtype, device=self.device)

        generated_ts = model.generate(static_context_vars, dynamic_context_vars).cpu().numpy()
        if generated_ts.ndim == 2:
            generated_ts = generated_ts.reshape(
                generated_ts.shape[0], -1, generated_ts.shape[1]
            )

        syn_data_subset = real_data_subset.copy()
        syn_data_subset["timeseries"] = list(generated_ts)

        # When normalize=False but a pretrained normalizer was applied via apply_pretrained_normalizer,
        # dataset.inverse_transform() is a no-op (it checks self.normalize). Do the inverse manually.
        normalizer = getattr(dataset, "_normalizer", None)
        if not getattr(dataset, "normalize", True) and normalizer is not None:
            def _inv(df):
                split = dataset.split_timeseries(df.copy())
                split = normalizer.inverse_transform(split)
                return dataset.merge_timeseries_columns(split)
            real_data_inv = _inv(real_data_subset)
            syn_data_inv = _inv(syn_data_subset)
        else:
            real_data_inv = dataset.inverse_transform(real_data_subset)
            syn_data_inv = dataset.inverse_transform(syn_data_subset)

        real_data_array = np.stack(real_data_inv["timeseries"])
        syn_data_array = np.stack(syn_data_inv["timeseries"])

        # Extract dynamic context as (N, T) numpy arrays for CFS/GCP
        context_np: Dict[str, np.ndarray] = {}
        for name, tensor in dynamic_context_vars.items():
            arr = tensor.cpu().numpy()  # (N,) static scalar or (N, T) time series
            if arr.ndim == 2:           # (N, T) — keep time-series context only
                context_np[name] = arr

        ts_col_names = list(getattr(dataset, "time_series_column_names", []))

        # Decide whether CFS/GCP should run: multivariate signal OR dynamic context present
        has_dynamic_context = len(context_np) > 0
        has_multivariate_signal = real_data_array.shape[-1] > 1
        cf_cfg = self.cfg.evaluator.get("eval_context_faithfulness", None)
        run_cf = (
            cf_cfg is not None
            and cf_cfg.get("enabled", False)
            and (has_dynamic_context or has_multivariate_signal)
        )

        # Build static context numpy arrays for Context Recovery Score
        static_context_np: Dict[str, np.ndarray] = {
            name: tensor.cpu().numpy() for name, tensor in static_context_vars.items()
        }
        dataset_continuous_vars: list = getattr(dataset, "continuous_vars", [])

        if self.cfg.evaluator.eval_metrics:
            rare_mask = None

            if (
                self.cfg.evaluator.eval_context_sparse
                and "is_rare" in real_data_subset.columns
            ):
                rare_mask = real_data_subset["is_rare"].values

            # Metrics in raw (un-normalized) domain
            self.compute_quality_metrics(
                real_data_array, syn_data_array, real_data_inv, rare_mask,
                log_prefix="[raw] ",
                context_arrays=context_np if run_cf else None,
                ts_column_names=ts_col_names if run_cf else None,
                static_context_arrays=static_context_np if static_context_np else None,
                continuous_vars=dataset_continuous_vars,
            )

            # Metrics in normalized (z) domain for cross-domain comparability.
            # Fires whenever a normalizer is available — whether data was pre-normalized by dataset
            # init (normalize=True) or normalized in-place via apply_pretrained_normalizer (normalize=False).
            if (
                getattr(dataset, "_normalizer", None) is not None
                and "timeseries" in real_data_subset.columns
            ):
                real_data_norm = np.stack(real_data_subset["timeseries"].values)
                syn_data_norm = generated_ts
                logger.info("[Cents] Computing metrics in normalized domain (z-space) for cross-domain comparison.")
                normalized_metrics = {}
                self.compute_quality_metrics(
                    real_data_norm, syn_data_norm, real_data_inv, rare_mask,
                    target=normalized_metrics,
                    log_prefix="[normalized] ",
                    context_arrays=context_np if run_cf else None,
                    ts_column_names=ts_col_names if run_cf else None,
                    static_context_arrays=static_context_np if static_context_np else None,
                    continuous_vars=dataset_continuous_vars,
                )
                self.current_results["metrics"]["normalized_domain"] = normalized_metrics

        if self.cfg.evaluator.eval_disentanglement:
            self.compute_disentanglement_metrics(static_context_vars, model)
