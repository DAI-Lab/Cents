from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from omegaconf import ListConfig


from cents.datasets.utils import split_timeseries
from cents.models.base import NormalizerModel
from cents.models.context import MLPContextModule, SepMLPContextModule, DynamicContextModule_CNN, DynamicContextModule_Transformer # Import to trigger registration
from cents.models.context_registry import get_context_module_cls
from cents.models.stats_head_registry import register_stats_head, get_stats_head_cls
from cents.models.registry import register_model
from cents.utils.utils import get_context_config


@register_stats_head("default", "mlp")
class MLPStatsHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        time_series_dims: int,
        do_scale: bool,
        n_layers: int = 3,
    ):
        super().__init__()
        self.time_series_dims = time_series_dims
        self.do_scale = do_scale
        out_dim = 4 * time_series_dims if do_scale else 2 * time_series_dims
        layers = []
        in_dim = embedding_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
        self._initialize_output_layer()
    
    def _initialize_output_layer(self, init_sigma: float = 1.0):
        D = self.time_series_dims
        K = 4 if self.do_scale else 2
        out_layer = self.net[-1]

        with torch.no_grad():
            nn.init.xavier_uniform_(out_layer.weight, gain=0.01)
            nn.init.zeros_(out_layer.bias)
            
            if self.do_scale:
                # 1. Initialize z_min to -2.0
                out_layer.bias[2 * D : 3 * D].fill_(-2.0)
                
                # 2. Initialize the RAW DELTA to ~4.0 
                # Softplus(4.0) is approx 4.018. 
                # z_max = -2.0 + 4.018 = 2.018 (Perfect starting point)
                out_layer.bias[3 * D : 4 * D].fill_(4.0)

    @staticmethod
    def _soft_clamp_tanh(x: torch.Tensor, bound: float) -> torch.Tensor:
        if bound <= 0:
            raise ValueError("bound must be > 0")
        return bound * torch.tanh(x / bound)    

    def forward(self, z: torch.Tensor):
        out = self.net(z)
        batch_size = out.size(0)
        
        if self.do_scale:
            out = out.view(batch_size, 4, self.time_series_dims)
            pred_mu = out[:, 0, :]
            pred_log_sigma = out[:, 1, :]
            
            # z_min is predicted normally
            pred_z_min = out[:, 2, :]
            
            # The 4th output is now the "raw delta"
            raw_delta = out[:, 3, :]
            
            # Ensure the range is strictly positive (minimum 1e-4)
            # F.softplus(x) = log(1 + exp(x))
            actual_range = F.softplus(raw_delta) + 1e-4
            
            # Structurally guarantee z_max > z_min
            pred_z_max = pred_z_min + actual_range
            
        else:
            out = out.view(batch_size, 2, self.time_series_dims)
            pred_mu = out[:, 0, :]
            pred_log_sigma = out[:, 1, :]
            pred_z_min = None
            pred_z_max = None
        
        pred_log_sigma_unclamped = pred_log_sigma
        pred_log_sigma_clamped = self._soft_clamp_tanh(pred_log_sigma, bound=10.0)
        pred_sigma = torch.exp(pred_log_sigma_clamped)
        
        return pred_mu, pred_sigma, pred_z_min, pred_z_max, pred_log_sigma_unclamped


class _NormalizerModule(nn.Module):
    """
    Wrapper module combining a context embedding and stats head for normalization.
    """

    def __init__(
        self,
        static_cond_module: nn.Module = None,
        dynamic_cond_module: nn.Module = None,
        hidden_dim: int = 512,
        time_series_dims: int = 2,
        do_scale: bool = True,
        stats_head_type: str = "mlp",
        n_layers: int = 3,
    ):
        super().__init__()
        self.static_cond_module = static_cond_module
        self.dynamic_cond_module = dynamic_cond_module
        
        # Determine embedding dimension from available modules
        if static_cond_module is not None:
            self.embedding_dim = static_cond_module.embedding_dim
        elif dynamic_cond_module is not None:
            self.embedding_dim = dynamic_cond_module.embedding_dim
        else:
            raise ValueError("At least one of static_cond_module or dynamic_cond_module must be provided")
        
        # If both modules exist, combine their embeddings
        if static_cond_module is not None and dynamic_cond_module is not None:
            combined_dim = static_cond_module.embedding_dim + dynamic_cond_module.embedding_dim
            self.combine_mlp = nn.Sequential(
                nn.Linear(combined_dim, self.embedding_dim),
                nn.ReLU(),
            )
        else:
            self.combine_mlp = None
        
        StatsHeadCls = get_stats_head_cls(stats_head_type)
        self.stats_head = StatsHeadCls(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            time_series_dims=time_series_dims,
            do_scale=do_scale,
            n_layers=n_layers,
        )

    def forward(self, static_context_vars_dict: dict = None, dynamic_context_vars_dict: dict = None):
        embeddings = []
        
        # Process static context variables
        if self.static_cond_module is not None:
            # static_vars = {
            #     k: v for k, v in context_vars_dict.items()
            #     if k not in getattr(self, '_dynamic_var_names', [])
            # }
            if static_context_vars_dict:
                device = next(self.static_cond_module.parameters()).device
                static_context_vars_dict = {
                    k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                    for k, v in static_context_vars_dict.items()
                }
                static_embedding, _ = self.static_cond_module(static_context_vars_dict)
                embeddings.append(static_embedding)
        
        # Process dynamic context variables
        if self.dynamic_cond_module is not None:
            # dynamic_var_names = getattr(self, '_dynamic_var_names', [])
            # dynamic_vars = {
            #     k: v for k, v in context_vars_dict.items()
            #     if k in dynamic_var_names
            # }
            if dynamic_context_vars_dict:
                device = next(self.dynamic_cond_module.parameters()).device
                dynamic_context_vars_dict = {
                    k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                    for k, v in dynamic_context_vars_dict.items()
                }
                dynamic_embedding, _ = self.dynamic_cond_module(dynamic_context_vars_dict)
                if torch.isnan(dynamic_embedding).any() or torch.isinf(dynamic_embedding).any():
                    raise ValueError(f"NaN/Inf detected in dynamic embedding.")
                embeddings.append(dynamic_embedding)
        
        # Combine embeddings
        if len(embeddings) == 2:
            combined = torch.cat(embeddings, dim=1)
            embedding = self.combine_mlp(combined)
        elif len(embeddings) == 1:
            embedding = embeddings[0]
        else:
            raise ValueError("No context variables provided")
        
        return self.stats_head(embedding)


@register_model("normalizer")
class Normalizer(NormalizerModel):
    """
    Learns group-wise normalization parameters (mean, std, optional min/max) for time series by context.
    """

    def __init__(
        self,
        dataset_cfg,
        normalizer_training_cfg,
        dataset,
        context_cfg,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.dataset_cfg = dataset_cfg
        self.normalizer_training_cfg = normalizer_training_cfg
        self.dataset = dataset

        # Get continuous variables from config if specified
        self.continuous_vars = [k for k, v in self.dataset_cfg.context_vars.items() if v[0] == "continuous"]
        self.categorical_vars = [k for k, v in self.dataset_cfg.context_vars.items() if v[0] == "categorical"]
        self.dynamic_context_vars = [k for k, v in self.dataset_cfg.context_vars.items() if v[0] == "time_series"]
        
        self.static_context_vars = self.categorical_vars + self.continuous_vars
        self.context_vars = self.static_context_vars + self.dynamic_context_vars

        # Normalizer-specific conditioning: subset of static vars for grouping / stats (e.g. per-station)
        self.normalizer_group_vars = getattr(self.dataset_cfg, "normalizer_group_vars", None)
        self.normalizer_static_vars = (
            list(self.normalizer_group_vars) if self.normalizer_group_vars is not None
            else self.static_context_vars
        )
        if self.normalizer_static_vars:
            bad = [v for v in self.normalizer_static_vars if v not in self.static_context_vars]
            if bad:
                raise ValueError(f"normalizer_group_vars {self.normalizer_group_vars} contains vars not in static_context_vars: {bad}")
        # When normalizer_group_vars is set it is always static-only (dynamic vars are rejected in
        # _build_training_samples), so exclude dynamic vars from normalizer conditioning entirely.
        self.normalizer_dynamic_vars = [] if self.normalizer_group_vars is not None else self.dynamic_context_vars
        self._group_bin_edges = {}  # filled in setup() when grouping by continuous vars
        
        self.time_series_cols = dataset_cfg.time_series_columns[: dataset_cfg.time_series_dims]
        self.time_series_dims = dataset_cfg.time_series_dims
        self.do_scale = dataset_cfg.scale
        self.seq_len = dataset_cfg.seq_len
        self.num_ts_steps = getattr(dataset_cfg, "num_ts_steps", None)

        self.static_module_type = self.dataset.static_module_type
        self.dynamic_module_type = self.dataset.dynamic_module_type
        self.stats_head_type = self.dataset.stats_head_type
        self.loss_type = getattr(self.normalizer_training_cfg, "loss_type", "mse")
        self.use_global_stats_preprocessing = bool(
            getattr(self.normalizer_training_cfg, "use_global_stats_preprocessing", True)
        )
        if hasattr(self.dataset_cfg, "normalizer_use_global_stats_preprocessing"):
            self.use_global_stats_preprocessing = bool(
                self.dataset_cfg.normalizer_use_global_stats_preprocessing
            )
        # When not using global preprocessing: mu is predicted in asinh space; clamp to avoid sinh explosion
        self.max_asinh_mu = float(getattr(self.normalizer_training_cfg, "max_asinh_mu", 10.0))
        # Floor sigma and scale range so normalized values (z) cannot explode
        self.min_sigma = float(getattr(self.normalizer_training_cfg, "min_sigma", 1e-3))
        self.min_scale_range = float(getattr(self.normalizer_training_cfg, "min_scale_range", 0.25))

        self.register_buffer("global_mu_mean", torch.tensor(0.0))
        self.register_buffer("global_mu_std", torch.tensor(1.0))
        self.register_buffer("global_log_sigma_mean", torch.tensor(0.0))

        # Create static context module (only normalizer_static_vars so it matches group conditioning)
        self.static_context_module = None
        if self.normalizer_static_vars:
            StaticContextModuleCls = get_context_module_cls(self.static_module_type)
            n_bins = getattr(self.dataset_cfg, "numeric_context_bins", 5)
            self.static_context_vars_dict = {}
            for k in self.normalizer_static_vars:
                if k in self.continuous_vars:
                    # Binned continuous: treat as categorical with n_bins for normalizer conditioning
                    self.static_context_vars_dict[k] = ["categorical", n_bins]
                else:
                    self.static_context_vars_dict[k] = self.dataset.context_var_dict[k]
            self.static_context_module = StaticContextModuleCls(
                self.static_context_vars_dict,
                256,
            )

        # Create dynamic context module only for vars the normalizer should condition on.
        # Filter to vars that are both in normalizer_dynamic_vars and are actual dynamic (time_series) vars.
        # When normalizer_group_vars is set, normalizer_dynamic_vars is empty so the dict is empty
        # and no dynamic module is created.
        self.dynamic_context_module = None
        _normalizer_dynamic_vars_dict = {
            k: v for k, v in self.dataset_cfg.context_vars.items()
            if k in self.normalizer_dynamic_vars and k in self.dynamic_context_vars
        }
        if _normalizer_dynamic_vars_dict and self.dynamic_module_type is not None:
            DynamicContextModuleCls = get_context_module_cls("dynamic", self.dynamic_module_type)
            dynamic_context_vars_dict = _normalizer_dynamic_vars_dict
            dynamic_seq_len = self.num_ts_steps if self.num_ts_steps is not None else self.seq_len
            self.dynamic_context_module = DynamicContextModuleCls(
                dynamic_context_vars_dict,
                256,
                seq_len=dynamic_seq_len,
            )
        
        self.normalizer_model = _NormalizerModule(
            static_cond_module=self.static_context_module,
            dynamic_cond_module=self.dynamic_context_module,
            hidden_dim=512,
            time_series_dims=self.time_series_dims,
            do_scale=self.do_scale,
            stats_head_type=self.stats_head_type,
            n_layers=context_cfg.normalizer.n_layers,
        )

        # Will be populated in setup()
        self.sample_stats = []
        self._verify_parameters()

    def _verify_parameters(self):
        all_param_names = [name for name, _ in self.named_parameters()]
        context_param_names = [name for name in all_param_names if 'cond_module' in name or 'context_module' in name]
        stats_head_param_names = [name for name in all_param_names if 'stats_head' in name]
        
        if not context_param_names:
            raise RuntimeError(
                "Context module parameters not found! "
                f"Found parameter names: {all_param_names[:10]}..."
            )
        
        print(f"[Normalizer] Found {len(context_param_names)} context module parameters")
        print(f"[Normalizer] Found {len(stats_head_param_names)} stats head parameters")
        print(f"[Normalizer] Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def setup(self, stage: Optional[str] = None):
        """
        Lightning hook: prepare training data before training.
        """
        # Compute per-sample statistics
        # Note: Using robust quantile scaling for targets to avoid outlier instability
        mode = getattr(self.dataset_cfg, "normalizer_stats_mode", "sample")
        group_vars = getattr(self.dataset_cfg, "normalizer_group_vars", None)
        self.sample_stats = self._build_training_samples(mode, use_quantile_scale=True, group_vars=group_vars)

        if self.use_global_stats_preprocessing:
            # --- COMPUTE GLOBAL TARGET STATS FOR SCALING ---
            # 1. Global Mu Stats (for Z-score scaling)
            all_mus = np.concatenate([s[2] for s in self.sample_stats])
            self.target_mu_mean = torch.tensor(all_mus.mean(), dtype=torch.float32)
            self.target_mu_std = torch.tensor(all_mus.std() + 1e-8, dtype=torch.float32)

            # 2. Global Sigma Stats (for Log-Space Centering)
            all_sigmas_concat = np.concatenate([s[3] for s in self.sample_stats])
            self.target_log_sigma_mean = torch.tensor(
                np.log(all_sigmas_concat + 1e-8).mean(), dtype=torch.float32
            )

            self.global_mu_mean.fill_(self.target_mu_mean)
            self.global_mu_std.fill_(self.target_mu_std)
            self.global_log_sigma_mean.fill_(self.target_log_sigma_mean)

            print(f"Global Target Stats: Mu Mean={self.target_mu_mean:.4f}, Mu Std={self.target_mu_std:.4f}")
            print(f"Global Target Log Sigma Mean: {self.target_log_sigma_mean:.4f}")
        else:
            # No global preprocessing: mu in asinh space, log(sigma) direct; identity for sigma centering
            self.global_mu_mean.zero_()
            self.global_mu_std.fill_(1.0)
            self.global_log_sigma_mean.zero_()
            print(f"Normalizer: use_global_stats_preprocessing=False — predicting asinh(mu) (clamp ±{self.max_asinh_mu}) and log(sigma) directly.")

        # Global range mean for safeguard: floor rng to 0.01 * global_rng_mean when do_scale
        if self.do_scale:
            all_rngs = []
            for s in self.sample_stats:
                zlow, zhigh = s[4], s[5]
                if zlow is not None and zhigh is not None:
                    all_rngs.extend((np.asarray(zhigh) - np.asarray(zlow)).flatten().tolist())

        # Log initial predictions
        if stage == "fit" or stage is None:
            self._log_initial_predictions()
    
    def _log_initial_predictions(self):
        """Log initial model predictions to diagnose initialization issues."""
        # self.eval()
        # with torch.no_grad():
        #     dataloader = self.train_dataloader()
        #     batch = next(iter(dataloader))
        #     cat_vars_dict, mu_t, sigma_t, zmin_t, zmax_t = batch
            
        #     device = next(self.parameters()).device
        #     cat_vars_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cat_vars_dict.items()}
        #     mu_t = mu_t.to(device)
        #     sigma_t = sigma_t.to(device)
            
        #     # Predict (Returns Real Unscaled values via Forward)
        #     pred_mu, pred_sigma, pred_z_min, pred_z_max, _ = self(cat_vars_dict)
            
        #     print(f"\n[Initial Predictions]")
        #     print(f"  Target mu: mean={mu_t.mean().item():.4f}, std={mu_t.std().item():.4f}")
        #     print(f"  Predicted mu: mean={pred_mu.mean().item():.4f}, std={pred_mu.std().item():.4f}")
        #     print(f"  Initial loss_mu: {F.mse_loss(pred_mu, mu_t).item():.6f}")
        #     print()
        
        self.train()
 
    def _raw_mu_to_real(self, pred_mu_raw: torch.Tensor) -> torch.Tensor:
        """Convert network mu output to real-world mu (handles both global and direct/asinh paths)."""
        if self.use_global_stats_preprocessing:
            return (pred_mu_raw * self.global_mu_std) + self.global_mu_mean
        # Direct path: network predicts asinh(mu); invert with sinh and clamp for stability
        return torch.sinh(torch.clamp(pred_mu_raw, -self.max_asinh_mu, self.max_asinh_mu))

    def forward(self, static_context_vars_dict: dict = None, dynamic_context_vars_dict: dict = None):
        """
        Predict normalization parameters.
        Applies UNSCALING logic to convert network outputs back to real-world range.

        Returns:
            Tuple of (pred_mu_real, pred_sigma_real, pred_z_min, pred_z_max, pred_log_sigma_raw).
        """
        pred_mu_raw, pred_sigma, pred_zmin, pred_zmax, pred_log_sigma_raw = self.normalizer_model(static_context_vars_dict, dynamic_context_vars_dict)

        pred_mu_real = self._raw_mu_to_real(pred_mu_raw)

        # Unscale Sigma: exp(NetworkLogOutput + GlobalLogMean) or exp(NetworkLogOutput) when no global
        pred_log_sigma_real = pred_log_sigma_raw + self.global_log_sigma_mean
        pred_sigma_real = torch.exp(pred_log_sigma_real).clamp(min=self.min_sigma)
        # Safeguard: sigma must be at least 0.01 * global sigma (exp(global_log_sigma_mean))
        sigma_global = torch.exp(self.global_log_sigma_mean)
        sigma_floor = max(self.min_sigma, (0.01 * sigma_global).item())
        pred_sigma_real = torch.clamp(pred_sigma_real, min=sigma_floor)

        return pred_mu_real, pred_sigma_real, pred_zmin, pred_zmax, pred_log_sigma_raw

    def _compute_loss_mse(self, pred_mu_raw, pred_log_sigma_raw, mu_t_scaled, target_log_sigma_centered):
        """
        Compute MSE loss in the SCALED space.
        """
        loss_mu = F.mse_loss(pred_mu_raw, mu_t_scaled)
        
        # Use Huber Loss for Log Sigma to be robust against outliers
        # Clamp targets slightly to prevent infinite loss if data is broken
        target_log_sigma_centered = torch.clamp(target_log_sigma_centered, min=-10.0, max=10.0)
        loss_sigma = F.smooth_l1_loss(pred_log_sigma_raw, target_log_sigma_centered, beta=1.0)
        
        return loss_mu, loss_sigma

    def training_step(self, batch, batch_idx: int):
        static_context_vars_dict, dynamic_context_vars_dict, mu_t, sigma_t, zmin_t, zmax_t = batch
        pred_mu_raw, _, pred_z_min, pred_z_max, pred_log_sigma_raw = self.normalizer_model(static_context_vars_dict, dynamic_context_vars_dict)

        # 2. Targets: scaled (global) or asinh(mu) + log(sigma) (direct)
        if self.use_global_stats_preprocessing:
            mu_t_scaled = (mu_t - self.global_mu_mean) / self.global_mu_std
        else:
            # asinh(mu) compresses full range to ~[-8, 8]; clamp so target matches forward clamp
            mu_t_scaled = torch.clamp(
                torch.asinh(mu_t),
                -self.max_asinh_mu,
                self.max_asinh_mu,
            )
        target_log_sigma_centered = torch.log(sigma_t + 1e-8) - self.global_log_sigma_mean

        # 3. Compute Loss
        loss_mu, loss_sigma = self._compute_loss_mse(
            pred_mu_raw, pred_log_sigma_raw, mu_t_scaled, target_log_sigma_centered
        )
        
        total_loss = loss_mu + loss_sigma

        # 4. Scaling parameters (z_min/z_max) - These are already naturally roughly scaled (-2 to 2)
        # We use SmoothL1Loss to be robust to outliers
        if self.do_scale:
            if torch.isnan(pred_z_min).any() or torch.isnan(pred_z_max).any():
                raise ValueError(f"NaN detected in scale predictions at batch {batch_idx}")
            loss_zmin = F.smooth_l1_loss(pred_z_min, zmin_t, beta=1.0)
            loss_zmax = F.smooth_l1_loss(pred_z_max, zmax_t, beta=1.0)
            total_loss += loss_zmin + loss_zmax
        else:
            loss_zmin = torch.tensor(0.0, device=total_loss.device)
            loss_zmax = torch.tensor(0.0, device=total_loss.device)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(f"NaN/Inf loss detected.")
        
        # Log metrics
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("loss_mu", loss_mu, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("loss_sigma", loss_sigma, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        if batch_idx % 100 == 0:
            with torch.no_grad():
                pred_mu_real = self._raw_mu_to_real(pred_mu_raw)
                self.log("pred_mu_mean_real", pred_mu_real.mean(), on_step=True, on_epoch=False)
                self.log("target_mu_mean_real", mu_t.mean(), on_step=True, on_epoch=False)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.normalizer_training_cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
        return optimizer
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.log("grad_norm", total_norm, on_step=True, on_epoch=False)

    def train_dataloader(self):
        ds = self._create_training_dataset()
        return DataLoader(
            ds,
            batch_size=self.normalizer_training_cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
        )

    def _compute_per_sample_stats(self) -> list:
        # Same implementation as before
        df = self.dataset.data.copy()
        sample_stats = []
        continuous_vars = getattr(self.dataset_cfg, "continuous_context_vars", None) or []
        
        for idx, row in df.iterrows():
            context_vars_dict = {}
            for var_name in self.static_context_vars:
                if var_name in row:
                    if var_name in continuous_vars:
                        # Normalize inputs using min-max if available, otherwise just cast
                        context_vars_dict[var_name] = torch.tensor(row[var_name], dtype=torch.float32)
                    else:
                        context_vars_dict[var_name] = torch.tensor(row[var_name], dtype=torch.long)
            
            dynamic_ctx_dict = {}
            for var_name in self.dynamic_context_vars:
                ts_data = row.get(var_name)
                if ts_data is not None:
                    if isinstance(ts_data, (np.ndarray, list)):
                        dynamic_ctx_dict[var_name] = np.array(ts_data)
                    else:
                        context_len = self.num_ts_steps if self.num_ts_steps is not None else self.seq_len
                        dynamic_ctx_dict[var_name] = np.full(context_len, ts_data)
            
            dimension_points = []
            for col_name in self.time_series_cols:
                arr = np.array(row[col_name], dtype=np.float32).flatten()
                dimension_points.append(arr)
            
            mu_array = np.array([pts.mean() for pts in dimension_points], dtype=np.float32)
            std_array = np.array([pts.std() + 1e-8 for pts in dimension_points], dtype=np.float32)

            if self.do_scale:
                z_min_array = np.array([(pts - mu).min() / std for pts, mu, std in zip(dimension_points, mu_array, std_array)], dtype=np.float32)
                z_max_array = np.array([(pts - mu).max() / std for pts, mu, std in zip(dimension_points, mu_array, std_array)], dtype=np.float32)
            else:
                z_min_array = z_max_array = None
            
            sample_stats.append((context_vars_dict, dynamic_ctx_dict, mu_array, std_array, z_min_array, z_max_array))
        
        return sample_stats

    def _create_training_dataset(self) -> Dataset:
        class _TrainSet(Dataset):
            def __init__(self, samples, dynamic_context_vars, do_scale, dataset_cfg):
                self.samples = samples
                self.dynamic_context_vars = dynamic_context_vars
                self.do_scale = do_scale
                self.dataset_cfg = dataset_cfg

            def __len__(self) -> int:
                return len(self.samples)

            def __getitem__(self, idx: int):
                static_context_vars_dict, dynamic_context_vars_dict, mu_arr, sigma_arr, zmin_arr, zmax_arr = self.samples[idx]
                for var_name in self.dynamic_context_vars:
                    if var_name in dynamic_context_vars_dict:
                        ts_data = np.array(dynamic_context_vars_dict[var_name])
                        var_info = self.dataset_cfg.context_vars.get(var_name, None)
                        if var_info and var_info[1] is not None:
                            dynamic_context_vars_dict[var_name] = torch.from_numpy(ts_data).long()
                        else:
                            dynamic_context_vars_dict[var_name] = torch.from_numpy(ts_data).float()
                
                mu_t = torch.from_numpy(mu_arr).float()
                sigma_t = torch.from_numpy(sigma_arr).float()
                if self.do_scale:
                    zmin_t = torch.from_numpy(zmin_arr).float()
                    zmax_t = torch.from_numpy(zmax_arr).float()
                else:
                    # Return dummy tensors so DataLoader collate does not see None
                    zmin_t = torch.zeros_like(mu_t)
                    zmax_t = torch.zeros_like(mu_t)
                return static_context_vars_dict, dynamic_context_vars_dict, mu_t, sigma_t, zmin_t, zmax_t

        return _TrainSet(self.sample_stats, self.dynamic_context_vars, self.do_scale, self.dataset_cfg)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.time_series_cols if c not in df.columns]
        if missing:
            df = split_timeseries(df, self.time_series_cols)
        
        df_out = df.copy()
        self.eval()
        continuous_vars = set(self.continuous_vars)
        categorical_ts = getattr(self.dataset, 'categorical_time_series', {})
        group_edges = getattr(self, "_group_bin_edges", {})

        with torch.no_grad():
            for i, row in tqdm(df_out.iterrows(), total=len(df_out), desc="Normalizing"):
                static_context_vars_dict = {}
                dynamic_context_vars_dict = {}
                for v in self.context_vars:
                    if v in self.normalizer_static_vars:
                        if v in continuous_vars and v in group_edges:
                            edges = group_edges[v]
                            bin_idx = np.digitize(np.asarray([float(row[v])]), edges[1:-1], right=False)
                            bin_idx = np.clip(bin_idx, 0, len(edges) - 2).item()
                            static_context_vars_dict[v] = torch.tensor(bin_idx, dtype=torch.long).unsqueeze(0)
                        elif v in continuous_vars:
                            static_context_vars_dict[v] = torch.tensor(row[v], dtype=torch.float32).unsqueeze(0)
                        else:
                            static_context_vars_dict[v] = torch.tensor(row[v], dtype=torch.long).unsqueeze(0)
                    elif v in self.static_context_vars:
                        continue
                    elif v in self.normalizer_dynamic_vars:
                        dynamic_context_vars_dict[v] = torch.tensor(row[v], dtype=torch.float32).unsqueeze(0)
                    elif v in self.dynamic_context_vars:
                        continue  # dynamic var excluded from normalizer conditioning
                    else:
                        raise ValueError(f"Variable {v} not found in context_vars")

                # self(ctx) calls forward, which automatically UNSCALES predictions
                pred_mu, pred_sigma, pred_zmin, pred_zmax, _ = self(static_context_vars_dict, dynamic_context_vars_dict)
                mu, sigma = pred_mu[0].cpu().numpy(), pred_sigma[0].cpu().numpy()

                for d, col in enumerate(self.time_series_cols):
                    if col in categorical_ts:
                        df_out.at[i, col] = np.asarray(row[col]).astype(np.int32)
                        continue

                    arr = np.asarray(row[col], dtype=np.float32)

                    sigma_floor = max(self.min_sigma, 0.01 * np.exp(self.global_log_sigma_mean.cpu().item()))
                    sigma_eff = max(float(sigma[d]), sigma_floor)
                    z = (arr - mu[d]) / sigma_eff
                    if self.do_scale:
                        zmin_, zmax_ = pred_zmin[0, d].item(), pred_zmax[0, d].item()
                        rng = (zmax_ - zmin_) + 1e-8
                        rng_floor = max(self.min_scale_range, .25)
                        rng_eff = max(rng, rng_floor)
                        z = (z - zmin_) / rng_eff

                    df_out.at[i, col] = z
        return df_out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.time_series_cols if c not in df.columns]
        if missing:
            df = split_timeseries(df, self.time_series_cols)

        df_out = df.copy()
        self.eval()
        # continuous_vars = getattr(self.dataset_cfg, "continuous_context_vars", None) or []
        
        continuous_vars = set(self.continuous_vars)
        categorical_ts = getattr(self.dataset, "categorical_time_series", {})
        group_edges = getattr(self, "_group_bin_edges", {})
        with torch.no_grad():
            for i, row in tqdm(df_out.iterrows(), total=len(df_out), desc="Inverse normalizing"):
                static_context_vars_dict = {}
                dynamic_context_vars_dict = {}
                for v in self.context_vars:
                    if v in self.normalizer_static_vars:
                        # Normalizer only conditions on normalizer_static_vars (e.g. station)
                        if v in continuous_vars and v in group_edges:
                            edges = group_edges[v]
                            bin_idx = np.digitize(np.asarray([float(row[v])]), edges[1:-1], right=False)
                            bin_idx = np.clip(bin_idx, 0, len(edges) - 2).item()
                            static_context_vars_dict[v] = torch.tensor(bin_idx, dtype=torch.long).unsqueeze(0)
                        elif v in continuous_vars:
                            static_context_vars_dict[v] = torch.tensor(row[v], dtype=torch.float32).unsqueeze(0)
                        else:
                            static_context_vars_dict[v] = torch.tensor(row[v], dtype=torch.long).unsqueeze(0)
                    elif v in self.static_context_vars:
                        continue  # not used by normalizer
                    elif v in self.normalizer_dynamic_vars:
                        # Dynamic var included in normalizer conditioning
                        val = row[v]
                        dtype_np = np.int64 if v in categorical_ts else np.float32
                        arr = np.asarray(val, dtype=dtype_np)
                        if arr.ndim == 0:
                            arr = arr[np.newaxis, np.newaxis]
                        elif arr.ndim == 1:
                            arr = arr[np.newaxis, :]
                        dtype = torch.long if v in categorical_ts else torch.float32
                        dynamic_context_vars_dict[v] = torch.tensor(arr, dtype=dtype)
                    elif v in self.dynamic_context_vars:
                        continue  # dynamic var excluded from normalizer conditioning
                
                # self(ctx) calls forward, which automatically UNSCALES predictions
                pred_mu, pred_sigma, pred_zmin, pred_zmax, _ = self(static_context_vars_dict, dynamic_context_vars_dict)
                mu, sigma = pred_mu[0].cpu().numpy(), pred_sigma[0].cpu().numpy()

                for d, col in enumerate(self.time_series_cols):
                    z = np.asarray(row[col], dtype=np.float32)
                    sigma_floor = max(self.min_sigma, 0.01 * np.exp(self.global_log_sigma_mean.cpu().item()))
                    sigma_eff = max(float(sigma[d]), sigma_floor)
                    if self.do_scale:
                        zmin_, zmax_ = pred_zmin[0, d].item(), pred_zmax[0, d].item()
                        rng = (zmax_ - zmin_) + 1e-8
                        rng_eff = max(rng, self.min_scale_range)
                        z = z * rng_eff + zmin_
                    arr = z * sigma_eff + mu[d]
                    df_out.at[i, col] = arr
        return df_out

    def _build_training_samples(
        self,
        mode: str = "sample",
        group_vars: Optional[list[str]] = None,
        use_quantile_scale: bool = False,
        q_low: float = 0.02,
        q_high: float = 0.98,
    ) -> list:
        """
        Build training samples.
        Note: Quantile scaling (q_low/q_high) only affects z_min/z_max calculation,
        not the mean/std targets.
        """
        assert mode in {"sample", "group"}, f"mode must be 'sample' or 'group', got {mode}"

        df = self.dataset.data.copy()

        continuous_vars = set(self.continuous_vars)
        dynamic_vars = set(self.dynamic_context_vars)
        static_vars = [v for v in self.static_context_vars]

        if group_vars is None:
            # Default: group by all static categorical (continuous in normalizer_static_vars get binned in group mode)
            group_vars = [v for v in self.normalizer_static_vars if (v not in continuous_vars and v not in dynamic_vars)]

        bad = [v for v in group_vars if v in dynamic_vars]
        if bad:
            raise ValueError(f"group_vars contains dynamic vars {bad}")

        def _row_stats(row) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
            dim_points = []
            for d, col_name in enumerate(self.time_series_cols):
                arr = np.asarray(row[col_name], dtype=np.float32).reshape(-1)
                dim_points.append(arr)

            mu = np.array([x.mean() for x in dim_points], dtype=np.float32)
            std = np.array([x.std() + 1e-8 for x in dim_points], dtype=np.float32)

            if not self.do_scale:
                return mu, std, None, None

            zlow = np.zeros(self.time_series_dims, dtype=np.float32)
            zhigh = np.zeros(self.time_series_dims, dtype=np.float32)

            for i, (x, m, s) in enumerate(zip(dim_points, mu, std)):
                z = (x - m) / s
                if use_quantile_scale:
                    zlow[i] = np.quantile(z, q_low).astype(np.float32)
                    zhigh[i] = np.quantile(z, q_high).astype(np.float32)
                else:
                    zlow[i] = z.min().astype(np.float32)
                    zhigh[i] = z.max().astype(np.float32)

            return mu, std, zlow, zhigh

        samples = []

        if mode == "sample":
            for _, row in df.iterrows():
                context_vars_dict = {}
                for v in self.normalizer_static_vars:
                    if v not in row:
                        continue
                    if v in continuous_vars:
                        context_vars_dict[v] = torch.tensor(row[v], dtype=torch.float32)
                    else:
                        context_vars_dict[v] = torch.tensor(row[v], dtype=torch.long)

                dynamic_ctx_dict = {}
                for v in self.normalizer_dynamic_vars:
                    if v not in row: continue
                    ts_data = row[v]
                    if isinstance(ts_data, (np.ndarray, list)):
                        dynamic_ctx_dict[v] = np.array(ts_data)
                    else:
                        L = self.num_ts_steps if self.num_ts_steps is not None else self.seq_len
                        dynamic_ctx_dict[v] = np.full(L, ts_data)

                mu, std, zlow, zhigh = _row_stats(row)
                samples.append((context_vars_dict, dynamic_ctx_dict, mu, std, zlow, zhigh))

            return samples

        # mode == "group"
        # Pre-process continuous vars for grouping by binning; store edges for inverse_transform
        self._group_bin_edges = {}
        for v in group_vars:
            if v in continuous_vars:
                n_bins = getattr(self.dataset_cfg, "numeric_context_bins", 5)
                binned, edges = pd.cut(
                    df[v], bins=n_bins, labels=False, include_lowest=True,
                    duplicates="drop", retbins=True
                )
                self._group_bin_edges[v] = np.asarray(edges, dtype=np.float64)
                df[v] = binned
        print(group_vars, "group_vars")
        grouped = df.groupby(list(group_vars), dropna=False)

        for group_key, gdf in grouped:
            if len(group_vars) == 1:
                group_key = (group_key,)

            context_vars_dict = {}
            for i, v in enumerate(group_vars):
                val = group_key[i]
                if v in continuous_vars:
                    if pd.isna(val): val = 0
                    context_vars_dict[v] = torch.tensor(int(val), dtype=torch.long)
                else:
                    context_vars_dict[v] = torch.tensor(val, dtype=torch.long)

            dynamic_ctx_dict = {} 

            dim_points = [[] for _ in range(self.time_series_dims)]
            for _, row in gdf.iterrows():
                for d, col_name in enumerate(self.time_series_cols):
                    arr = np.asarray(row[col_name], dtype=np.float32).reshape(-1)
                    dim_points[d].append(arr)

            dim_points = [np.concatenate(xs, axis=0) if len(xs) else np.zeros((0,), dtype=np.float32)
                        for xs in dim_points]

            mu = np.array([x.mean() if x.size else 0.0 for x in dim_points], dtype=np.float32)
            std = np.array([x.std() + 1e-8 if x.size else 1.0 for x in dim_points], dtype=np.float32)

            if self.do_scale:
                zlow = np.zeros(self.time_series_dims, dtype=np.float32)
                zhigh = np.zeros(self.time_series_dims, dtype=np.float32)
                for i, (x, m, s) in enumerate(zip(dim_points, mu, std)):
                    if x.size == 0:
                        zlow[i], zhigh[i] = -2.0, 2.0
                        continue
                    z = (x - m) / s
                    if use_quantile_scale:
                        zlow[i] = np.quantile(z, q_low).astype(np.float32)
                        zhigh[i] = np.quantile(z, q_high).astype(np.float32)
                    else:
                        zlow[i] = z.min().astype(np.float32)
                        zhigh[i] = z.max().astype(np.float32)
            else:
                zlow = zhigh = None

            samples.append((context_vars_dict, dynamic_ctx_dict, mu, std, zlow, zhigh))

        return samples