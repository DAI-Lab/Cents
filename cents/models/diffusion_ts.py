import copy
import math
from typing import Any, Optional, Tuple

from pyparsing import alphas

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from contextlib import contextmanager



from cents.models.base import GenerativeModel
from cents.models.model_utils import (
    Transformer,
    cosine_beta_schedule,
    default,
    linear_beta_schedule,
    total_correlation,
    cosine_beta_schedule_logsnr,
)
from cents.models.registry import register_model

def _randn_like_correlated(
    x: torch.Tensor, correlated: bool
) -> torch.Tensor:
    """White noise with same shape as x. If correlated and C>1, same noise broadcast across last dim."""
    if not correlated or x.dim() < 3 or x.shape[-1] == 1:
        return torch.randn_like(x)
    return torch.randn(*x.shape[:-1], 1, device=x.device, dtype=x.dtype).expand_as(x).clone()


def _randn_shape_correlated(
    shape: tuple, device: torch.device, dtype: torch.dtype, correlated: bool
) -> torch.Tensor:
    """Randn with given shape. If correlated and shape[-1]>1, same noise broadcast across last dim."""
    if not correlated or len(shape) < 3 or shape[-1] == 1:
        return torch.randn(shape, device=device, dtype=dtype)
    B, L, C = shape[0], shape[1], shape[2]
    return torch.randn(B, L, 1, device=device, dtype=dtype).expand(shape).clone()


def blueish_noise_like(
    x: torch.Tensor, power: float = 1.0, eps: float = 1e-6, correlated: bool = False
) -> torch.Tensor:
    """
    Generate 'blue-ish' noise: more energy at higher frequencies.
    - power = 0.0 -> white noise
    - power > 0.0 -> increasingly high-frequency-heavy (blue/violet-ish)
    Returns noise with ~unit std per sample/channel so diffusion scaling stays consistent.
    - correlated: if True and x has multiple channels (C>1), same noise is used for all channels.

    x: (B, L, C) where L is time dimension.
    """
    B, L, C = x.shape

    if power == 0.0:
        return _randn_like_correlated(x, correlated)

    # When correlated and C>1, generate (B, L, 1) then expand after FFT shaping
    if correlated and C > 1:
        n = torch.randn(B, L, 1, device=x.device, dtype=torch.float32)
    else:
        n = torch.randn(B, L, C, device=x.device, dtype=torch.float32)

    # real FFT over time
    N = torch.fft.rfft(n, dim=1)  # (B, F, C) or (B, F, 1)
    freqs = torch.fft.rfftfreq(L, d=1.0).to(x.device)  # (F,)

    # Amplitude shaping:
    amp = (freqs.clamp_min(eps) ** (power / 2.0)).view(1, -1, 1)
    N = N * amp

    n_blue = torch.fft.irfft(N, n=L, dim=1)  # (B, L, C) or (B, L, 1)

    # Re-normalize per (B,C) to unit std across time
    n_blue = n_blue / n_blue.std(dim=1, keepdim=True).clamp_min(1e-6)

    if correlated and C > 1:
        n_blue = n_blue.expand(B, L, C).clone()

    out = n_blue.to(dtype=x.dtype)
    return out


@register_model("diffusion_ts", "Watts_2_1D", "Watts_2_2D")
class Diffusion_TS(GenerativeModel):
    """
    PyTorch-Lightning module for a conditional time-series diffusion model.

    Uses a Transformer backbone to predict and denoise time series over
    discrete diffusion timesteps. Supports EMA smoothing and configurable
    beta schedules. Optional reconstruction-guided sampling (Algorithms 1 & 2)
    via sample_reconstruction_guided(shape, static_context_vars, dynamic_context_vars, algorithm="alg1"|"alg2")
    when conditional observed data x_a is provided.

    Training objective (config model.training_objective): x0, epsilon, or v.
      - x0: predict clean sample; loss = L1/L2(model_out, x_clean).
      - epsilon: predict noise; loss = L1/L2(pred_epsilon, noise); pred_epsilon derived from model x0.
      - v: v-parameterization; loss = L1/L2(pred_v, true_v); pred_v derived from model x0.
    The network always outputs x0 (fc layer); sampling uses x0 and q_posterior unchanged.
    Variance-debugging reference:
      (a) Sampling: model predicts x0; reverse step uses q_posterior(x0,x_t,t);
          noise term = sqrt(posterior_variance). Same beta schedule in train and sample.
      (b) Normalization: per-group (context) z-score + optional [zmin,zmax] scale;
          denorm must use the same normalizer.inverse_transform. Run
          scripts/check_diffusion_consistency.py to verify norm/denorm identity.
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: OmegaConf config with fields:
                dataset.seq_len: sequence length
                dataset.time_series_dims: number of series dims
                model.*: diffusion-specific hyperparameters
                trainer: optimizer and scheduler configs
        """
        super().__init__(cfg)
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.seq_len = cfg.dataset.seq_len
        self.time_series_dims = cfg.dataset.time_series_dims
        self.eta = cfg.model.eta
        self.use_ff = cfg.model.use_ff
        self.ff_weight = default(cfg.model.reg_weight, math.sqrt(self.seq_len) / 5)
        self.embedding_dim = cfg.model.cond_emb_dim
        self.context_reconstruction_loss_weight = (
            cfg.model.context_reconstruction_loss_weight
        )
        # Reconstruction-guided sampling (Algorithms 1 & 2)
        self.recon_guide_eta = getattr(cfg.model, "recon_guide_eta", 0.1)
        self.recon_guide_gamma = getattr(cfg.model, "recon_guide_gamma", 1.0)
        self.recon_guide_algorithm = getattr(cfg.model, "recon_guide_algorithm", "none")
        self.recon_guide_K = getattr(cfg.model, "recon_guide_K", 3)
        # Verify context modules are initialized (static, dynamic, or both)
        if not hasattr(self, 'static_context_module') and not hasattr(self, 'dynamic_context_module'):
            raise ValueError("At least one context module (static or dynamic) must be initialized")

        # Context embedding dropout (training only): zeros the *entire* embedding for a random
        # subset of samples (Bernoulli with prob p).  This is CFG-compatible — the model learns
        # to denoise both with and without context, enabling guidance-scale inference later.
        self.context_embed_dropout_p = getattr(cfg.model, "context_embed_dropout", 0.0)
        # Keep the nn.Dropout attribute so old checkpoints that saved it don't break on load.
        self.context_embed_dropout = nn.Dropout(p=0.0)

        # Optional dual head for x̂_a / x̂_b: separate output heads for conditional vs rest of sequence
        self.recon_cond_len = getattr(cfg.model, "recon_cond_len", None)
        if self.recon_cond_len is not None:
            cond_len = int(self.recon_cond_len)
            assert 0 < cond_len < self.seq_len, "recon_cond_len must be in (0, seq_len)"
            self.fc_a = nn.Linear(self.time_series_dims, self.time_series_dims)
            self.fc_b = nn.Linear(self.time_series_dims, self.time_series_dims)
            self.fc = None
        else:
            self.fc_a = None
            self.fc_b = None
            self.fc = nn.Linear(
                self.time_series_dims, self.time_series_dims
            )
        # Transformer backbone (now uses AdaLN conditioning instead of input concatenation)
        self.model = Transformer(
            n_feat=self.time_series_dims,
            n_channel=self.seq_len,
            n_layer_enc=cfg.model.n_layer_enc,
            n_layer_dec=cfg.model.n_layer_dec,
            n_heads=cfg.model.n_heads,
            attn_pdrop=cfg.model.attn_pd,
            resid_pdrop=cfg.model.resid_pd,
            mlp_hidden_times=cfg.model.mlp_hidden_times,
            max_len=self.seq_len,
            n_embd=cfg.model.d_model,
            conv_params=[cfg.model.kernel_size, cfg.model.padding_size],
            cond_dim=self.embedding_dim,
            has_dynamic_ctx=self.dynamic_context_module is not None,
        )

        self.blue_noise_power = cfg.model.blue_noise_power
        self.correlated_noise = bool(getattr(cfg.model, "correlated_noise", False))

        # EMA helper will be initialized on train start
        self._ema: Optional[EMA] = None

        # set up beta schedule
        if cfg.model.beta_schedule == "linear":
            betas = linear_beta_schedule(cfg.model.n_steps)
        elif cfg.model.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.model.n_steps)
        elif cfg.model.beta_schedule == "cosine_logsnr":
            betas = cosine_beta_schedule_logsnr(cfg.model.n_steps)
        else:
            raise ValueError("Unknown beta schedule")

        eps = 1e-5
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).clamp(min=eps, max=1.0 - eps)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0 - eps)

        self.num_timesteps = betas.shape[0]
        self.sampling_timesteps = default(
            cfg.model.sampling_timesteps, self.num_timesteps
        )
        self.fast_sampling = self.sampling_timesteps < self.num_timesteps
        self.loss_type = cfg.model.loss_type
        self.training_objective = getattr(
            cfg.model, "training_objective", "x0"
        ).lower()
        if self.training_objective not in ("x0", "eps", "v"):
            raise ValueError(
                f"training_objective must be one of x0, eps, v; got {self.training_objective}"
            )

        # register buffers for diffusion coefficients
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        # posterior coefficients
        pmc1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        pmc2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", pmc1)
        self.register_buffer("posterior_mean_coef2", pmc2)

        # Loss weighting: default (legacy), uniform, snr, or min_snr (via compute_snr_weights)
        loss_weighting = getattr(cfg.model, "loss_weighting", "default")
        min_snr_gamma = getattr(cfg.model, "min_snr_gamma", 5.0)
        if loss_weighting == "default":
            lw = torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas.clamp(min=1e-8) / 100
        else:
            lw = self.compute_snr_weights(
                alphas_cumprod,
                loss_weighting=loss_weighting,
                objective=self.training_objective,
                gamma=min_snr_gamma,
            )
        self.register_buffer("loss_weight", lw)

        # choose reconstruction loss
        if self.loss_type == "l1":
            self.recon_loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.recon_loss_fn = F.mse_loss  # MSE for continuous RVs
        else:
            raise ValueError("Invalid loss type")

        self.auxiliary_loss = nn.CrossEntropyLoss()
        
        # Get continuous variables from config to distinguish them in loss computation
        self.continuous_context_vars = [k for k, v in cfg.dataset.context_vars.items() if v[0] == "continuous"]
        self.categorical_context_vars = [k for k, v in cfg.dataset.context_vars.items() if v[0] == "categorical"]

    def _get_context_embedding(
        self, static_context_vars: dict, dynamic_context_vars: dict = None,
        batch_size: int = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Get context embeddings from static and/or dynamic context modules.

        Returns:
            static_emb: (B, embedding_dim) — fed into AdaLN via the Transformer's cond path.
            dyn_ctx_seq: (B, T, embedding_dim) or None — fed into cross-attention in each
                         DecoderBlock when the dynamic module returns_sequence=True.
            all_logits: Dict of auxiliary classification/regression logits (static only).
        """
        all_logits = {}
        static_emb = None
        dyn_ctx_seq = None

        # --- Static context (categorical + continuous) → (B, emb_dim) for AdaLN ---
        if self.static_context_module is not None and static_context_vars:
            device = next(self.static_context_module.parameters()).device
            static_vars = {
                k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                for k, v in static_context_vars.items()
            }
            static_emb, static_logits = self.static_context_module(static_vars)
            all_logits.update(static_logits)

        # --- Dynamic context (time series) → (B, T, emb_dim) for cross-attention ---
        if self.dynamic_context_module is not None and dynamic_context_vars:
            device = next(self.dynamic_context_module.parameters()).device
            dyn_vars = {
                k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                for k, v in dynamic_context_vars.items()
            }
            dyn_out, dyn_logits = self.dynamic_context_module(dyn_vars)
            all_logits.update(dyn_logits)

            if getattr(self.dynamic_context_module, "returns_sequence", False):
                # (B, T, emb_dim) — routed to cross-attention, not AdaLN
                dyn_ctx_seq = dyn_out.float() if dyn_out.is_floating_point() else dyn_out
            else:
                # Legacy pooled vector: fuse with static embedding via combine_mlp
                if static_emb is not None and self.combine_mlp is not None:
                    combined = torch.cat([static_emb, dyn_out], dim=1)
                    static_emb = self.combine_mlp(combined)
                elif static_emb is None:
                    static_emb = dyn_out

        if static_emb is None:
            if batch_size is None:
                raise ValueError("No context provided and batch_size not given — cannot create unconditional embedding")
            device = next(self.parameters()).device
            static_emb = torch.zeros(batch_size, self.embedding_dim, device=device)

        if static_emb.is_floating_point():
            static_emb = static_emb.float()
        if self.training and self.context_embed_dropout_p > 0:
            mask = torch.bernoulli(
                torch.full((static_emb.shape[0], 1), 1.0 - self.context_embed_dropout_p,
                           device=static_emb.device, dtype=static_emb.dtype)
            )
            static_emb = static_emb * mask
        return static_emb, dyn_ctx_seq, all_logits

    def _decode_to_x0(self, backbone: torch.Tensor) -> torch.Tensor:
        """
        Map backbone output (trend+season) to x0 prediction. Uses single fc or dual fc_a/fc_b when recon_cond_len is set.
        backbone: (B, L, time_series_dims).
        """
        if self.fc is not None:
            out = self.fc(backbone)
        else:
            cond_len = self.recon_cond_len
            out = torch.cat([
                self.fc_a(backbone[:, :cond_len]),
                self.fc_b(backbone[:, cond_len:]),
            ], dim=1)
        return out

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate noise given a noised sample x_t and predicted x0.

        Args:
            x_t: Noised input at timestep t, shape (B,L,C).
            t: Tensor of timesteps, shape (B,).
            x0: Predicted clean sample at t=0, shape (B,L,C).

        Returns:
            Noise prediction tensor same shape as x_t.
        """
        out = (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t - x0
        ) / self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)
        return out

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct x0 from noisy x_t and noise.

        Args:
            x_t: Noised input at timestep t, shape (B,L,C).
            t: Tensor of timesteps, shape (B,).
            noise: Actual noise added, shape (B,L,C).

        Returns:
            Reconstructed x0 tensor same shape as x_t.
        """
        out = (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t
            - self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise
        )
        return out

    def predict_start_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct x0 from x_t and v-parameterization.
        v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x0  =>  x0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v
        """
        out = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x_t
            - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * v
        )
        return out

    def predict_noise_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct epsilon from x_t and v-parameterization.
        v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x0  =>  epsilon = sqrt(1 - alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v
        """
        out = (
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * x_t
            + self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * v
        )
        return out


    def compute_snr_weights(
        self,
        alphas_cumprod: torch.Tensor,
        *,
        loss_weighting: str,
        objective: str,
        gamma: float = 5.0,
    ) -> torch.Tensor:
        """
        SNR-based loss weighting per timestep.

        Args:
            alphas_cumprod: Cumulative product of alphas, shape (n_steps,).
            loss_weighting: "uniform" | "snr" | "min_snr".
            objective: "eps" | "x0" | "v" — must match training objective.
            gamma: Cap for SNR when loss_weighting == "min_snr".

        Returns:
            Weight tensor same shape as alphas_cumprod.
        """
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)

        if loss_weighting == "uniform":
            return torch.ones_like(snr)

        if loss_weighting == "snr":
            if objective == "eps":
                return 1.0 / (snr + 1.0)
            elif objective == "x0":
                return snr / (snr + 1.0)
            elif objective == "v":
                return 1.0 / torch.sqrt(snr + 1.0)
            else:
                raise ValueError(objective)

        if loss_weighting == "min_snr":
            snr_c = torch.minimum(snr, torch.full_like(snr, gamma))
            if objective == "eps":
                return snr_c / snr.clamp(min=1e-8)
            elif objective == "x0":
                return snr_c
            elif objective == "v":
                return snr_c / (snr + 1.0)
            else:
                raise ValueError(objective)

        raise ValueError(loss_weighting)

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Predicted x0, shape (B,L,C).
            x_t: Noised sample at t, shape (B,L,C).
            t: Tensor of timesteps, shape (B,).

        Returns:
            Tuple of (mean, variance, log_variance) tensors for posterior.
        """
        pm = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x_start
            + self.posterior_mean_coef2[t].view(-1, 1, 1) * x_t
        )
        pv = self.posterior_variance[t].view(-1, 1, 1)
        plv = self.posterior_log_variance_clipped[t].view(-1, 1, 1)
        return pm, pv, plv

    def forward(self, x: torch.Tensor, static_context_vars: dict, dynamic_context_vars: dict = None) -> Tuple[torch.Tensor, dict]:
        """
        Single forward pass: add noise, predict denoised output and compute reconstruction loss.

        Args:
            x: Clean input batch, shape (B,L,C).
            context_vars: Dict of context tensors used for conditioning.

        Returns:
            rec_loss: Reconstruction loss tensor.
            cond_logits: Classification logits dict from context module.
        """

        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device)
        embedding, dyn_ctx_seq, cond_classification_logits = self._get_context_embedding(static_context_vars, dynamic_context_vars, batch_size=b)

        noise = blueish_noise_like(
            x, power=self.blue_noise_power, correlated=self.correlated_noise
        )
        x_noisy = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x
            + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
        )
        trend, season = self.model(x_noisy, t, padding_masks=None, cond=embedding, dyn_ctx=dyn_ctx_seq)
        x_start_pred = self._decode_to_x0((trend + season).contiguous())
        # Compute loss based on training objective (network always predicts x0; we derive epsilon/v as needed)
        if self.training_objective == "x0":
            loss_per_elem = self.recon_loss_fn(x_start_pred, x, reduction="none")
        elif self.training_objective == "eps":
            pred_noise = self.predict_noise_from_start(x_noisy, t, x_start_pred)
            loss_per_elem = self.recon_loss_fn(pred_noise, noise, reduction="none")
        else:  # v
            sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
            sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1).clamp(min=1e-3)
            pred_v = (sqrt_ab * x_noisy - x_start_pred) / sqrt_1mab
            true_v = (
                self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * noise
                - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * x
            )
            loss_per_elem = self.recon_loss_fn(pred_v, true_v, reduction="none")
        rec_loss = (
            self.loss_weight[t].view(-1, 1, 1) * loss_per_elem
        ).mean()
        fourier_loss = torch.tensor(0.0, device=self.device)
        if self.use_ff:
            # FFT is not generally supported in fp16 for non power-of-2 sizes on cuFFT.
            # Run FFT in fp32 outside autocast.
            with torch.autocast(device_type=x.device.type, enabled=False):
                x1 = x_start_pred.transpose(1, 2).float()
                x2 = x.transpose(1, 2).float()

                fft1 = torch.fft.fft(x1, norm="forward")
                fft2 = torch.fft.fft(x2, norm="forward")

            mag1 = torch.abs(fft1)
            mag2 = torch.abs(fft2)

            fourier_loss = self.recon_loss_fn(mag1, mag2, reduction="none")

            fourier_loss = (self.loss_weight[t].view(-1, 1, 1) * fourier_loss).mean()


        return rec_loss, cond_classification_logits, fourier_loss.mean()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step: compute total loss including context reconstruction.

        Args:
            batch: Tuple(ts_batch, cond_batch).
            batch_idx: Index of batch (unused).

        Returns:
            total_loss: Scalar training loss.
        """
        ts_batch, static_context_batch, dynamic_context_batch = batch
        rec_loss, cond_class_logits, fourier_loss = self(ts_batch, static_context_batch, dynamic_context_batch)

        cond_loss = 0.0
        for var_name, outputs in cond_class_logits.items():
            labels = static_context_batch[var_name]
            if var_name in self.continuous_context_vars:
                loss = F.mse_loss(outputs, labels.float())
            elif var_name in self.categorical_context_vars:
                loss = self.auxiliary_loss(outputs, labels)
            cond_loss += loss.mean()

        h, _, _ = self._get_context_embedding(static_context_batch, dynamic_context_batch, batch_size=ts_batch.shape[0])
        tc_term = (
            self.cfg.model.tc_loss_weight * total_correlation(h)
            if self.cfg.model.tc_loss_weight > 0.0
            else torch.tensor(0.0, device=self.device)
        )

        total_loss = (
            rec_loss + self.context_reconstruction_loss_weight * cond_loss + tc_term + fourier_loss * self.ff_weight
        )

        # Skip this batch entirely if loss is bad — avoids corrupting weights before EMA can help

        self.log_dict(
            {
                "train_loss": total_loss.item(),
                "rec_loss": rec_loss.item(),
                "cond_loss": cond_loss.item() if isinstance(cond_loss, torch.Tensor) else float(cond_loss),
                "tc_loss": tc_term,
                "fourier_loss": fourier_loss.item(),
            },
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
        )
        return total_loss

    def configure_optimizers(self) -> dict:
        """
        Set up optimizer and LR scheduler.

        Returns:
            Dict with "optimizer" and "lr_scheduler" entries.
        """
        optimizer = Adam(
            self.parameters(), lr=self.cfg.trainer.base_lr, betas=(0.9, 0.96)
        )
        scheduler = ReduceLROnPlateau(optimizer, **self.cfg.trainer.lr_scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }

    def on_train_start(self) -> None:
        """
        Initialize EMA helper at start of training.
        """
        if self._ema is None:
            object.__setattr__(self, '_ema', EMA(
                self.model,
                beta=self.cfg.model.ema_decay,
                update_every=self.cfg.model.ema_update_interval,
            ))
        # EMA is not a registered submodule so PL won't move it automatically
        self._ema.to(self.device)

    def on_train_epoch_start(self) -> None:
        """
        Apply linear LR warmup for the first warmup_epochs epochs.
        After warmup, ReduceLROnPlateau takes over untouched.
        """
        warmup_epochs = self.cfg.trainer.get("warmup_epochs", 0)
        if warmup_epochs <= 0:
            return
        epoch = self.current_epoch
        if epoch >= warmup_epochs:
            return
        target_lr = self.cfg.trainer.base_lr
        warmup_lr = target_lr * max(0.01, epoch / warmup_epochs)
        for opt in self.trainer.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = warmup_lr

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log global gradient norm each step for observability."""
        total_norm_sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().float().norm(2).item() ** 2
        grad_norm = total_norm_sq ** 0.5
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """
        Apply EMA update after each batch end.
        Skip if the step was skipped due to NaN loss (outputs is None).
        """
        if outputs is None:
            return
        if hasattr(self, '_ema') and self._ema:
            self._ema.update()


    def load_state_dict(self, state_dict, strict=True):
        # Strip legacy _ema.* keys — EMA is restored separately via on_load_checkpoint.
        # Old checkpoints have these because _ema was previously a registered submodule.
        filtered = {k: v for k, v in state_dict.items() if not k.startswith('_ema.')}
        return super().load_state_dict(filtered, strict=strict)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if 'ema_state_dict' in checkpoint and False:
            if self._ema is None:
                object.__setattr__(self, '_ema', EMA(
                    self.model,
                    beta=self.cfg.model.ema_decay,
                    update_every=self.cfg.model.ema_update_interval,
                ))
            self._ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            print(f"[EMA] Restored EMA weights from checkpoint")
        else:
            print(f"[EMA] No EMA weights in checkpoint, initializing fresh")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self._ema is not None:
            checkpoint['ema_state_dict'] = self._ema.ema_model.state_dict()


    def _predict_x0_from_xt_with_grad(
        self, x_t: torch.Tensor, t: torch.Tensor, embedding: torch.Tensor,
        dyn_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict x0 from x_t with gradients enabled (for reconstruction-guided sampling).
        Returns x_start of shape (B, L, C). Call with x_t.requires_grad_(True).
        """
        trend, season = self.model(x_t, t, padding_masks=None, cond=embedding, dyn_ctx=dyn_ctx)
        x_start = self._decode_to_x0((trend + season).contiguous())
        return x_start

    @torch.no_grad()
    def model_predictions(
        self, x: torch.Tensor, t: torch.Tensor, embedding: torch.Tensor,
        dyn_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict both noise and clean sample from current x.

        Returns:
            pred_noise: predicted noise tensor.
            x_start: predicted clean sample tensor.
        """
        trend, season = self.model(x, t, padding_masks=None, cond=embedding, dyn_ctx=dyn_ctx)
        x_start = self._decode_to_x0((trend + season).contiguous())
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    @staticmethod
    def _replace_conditional(
        x_a: torch.Tensor, x_prev: torch.Tensor, cond_len: int
    ) -> torch.Tensor:
        """
        Replace the first cond_len time steps of x_prev with conditional data x_a.
        x_a: (B, cond_len, C), x_prev: (B, L, C). Returns (B, L, C).
        """
        out = x_prev.clone()
        out[:, :cond_len] = x_a
        return out

    @torch.no_grad()
    def p_mean_variance(
        self, x: torch.Tensor, t: torch.Tensor, embedding: torch.Tensor,
        dyn_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).

        Returns:
            pm, pv, plv, x_start: posterior parameters and predicted x0.
        """
        pred_noise, x_start = self.model_predictions(x, t, embedding, dyn_ctx=dyn_ctx)
        pm, pv, plv = self.q_posterior(x_start, x, t)
        return pm, pv, plv, x_start

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, t: int, embedding: torch.Tensor,
        dyn_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t using posterior distribution.
        """
        bt = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        pm, pv, plv, _ = self.p_mean_variance(x, bt, embedding, dyn_ctx=dyn_ctx)
        noise = (
            blueish_noise_like(x, power=self.blue_noise_power, correlated=self.correlated_noise)
            if t > 0
            else 0
        )
        out = pm + (0.5 * plv).exp() * noise
        return out

    def _reconstruction_guided_step_alg1(
        self,
        x_t: torch.Tensor,
        t: int,
        embedding: torch.Tensor,
        x_a: torch.Tensor,
        cond_len: int,
        eta: float,
        gamma: float,
        dyn_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One step of Algorithm 1: predict x̂_0, compute L_1 + γ*L_2, then
        x̃_0 = x̂_0 + η ∇_{x_t}(L_1 + γ*L_2); sample x_{t-1} ~ N(μ(x̃_0, x_t), Σ) and Replace(x_a, x_{t-1}).
        """
        bt = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        x_t = x_t.detach().requires_grad_(True)

        x_start = self._predict_x0_from_xt_with_grad(x_t, bt, embedding, dyn_ctx=dyn_ctx)
        x_hat_a = x_start[:, :cond_len]
        L_1 = (x_a - x_hat_a).pow(2).mean()

        pm, pv, plv = self.q_posterior(x_start, x_t, bt)
        noise = (
            blueish_noise_like(x_t, power=self.blue_noise_power, correlated=self.correlated_noise)
            if t > 0
            else 0
        )
        x_prev_initial = (pm + (0.5 * plv).exp() * noise).detach()
        L_2 = ((x_prev_initial - pm).pow(2) / pv.clamp(min=1e-8)).mean()

        loss = L_1 + gamma * L_2
        loss.backward()
        with torch.no_grad():
            x_tilde_0 = x_start.detach() + eta * x_t.grad
            pm_final, pv_final, plv_final = self.q_posterior(x_tilde_0, x_t.detach(), bt)
            noise_final = (
                _randn_like_correlated(x_t, self.correlated_noise)
                if t > 0
                else torch.zeros_like(x_t, device=x_t.device)
            )
            x_prev = pm_final + (0.5 * plv_final).exp() * noise_final
            x_prev = self._replace_conditional(x_a, x_prev, cond_len)
        return x_prev

    def _reconstruction_guided_step_alg2(
        self,
        x_t: torch.Tensor,
        t: int,
        embedding: torch.Tensor,
        x_a: torch.Tensor,
        cond_len: int,
        eta: float,
        gamma: float,
        K: int,
        dyn_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One step of Algorithm 2: K inner gradient updates on x_t, then one final sample and Replace.
        """
        bt = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        embedding_detach = embedding.detach()
        dyn_ctx_detach = dyn_ctx.detach() if dyn_ctx is not None else None
        x_t = x_t.detach().clone()

        for _ in range(K):
            x_t = x_t.requires_grad_(True)
            x_start = self._predict_x0_from_xt_with_grad(x_t, bt, embedding_detach, dyn_ctx=dyn_ctx_detach)
            x_hat_a = x_start[:, :cond_len]
            L_1 = (x_a - x_hat_a).pow(2).mean()
            pm, pv, plv = self.q_posterior(x_start, x_t, bt)
            noise = (
                blueish_noise_like(x_t, power=self.blue_noise_power, correlated=self.correlated_noise)
                if t > 0
                else 0
            )
            x_prev_initial = (pm + (0.5 * plv).exp() * noise).detach()
            L_2 = ((x_prev_initial - pm).pow(2) / pv.clamp(min=1e-8)).mean()
            loss = L_1 + gamma * L_2
            loss.backward()
            with torch.no_grad():
                x_t = x_t + eta * x_t.grad
                x_t = x_t.detach()

        with torch.no_grad():
            x_start_final = self._predict_x0_from_xt_with_grad(x_t, bt, embedding_detach, dyn_ctx=dyn_ctx_detach)
            pm_final, pv_final, plv_final = self.q_posterior(x_start_final, x_t, bt)
            noise_final = (
                blueish_noise_like(x_t, power=self.blue_noise_power, correlated=self.correlated_noise)
                if t > 0
                else 0
            )
            x_prev = pm_final + (0.5 * plv_final).exp() * noise_final
            x_prev = self._replace_conditional(x_a, x_prev, cond_len)
        return x_prev

    def sample_reconstruction_guided(
        self,
        x_a: torch.Tensor,
        shape: Tuple[int, int, int],
        static_context_vars: dict,
        dynamic_context_vars: dict = None,
        algorithm: str = "alg1",
    ) -> torch.Tensor:
        """
        Full reverse-pass sampling with reconstruction guidance (Algorithm 1 or 2).

        Args:
            shape: (batch_size, seq_len, time_series_dims).
            static_context_vars: static context conditioning dict.
            dynamic_context_vars: dynamic context conditioning dict.
            x_a: Conditional (observed) data, shape (B, cond_len, C). First cond_len
                 time steps to reconstruct; model output is split as x̂_0 = [x̂_a, x̂_b].
            algorithm: "alg1" (one gradient step per t) or "alg2" (K inner steps per t).

        Returns:
            Generated samples (B, L, C) with first cond_len steps equal to x_a.

        Architecture / config notes (optional improvements):
        - x̂_0 split: Currently x̂_0 is the single model output (B,L,C); we split by time
          so x̂_a = x̂_0[:, :cond_len], x̂_b = x̂_0[:, cond_len:]. Optionally use two output
          heads (one for x̂_a, one for x̂_b) if you want different capacities.
        - Context dropout: Consider adding dropout on the context embedding (or on
          context encoder outputs) during training to improve robustness of
          reconstruction-guided sampling at test time.
        - Config: recon_guide_eta (gradient scale), recon_guide_gamma (L1 vs L2 trade-off),
          recon_guide_K (int or list of length num_timesteps for per-t inner steps in alg2).
        """
        cond_len = x_a.shape[1]
        assert cond_len < shape[1], "x_a length must be < seq_len"
        assert x_a.shape[0] == shape[0] and x_a.shape[2] == shape[2]
        eta = self.recon_guide_eta
        gamma = self.recon_guide_gamma
        x = _randn_shape_correlated(
            shape, self.device, torch.float32, self.correlated_noise
        )
        embedding, dyn_ctx_seq, _ = self._get_context_embedding(static_context_vars, dynamic_context_vars, batch_size=shape[0])
        x_a = x_a.to(self.device)

        for t in reversed(range(self.num_timesteps)):
            K_t = (
                self.recon_guide_K[t] if t < len(self.recon_guide_K) else self.recon_guide_K[-1]
                if isinstance(self.recon_guide_K, (list, tuple))
                else self.recon_guide_K
            )
            if algorithm == "alg1":
                x = self._reconstruction_guided_step_alg1(
                    x, t, embedding, x_a, cond_len, eta, gamma, dyn_ctx=dyn_ctx_seq
                )
            else:
                x = self._reconstruction_guided_step_alg2(
                    x, t, embedding, x_a, cond_len, eta, gamma, K_t, dyn_ctx=dyn_ctx_seq
                )
        return x

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int], static_context_vars: dict, dynamic_context_vars: dict = None) -> torch.Tensor:
        """
        Full reverse-pass sampling over all timesteps.

        Args:
            shape: (batch_size, seq_len, dims)
            static_context_vars: static context conditioning dict
            dynamic_context_vars: dynamic context conditioning dict

        Returns:
            Generated samples tensor.
        """
        x = _randn_shape_correlated(
            shape, self.device, torch.float32, self.correlated_noise
        )
        embedding, dyn_ctx_seq, _ = self._get_context_embedding(static_context_vars, dynamic_context_vars, batch_size=shape[0])
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, embedding, dyn_ctx=dyn_ctx_seq)
        return x

    @torch.no_grad()
    def fast_sample(
        self, shape: Tuple[int, int, int], static_context_vars: dict,
        dynamic_context_vars: dict = None, cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        DDIM sampling with optional classifier-free guidance.

        cfg_scale=1.0 → standard conditional sampling (no guidance).
        cfg_scale>1.0 → CFG: runs both conditional and unconditional passes each step
                         and blends pred_noise = uncond + scale*(cond - uncond).
                         Requires the model to have been trained with context_embed_dropout > 0.
        """
        x = _randn_shape_correlated(
            shape, self.device, torch.float32, self.correlated_noise
        )
        embedding, dyn_ctx_seq, _ = self._get_context_embedding(static_context_vars, dynamic_context_vars, batch_size=shape[0])

        use_cfg = cfg_scale > 1.0
        if use_cfg:
            uncond_emb = torch.zeros_like(embedding)
            uncond_dyn = torch.zeros_like(dyn_ctx_seq) if dyn_ctx_seq is not None else None

        times = torch.linspace(
            -1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1
        )
        times = list(reversed(times.int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))
        for time, time_next in pairs:
            bt = torch.full((x.shape[0],), time, device=self.device, dtype=torch.long)
            if use_cfg:
                pred_noise_u, _ = self.model_predictions(x, bt, uncond_emb, dyn_ctx=uncond_dyn)
                pred_noise_c, _ = self.model_predictions(x, bt, embedding, dyn_ctx=dyn_ctx_seq)
                pred_noise = pred_noise_u + cfg_scale * (pred_noise_c - pred_noise_u)
                x_start = self.predict_start_from_noise(x, bt, pred_noise)
            else:
                pred_noise, x_start = self.model_predictions(x, bt, embedding, dyn_ctx=dyn_ctx_seq)
            if time_next < 0:
                x = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                self.eta
                * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = _randn_like_correlated(x, self.correlated_noise)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return x

    @contextmanager
    def ema_scope(self):
        if hasattr(self, "_ema") and self._ema and getattr(self.cfg.model, "use_ema_sampling", False):
            self._ema.store(self.model.parameters())
            self._ema.copy_to(self.model.parameters())
            try:
                yield
            finally:
                self._ema.restore(self.model.parameters())
        else:
            yield

    def generate(self, static_context_vars: dict, dynamic_context_vars: dict = None, n: int = None) -> torch.Tensor:
        """
        Public entry to generate conditioned samples in batches.

        Args:
            static_context_vars: dict of context tensors for each sample. Pass {} for unconditional generation.
            dynamic_context_vars: dict of dynamic context tensors for each sample.
            n: number of samples to generate; required when static_context_vars is empty.

        Returns:
            Complete generated tensor of shape (N, seq_len, dims).
        """
        bs = self.cfg.model.sampling_batch_size
        if static_context_vars:
            total = len(next(iter(static_context_vars.values())))
        elif n is not None:
            total = n
        else:
            raise ValueError("Pass n= when generating unconditionally (empty static_context_vars)")
        generated_samples = []

        with self.ema_scope():
            for start_idx in tqdm(
                range(0, total, bs),
                unit="seq",
                desc="[CENTS] Generating samples",
                leave=True,
            ):
                end_idx = min(start_idx + bs, total)
                batch_static_context_vars = {
                    var_name: var_tensor[start_idx:end_idx]
                    for var_name, var_tensor in static_context_vars.items()
                }
                batch_dynamic_context_vars = {
                    var_name: var_tensor[start_idx:end_idx]
                    for var_name, var_tensor in (dynamic_context_vars or {}).items()
                }

                current_bs = end_idx - start_idx
                shape = (current_bs, self.seq_len, self.time_series_dims)

                with torch.no_grad():
                    cfg_scale = getattr(self, '_cfg_scale', 1.0)
                    if self.fast_sampling:
                        samples = self.fast_sample(shape, batch_static_context_vars, batch_dynamic_context_vars, cfg_scale=cfg_scale)
                    else:
                        samples = self.sample(shape, batch_static_context_vars, batch_dynamic_context_vars)


                generated_samples.append(samples.cpu())

        return torch.cat(generated_samples, dim=0)
        
    def _ensure_ema_helper(self) -> None:
        """
        Ensure EMA helper is initialized if needed for inference.
        """
        if not hasattr(self, '_ema') or self._ema is None:
            print("Initializing EMA helper for inference...")
            object.__setattr__(self, '_ema', EMA(
                self.model,
                beta=self.cfg.model.ema_decay,
                update_every=self.cfg.model.ema_update_interval,
            ))
    def stratified_timesteps(self, batch_size: int, num_timesteps: int, k_bins: int, device=None) -> torch.Tensor:
        device = device or "cpu"
        k_bins = min(k_bins, batch_size)
        edges = torch.linspace(0, num_timesteps, steps=k_bins + 1, device=device)

        # sample one t per bin
        u = torch.rand(k_bins, device=device)
        t_bins = (edges[:-1] + u * (edges[1:] - edges[:-1])).floor().clamp_(0, num_timesteps - 1).long()

        # repeat to fill batch, then shuffle
        reps = (batch_size + k_bins - 1) // k_bins
        t = t_bins.repeat(reps)[:batch_size]
        t = t[torch.randperm(batch_size, device=device)]
        return t


class EMA(nn.Module):
    def __init__(self, model, beta, update_every):
        super().__init__()
        self.beta = beta
        self.update_every = update_every
        self.step = 0
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # Store as plain python attribute, not nn.Module attribute
        # This prevents it being registered as a submodule and saved in state_dict
        object.__setattr__(self, '_source_model', model)
        
        self.collected_params = []

    def update(self):
        self.step += 1
        if self.step % self.update_every != 0:
            return
        with torch.no_grad():
            for ema_p, src_p in zip(
                self.ema_model.parameters(), 
                self._source_model.parameters()
            ):
                ema_p.data.mul_(self.beta).add_(src_p.data, alpha=1.0 - self.beta)

    def store(self, parameters):
        """
        Save the current parameters (of the live model) to a temporary list.
        Used by the context manager to back up weights before swapping.
        """
        self.collected_params = [param.clone().cpu() for param in parameters]

    def restore(self, parameters):
        """
        Restore the saved parameters back to the live model.
        """
        if not self.collected_params:
            raise RuntimeError("No parameters stored to restore.")
            
        for param, saved_param in zip(parameters, self.collected_params):
            param.data.copy_(saved_param.data.to(param.device))
            
        self.collected_params = [] # Clear memory

    def copy_to(self, parameters):
        """
        Copy the EMA shadow weights INTO the live model parameters.
        """
        for param, ema_param in zip(parameters, self.ema_model.parameters()):
            param.data.copy_(ema_param.data.to(param.device))