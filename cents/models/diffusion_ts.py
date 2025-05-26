import copy
import math
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from cents.models.base import GenerativeModel
from cents.models.model_utils import (
    Transformer,
    cosine_beta_schedule,
    default,
    linear_beta_schedule,
    total_correlation,
)
from cents.models.registry import register_model


@register_model("diffusion_ts", "Watts_2_1D", "Watts_2_2D")
class Diffusion_TS(GenerativeModel):
    """
    PyTorch-Lightning module for a conditional time-series diffusion model.

    Uses a Transformer backbone to predict and denoise time series over
    discrete diffusion timesteps. Supports EMA smoothing and configurable
    beta schedules.
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
        _ = self.context_module

        # linear layer for denoised output
        self.fc = nn.Linear(
            self.time_series_dims + self.embedding_dim, self.time_series_dims
        )
        # Transformer backbone
        self.model = Transformer(
            n_feat=self.time_series_dims + self.embedding_dim,
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
        )

        # EMA helper will be initialized on train start
        self._ema_helper: Optional[EMA] = None

        # set up beta schedule
        if cfg.model.beta_schedule == "linear":
            betas = linear_beta_schedule(cfg.model.n_steps)
        elif cfg.model.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.model.n_steps)
        else:
            raise ValueError("Unknown beta schedule")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.num_timesteps = betas.shape[0]
        self.sampling_timesteps = default(
            cfg.model.sampling_timesteps, self.num_timesteps
        )
        self.fast_sampling = self.sampling_timesteps < self.num_timesteps
        self.loss_type = cfg.model.loss_type

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

        lw = torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100
        self.register_buffer("loss_weight", lw)

        # choose reconstruction loss
        if self.loss_type == "l1":
            self.recon_loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.recon_loss_fn = F.mse_loss
        else:
            raise ValueError("Invalid loss type")

        self.auxiliary_loss = nn.CrossEntropyLoss()

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
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t - x0
        ) / self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)

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
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t
            - self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise
        )

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

    def forward(self, x: torch.Tensor, context_vars: dict) -> Tuple[torch.Tensor, dict]:
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
        embedding, cond_classification_logits = self.context_module(context_vars)
        noise = torch.randn_like(x)
        x_noisy = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x
            + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
        )
        c = torch.cat(
            [x_noisy, embedding.unsqueeze(1).repeat(1, self.seq_len, 1)],
            dim=-1,
        )
        trend, season = self.model(c, t, padding_masks=None)
        x_recon = self.fc(trend + season)
        rec_loss = self.recon_loss_fn(x_recon, x)
        return rec_loss, cond_classification_logits

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step: compute total loss including context reconstruction.

        Args:
            batch: Tuple(ts_batch, cond_batch).
            batch_idx: Index of batch (unused).

        Returns:
            total_loss: Scalar training loss.
        """
        ts_batch, cond_batch = batch
        rec_loss, cond_class_logits = self(ts_batch, cond_batch)
        cond_loss = 0.0

        for var_name, logits in cond_class_logits.items():
            labels = cond_batch[var_name]
            cond_loss += self.auxiliary_loss(logits, labels)

        h, _ = self.context_module(cond_batch)
        tc_term = (
            self.cfg.model.tc_loss_weight * total_correlation(h)
            if self.cfg.model.tc_loss_weight > 0.0
            else torch.tensor(0.0, device=self.device)
        )

        total_loss = (
            rec_loss + self.context_reconstruction_loss_weight * cond_loss + tc_term
        )
        self.log_dict(
            {
                "train_loss": total_loss,
                "rec_loss": rec_loss,
                "cond_loss": cond_loss,
                "tc_loss": tc_term,
            },
            prog_bar=True,
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
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def on_train_start(self) -> None:
        """
        Initialize EMA helper at start of training.
        """
        self._ema = EMA(
            self.model,
            beta=self.cfg.model.ema_decay,
            update_every=self.cfg.model.ema_update_interval,
        )

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """
        Apply EMA update after each batch end.
        """
        if self._ema_helper:
            self._ema_helper.update()

    @torch.no_grad()
    def model_predictions(
        self, x: torch.Tensor, t: torch.Tensor, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict both noise and clean sample from current x.

        Returns:
            pred_noise: predicted noise tensor.
            x_start: predicted clean sample tensor.
        """
        c = torch.cat([x, embedding.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1)
        trend, season = self.model(c, t, padding_masks=None)
        x_start = self.fc(trend + season)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    @torch.no_grad()
    def p_mean_variance(
        self, x: torch.Tensor, t: torch.Tensor, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).

        Returns:
            pm, pv, plv, x_start: posterior parameters and predicted x0.
        """
        pred_noise, x_start = self.model_predictions(x, t, embedding)
        pm, pv, plv = self.q_posterior(x_start, x, t)
        return pm, pv, plv, x_start

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, t: int, embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t using posterior distribution.
        """
        bt = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        pm, pv, plv, _ = self.p_mean_variance(x, bt, embedding)
        noise = torch.randn_like(x) if t > 0 else 0
        return pm + (0.5 * plv).exp() * noise

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int], context_vars: dict) -> torch.Tensor:
        """
        Full reverse-pass sampling over all timesteps.

        Args:
            shape: (batch_size, seq_len, dims)
            context_vars: context conditioning dict

        Returns:
            Generated samples tensor.
        """
        x = torch.randn(shape, device=self.device)
        embedding, _ = self.context_module(context_vars)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, embedding)
        return x

    @torch.no_grad()
    def fast_sample(
        self, shape: Tuple[int, int, int], context_vars: dict
    ) -> torch.Tensor:
        """
        Faster sampling using a reduced number of timesteps.
        """
        x = torch.randn(shape, device=self.device)
        embedding, _ = self.context_module(context_vars)
        times = torch.linspace(
            -1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1
        )
        times = list(reversed(times.int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))
        for time, time_next in pairs:
            bt = torch.full((x.shape[0],), time, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(x, bt, embedding)
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
            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return x

    def generate(self, context_vars: dict) -> torch.Tensor:
        """
        Public entry to generate conditioned samples in batches.

        Args:
            context_vars: dict of context tensors for each sample.

        Returns:
            Complete generated tensor of shape (N, seq_len, dims).
        """
        bs = self.cfg.model.sampling_batch_size
        total = len(next(iter(context_vars.values())))
        generated_samples = []

        for start_idx in tqdm(
            range(0, total, bs),
            unit="seq",
            desc="[CENTS] Generating samples",
            leave=True,
        ):
            end_idx = min(start_idx + bs, total)
            batch_context_vars = {
                var_name: var_tensor[start_idx:end_idx]
                for var_name, var_tensor in context_vars.items()
            }
            current_bs = end_idx - start_idx
            shape = (current_bs, self.seq_len, self.time_series_dims)

            with torch.no_grad():
                if getattr(self.cfg.model, "use_ema_sampling", False) and hasattr(
                    self, "_ema_helper"
                ):
                    samples = self._ema_helper.ema_model._generate(
                        shape, batch_context_vars
                    )
                else:
                    samples = (
                        self.fast_sample(shape, batch_context_vars)
                        if self.fast_sampling
                        else self.sample(shape, batch_context_vars)
                    )

            generated_samples.append(samples)

        return torch.cat(generated_samples, dim=0)


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) helper for model parameters.

    Maintains a shadow copy of the model weights that are updated
    via EMA every `update_every` steps.
    """

    def __init__(self, model: nn.Module, beta: float, update_every: int):
        """
        Args:
            model: Base model to copy for EMA tracking.
            beta: EMA decay rate (0 < beta < 1).
            update_every: Frequency (in steps) to apply EMA update.
        """
        super().__init__()
        self.model = copy.deepcopy(model)
        self.ema_model = self.model.eval()
        self.beta = beta
        self.update_every = update_every
        self.step = 0
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self) -> None:
        """
        Perform an EMA update of the shadow model parameters.
        Called typically at end of each training batch.
        """
        self.step += 1
        if self.step % self.update_every != 0:
            return
        with torch.no_grad():
            for ema_p, model_p in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_p.data.mul_(self.beta).add_(model_p.data, alpha=1.0 - self.beta)
