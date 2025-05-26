"""
This module defines the core GAN-based models for time-series generation:

- **Generator**: Builds synthetic time-series via transposed convolutions.
- **Discriminator**: Classifies real vs. fake samples, with optional auxiliary context predictions.
- **ACGAN**: PyTorch Lightning wrapper orchestrating adversarial training.

Inspired by the synthetic-timeseries-smart-grid repository (Chi Zhang, MIT License),
with modifications to hyperparameters, network structure, and conditioning logic.
"""

from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from cents.models.base import GenerativeModel
from cents.models.context import ContextModule
from cents.models.model_utils import total_correlation
from cents.models.registry import register_model


class Generator(nn.Module):
    """
    Generator network for time-series data using transposed convolutions.

    Attributes:
        noise_dim (int): Dimensionality of input noise vector.
        embedding_dim (int): Dimensionality of context embedding.
        final_window_length (int): Length of output sequence divided by 8.
        time_series_dims (int): Number of features per time step.
        base_channels (int): Number of channels in the hidden layers.
        context_module (ContextModule): Module to produce context embeddings.
        context_vars (dict): Mapping of context variable names to category counts.
    """

    def __init__(
        self,
        noise_dim: int,
        embedding_dim: int,
        final_window_length: int,
        time_series_dims: int,
        context_module: ContextModule,
        context_vars: Optional[dict] = None,
        base_channels: int = 256,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        # we halve the length in three ConvTranspose layers (×2 each)
        self.final_window_length = final_window_length // 8
        self.time_series_dims = time_series_dims
        self.base_channels = base_channels

        self.context_vars = context_vars
        self.context_module = context_module

        in_dim = noise_dim + (embedding_dim if context_vars else 0)
        self.fc = nn.Linear(in_dim, self.final_window_length * base_channels)

        self.conv_transpose_layers = nn.Sequential(
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels // 2,
                base_channels // 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(base_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels // 4, time_series_dims, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(
        self, noise: torch.Tensor, context_vars: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate synthetic time-series given noise and optional context.

        Args:
            noise (Tensor): Input noise of shape (B, noise_dim).
            context_vars (dict): Mapping from var names to category indices.

        Returns:
            Tensor: Generated series of shape (B, seq_len, time_series_dims).
            dict: Context reconstruction logits for auxiliary loss.
        """
        if context_vars:
            embedding, cond_logits = self.context_module(context_vars)
            x = torch.cat([noise, embedding], dim=1)
        else:
            cond_logits = {}
            x = noise

        x = self.fc(x)
        x = x.view(-1, self.base_channels, self.final_window_length)
        x = self.conv_transpose_layers(x)
        # reshape to (B, L, C)
        x = x.permute(0, 2, 1)
        return x, cond_logits


class Discriminator(nn.Module):
    """
    Discriminator network to distinguish real from fake series and predict context.

    Attributes:
        conv (nn.Sequential): Convolutional feature extractor.
        fc_real_fake (nn.Linear): Real/fake classification head.
        aux_cls (nn.ModuleDict): Auxiliary context classification heads.
    """

    def __init__(
        self,
        window_length: int,
        time_series_dims: int,
        context_var_n_categories: Optional[dict] = None,
        base_channels: int = 256,
    ):
        super().__init__()
        self.context_var_n_categories = context_var_n_categories or {}
        self.conv = nn.Sequential(
            nn.Conv1d(time_series_dims, base_channels // 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels // 4, base_channels // 2, 4, 2, 1),
            nn.BatchNorm1d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels // 2, base_channels, 4, 2, 1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        feat_dim = (window_length // 8) * base_channels
        self.fc_real_fake = nn.Linear(feat_dim, 1)
        self.aux_cls = nn.ModuleDict(
            {
                name: nn.Linear(feat_dim, n_cls)
                for name, n_cls in self.context_var_n_categories.items()
            }
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Classify input series as real/fake and predict context variables.

        Args:
            x (Tensor): Input series of shape (B, L, C).

        Returns:
            Tensor: Real/fake logits of shape (B,1).
            dict: Auxiliary logits per context variable.
        """
        # convert to (B, C, L)
        feat = self.conv(x.permute(0, 2, 1)).flatten(1)
        rf_logits = self.fc_real_fake(feat)
        aux_logits = {k: head(feat) for k, head in self.aux_cls.items()}
        return rf_logits, aux_logits


@register_model("acgan", "Watts_1_1D", "Watts_1_2D")
class ACGAN(GenerativeModel):
    """
    Auxiliary Classifier GAN LightningModule combining Generator and Discriminator.

    Args:
        cfg (DictConfig): Hydra configuration mapping model and trainer hyperparameters.

    Methods:
        forward(noise, ctx): produce synthetic series and context logits.
        training_step(batch, idx): one iteration of GAN training (G then D).
        configure_optimizers(): set up separate optimizers for G and D.
        generate(ctx): batched generation of synthetic series.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.automatic_optimization = False

        # shared context embedding + classification
        # self.context_module = ContextModule(
        #     cfg.dataset.context_vars, cfg.model.cond_emb_dim
        # )
        self.generator = Generator(
            noise_dim=cfg.model.noise_dim,
            embedding_dim=cfg.model.cond_emb_dim,
            final_window_length=cfg.dataset.seq_len,
            time_series_dims=cfg.dataset.time_series_dims,
            context_module=self.context_module,
            context_vars=cfg.dataset.context_vars,
        )
        self.discriminator = Discriminator(
            window_length=cfg.dataset.seq_len,
            time_series_dims=cfg.dataset.time_series_dims,
            context_var_n_categories=cfg.dataset.context_vars,
        )
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.aux_loss = nn.CrossEntropyLoss()

    def forward(self, noise: torch.Tensor, context_vars: dict):
        """
        Proxy forward to the Generator.

        Returns:
            see Generator.forward
        """
        return self.generator(noise, context_vars)

    def configure_optimizers(self):
        """
        Set up two Adam optimizers: one for G, one for D.

        Returns:
            Tuple[List[Optimizer], list]: ([opt_G, opt_D], [])
        """
        opt_G = optim.Adam(
            self.generator.parameters(),
            lr=self.cfg.trainer.optimizer.generator.lr,
            betas=self.cfg.trainer.optimizer.generator.betas,
        )
        opt_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.trainer.optimizer.discriminator.lr,
            betas=self.cfg.trainer.optimizer.discriminator.betas,
        )
        return [opt_G, opt_D], []

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """
        Perform one GAN training iteration: Generator update, then Discriminator.
        Logs `loss_G` and `loss_D`.
        """
        ts_real, ctx = batch
        bsz = ts_real.size(0)
        noise = torch.randn(bsz, self.cfg.model.noise_dim, device=self.device)

        opt_G, opt_D = self.optimizers()
        smooth_pos, smooth_neg = 0.95, 0.0

        # Generator step
        opt_G.zero_grad()
        ts_fake, logits_ctx = self.generator(noise, ctx)
        logits_fake, aux_fake = self.discriminator(ts_fake)
        g_adv = self.adv_loss(logits_fake, torch.full_like(logits_fake, smooth_pos))
        g_aux = (
            sum(self.aux_loss(aux_fake[v], ctx[v]) for v in aux_fake)
            if self.cfg.model.include_auxiliary_losses
            else 0.0
        )
        g_ctx = sum(self.aux_loss(logits_ctx[v], ctx[v]) for v in logits_ctx)

        h, _ = self.context_module(ctx)
        tc_term = (
            self.cfg.model.tc_loss_weight * total_correlation(h)
            if self.cfg.model.tc_loss_weight > 0.0
            else torch.tensor(0.0, device=self.device)
        )
        self.log_dict(
            {"g_adv": g_adv, "g_aux": g_aux, "g_ctx": g_ctx, "tc_loss": tc_term},
            prog_bar=True,
            on_step=True,
        )

        g_total = (
            g_adv
            + g_aux
            + self.cfg.model.context_reconstruction_loss_weight * g_ctx
            + tc_term
        )

        self.manual_backward(g_total)
        opt_G.step()
        self.log("loss_G", g_total, prog_bar=True, on_step=True)

        # Discriminator step
        opt_D.zero_grad()
        logits_real, aux_real = self.discriminator(ts_real)
        d_real = self.adv_loss(logits_real, torch.full_like(logits_real, smooth_pos))
        logits_fake_d, aux_fake_d = self.discriminator(ts_fake.detach())
        d_fake = self.adv_loss(
            logits_fake_d, torch.full_like(logits_fake_d, smooth_neg)
        )
        d_aux = 0.0
        if self.cfg.model.include_auxiliary_losses:
            for v in aux_real:
                d_aux += self.aux_loss(aux_real[v], ctx[v])
                d_aux += self.aux_loss(aux_fake_d[v], ctx[v])

        d_total = d_real + d_fake + d_aux
        self.manual_backward(d_total)
        opt_D.step()
        self.log("loss_D", d_total, prog_bar=True, on_step=True)

    @torch.no_grad()
    def generate(self, context_vars: dict) -> torch.Tensor:
        """
        Generate synthetic series in batches from stored context.

        Args:
            context_vars (dict): Mapping of variables to Tensors of length N.

        Returns:
            Tensor: Generated series of shape (N, seq_len, time_series_dims).
        """
        batch_size = self.cfg.model.sampling_batch_size
        total = len(next(iter(context_vars.values())))
        results = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            sub_ctx = {k: v[start:end].to(self.device) for k, v in context_vars.items()}
            noise = torch.randn(
                end - start, self.cfg.model.noise_dim, device=self.device
            )
            ts_batch, _ = self.generator(noise, sub_ctx)
            results.append(ts_batch)
        return torch.cat(results, dim=0)
